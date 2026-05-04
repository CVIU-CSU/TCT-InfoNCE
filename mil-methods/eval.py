import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from dataloader import *
from modules import attmil, clam, dsmil, mean_max, mhim, transmil
from utils import *


TARGET_SENSITIVITIES = (0.9444, 0.9784)


def parse_tct_dataset(args, file_name: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    test_label_path = os.path.join(args.label_path, file_name)
    patients, labels = [], []
    with open(test_label_path, "r") as handle:
        for line in handle.readlines():
            patients.append(line.split(",")[0])
            labels.append(line.split(",")[1])
    return [np.array(patients)], [np.array(labels)]


def build_loaders(args):
    k = 0
    if args.datasets.lower() != "tct":
        print("##info##: 暂时只支持tct数据集，tct为通用的tct-ngc和tct-gc")
        return None, None, None

    test_p, test_l = parse_tct_dataset(args, "test_label.csv")
    test_set = C16Dataset(
        test_p[k],
        test_l[k],
        root=args.dataset_root,
        persistence=args.persistence,
        keep_same_psize=args.same_psize,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_c_loader, test_h_loader = None, None
    if args.c_h:
        test_c_p, test_c_l = parse_tct_dataset(args, "test_c.csv")
        test_c_set = C16Dataset(
            test_c_p[k],
            test_c_l[k],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
        )
        test_c_loader = DataLoader(test_c_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        test_h_p, test_h_l = parse_tct_dataset(args, "test_h.csv")
        test_h_set = C16Dataset(
            test_h_p[k],
            test_h_l[k],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
        )
        test_h_loader = DataLoader(test_h_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader, test_c_loader, test_h_loader


def build_model(args, device):
    if args.model == "mhim":
        model = mhim.MHIM(
            input_dim=args.input_dim,
            baseline=args.baseline,
            dropout=args.dropout,
            mask_ratio=args.mask_ratio,
            n_classes=args.n_classes,
            temp_t=args.temp_t,
            act=args.act,
            head=args.n_heads,
            msa_fusion=args.msa_fusion,
            mask_ratio_h=args.mask_ratio_h,
            mask_ratio_hr=args.mask_ratio_hr,
            mask_ratio_l=args.mask_ratio_l,
            mrh_sche=None,
            da_act=args.da_act,
            attn_layer=args.attn_layer,
        ).to(device)
    elif args.model == "pure":
        model = mhim.MHIM(
            input_dim=args.input_dim,
            select_mask=False,
            n_classes=args.n_classes,
            act=args.act,
            head=args.n_heads,
            da_act=args.da_act,
            baseline=args.baseline,
        ).to(device)
    elif args.model == "attmil":
        model = attmil.DAttention(n_classes=args.n_classes, dropout=args.dropout, act=args.act).to(device)
    elif args.model == "gattmil":
        model = attmil.AttentionGated(dropout=args.dropout).to(device)
    elif args.model == "clam_sb":
        model = clam.CLAM_SB(n_classes=args.n_classes, dropout=args.dropout, act=args.act).to(device)
    elif args.model == "clam_mb":
        model = clam.CLAM_MB(n_classes=args.n_classes, dropout=args.dropout, act=args.act).to(device)
    elif args.model == "transmil":
        model = transmil.TransMIL(n_classes=args.n_classes, dropout=args.dropout, act=args.act).to(device)
    elif args.model == "dsmil":
        model = dsmil.MILNet(n_classes=args.n_classes, dropout=args.dropout, act=args.act).to(device)
        state_dict_weights = torch.load("./modules/init_cpk/dsmil_init.pth")
        info = model.load_state_dict(state_dict_weights, strict=False)
        if not args.no_log:
            print(info)
    elif args.model == "meanmil":
        model = mean_max.MeanMIL(n_classes=args.n_classes, dropout=args.dropout, act=args.act, input_dim=args.input_dim).to(device)
    elif args.model == "maxmil":
        model = mean_max.MaxMIL(n_classes=args.n_classes, dropout=args.dropout, act=args.act, input_dim=args.input_dim).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model


def _binary_metrics_at_threshold(probs_pos: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, Any]:
    labels = labels.astype(int).reshape(-1)
    probs_pos = probs_pos.reshape(-1)
    preds = (probs_pos >= threshold).astype(int)

    sens = recall_score(labels, preds, pos_label=1, average="binary")
    spec = recall_score(labels, preds, pos_label=0, average="binary", zero_division=0)
    prec = precision_score(labels, preds, pos_label=1, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "threshold": float(threshold),
        "recall": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "accuracy": float(acc),
        "f1": float(f1),
        "confusion_mat": cm,
    }


def compute_metrics_with_target_rounded_sensitivity(
    probs: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    target_sensitivity: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> Dict[str, Any]:
    probs_arr = np.asarray(probs)
    labels_arr = np.asarray(labels).astype(int).reshape(-1)

    if probs_arr.ndim == 1:
        probs_pos = probs_arr
    elif probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
        probs_pos = probs_arr[:, 1]
    else:
        raise ValueError("compute_metrics_with_target_rounded_sensitivity 目前仅支持二分类")

    lo, hi = 0.0, 1.0
    best_metrics = None
    best_diff = 1.0
    target_rounded = round(target_sensitivity * 10000) / 10000.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        metrics = _binary_metrics_at_threshold(probs_pos, labels_arr, mid)
        sens = metrics["recall"]
        sens_rounded = round(sens * 10000) / 10000.0
        diff = abs(sens_rounded - target_rounded)

        if diff < best_diff:
            best_diff = diff
            best_metrics = metrics

        if sens < target_sensitivity:
            hi = mid
        else:
            lo = mid

        if diff <= tol:
            break

    result = dict(best_metrics)
    result["target_sensitivity"] = float(target_sensitivity)
    result["rounded_recall"] = round(result["recall"] * 10000) / 10000.0
    return result


def forward_eval_logits(args, model, bag):
    if args.model in ("mhim", "pure"):
        return model.forward_test(bag)
    if args.model == "dsmil":
        logits, _ = model(bag)
        return logits
    return model(bag)


def collect_probabilities(args, logits) -> List[float]:
    if args.loss == "ce":
        if (args.model == "dsmil" and args.ds_average) or (args.model == "mhim" and isinstance(logits, (list, tuple))):
            probs = 0.5 * torch.softmax(logits[1], dim=-1) + 0.5 * torch.softmax(logits[0], dim=-1)
            return probs[:, 1].detach().cpu().numpy().reshape(-1).tolist()
        probs = torch.softmax(logits, dim=-1)
        if args.n_classes == 2:
            return probs[:, 1].detach().cpu().numpy().reshape(-1).tolist()
        return probs.detach().cpu().numpy().tolist()

    if args.model == "dsmil" and args.ds_average:
        probs = 0.5 * torch.sigmoid(logits[1]) + 0.5 * torch.sigmoid(logits[0])
    else:
        probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy().reshape(-1).tolist()


def gather_outputs(args, model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    bag_probs, bag_labels = [], []
    model.eval()

    with torch.no_grad():
        for data in loader:
            labels = data[1]
            if len(labels) > 1:
                bag_labels.extend(labels.tolist())
            else:
                bag_labels.append(labels.item())

            if isinstance(data[0], (list, tuple)):
                for idx in range(len(data[0])):
                    data[0][idx] = data[0][idx].to(device)
                bag = data[0]
            else:
                bag = data[0].to(device)

            logits = forward_eval_logits(args, model, bag)
            probs = collect_probabilities(args, logits)
            bag_probs.extend(probs)

    return np.asarray(bag_labels).astype(int), np.asarray(bag_probs)


def compute_default_metrics(args, labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    if args.n_classes != 2:
        auc_value, accuracy, recall, precision, fscore = multi_class_scores(labels.tolist(), probs.tolist())
        return {
            "accuracy": float(accuracy),
            "auc": float(auc_value),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": 0.0,
            "f1": float(fscore),
            "threshold": None,
        }

    threshold = args.threshold
    if threshold == 0:
        fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
        _, _, threshold = optimal_thresh(fpr, tpr, thresholds)
    metrics = _binary_metrics_at_threshold(probs, labels, threshold)
    metrics["auc"] = float(roc_auc_score(labels, probs))
    return metrics


def maybe_dump_roc(args, labels: np.ndarray, probs: np.ndarray, seed_tag: str):
    if not args.output_auc or args.n_classes != 2:
        return
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
    auc_value = roc_auc_score(labels, probs)
    roc_output_dir = "output_roc"
    feat_dir = args.dataset_root.split("/")[-1]
    roc_par_dir = os.path.join(roc_output_dir, feat_dir, seed_tag)
    os.makedirs(roc_par_dir, exist_ok=True)
    np.save(os.path.join(roc_par_dir, "fpr.npy"), fpr)
    np.save(os.path.join(roc_par_dir, "tpr.npy"), tpr)
    np.save(os.path.join(roc_par_dir, "auc.npy"), auc_value)


def summarize_numeric_dicts(records: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key in keys:
        values = [float(record[key]) for record in records]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return summary


def format_metric_block(metrics: Dict[str, Any], prefix: str = "") -> str:
    threshold = metrics.get("threshold")
    threshold_str = "None" if threshold is None else f"{float(threshold):.6f}"
    return (
        f"{prefix}threshold={threshold_str}, "
        f"acc={float(metrics['accuracy']):.4f}, "
        f"auc={float(metrics['auc']):.4f}, "
        f"precision={float(metrics['precision']):.4f}, "
        f"recall={float(metrics['recall']):.4f}, "
        f"specificity={float(metrics['specificity']):.4f}, "
        f"f1={float(metrics['f1']):.4f}"
    )


def resolve_seed_list(args) -> List[int]:
    if args.seeds:
        return [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    discovered = []
    if os.path.isdir(args.ckp_path):
        for name in os.listdir(args.ckp_path):
            if name.startswith("seed_"):
                try:
                    discovered.append(int(name.split("_", 1)[1]))
                except ValueError:
                    continue
    return sorted(discovered) if discovered else [args.seed]


def resolve_checkpoint_path(base_path: str, seed: int, fold: int) -> Tuple[str, str]:
    candidates = [
        os.path.join(base_path, f"seed_{seed}", f"fold_{fold}", f"fold_{fold}_model_best_auc.pt"),
        os.path.join(base_path, f"seed_{seed}", f"fold_{fold}_model_best_auc.pt"),
        os.path.join(base_path, f"fold_{fold}", f"fold_{fold}_model_best_auc.pt"),
        os.path.join(base_path, f"fold_{fold}_model_best_auc.pt"),
        base_path,
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate, f"seed_{seed}"
    raise FileNotFoundError(f"Can not find checkpoint for seed={seed}, fold={fold} under {base_path}")


def evaluate_single_seed(args, seed: int, model, test_loader, test_c_loader, test_h_loader, device) -> Dict[str, Any]:
    checkpoint_path, seed_tag = resolve_checkpoint_path(args.ckp_path, seed, args.fold)
    best_std = torch.load(checkpoint_path, map_location=device)
    state_dict = best_std["model"] if isinstance(best_std, dict) and "model" in best_std else best_std
    info = model.load_state_dict(state_dict, strict=False)
    if not args.no_log:
        print(f"\n===== Seed {seed} =====")
        print(f"checkpoint: {checkpoint_path}")
        print(info)

    labels, probs = gather_outputs(args, model, test_loader, device)
    default_metrics = compute_default_metrics(args, labels, probs)
    maybe_dump_roc(args, labels, probs, seed_tag)

    target_metrics = []
    if args.n_classes == 2:
        for target_sensitivity in TARGET_SENSITIVITIES:
            metrics = compute_metrics_with_target_rounded_sensitivity(probs, labels, target_sensitivity)
            metrics["auc"] = float(default_metrics["auc"])
            target_metrics.append(metrics)

    ch_metrics = None
    if args.c_h:
        ch_metrics = {}
        if test_c_loader is not None:
            c_labels, c_probs = gather_outputs(args, model, test_c_loader, device)
            ch_metrics["sen_c"] = float(_binary_metrics_at_threshold(c_probs, c_labels, args.threshold)["recall"])
        if test_h_loader is not None:
            h_labels, h_probs = gather_outputs(args, model, test_h_loader, device)
            ch_metrics["sen_h"] = float(_binary_metrics_at_threshold(h_probs, h_labels, args.threshold)["recall"])

    if not args.no_log:
        print(format_metric_block(default_metrics, prefix="default: "))
        for metrics in target_metrics:
            print(
                f"target_sens={metrics['target_sensitivity']:.4f}, "
                f"rounded_recall={metrics['rounded_recall']:.4f}, "
                f"threshold={metrics['threshold']:.6f}, "
                f"specificity={metrics['specificity']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"f1={metrics['f1']:.4f}"
            )
        if ch_metrics is not None:
            print(ch_metrics)

    return {
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "default_metrics": default_metrics,
        "target_metrics": target_metrics,
        "ch_metrics": ch_metrics,
    }


def print_multi_seed_summary(seed_results: List[Dict[str, Any]]):
    default_records = [result["default_metrics"] for result in seed_results]
    default_summary = summarize_numeric_dicts(default_records, ["accuracy", "auc", "precision", "recall", "specificity", "f1"])
    print("\n===== Multi-seed summary =====")
    for key, stats in default_summary.items():
        print(f"default/{key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    for target_sensitivity in TARGET_SENSITIVITIES:
        matched = []
        for result in seed_results:
            for metrics in result["target_metrics"]:
                if abs(metrics["target_sensitivity"] - target_sensitivity) < 1e-8:
                    matched.append(metrics)
                    break
        if not matched:
            continue
        summary = summarize_numeric_dicts(matched, ["threshold", "precision", "recall", "rounded_recall", "specificity", "accuracy", "f1"])
        print(f"\nfixed sensitivity {target_sensitivity:.4f}")
        for key, stats in summary.items():
            print(f"{key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")


def main(args):
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed_torch(args.seed)

    test_loader, test_c_loader, test_h_loader = build_loaders(args)
    if test_loader is None:
        return

    seeds = resolve_seed_list(args)
    if not args.no_log:
        print("evaluate seeds:", seeds)

    seed_results = []
    for seed in seeds:
        model = build_model(args, device)
        seed_results.append(evaluate_single_seed(args, seed, model, test_loader, test_c_loader, test_h_loader, device))

    print_multi_seed_summary(seed_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIL eval script")

    parser.add_argument("--datasets", default="tct", type=str, help="[camelyon16, tcga, tct]")
    parser.add_argument("--dataset_root", default="/home1/wsi/gc-all-features/contrastive/clip3/pt", type=str, help="Dataset root path")
    parser.add_argument("--label_path", default="datatools/gc/labels", type=str, help="label path root")
    parser.add_argument("--fix_loader_random", action="store_true", help="Fix random seed of dataloader")
    parser.add_argument("--fix_train_random", action="store_true", help="Fix random seed of training")
    parser.add_argument("--persistence", action="store_true", help="Load data into memory")
    parser.add_argument("--same_psize", default=0, type=int, help="Keep the same size of all patches")

    parser.add_argument("--ds_average", action="store_true", help="DSMIL hyperparameter")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of batch size")
    parser.add_argument("--loss", default="ce", type=str, help="Classification Loss [ce, bce]")
    parser.add_argument("--n_classes", default=2, type=int, help="Number of classes")
    parser.add_argument("--model", default="pure", type=str, help="Model name")
    parser.add_argument("--seed", default=2024, type=int, help="fallback single seed")
    parser.add_argument("--seeds", default="", type=str, help="comma separated seeds, e.g. 2023,2024,2025")
    parser.add_argument("--fold", default=0, type=int, help="fold index to evaluate")
    parser.add_argument("--baseline", default="attn", type=str, help="Baselin model [attn,selfattn]")
    parser.add_argument("--act", default="relu", type=str, help="Activation func in the projection head [gelu,relu]")
    parser.add_argument("--dropout", default=0.25, type=float, help="Dropout in the projection head")
    parser.add_argument("--n_heads", default=8, type=int, help="Number of head in the MSA")
    parser.add_argument("--da_act", default="relu", type=str, help="Activation func in the DAttention [gelu,relu]")
    parser.add_argument("--input_dim", default=512, type=int, help="The dimention of patch feature")

    parser.add_argument("--patch_shuffle", action="store_true", help="2-D group shuffle")
    parser.add_argument("--group_shuffle", action="store_true", help="Group shuffle")
    parser.add_argument("--shuffle_group", default=0, type=int, help="Number of the shuffle group")

    parser.add_argument("--mask_ratio", default=0.0, type=float, help="Random mask ratio")
    parser.add_argument("--mask_ratio_l", default=0.0, type=float, help="Low attention mask ratio")
    parser.add_argument("--mask_ratio_h", default=0.0, type=float, help="High attention mask ratio")
    parser.add_argument("--mask_ratio_hr", default=1.0, type=float, help="Randomly high attention mask ratio")
    parser.add_argument("--mrh_sche", action="store_true", help="Decay of HAM")
    parser.add_argument("--msa_fusion", default="vote", type=str, help="[mean,vote]")
    parser.add_argument("--attn_layer", default=0, type=int)

    parser.add_argument("--cl_alpha", default=0.0, type=float, help="Auxiliary loss alpha")
    parser.add_argument("--temp_t", default=0.1, type=float, help="Temperature")
    parser.add_argument("--teacher_init", default="none", type=str, help="Path to initial teacher model")
    parser.add_argument("--no_tea_init", action="store_true", help="Without teacher initialization")
    parser.add_argument("--init_stu_type", default="none", type=str, help="Student initialization [none,fc,all]")
    parser.add_argument("--tea_type", default="none", type=str, help="[none,same]")
    parser.add_argument("--mm", default=0.9999, type=float, help="Ema decay [0.9997]")
    parser.add_argument("--mm_final", default=1.0, type=float, help="Final ema decay [1.]")
    parser.add_argument("--mm_sche", action="store_true", help="Cosine schedule of ema decay")

    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers in the dataloader")
    parser.add_argument("--threshold", default=0.0, type=float, help="the threshold of classification, 0 means auto threshold")
    parser.add_argument("--no_log", action="store_true", help="Without log")
    parser.add_argument("--c_h", action="store_true")
    parser.add_argument("--output_auc", action="store_true")
    parser.add_argument(
        "--ckp_path",
        type=str,
        default="/home/huangjialong/projects/TCT-InfoNCE/mil-methods/output-model/mil-methods-info/clip3-mhim(abmil)-gc-trainval-3seed",
        help="Checkpoint root path or a single checkpoint file",
    )

    main(parser.parse_args())
