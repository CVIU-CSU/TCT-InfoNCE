import argparse
import os
import random
import time
from contextlib import suppress
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, RandomSampler

from dataloader import C16Dataset, GcDataset, TCGADataset, get_kflod, get_patient_label
from modules import attmil, clam, dsmil, mean_max, mhim, transmil, vit
from timm.models import model_parameters
from timm.utils import AverageMeter, dispatch_clip_grad
from utils import (
    EarlyStopping,
    cosine_scheduler,
    ema_update,
    group_shuffle,
    multi_class_scores,
    patch_shuffle,
    seed_torch,
    six_scores,
)


SEEDS = [2023, 2024, 2025]
METRIC_KEYS = ("acc", "spec", "sen", "f1", "auc")
FIXED_SPLIT_DATASETS = {"ngc", "gc", "fnac"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values))


def summarize_metrics(metric_items):
    summary = {}
    for key in METRIC_KEYS:
        metric_values = [item[key] for item in metric_items]
        summary[f"{key}_mean"], summary[f"{key}_std"] = mean_std(metric_values)
    return summary


def format_metric(value):
    return -1.0 if isinstance(value, float) and np.isnan(value) else value


def save_dataframe(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_excel(path, index=False)


def save_fold_log(seed_dir, fold_idx, seed, args, records):
    fold_dir = os.path.join(seed_dir, f"fold_{fold_idx}")
    ensure_dir(fold_dir)
    log_path = os.path.join(
        fold_dir,
        f"Useful_Log_seed{seed}_{args.datasets}_{args.model}.xlsx",
    )
    save_dataframe(
        log_path,
        records,
        ["epoch", "train_loss", "val_loss", "val_acc", "val_spec", "val_sen", "val_f1", "val_auc"],
    )


def save_seed_summary(seed_dir, fold_metrics, fold_count):
    rows = []
    for fold_metric in fold_metrics:
        rows.append([f"fold_{fold_metric['fold']}"] + [fold_metric[key] for key in METRIC_KEYS])

    summary = summarize_metrics(fold_metrics)
    rows.append(["mean"] + [summary[f"{key}_mean"] for key in METRIC_KEYS])
    rows.append(["std"] + [summary[f"{key}_std"] for key in METRIC_KEYS])

    save_dataframe(
        os.path.join(seed_dir, f"useful_{fold_count}_fold_metrics.xlsx"),
        rows,
        ["fold", "acc", "spec", "sen", "f1", "auc"],
    )
    return summary


def save_seed_aggregate(root_dir, seed_summaries):
    rows = []
    for seed_summary in seed_summaries:
        rows.append(
            [f"seed_{seed_summary['seed']}"] + [seed_summary[f"{key}_mean"] for key in METRIC_KEYS]
        )

    rows.append(["mean"] + [mean_std([item[f"{key}_mean"] for item in seed_summaries])[0] for key in METRIC_KEYS])
    rows.append(["std"] + [mean_std([item[f"{key}_mean"] for item in seed_summaries])[1] for key in METRIC_KEYS])

    save_dataframe(
        os.path.join(root_dir, f"useful_{len(seed_summaries)}_seed_metrics.xlsx"),
        rows,
        ["seed", "acc", "spec", "sen", "f1", "auc"],
    )


def print_seed_summary(seed_summary):
    print(
        "Seed {} summary: acc {:.4f}±{:.4f}, spec {:.4f}±{:.4f}, sen {:.4f}±{:.4f}, f1 {:.4f}±{:.4f}, auc {:.4f}±{:.4f}".format(
            seed_summary["seed"],
            seed_summary["acc_mean"],
            seed_summary["acc_std"],
            seed_summary["spec_mean"],
            seed_summary["spec_std"],
            seed_summary["sen_mean"],
            seed_summary["sen_std"],
            seed_summary["f1_mean"],
            seed_summary["f1_std"],
            seed_summary["auc_mean"],
            seed_summary["auc_std"],
        )
    )


def print_final_summary(seed_summaries):
    print("\n===== Final 3-seed summary =====")
    for key in METRIC_KEYS:
        metric_mean, metric_std = mean_std([item[f"{key}_mean"] for item in seed_summaries])
        print(f"{key:<5}: {metric_mean:.4f} ± {metric_std:.4f}")


def read_split_csv(path):
    patients, labels = [], []
    with open(path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            patient, label = line.strip().split(",")[:2]
            patients.append(patient)
            labels.append(label)
    return [np.array(patients)], [np.array(labels)]


def shuffle_patient_labels(patients, labels):
    indices = np.arange(len(patients))
    random.shuffle(indices)
    return patients[indices], labels[indices]


def prepare_split_data(args):
    dataset_name = args.datasets.lower()

    if dataset_name in {"camelyon16", "tcga"}:
        label_path = os.path.join(args.dataset_root, "label.csv")
        patients, labels = get_patient_label(label_path)
        patients, labels = shuffle_patient_labels(patients, labels)
        return patients, labels, None

    if dataset_name in FIXED_SPLIT_DATASETS:
        train_p, train_l = read_split_csv(args.train_label_path)
        val_p, val_l = read_split_csv(args.val_label_path)
        test_p, test_l = read_split_csv(args.test_label_path)
        return None, None, (train_p, train_l, test_p, test_l, val_p, val_l)

    raise ValueError(f"Unsupported dataset: {args.datasets}")


def build_fold_split(args, split_data):
    patients, labels, preset_split = split_data
    if preset_split is not None:
        return preset_split
    if args.cv_fold > 1:
        return get_kflod(args.cv_fold, patients, labels, args.val_ratio)
    return [patients], [labels], [patients], [labels], [patients], [labels]


def build_datasets(args, fold_idx, split_data):
    train_p, train_l, test_p, test_l, val_p, val_l = split_data
    dataset_name = args.datasets.lower()

    if dataset_name == "camelyon16":
        train_set = C16Dataset(
            train_p[fold_idx],
            train_l[fold_idx],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
            is_train=True,
        )
        test_set = C16Dataset(
            test_p[fold_idx],
            test_l[fold_idx],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
        )
        val_set = (
            C16Dataset(
                val_p[fold_idx],
                val_l[fold_idx],
                root=args.dataset_root,
                persistence=args.persistence,
                keep_same_psize=args.same_psize,
            )
            if args.val_ratio != 0.0
            else test_set
        )
        return train_set, val_set, test_set

    if dataset_name in {"ngc", "fnac"}:
        train_set = C16Dataset(
            train_p[fold_idx],
            train_l[fold_idx],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
            is_train=True,
        )
        test_set = C16Dataset(
            test_p[fold_idx],
            test_l[fold_idx],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
        )
        val_set = C16Dataset(
            val_p[fold_idx],
            val_l[fold_idx],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
        )
        return train_set, val_set, test_set

    if dataset_name == "gc":
        train_set = GcDataset(
            train_p[fold_idx],
            train_l[fold_idx],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
            high_weight=args.high_weight,
            is_train=True,
        )
        test_set = GcDataset(
            test_p[fold_idx],
            test_l[fold_idx],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
            high_weight=args.high_weight,
        )
        val_set = GcDataset(
            val_p[fold_idx],
            val_l[fold_idx],
            root=args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
            high_weight=args.high_weight,
        )
        return train_set, val_set, test_set

    if dataset_name == "tcga":
        train_set = TCGADataset(
            train_p[fold_idx],
            train_l[fold_idx],
            args.tcga_max_patch,
            args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
            is_train=True,
        )
        test_set = TCGADataset(
            test_p[fold_idx],
            test_l[fold_idx],
            args.tcga_max_patch,
            args.dataset_root,
            persistence=args.persistence,
            keep_same_psize=args.same_psize,
        )
        val_set = (
            TCGADataset(
                val_p[fold_idx],
                val_l[fold_idx],
                args.tcga_max_patch,
                args.dataset_root,
                persistence=args.persistence,
                keep_same_psize=args.same_psize,
            )
            if args.val_ratio != 0.0
            else test_set
        )
        return train_set, val_set, test_set

    raise ValueError(f"Unsupported dataset: {args.datasets}")


def build_loaders(args, train_set, val_set, test_set):
    if args.fix_loader_random:
        generator = torch.Generator()
        generator.manual_seed(7784414403328510413)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=generator,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_set),
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, test_loader


def resolve_teacher_init(args, fold_idx):
    if args.teacher_init.endswith(".pt"):
        return args.teacher_init

    candidates = [
        os.path.join(args.teacher_init, f"seed_{args.seed}", f"fold_{fold_idx}", f"fold_{fold_idx}_model_best_auc.pt"),
        os.path.join(args.teacher_init, f"fold_{fold_idx}", f"fold_{fold_idx}_model_best_auc.pt"),
        os.path.join(args.teacher_init, f"fold_{fold_idx}_model_best_auc.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def build_mhim_model(args, device, train_loader):
    mrh_sche = None
    mm_sche = None
    if args.mrh_sche:
        mrh_sche = cosine_scheduler(
            args.mask_ratio_h,
            0.0,
            epochs=args.num_epoch,
            niter_per_ep=len(train_loader),
        )
    if args.mm_sche:
        mm_sche = cosine_scheduler(
            args.mm,
            args.mm_final,
            epochs=args.num_epoch,
            niter_per_ep=len(train_loader),
            start_warmup_value=1.0,
        )

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
        mrh_sche=mrh_sche,
        da_act=args.da_act,
        attn_layer=args.attn_layer,
    ).to(device)
    return model, mm_sche


def build_model(args, device, train_loader, fold_idx):
    mm_sche = None
    teacher_init = resolve_teacher_init(args, fold_idx)

    if args.model == "mhim":
        model, mm_sche = build_mhim_model(args, device, train_loader)
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
        init_weights = torch.load("./modules/init_cpk/dsmil_init.pth")
        info = model.load_state_dict(init_weights, strict=False)
        if not args.no_log:
            print(info)
    elif args.model == "meanmil":
        model = mean_max.MeanMIL(
            n_classes=args.n_classes,
            dropout=args.dropout,
            act=args.act,
            input_dim=args.input_dim,
        ).to(device)
    elif args.model == "maxmil":
        model = mean_max.MaxMIL(
            n_classes=args.n_classes,
            dropout=args.dropout,
            act=args.act,
            input_dim=args.input_dim,
        ).to(device)
    elif args.model == 'vit':
        model = vit.vit_base_patch16_224_in21k(num_classes=2, has_logits=False, input_dim=args.input_dim).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.init_stu_type != "none":
        pretrained = torch.load(teacher_init)
        if "model" in pretrained:
            pretrained = pretrained["model"]
        if args.init_stu_type == "fc":
            new_state_dict = {}
            for key, value in pretrained.items():
                clean_key = key.replace("patch_to_emb.", "") if "patch_to_emb" in key else key
                new_state_dict[clean_key] = value
            info = model.patch_to_emb.load_state_dict(new_state_dict, strict=False)
        else:
            info = model.load_state_dict(pretrained, strict=False)
        if not args.no_log:
            print(info)

    if args.model != "mhim":
        return model, None, mm_sche

    teacher_model = deepcopy(model)
    if not args.no_tea_init and args.tea_type != "same":
        try:
            pretrained = torch.load(teacher_init)
            if "model" in pretrained:
                pretrained = pretrained["model"]
            info = teacher_model.load_state_dict(pretrained, strict=False)
            if not args.no_log:
                print(info)
        except Exception:
            if not args.no_log:
                print("########## Init Error")

    if args.tea_type == "same":
        teacher_model = model

    return model, teacher_model, mm_sche


def build_criterion(args):
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if args.loss == "ce":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {args.loss}")


def build_optimizer(args, model):
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    if args.opt == "adamw":
        return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.opt == "adam":
        return torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.opt}")


def build_scheduler(args, optimizer, train_loader):
    if args.lr_sche == "cosine":
        t_max = args.num_epoch * len(train_loader) if args.lr_supi else args.num_epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, 0)
    if args.lr_sche == "step":
        assert not args.lr_supi
        return torch.optim.lr_scheduler.StepLR(optimizer, args.num_epoch / 2, 0.2)
    if args.lr_sche == "const":
        return None
    raise ValueError(f"Unsupported lr scheduler: {args.lr_sche}")


def build_early_stopping(args):
    if not args.early_stopping:
        return None
    return EarlyStopping(
        patience=30 if args.datasets == "camelyon16" else 20,
        stop_epoch=args.max_epoch if args.datasets == "camelyon16" else 70,
        save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch),
    )


def extract_label_list(labels):
    if labels.ndim > 1:
        return torch.argmax(labels, dim=-1).tolist()
    return labels.tolist() if len(labels) > 1 else [labels.item()]


def unpack_batch(data, device):
    if len(data) < 2:
        raise ValueError(f"Unexpected batch format with {len(data)} items")

    bag, labels = data[0], data[1]
    label_list = extract_label_list(labels)

    if isinstance(bag, (list, tuple)):
        bag = [item.to(device) for item in bag]
        batch_size = bag[0].size(0)
    else:
        bag = bag.to(device)
        batch_size = bag.size(0)

    return bag, labels.to(device), batch_size, label_list


def maybe_shuffle_bag(args, bag):
    if args.patch_shuffle:
        return patch_shuffle(bag, args.shuffle_group)
    if args.group_shuffle:
        return group_shuffle(bag, args.shuffle_group)
    return bag


def compute_train_outputs(args, model, model_tea, bag, label, criterion, loader_len, epoch, batch_idx):
    logit_loss = None
    iter_idx = epoch * loader_len + batch_idx

    if args.model == "mhim":
        if model_tea is not None:
            cls_tea, attn = model_tea.forward_teacher(bag, return_attn=True)
        else:
            attn, cls_tea = None, None

        cls_tea = None if args.cl_alpha == 0.0 else cls_tea
        logits, cls_loss, patch_num, keep_num = model(bag, attn, cls_tea, i=iter_idx)

    elif args.model == "pure":
        logits, cls_loss, patch_num, keep_num = model.pure(bag)

    elif args.model in ("clam_sb", "clam_mb", "dsmil"):
        logits, cls_loss, patch_num = model(bag, label, criterion)
        keep_num = patch_num

    else:
        logits = model(bag)
        cls_loss, patch_num, keep_num = 0.0, 0.0, 0.0

    if logit_loss is None:
        if args.loss == "ce":
            logit_loss = criterion(logits.view(label.size(0), -1), label)
        elif args.loss == "bce":
            logit_loss = criterion(
                logits.view(label.size(0), -1),
                one_hot(label.view(label.size(0), -1).float(), num_classes=2),
            )
        else:
            raise ValueError(f"Unsupported loss: {args.loss}")

    return logits, logit_loss, cls_loss, patch_num, keep_num


def forward_eval_logits(args, model, bag):
    if args.model in ("mhim", "pure"):
        return model.forward_test(bag)
    if args.model == "dsmil":
        logits, _ = model(bag)
        return logits
    return model(bag)


def should_average_eval_logits(args, logits):
    return (args.model == "dsmil" and args.ds_average) or (
        args.model == "mhim" and isinstance(logits, (list, tuple))
    )


def collect_eval_outputs(args, criterion, logits, label, batch_size):
    if args.loss == "ce":
        if should_average_eval_logits(args, logits):
            loss = criterion(logits[0].view(batch_size, -1), label)
            probs = 0.5 * torch.softmax(logits[1], dim=-1) + 0.5 * torch.softmax(logits[0], dim=-1)
            if args.n_classes == 2:
                return loss, probs[:, 1].cpu().numpy().tolist()
            return loss, probs.cpu().numpy().tolist()

        loss = criterion(logits.view(batch_size, -1), label)
        probs = torch.softmax(logits, dim=-1)
        if args.n_classes == 2:
            return loss, probs[:, 1].cpu().numpy().tolist()
        return loss, probs.cpu().numpy().tolist()

    if args.loss == "bce":
        if args.model == "dsmil" and args.ds_average:
            loss = criterion(logits.view(batch_size, -1), label)
            probs = 0.5 * torch.sigmoid(logits[1]) + 0.5 * torch.sigmoid(logits[0])
            return loss, probs.cpu().numpy().tolist()

        logits_for_loss = logits[0] if isinstance(logits, (list, tuple)) else logits
        loss = criterion(logits_for_loss.view(batch_size, -1), label.view(batch_size, -1).float())
        return loss, torch.sigmoid(logits_for_loss).cpu().numpy().tolist()

    raise ValueError(f"Unsupported loss: {args.loss}")


def compute_metrics(args, bag_labels, bag_outputs):
    if args.n_classes == 2:
        acc, auc, _, sen, spec, f1 = six_scores(bag_labels, bag_outputs, 0)
        return {
            "acc": float(acc),
            "spec": float(spec),
            "sen": float(sen),
            "f1": float(f1),
            "auc": float(auc),
        }

    auc, acc, sen, _, f1 = multi_class_scores(bag_labels, bag_outputs)
    return {
        "acc": float(acc),
        "spec": np.nan,
        "sen": float(sen),
        "f1": float(f1),
        "auc": float(auc),
    }


def train_loop(args, model, model_tea, loader, optimizer, device, amp_autocast, criterion, scheduler, mm_sche, epoch):
    start = time.time()
    cls_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()
    train_loss_log = 0.0

    model.train()
    if model_tea is not None:
        model_tea.train()

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()
        bag, label, batch_size, _ = unpack_batch(data, device)

        with amp_autocast():
            bag = maybe_shuffle_bag(args, bag)
            _, logit_loss, cls_loss, patch_num, keep_num = compute_train_outputs(
                args,
                model,
                model_tea,
                bag,
                label,
                criterion,
                len(loader),
                epoch,
                batch_idx,
            )
            train_loss = args.cls_alpha * logit_loss + args.cl_alpha * cls_loss

        train_loss = train_loss / args.accumulation_steps
        train_loss.backward()

        if args.clip_grad > 0.0:
            dispatch_clip_grad(model_parameters(model), value=args.clip_grad, mode="norm")

        if (batch_idx + 1) % args.accumulation_steps == 0:
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
            if args.model == "mhim" and model_tea is not None and args.tea_type != "same":
                mm = mm_sche[epoch * len(loader) + batch_idx] if mm_sche is not None else args.mm
                ema_update(model, model_tea, mm)

        cls_loss_meter.update(logit_loss.item(), 1)
        aux_loss_meter.update(cls_loss.item() if isinstance(cls_loss, torch.Tensor) else float(cls_loss), 1)
        patch_num_meter.update(float(patch_num), 1)
        keep_num_meter.update(float(keep_num), 1)

        if (batch_idx % args.log_iter == 0 or batch_idx == len(loader) - 1) and not args.no_log:
            print(
                "[{}/{}] logit_loss:{:.6f}, cls_loss:{:.6f}, patch_num:{:.2f}, keep_num:{:.2f}".format(
                    batch_idx,
                    len(loader) - 1,
                    cls_loss_meter.avg,
                    aux_loss_meter.avg,
                    patch_num_meter.avg,
                    keep_num_meter.avg,
                )
            )

        train_loss_log += train_loss.item()

    end = time.time()
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    return train_loss_log / len(loader), start, end


def val_loop(args, model, loader, device, criterion, early_stopping=None, epoch=0, test_mode=False):
    model.eval()
    loss_meter = AverageMeter()
    bag_labels = []
    bag_outputs = []

    with torch.no_grad():
        for data in loader:
            bag, label, batch_size, label_list = unpack_batch(data, device)
            bag_labels.extend(label_list)

            logits = forward_eval_logits(args, model, bag)
            loss, outputs = collect_eval_outputs(args, criterion, logits, label, batch_size)
            bag_outputs.extend(outputs)
            loss_meter.update(loss.item(), 1)

    metrics = compute_metrics(args, bag_labels, bag_outputs)
    if test_mode:
        return metrics, loss_meter.avg

    if early_stopping is not None:
        early_stopping(epoch, -metrics["auc"], model)
        stop = early_stopping.early_stop
    else:
        stop = False
    return stop, metrics, loss_meter.avg


def save_best_model(path, model, model_tea):
    torch.save(
        {
            "model": model.state_dict(),
            "teacher": model_tea.state_dict() if model_tea is not None else None,
        },
        path,
    )


def run_one_fold(args, seed_dir, fold_idx, split_data):
    seed_torch(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress

    train_set, val_set, test_set = build_datasets(args, fold_idx, split_data)
    train_loader, val_loader, test_loader = build_loaders(args, train_set, val_set, test_set)
    model, model_tea, mm_sche = build_model(args, device, train_loader, fold_idx)
    criterion = build_criterion(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, train_loader)
    early_stopping = build_early_stopping(args)

    fold_dir = os.path.join(seed_dir, f"fold_{fold_idx}")
    ensure_dir(fold_dir)
    best_model_path = os.path.join(fold_dir, f"fold_{fold_idx}_model_best_auc.pt")
    best_auc = -1.0
    train_time_meter = AverageMeter()
    fold_records = []

    for epoch in range(args.num_epoch):
        train_loss, start, end = train_loop(
            args,
            model,
            model_tea,
            train_loader,
            optimizer,
            device,
            amp_autocast,
            criterion,
            scheduler,
            mm_sche,
            epoch,
        )
        train_time_meter.update(end - start)

        stop, val_metrics, val_loss = val_loop(
            args,
            model,
            val_loader,
            device,
            criterion,
            early_stopping=early_stopping,
            epoch=epoch,
        )

        fold_records.append(
            [
                epoch + 1,
                train_loss,
                val_loss,
                val_metrics["acc"],
                val_metrics["spec"],
                val_metrics["sen"],
                val_metrics["f1"],
                val_metrics["auc"],
            ]
        )

        if not args.no_log:
            print(
                "Epoch [{}/{}] train_loss:{:.6f} val_loss:{:.6f} val_acc:{:.4f} val_spec:{:.4f} val_sen:{:.4f} val_f1:{:.4f} val_auc:{:.4f} time:{:.3f}({:.3f})".format(
                    epoch + 1,
                    args.num_epoch,
                    train_loss,
                    val_loss,
                    val_metrics["acc"],
                    format_metric(val_metrics["spec"]),
                    val_metrics["sen"],
                    val_metrics["f1"],
                    val_metrics["auc"],
                    train_time_meter.val,
                    train_time_meter.avg,
                )
            )

        if val_metrics["auc"] > best_auc and epoch >= args.save_best_model_stage * args.num_epoch:
            best_auc = val_metrics["auc"]
            save_best_model(best_model_path, model, model_tea)

        if stop:
            break

    best_state = torch.load(best_model_path)
    model.load_state_dict(best_state["model"])
    if model_tea is not None and best_state["teacher"] is not None:
        model_tea.load_state_dict(best_state["teacher"])

    test_metrics, test_loss = val_loop(args, model, test_loader, device, criterion, test_mode=True)
    save_fold_log(seed_dir, fold_idx, args.seed, args, fold_records)

    return {
        "fold": fold_idx + 1,
        "acc": test_metrics["acc"],
        "spec": test_metrics["spec"],
        "sen": test_metrics["sen"],
        "f1": test_metrics["f1"],
        "auc": test_metrics["auc"],
        "test_loss": test_loss,
    }


def run_one_seed(args):
    seed_torch(args.seed)
    split_data = build_fold_split(args, prepare_split_data(args))
    actual_fold_count = len(split_data[0])
    seed_dir = os.path.join(args.model_path, f"seed_{args.seed}")
    ensure_dir(seed_dir)

    if not args.no_log:
        print(f"\n===== Seed {args.seed} =====")

    fold_metrics = []
    for fold_idx in range(args.fold_start, actual_fold_count):
        if not args.no_log:
            print(f"Start {actual_fold_count}-fold cross validation: fold {fold_idx}")
        fold_metrics.append(run_one_fold(args, seed_dir, fold_idx, split_data))

    seed_summary = {"seed": args.seed}
    seed_summary.update(save_seed_summary(seed_dir, fold_metrics, actual_fold_count))

    if not args.no_log:
        print_seed_summary(seed_summary)

    return seed_summary


def main(args):
    seed_summaries = []
    for seed in SEEDS:
        args.seed = seed
        seed_summaries.append(run_one_seed(args))

    save_seed_aggregate(args.model_path, seed_summaries)
    if not args.no_log:
        print_final_summary(seed_summaries)


def build_parser():
    parser = argparse.ArgumentParser(description="MIL Training Script")

    parser.add_argument("--datasets", default="camelyon16", type=str, help="[camelyon16, tcga, ngc, gc, fnac]")
    parser.add_argument("--dataset_root", default="/data/xxx/TCGA", type=str, help="Dataset root path")
    parser.add_argument("--label_path", default="/root/project/MHIM-MIL/ngc-labels/train_label.csv", type=str, help="Label path root")
    parser.add_argument("--tcga_max_patch", default=-1, type=int, help="Max number of patches in TCGA")
    parser.add_argument("--fix_loader_random", action="store_true", help="Fix random seed of dataloader")
    parser.add_argument("--fix_train_random", action="store_true", help="Fix random seed of training")
    parser.add_argument("--val_ratio", default=0.0, type=float, help="Val-set ratio")
    parser.add_argument("--fold_start", default=0, type=int, help="Start validation fold")
    parser.add_argument("--cv_fold", default=3, type=int, help="Number of cross validation folds")
    parser.add_argument("--persistence", action="store_true", help="Load data into memory")
    parser.add_argument("--same_psize", default=0, type=int, help="Keep the same size of all patches")
    parser.add_argument("--high_weight", default=1.0, type=float, help="Weight loss for high-risk WSI in GC")
    parser.add_argument("--train_val", action="store_true", help="Use train and val set together for training")

    parser.add_argument("--cls_alpha", default=1.0, type=float, help="Main loss alpha")
    parser.add_argument("--num_epoch", default=200, type=int, help="Number of total training epochs")
    parser.add_argument("--early_stopping", action="store_false", help="Enable early stopping")
    parser.add_argument("--max_epoch", default=130, type=int, help="Max epoch for early stopping")
    parser.add_argument("--n_classes", default=2, type=int, help="Number of classes")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--loss", default="ce", type=str, help="Classification loss [ce, bce]")
    parser.add_argument("--opt", default="adam", type=str, help="Optimizer [adam, adamw]")
    parser.add_argument("--save_best_model_stage", default=0.0, type=float, help="Start saving best model after this ratio")
    parser.add_argument("--model", default="mhim", type=str, help="Model name")
    parser.add_argument("--seed", default=2023, type=int, help="Overwritten by the fixed 3-seed loop")
    parser.add_argument("--lr", default=2e-4, type=float, help="Initial learning rate")
    parser.add_argument("--lr_sche", default="cosine", type=str, help="LR schedule [cosine, step, const]")
    parser.add_argument("--lr_supi", action="store_true", help="Update LR schedule per iteration")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay")
    parser.add_argument("--accumulation_steps", default=1, type=int, help="Gradient accumulation")
    parser.add_argument("--clip_grad", default=0.0, type=float, help="Gradient clip")

    parser.add_argument("--ds_average", action="store_true", help="DSMIL hyperparameter")
    parser.add_argument("--baseline", default="selfattn", type=str, help="Baseline model [attn, selfattn, dsmil]")
    parser.add_argument("--act", default="relu", type=str, help="Activation function in the projection head")
    parser.add_argument("--dropout", default=0.25, type=float, help="Dropout in the projection head")
    parser.add_argument("--n_heads", default=8, type=int, help="Number of heads in MSA")
    parser.add_argument("--da_act", default="relu", type=str, help="Activation function in DAttention")
    parser.add_argument("--input_dim", default=1024, type=int, help="Patch feature dimension")

    parser.add_argument("--patch_shuffle", action="store_true", help="2-D group shuffle")
    parser.add_argument("--group_shuffle", action="store_true", help="Group shuffle")
    parser.add_argument("--shuffle_group", default=0, type=int, help="Number of shuffle groups")

    parser.add_argument("--mask_ratio", default=0.0, type=float, help="Random mask ratio")
    parser.add_argument("--mask_ratio_l", default=0.0, type=float, help="Low-attention mask ratio")
    parser.add_argument("--mask_ratio_h", default=0.0, type=float, help="High-attention mask ratio")
    parser.add_argument("--mask_ratio_hr", default=1.0, type=float, help="Random high-attention mask ratio")
    parser.add_argument("--mrh_sche", action="store_true", help="Decay high-attention mask ratio")
    parser.add_argument("--msa_fusion", default="vote", type=str, help="[mean, vote]")
    parser.add_argument("--attn_layer", default=0, type=int, help="Attention layer index")

    parser.add_argument("--cl_alpha", default=0.0, type=float, help="Auxiliary loss alpha")
    parser.add_argument("--temp_t", default=0.1, type=float, help="Temperature")
    parser.add_argument("--teacher_init", default="none", type=str, help="Path to initial teacher model")
    parser.add_argument("--no_tea_init", action="store_true", help="Disable teacher initialization")
    parser.add_argument("--init_stu_type", default="none", type=str, help="Student initialization [none, fc, all]")
    parser.add_argument("--tea_type", default="none", type=str, help="[none, same]")
    parser.add_argument("--mm", default=0.9999, type=float, help="EMA decay")
    parser.add_argument("--mm_final", default=1.0, type=float, help="Final EMA decay")
    parser.add_argument("--mm_sche", action="store_true", help="Use cosine schedule for EMA")

    parser.add_argument("--title", default="default", type=str, help="Experiment title")
    parser.add_argument("--project", default="output-test", type=str, help="Experiment project")
    parser.add_argument("--log_iter", default=100, type=int, help="Log frequency")
    parser.add_argument("--amp", action="store_true", help="Automatic mixed precision training")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of dataloader workers")
    parser.add_argument("--no_log", action="store_true", help="Disable stdout logging")
    parser.add_argument("--model_path", type=str, default="output-model", help="Output path")
    return parser


def configure_args(args):
    if args.train_val:
        args.train_label_path = os.path.join(args.label_path, "train_val.csv")
        args.val_label_path = os.path.join(args.label_path, "test_label.csv")
        args.test_label_path = os.path.join(args.label_path, "test_label.csv")
    else:
        args.train_label_path = os.path.join(args.label_path, "train_label.csv")
        args.val_label_path = os.path.join(args.label_path, "val_label.csv")
        args.test_label_path = os.path.join(args.label_path, "test_label.csv")

    args.model_path = os.path.join(args.model_path, args.project, args.title)
    ensure_dir(args.model_path)

    if args.model == "pure":
        args.cl_alpha = 0.0
    elif args.model in ("clam_sb", "clam_mb"):
        args.cls_alpha = 0.7
        args.cl_alpha = 0.3
    elif args.model == "dsmil":
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5

    if args.datasets == "camelyon16":
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets == "tcga":
        args.num_workers = 0

    return args


if __name__ == "__main__":
    parser = build_parser()
    args = configure_args(parser.parse_args())

    print(args)
    print(time.asctime(time.localtime(time.time())))
    print("Fixed seeds:", SEEDS)
    main(args)
