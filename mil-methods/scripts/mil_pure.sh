#!/usr/bin/env bash

set -euo pipefail

FEATURE_SOURCE="${1:-frozen}"
FEATURE_NAME="${2:-resnet50}"
DATASET="${3:-gc}"
NUM_DIM="${4:-512}"
HIGH_WEIGHT="${5:-1.0}"
EXPERIMENT="${6:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

DATASET_PATH="/home1/wsi/gc-all-features/${FEATURE_SOURCE}/${FEATURE_NAME}"
# DATASET_PATH="/home/huangjialong/projects/BiomedCLIP-PUNCE/PatchEncoder/result-final-fnac-features/${FEATURE_NAME}"
LABEL_PATH="../datatools/${DATASET}/labels"
OUTPUT_PATH="output-model"
PROJECT_NAME="mil-methods-info"

echo "特征来源 ：${FEATURE_SOURCE}"
echo "特征名称 ：${FEATURE_NAME}"
echo "数据集 ：${DATASET}"
echo "输入维度 ：${NUM_DIM}"
echo "高风险权重 ：${HIGH_WEIGHT}"
echo "实验 ：${EXPERIMENT}"

run_train() {
    local title_name="$1"
    local model_name="$2"
    local baseline_name="$3"
    local teacher_init="$4"
    shift 4

    local cmd=(
        python3 mil_pure.py
        --project="${PROJECT_NAME}"
        --dataset_root="${DATASET_PATH}"
        --label_path="${LABEL_PATH}"
        --model_path="${OUTPUT_PATH}"
        --datasets="${DATASET}"
        --input_dim="${NUM_DIM}"
        --cv_fold=1
        --title="${title_name}"
        --model="${model_name}"
        --train_val
        --baseline="${baseline_name}"
        --high_weight="${HIGH_WEIGHT}"
    )

    if [[ -n "${teacher_init}" ]]; then
        cmd+=(--teacher_init="${teacher_init}")
    fi

    cmd+=("$@")

    echo
    echo "========== ${title_name} =========="
    "${cmd[@]}"
}

run_abmil() {
    run_train \
        "${FEATURE_NAME}-abmil-${DATASET}-trainval-3seed" \
        "pure" \
        "attn" \
        ""
}

run_transmil() {
    run_train \
        "${FEATURE_NAME}-transmil-${DATASET}-trainval-3seed" \
        "pure" \
        "selfattn" \
        ""
}

run_vit() {
    run_train \
        "${FEATURE_NAME}-vit-${DATASET}-trainval-3seed" \
        "vit" \
        "selfattn" \
        ""
}

run_mhim_abmil() {
    local teacher_init="./${OUTPUT_PATH}/${PROJECT_NAME}/${FEATURE_NAME}-abmil-${DATASET}-trainval-3seed"
    run_train \
        "${FEATURE_NAME}-mhim(abmil)-${DATASET}-trainval-3seed" \
        "mhim" \
        "attn" \
        "${teacher_init}" \
        --num_workers=0 \
        --cl_alpha=0.1 \
        --mask_ratio_h=0.01 \
        --mask_ratio_hr=0.5 \
        --mrh_sche \
        --init_stu_type=fc \
        --mask_ratio=0.5 \
        --mask_ratio_l=0.0
}

run_mhim_transmil() {
    local teacher_init="./${OUTPUT_PATH}/${PROJECT_NAME}/${FEATURE_NAME}-transmil-${DATASET}-trainval-3seed"
    run_train \
        "${FEATURE_NAME}-mhim(transmil)-${DATASET}-trainval-3seed" \
        "mhim" \
        "selfattn" \
        "${teacher_init}" \
        --mask_ratio_h=0.03 \
        --mask_ratio_hr=0.5 \
        --mrh_sche \
        --mask_ratio=0.0 \
        --mask_ratio_l=0.8 \
        --cl_alpha=0.1 \
        --mm_sche \
        --init_stu_type=fc \
        --attn_layer=0
}

case "${EXPERIMENT}" in
    abmil)
        run_abmil
        ;;
    transmil)
        run_transmil
        ;;
    vit)
        run_vit
        ;;
    mhim_abmil)
        run_mhim_abmil
        ;;
    mhim_transmil)
        run_mhim_transmil
        ;;
    all)
        run_abmil
        run_transmil
        run_mhim_abmil
        run_mhim_transmil
        ;;
    *)
        echo "Unknown EXPERIMENT: ${EXPERIMENT}"
        echo "Use one of: abmil, transmil, vit, mhim_abmil, mhim_transmil, all"
        exit 1
        ;;
esac
