#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

FEATURE_SOURCE=frozen # frozen contrastive
FEAT_NAME=resnet1 # resnet50-tuneR biomed1 biomed3 clip3 plip3 biomed20 biomed200 biomed1 biomed3 resnet1
DATASET=gc
INPUT_DIM=1024 # 128 512 1024
HIGH_WEIGHT=1.0
EXPERIMENT=all # abmil transmil vit mhim_abmil mhim_transmil all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}/../mil-methods/scripts"
bash mil_pure.sh "${FEATURE_SOURCE}" "${FEAT_NAME}" "${DATASET}" "${INPUT_DIM}" "${HIGH_WEIGHT}" "${EXPERIMENT}"
