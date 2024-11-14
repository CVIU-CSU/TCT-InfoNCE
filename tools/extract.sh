#!/usr/bin/env bash

cd extract-features
export CUDA_VISIBLE_DEVICES=1,2,3
GPU_NUMBERS=3
FEAT_DIR=clip-test
DATASET=fnac
INPUT_DIM=512
WSI_ROOT=../data/FNAC-2019/split_data
OUTPUT_PATH=../data/fnac-features
K=20 # filted patch number

python -m torch.distributed.launch --master_port=10000 --nproc_per_node=${GPU_NUMBERS} \
        extract_features_tct.py \
        --base_model=biomedclip \
        --dataset=${DATASET} \
        --output_path=${OUTPUT_PATH} \
        --feat_dir=${FEAT_DIR} \
        --wsi_root=${WSI_ROOT} \
        --ckp_path=${CKP_PATH} \
        --multi_gpu \
        --batch_size=32 \
        --num_workers=64