#!/usr/bin/env bash

cd extract-features
# GPU_NUMBERS=3
# FEAT_DIR=biomedclip-test
# DATASET=fnac
# INPUT_DIM=512
# WSI_ROOT=../data/FNAC-2019/split_data
# OUTPUT_PATH=../data/fnac-features
# K=20 # filted patch number

# python -m torch.distributed.launch --master_port=10000 --nproc_per_node=${GPU_NUMBERS} \
#         extract_features_tct.py \
#         --base_model=biomedclip \
#         --dataset=${DATASET} \
#         --output_path=${OUTPUT_PATH} \
#         --feat_dir=${FEAT_DIR} \
#         --wsi_root=${WSI_ROOT} \
#         --ckp_path=${CKP_PATH} \
#         --multi_gpu \
#         --batch_size=32 \
#         --num_workers=64

GPU_NUMBERS=4
FEAT_DIR=biomedclip-adapter
WSI_ROOT=../data/FNAC-2019/split_data
OUTPUT_PATH=../data/fnac-features
BASE_MODEL=biomedclip
DATASET=fnac
CKP_PATH='./output-model/simclr-infonce/_simclr_infonce_filterfnac_20_224_4*256/_simclr_infonce_filterfnac_20_224_4*256_epoch200.pt'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS --master_port=12345 extract_features_tct.py \
        --base_model=${BASE_MODEL} \
        --dataset=${DATASET} \
        --output_path=${OUTPUT_PATH} \
        --feat_dir=${FEAT_DIR} \
        --wsi_root=${WSI_ROOT} \
        --ckp_path=${CKP_PATH} \
        --with_adapter \
        --target_patch_size 224 224 \
        --multi_gpu \
        --batch_size=32 \
        --num_workers=64