#!/usr/bin/env bash

######################## step1: extract features
cd extract-features
export CUDA_VISIBLE_DEVICES=1,2,3
GPU_NUMBERS=3
FEAT_DIR=biomedclip-test
DATASET=fnac
BASE_MODEL=biomedclip # biomedclip, clip, plip
INPUT_DIM=512
WSI_ROOT=../data/FNAC-2019/split_data
OUTPUT_PATH=../data/fnac-features
K=20 # filted patch number

python -m torch.distributed.launch --master_port=10000 --nproc_per_node=${GPU_NUMBERS} \
        extract_features_tct.py \
        --base_model=${BASE_MODEL} \
        --dataset=${DATASET} \
        --output_path=${OUTPUT_PATH} \
        --feat_dir=${FEAT_DIR} \
        --wsi_root=${WSI_ROOT} \
        --ckp_path=${CKP_PATH} \
        --multi_gpu \
        --batch_size=32 \
        --num_workers=64


######################## step2: train patch classifier
cd ../mil-methods
DATASET_PATH=${OUTPUT_PATH}/${FEAT_DIR}
LABEL_PATH=../datatools/${DATASET}/labels
MIL_OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods-info
TITLE_NAME=${FEAT_DIR}-meanmil-$DATASET--trainval
python3 mil.py --project=${PROJECT_NAME} \
                --dataset_root=${DATASET_PATH} \
                --label_path=${LABEL_PATH} \
                --model_path=${MIL_OUTPUT_PATH} \
                --datasets=${DATASET} \
                --input_dim=${INPUT_DIM} \
                --cv_fold=1 \
                --title=$TITLE_NAME \
                --model=meanmil --seed=2024 --train_val \
                # --wandb


######################## step3: filter patches
FILTER_OUTPUT_ROOT=../data/${DATASET}/${FEAT_DIR}-meanmil-${K}
TRAIN_LABEL=${LABEL_PATH}/train_val.csv
CKP_PATH=${MIL_OUTPUT_PATH}/${PROJECT_NAME}/${TITLE_NAME}/fold_0_model_best_auc.pt
python inference-multi.py --input_dim=${INPUT_DIM} \
                            --datasets=${DATASET} \
                            --feature_root=${DATASET_PATH} \
                            --wsi_root=${WSI_ROOT} \
                            --output_root=${FILTER_OUTPUT_ROOT} \
                            --train_label=${TRAIN_LABEL} \
                            --ckp_path=${CKP_PATH} \
                            --topk_num=${K}  \
                            --model=meanmil
