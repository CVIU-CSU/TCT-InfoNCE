#!/bin/bash

echo "提取特征方式 ：$1"
echo "数据集 ：$2"
echo "输入维度 ：$3"

FEATURE_NAME=$1
DATASET=$2
NUM_DIM=$3

cd ../
DATASET_PATH=../data/$DATASET-features/$FEATURE_NAME
LABEL_PATH=../datatools/$DATASET/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods-info
TITLE_NAME=$FEATURE_NAME-abmil-$DATASET-trainval
python3 mil.py --project=$PROJECT_NAME \
                --dataset_root=$DATASET_PATH \
                --label_path=$LABEL_PATH \
                --model_path=$OUTPUT_PATH \
                --datasets=$DATASET \
                --input_dim=$NUM_DIM \
                --cv_fold=1 \
                --title=$TITLE_NAME \
                --model=pure \
                --baseline=attn \
                --train_val \
                --seed=2024 \
                # --wandb

CHECKPOINT_PATH=$OUTPUT_PATH/$PROJECT_NAME/$FEATURE_NAME-abmil-$DATASET-trainval
python3 eval.py --label_path=$LABEL_PATH  \
                    --dataset_root=$DATASET_PATH \
                    --ckp_path=$CHECKPOINT_PATH \
                    --datasets=tct \
                    --input_dim=$NUM_DIM \
                    --model=pure \
                    --baseline=attn \
                    --seed=2024
