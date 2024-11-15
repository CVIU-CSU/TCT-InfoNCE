#!/usr/bin/env bash

# Extract feature
cd mil-methods
FEATURE_NAME=plip3
DATASET=gc
NUM_DIM=512

DATASET_PATH=../data/$DATASET-features/$FEATURE_NAME
LABEL_PATH=../datatools/$DATASET/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods-info

CHECKPOINT_PATH=$OUTPUT_PATH/$PROJECT_NAME/$FEATURE_NAME-mhim\(transmil\)-$DATASET-trainval

python3 eval.py --label_path=$LABEL_PATH  \
                                        --dataset_root=$DATASET_PATH \
                                        --ckp_path=$CHECKPOINT_PATH \
                                        --datasets=tct \
                                        --input_dim=$NUM_DIM \
                                        --model=pure \
                                        --baseline=selfattn \
                                        --seed=2024 