#!/usr/bin/env bash

cd extract-features
BACKBONE=biomedclip
DATASET=fnac
K=20

DATA_DIR=../data/${DATASET}/biomedclip-test-meanmil-${K}
TRAIN_LABEL=../datatools/${DATASET}/labels/train_val.csv
PROJECT_NAME=simclr-infonce
OUTPUT_PATH=output-model
TITLE_NAME=${FEAT_DIR}_simclr_infonce_filter${DATASET}_${K}_224_4*256
python -m torch.distributed.launch --master_port=10000 --nproc_per_node=4 simclr.py --ddp \
                                                            --dataset=${DATASET} \
                                                            --backbone=${BACKBONE} \
                                                            --data_dir=${DATA_DIR} \
                                                            --train_label_path=${TRAIN_LABEL} \
                                                            --project=${PROJECT_NAME} \
                                                            --model_path=${OUTPUT_PATH} \
                                                            --title=${TITLE_NAME} \
                                                            --workers=2 \
                                                            --seed=2024 \
                                                            --batch_size=256 \
                                                            --epochs=200 \
                                                            # --wandb
