# TCT-InfoNCE


## An efficient framework based on large foundation model for cervical cytopathology whole slide image classification

[[`Model`](#wait)] [[`Paper`](#wait)] [[`BibTeX`](#wait)]


## Model Overview

<p align="center">
    <img src="imgs/structure.png" width="100%"> <br>
</p>

## Install

On an NVIDIA Tensor Core GPU machine, with CUDA toolkit enabled.

1. Download our repository and open the TCT-InfoNCE
```
git clone git@github.com:CVIU-CSU/TCT-InfoNCE.git
cd TCT-InfoNCE
```

2. Requirements

```bash
conda create -n biomed python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install jupyter
pip install open-clip-torch transformers matplotlib
pip install h5py scikit-learn==0.22.1 future==0.18.3
pip install wandb==0.15 torchsummary==1.5.1 torchmetrics
pip install einops chardet omegaconf
```


## Results and Models

- **Result**

<p align="center">
    <img src="imgs/results.png" width="90%"> <br>

- **Model Download**

## Train
The CSD dataset labels are based on the WSI as the fundamental unit, and the details can be found in the `datatools` folder.


The training process consists of three stages: filtering patches, training adapter, and training the MIL method. The subsequent process is specifically tailored for the `PLIP` foundation model in conjunction with the `MHIM(TransMIL)` approach.
 
- **Filter patches**
```bash 
# extract patch features use the frozen image encoder

cd extract-features
GPU_NUMBERS=4
FEAT_DIR='clip1-test'
WSI_ROOT='/home1/wsi/gc-224'
OUTPUT_PATH='result-final-gc-features'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS extract_features_tct.py --base_model=clip --dataset=gc --output_path=$OUTPUT_PATH --feat_dir=$FEAT_DIR --wsi_root=$WSI_ROOT --ckp_path=$CKP_PATH --target_patch_size 224 224 --multi_gpu --batch_size=32 --num_workers=64
---------------------------------------------

# train patch classifier
cd mil-methods/scripts
bash meanmil.sh clip1-test gc 512
---------------------------------------------

# filter patches
cd mil-methods
FEAT_DIR=clip1-test
K=20

OUTPUT_ROOT=/home1/wsi/gc-output-filter/$FEAT_DIR-meanmil-$K
FEATURE_ROOT=../extract-features/result-final-gc-features/$FEAT_DIR
WSI_ROOT=/home1/wsi/gc-224
TRAIN_LABEL=../datatools/gc/labels/train_val.csv
CKP_PATH=./output-model/mil-methods/$FEAT_DIR-meanmil-gc-trainval/fold_0_model_best_auc.pt
python inference-multi.py --input_dim=512 --datasets=gc --feature_root=$FEATURE_ROOT --wsi_root=$WSI_ROOT --output_root=$OUTPUT_ROOT --train_label=$TRAIN_LABEL --ckp_path=$CKP_PATH --topk_num=$K  --model=meanmil
```

- **Train adapter**
```bash 
cd extract-features
BACKBONE=clip
K=50

DATA_DIR=/home1/wsi/gc-output-filter/clip1-test-meanmil-${K}
TRAIN_LABEL=../datatools/gc/labels/train_label.csv
PROJECT_NAME=simclr-infonce
OUTPUT_PATH=output-model
TITLE_NAME=${BACKBONE}_simclr_infonce_filterGC_${K}_224_4*256
python -m torch.distributed.launch --nproc_per_node=4 simclr.py --ddp --dataset=gc --backbone=$BACKBONE --data_dir=$DATA_DIR --train_label_path=$TRAIN_LABEL --project=$PROJECT_NAME --model_path=$OUTPUT_PATH --title=$TITLE_NAME --workers=2 --seed=2024 --batch_size=256 --epochs=200 --wandb
```

- **Train MIL**
```bash 
# extract patch features use the trained image encoder
cd extract-features

GPU_NUMBERS=4
FEAT_DIR=clip3-test
WSI_ROOT='/home1/wsi/gc-224'
OUTPUT_PATH='result-final-gc-features'
CKP_PATH=./output-model/simclr-infonce/clip_simclr_infonce_filterGC_50_224_4*256_200/clip_simclr_infonce_filterGC_50_224_4*256_200_epoch200.pt
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS --master_port=12345 extract_features_tct.py --base_model=clip --dataset=gc --output_path=$OUTPUT_PATH --feat_dir=$FEAT_DIR --wsi_root=$WSI_ROOT --ckp_path=$CKP_PATH --with_adapter --target_patch_size 224 224 --multi_gpu --batch_size=32 --num_workers=64
---------------------------------------------

# train TransMIL
cd mil-methods/scripts
bash transmil.sh clip3-test gc 512
bash mhim\(transmil).sh clip3-test gc 512

```

## Test

The adapter modules' and mil methods' weights can be accessed from [Baiduyun](https://pan.baidu.com/s/1XPT2WjrzBC13NMnwG2lEAA?pwd=j1u4).





## Acknowledgements
We would like to express our gratitude to the authors and developers of the exceptional repositories that this project is built upon: SAM. Their contributions have been invaluable to our work.

## Citation
