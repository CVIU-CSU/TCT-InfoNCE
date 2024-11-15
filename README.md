# TCT-InfoNCE


## An efficient framework based on large foundation model for cervical cytopathology whole slide image classification

[[`Model`](https://pan.baidu.com/s/1iY9U4UzX-lusHYCQN7fwTw?pwd=5367)] [[`Paper`](https://arxiv.org/abs/2407.11486)] [[`BibTeX`](#wait)]


## Model Overview

<p align="center">
    <img src="imgs/structure.png" width="100%"> <br>
</p>

## Install

On an NVIDIA Tensor Core GPU machine, with CUDA toolkit enabled.

1. Download our repository and open the TCT-InfoNCE
```
git clone https://github.com/CVIU-CSU/TCT-InfoNCE.git
cd TCT-InfoNCE
```

2. Requirements

```bash
conda create -n biomed python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib
pip install h5py scikit-learn==0.22.1 future==0.18.3
pip install wandb==0.15 torchsummary==1.5.1 torchmetrics
pip install einops chardet omegaconf
pip install jupyter
```

3. Make directories
```bash
mkdir data
mkdir mil-methods/output-model
mkdir extract-features/output-model
```

## Results and Models

- **Result**

<p align="center">
    <img src="imgs/results.png" width="90%"> <br>

- **Model Download**
The models and gc-features can be accessed from [Baiduyun](https://pan.baidu.com/s/1iY9U4UzX-lusHYCQN7fwTw?pwd=5367)

## Train FNAC-2019
The CSD dataset labels are based on the WSI as the fundamental unit, and the details can be found in the `datatools` folder. Owing to the absence of permission to publicly disclose the CSD dataset, we have presented the experimental procedure for the FNAC-2019 dataset, which exhibits a similar procedure to that of the CSD dataset.


Befor Training, need to download the [FNAC-2019](https://pan.baidu.com/s/1Yh_nQH02xWan0ck9h6hMzQ?pwd=9h39) dataset and put it in the `data` folder.

The training process consists of three stages: filtering patches, training adapter, and training the MIL method. The subsequent process is specifically tailored for the `PLIP` foundation model in conjunction with the `MHIM(TransMIL)` approach.
 
- **Filter patches** 
```bash
# default: 4 gpu; output: ./data/fnac/biomedclip-test-meanmil-20
bash tools/filter.sh
```

- **Train adapter**
```bash 
# default: 4 gpu; output: ./extract-features/output-model/simclr-infonce
bash tools/adapter.sh
```

- **Train MIL**
```bash 
# extract patch features use the trained image encoder
# default: 4 gpu; output: ./data/fnac-features/biomedclip-adapter
bash tools/extract.sh

# train TransMIL
# default: 1 gpu; output: ./mil-methods/output-model/biomedclip-adapter-mhim(transmil)-fnac-trainval
bash tools/mil.sh
```

## Test CSD
You can eval our method on CSD dataset by downloading the WSI features and MIL methods' weights from [Baiduyun](https://pan.baidu.com/s/1iY9U4UzX-lusHYCQN7fwTw?pwd=5367). Put the download file to `TCT-InfoNCEI` folder, which is this project root.
```bash
bash tools/test.sh
```

## Acknowledgements


## Citation
