# Implementation of A Skin Segmentation Model Based on Synthetic Face Images Using Deep Learning

> This experiment is based on training a skin segmentation model with best performance using synthetic face data, and investigates its practicality in the PPGI task involving real face videos. This repository contains code for training/evaluating a face segmentation model using different metrics and visualizing prediction results, as well as applying it to some PPGI tasks [Thesis view](https://github.com/Supcode123/Skin-Segmentation-4-iPPG/blob/main/report/Implementation_of_A_Skin_Segmentation_Model_Based_on_Synthetic_Face_Images_Using_Deep_Learning%20(1).pdf).

## 🛠 Installation

The code is tested with Python Version 3.10. We recommend using Miniconda: [Installing Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)

```
git clone <repo>

cd <repo>

conda create -n <env_name> python=3.10

conda activate <env_name>

pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Then install all necessary packages:
`pip install -r requirements.txt`

> 📌 **Note**: Model Mask2Former and SegNeXt models are implemented and pretrained in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please refer to their setup instructions for environment and checkpoint usage.

## 💡 Usage

Run the `train` script:
```
python train.py --data_path /path/to/data_folder \
                --data_conf data_config.yaml \
                --model_conf model_config.yaml \
                --train_conf train_config.yaml \
                --log  /path/to/save/logs&results \
                --device cuda
```
Run the `eval` script:
```
python model_eval.py --data_path /path/to/data_folder \
                     --test_conf test_config.yaml \
                     --chkpt_conf /path/to/checkpoint/dir \
                     --save_path /path/to/save \
                     --device cuda
```
for `visualization`:
```
python tools/eval_pics.py --data_path /path/to/data_folder \
                          --test_conf test_config.yaml \
                          --chkpt_conf /path/to/checkpoint/dir \
                          --save_path /path/to/save \
                          --device cuda
```
for `fps` calculation:
```
python tools/fps.py --data_path /path/to/data_folder \
                     --test_conf test_config.yaml\
                     --chkpt_conf /path/to/checkpoint/dir \
                     --device cuda
```
for PPGI on `UBFC/PURE dataset`:
```
python tools/video_ppgi/main.py --data_path /path/to/data_folder \
                                --test_conf test_config.yaml\
                                --chkpt_conf /path/to/checkpoint/dir \
                                --save_path /path/to/save \
                                --device cuda
```
for PPGI on `KISMED dataset`:
```
python tools/video_ppgi/main.py --data_path /path/to/data_folder \
                                --test_conf test_config.yaml\
                                --chkpt_conf /path/to/checkpoint/dir \
                                --save_path /path/to/save \
                                --device cuda
```
## ⚙️ Configuration Parameters
exsample:
```
DATA:
  FILTER_MISLABELED: False
  IMG_SIZE: (3, 256, 256)
  MEAN: [ 0.485, 0.456, 0.406 ]
  STD: [ 0.229, 0.224, 0.225 ]
  CLASSES: 2                   # could be =2 or =19
  EXP: EXP2                    # coressponding be 'EXP2' or 'EXP1' 
  SWIN_UNET: False             # if model is Swin_Unet, set it be 'True'
PPGi:
  DATA_NAME: PURE              # processing type : only be 'UBFC' or 'PURE'
  DATA_PROJECT_ORDER: [ 0 ]    # null -> traverse all projects(videos),[0, 1] -> process only the choosed projects(videos), here represents the 1st and 2nd project are choosed.
  TASK_ORDER: null #[ 0 ]      # same as above, specify the target tasks(for KISMED dataset).
  WIN_SIZE: 20                 # window size for HR estimation
  STEP: 10                     # sliding step for window
  ROI_OUTPUT: False            # if output the coressponding ROI extracted video
  METRICS_CAL: True            # if calculate the coressponding HR estimation metrics
MODEL:
  NAME: EfficientNetb0_UNet
  BACKBONE: efficientnet-b0
  HEAD: Unet
  ENCODER_WEIGHTS: null        # if use pretrained weights 
  ENCODER_DEPTH: 5
  DECODER_CHANNELS: [ 256, 128, 64, 32, 16 ]
TRAIN:
  WORKERS: 8
  BATCH_SIZE: 100
```
