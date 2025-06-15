# Implementation of A Skin Segmentation Model Based on Synthetic Face Images Using Deep Learning

> This experiment is based on training a skin segmentation model with best performance using synthetic face data, and investigates its practicality in the PPGI task involving real face videos. This repository contains code for training/evaluating a face segmentation model using different metrics and visualizing prediction results, as well as applying it to some PPGI tasks.

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

