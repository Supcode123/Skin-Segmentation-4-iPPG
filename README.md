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

> 📌 **Note**: Mask2Former and SegNeXt models are implemented and pretrained in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please refer to their setup instructions for environment and checkpoint usage.

## Usage

> Put instructions on how to use your project code here. Best practice is to prepare a separate scripts for generating data and another script that creates plots and visualizations

## Configuration Parameters
> If your code is parameterized, you can explain the most important parameters here
