# Finetuning Mask2Former on custom Dataset

## Introduction
Mask2Former is a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components include masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most notably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU on ADE20K).

### [Mask2Former Project page](https://github.com/facebookresearch/Mask2Former) | [Mask2Former Paper](https://arxiv.org/abs/2112.10764) | 
Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)
### [Mask2Former Hugging Face](https://huggingface.co/docs/transformers/model_doc/mask2former)

## Purpose
The purpose of this document is to build a process of finetuning Mask2Former for custom dataset on semantic segmentation. The code is done using Pytorch Lightning and the model can be imported from hugging face.

1. Create a virtual environment: `conda create -n Mask2Former python=3.10 -y` and `conda activate Mask2Former `
2. Install [Pytorch CUDA 12.1](https://pytorch.org/): ` pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 `
3. Download code: `git clone https://github.com/sleepreap/Finetune-Mask2Former.git` 
4. `cd Finetune-Mask2Former` and run `pip install -e .`


## Citation
```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```
