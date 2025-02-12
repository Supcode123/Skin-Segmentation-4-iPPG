import os
import shutil

import pytorch_lightning as pl
import torch
import yaml

from models.Mask2Former.scripts.mask2former import Mask2FormerFinetuner
from models.Mask2Former.scripts.mask2former.config import test_args
from pytorch_lightning.loggers import CSVLogger

torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from mask2former import SegmentationDataModule


if __name__=="__main__":
    args = test_args()
    print(args)
    eval_path = os.path.join(args.chkpt_path, "eval")
    if os.path.isdir(eval_path):
        shutil.rmtree(eval_path)
    os.makedirs(eval_path, exist_ok=False)

    print("##### Load config")
    with open(args.test_conf, "r") as doc:
        cfg_info = yaml.load(doc, Loader=yaml.Loader)
    model_info = cfg_info["MODEL"]
    train_info = cfg_info["TRAIN"]

    data_module = SegmentationDataModule(dataset_dir=args.data_path, train_conf=train_info,
                                         model_conf=model_info)

    print("##### Load models ...###")
    file_path = os.path.join(args.chkpt_path, "epoch_epoch=07-iou_SKIN_iou_SKIN=0.87.ckpt")
    checkpoint = torch.load(file_path)
    model = Mask2FormerFinetuner(model_info, train_info)
    model.load_state_dict(checkpoint['state_dict'])
    LOGGER = CSVLogger(save_dir=eval_path, name="test_results")
    trainer = pl.Trainer(
        logger=LOGGER,
        accelerator='cuda',
        devices=train_info['DEVICES'],
    )

    print("Test starts!!")
    trainer.test(model,data_module)
