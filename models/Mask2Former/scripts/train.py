import sys
import os

from pytorch_lightning.callbacks import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import os
import pytorch_lightning as pl
import torch
from mask2former import SegmentationDataModule
from mask2former.config import _args, _config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from mask2former import Mask2FormerFinetuner


if __name__ == "__main__":

    print("##### config Load ... #####")

    args = _args()
    print(args)
    output_dir, data_config, model_config, train_config = _config(args)
    # data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    data_module = SegmentationDataModule(dataset_dir=args.data_path, train_conf=train_config,
                                         model_conf=model_config)


    print("##### Load models ...###")
    model = Mask2FormerFinetuner(model_config, train_config, output_dir)

    # model=Mask2FormerFinetuner(ID2LABEL, LEARNING_RATE)
    Early_Stopping = EarlyStopping(
                                   monitor="iou_SKIN",
                                   patience=train_config['PATIENCE'],
                                   mode='max',
                                   verbose=True,
                                   min_delta=train_config['THRESHOLD']
    )

    CHECKPOINT_CALLBACK = ModelCheckpoint(
                                          dirpath=os.path.join(output_dir,"checkpoints/"),
                                          filename="epoch_{epoch:02d}-iou_SKIN_{iou_SKIN:.2f}",
                                          save_top_k=1,
                                          monitor="iou_SKIN",
                                          mode="max",
                                           # Save the model at every epoch
                                          )
    LOGGER = TensorBoardLogger(save_dir=output_dir,
                               name="lightning_tensorboard",
                               version='',
                               log_graph=True)
    # data_module.setup("fit")
    trainer = pl.Trainer(
            logger=LOGGER,
            accelerator='cuda',
            devices=train_config['DEVICES'],
            # strategy="ddp",
            callbacks=[Early_Stopping, CHECKPOINT_CALLBACK],
            max_epochs=train_config['EPOCH'],
        )
    print("Training starts!!")
    trainer.fit(model, datamodule=data_module)

    # trainer.save_checkpoint("mask2former.ckpt")
