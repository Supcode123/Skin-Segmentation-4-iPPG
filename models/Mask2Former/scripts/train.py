import sys
import os
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


if __name__=="__main__":

    print("##### config Load ... #####")

    args = _args()
    print(args)
    output_dir, data_config, model_config, train_config = _config(args)
    # data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    data_module = SegmentationDataModule(dataset_dir=args.data_path, model_conf=model_config, train_conf=train_config)

    # train_loader = data_module.train_dataloader()
    # val_loader = data_module.val_dataloader()
    #
    # # 尝试从数据加载器获取第一个批次
    # try:
    #     first_batch = next(iter(train_loader))
    #     print(f"First batch from train_dataloader: {first_batch}")
    # except Exception as e:
    #     print(f"Error fetching first batch from train_dataloader: {str(e)}")
    #
    # try:
    #     first_batch = next(iter(val_loader))
    #     print(f"First batch from val_dataloader: {first_batch}")
    # except Exception as e:
    #     print(f"Error fetching first batch from val_dataloader: {str(e)}")

    print("##### Load models ...###")
    model = Mask2FormerFinetuner(model_config, train_config, output_dir)
    # model=Mask2FormerFinetuner(ID2LABEL, LEARNING_RATE)
    print(model.id2label)
    print(model.label2id)
    CHECKPOINT_CALLBACK = ModelCheckpoint(
                                          dirpath=os.path.join(output_dir,"checkpoints/"),
                                          filename="epoch_{epoch:02d}-valLoss_{valLoss:.2f}",
                                          save_top_k=1,
                                          monitor="valLoss",
                                          mode="min",
                                          every_n_epochs=1,  # Save the model at every epoch
                                          save_weights_only=True,
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
            callbacks=[CHECKPOINT_CALLBACK],
            log_every_n_steps=train_config['LOG_INTERVALS'],
            max_epochs=train_config['EPOCH']
        )
    print("Training starts!!")
    trainer.fit(model, datamodule=data_module)

    # trainer.save_checkpoint("mask2former.ckpt")
