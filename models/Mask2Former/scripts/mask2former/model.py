import os
import time

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torchmetrics import MeanMetric
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from transformers import Mask2FormerImageProcessor
from torch.optim.lr_scheduler import PolynomialLR
import evaluate
import json 
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class Mask2FormerFinetuner(pl.LightningModule):

    def __init__(self, model_config, train_config, output_dir=None):
        super(Mask2FormerFinetuner, self).__init__()
        self.train_loss_metric = MeanMetric(sync_on_compute=True)
        self.val_loss_metric = MeanMetric(sync_on_compute=True)
        # for multiple gpu synchronization
        self.val_iou_metric = MeanMetric(sync_on_compute=True)
        self.val_dice_metric = MeanMetric(sync_on_compute=True)
        self.model_config = model_config
        self.train_config = train_config
        self.output_dir = output_dir
        self.id2label = self.train_config["ID2LABEL"]
        self.lr = self.train_config["LR"]
        self.num_classes = len(self.id2label.keys())
        # self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_config['PRETRAIN'],
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
        )
        self.processor = Mask2FormerImageProcessor(ignore_index=254, do_resize=False,
                                                   do_rescale=False, do_normalize=True,
                                                   do_reduce_labels=True)

        self.epoch_metrics = {}
        self.validation_outputs = []
        self.val_iou_skin = []
        self.val_dice_skin = []
        self.test_iou_skin = []
        self.test_dice_skin = []

    # def lr_lambda(self, epoch):
    #     return max(
    #         (1 - epoch / self.train_config['EPOCH']) ** self.train_config['POLYNOMIAL_POWER'],
    #         self.train_config['MIN_LR'])
    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        # Your model's forward method
        return self.model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        batch['pixel_values'] = batch['pixel_values'].to(device)
        batch['mask_labels'] = [label.to(device) for label in batch['mask_labels']]
        batch['class_labels'] = [label.to(device) for label in batch['class_labels']]
        return batch

    @rank_zero_only
    def on_train_end(self):
        file_path = os.path.join(self.output_dir, "metrics.json")
        if not self.epoch_metrics:
            print("Warning: No epoch metrics to save!")
            return
        with open(file_path, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=4)
        print("****  training end  ****")

    def training_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )
        loss = outputs.loss
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, sync_dist=self.trainer.num_devices > 1,
                 batch_size=self.train_config['BATCH_SIZE'], logger=True, prog_bar=True)
        self.log("trainLoss", loss, sync_dist=self.trainer.num_devices > 1,
                 batch_size=self.train_config['BATCH_SIZE'],logger=True)
        self.train_loss_metric.update(loss.detach())

        return loss

    def on_train_epoch_end(self):

        train_loss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        # print(f"|train_epoch_loss: {train_loss} |")
        epoch_key = f"{self.current_epoch + 1} epoch"
        self.epoch_metrics.setdefault(epoch_key, {})
        self.epoch_metrics[epoch_key]["train_loss"] = train_loss.item()
        return self.epoch_metrics



    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        self.log("valLoss", loss, sync_dist=self.trainer.num_devices > 1,
                 batch_size=self.train_config['BATCH_SIZE'], logger=True)
        self.val_loss_metric.update(loss.detach())

        original_images = batch["original_images"]
        ground_truth = batch["original_segmentation_maps"]  # list[array]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]

        # predict segmentation maps
        predicted_segmentation_maps = \
            self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)  # list[tensor]
        mean_iou, mean_dice = self.score(predicted_segmentation_maps, ground_truth)
        self.val_iou_metric.update(mean_iou)
        self.val_dice_metric.update(mean_dice)
        return loss

    def on_validation_epoch_end(self):
        print(f"Running validation for epoch {self.current_epoch}")
        val_loss = self.val_loss_metric.compute()
        self.val_loss_metric.reset()
        mean_iou_skin = self.val_iou_metric.compute()
        mean_dice_skin = self.val_dice_metric.compute()

        epoch_key = f"{self.current_epoch + 1} epoch"
        if epoch_key not in self.epoch_metrics:
            print(f"Warning: {epoch_key} not found in epoch_metrics, initializing...")
            self.epoch_metrics[epoch_key] = {"val_loss": 0.0, "iou_Skin": 0.0, "dice_SKIN": 0.0}

        self.epoch_metrics[epoch_key].update({
            "val_loss": val_loss.item(),
            f"iou_{self.id2label[0]}": mean_iou_skin.item(),  #
            f"dice_{self.id2label[0]}": mean_dice_skin.item()
        })

        # log
        for k, v in self.epoch_metrics[epoch_key].items():
            self.log(k, v, sync_dist=self.trainer.num_devices > 1, on_epoch=True, logger=True)
            print(f"| {k}: {v} |")

        print(f"##### epoch {self.current_epoch + 1} metrics saved #####")

    def on_test_start(self) -> None:
        self.start_time = time.time()
        self.step_num = 0
    def test_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        original_images = batch["original_images"]
        ground_truth = batch["original_segmentation_maps"]  # list[array]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]

        # predict segmentation maps
        predicted_segmentation_maps =\
            self.processor.post_process_semantic_segmentation(outputs,target_sizes=target_sizes)  # list[tensor]
        mean_iou, mean_dice = self.score(predicted_segmentation_maps, ground_truth)
        self.step_num += 1
        print(f"***********len(iou_list): {len(self.test_iou_skin)},"
              f" len(dice_list): {len(self.test_dice_skin)}")
        self.test_iou_skin.append(mean_iou)
        self.test_dice_skin.append(mean_dice)
        self.log("iou_skin", mean_iou, logger=True, prog_bar=True,
                 batch_size=self.train_config['BATCH_SIZE'])
        self.log("dice_skin", mean_dice, logger=True, prog_bar=True,
                 batch_size=self.train_config['BATCH_SIZE'])
        return self.test_iou_skin, self.test_dice_skin

    def on_test_end(self):
        mean_iou_skin = np.mean(self.test_iou_skin)
        std_iou_skin = np.std(self.test_iou_skin)
        mean_dice_skin = np.mean(self.test_dice_skin)
        std_dice_skin = np.std(self.test_dice_skin)
        inference_time =((time.time() - self.start_time) / ( self.step_num * self.train_config['BATCH_SIZE'])) * 1000
        print("****  test end  ****")

        print(f"Mean IoU (Skin): {mean_iou_skin:.6f}")
        print(f"Std IoU (Skin): {std_iou_skin:.6f}")
        print(f"Mean Dice (Skin): {mean_dice_skin:.6f}")
        print(f"Std Dice (Skin): {std_dice_skin:.6f}")
        print(f"Inference Time (ms per sample): {inference_time:.6f}")

        # 组织结果数据
        results = {
            "Mean IoU (Skin)": [mean_iou_skin],
            "Std IoU (Skin)": [std_iou_skin],
            "Mean Dice (Skin)": [mean_dice_skin],
            "Std Dice (Skin)": [std_dice_skin],
            "Inference Time (ms per sample)": [inference_time]
        }
        df = pd.DataFrame(results)
        output_path = "test_results.csv"

        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, mode='w', header=True, index=False)

    def score(self, pred, ground_truth):
        ious = []
        dice_list = []
        for i in range(len(pred)):
            device = pred[i].device
            ground_truth = torch.from_numpy(ground_truth[i]).squeeze(1).to(device)
            mask0 = (ground_truth != 254)
            mask1 = (ground_truth != 255)
            pred_0 = (pred[i] == 0) & mask0
            true_count0 = torch.sum(pred_0).item()  # do_reduce_labels 1->0
            gt_1 = (ground_truth == 1) & mask1
            true_count1 = torch.sum(gt_1).item()  # Skin
            intersection = torch.sum(pred_0 & gt_1).float()
            union = torch.sum(pred_0 | gt_1).float()
            iou = intersection / (union + 1e-6)
            ious.append(iou)

            # dice
            # Compute Skin Dice coefficients
            dice = (2 * intersection) / (union + 1e-6) if union > 0 else 0.0
            dice_list.append(dice)

        mean_iou = torch.mean(torch.tensor(ious)).item()
        mean_dice = torch.mean(torch.tensor(dice_list)).item()
        return mean_iou, mean_dice


    def configure_optimizers(self):
        # AdamW optimizer with specified learning rate
        optimizer = AdamW(
            self.parameters(),
            lr=self.train_config['LR'],
            weight_decay=self.train_config['WEIGHT_DECAY'],
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        num_devices = torch.cuda.device_count()
        total_iters = (self.train_config['EPOCH'] * self.train_config['INTERVALS']) // num_devices
        scheduler = {
            'scheduler': PolynomialLR(optimizer, total_iters=total_iters,
                                      power=self.train_config['POWER']),
            "interval": "step",
            "frequency": self.train_config['STEP'],
        }
        # ReduceLROnPlateau scheduler
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                                             factor=self.train_config['FACTOR'],
        #                                                             patience=self.train_config['PATIENCE']),
        #     'reduce_on_plateau': True,  # Necessary for ReduceLROnPlateau
        #     'monitor': 'mean_iou'  # Metric to monitor for reducing learning rate
        # }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}



