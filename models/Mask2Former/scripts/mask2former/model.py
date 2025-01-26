import os.path

import pytorch_lightning as pl
import torch
from transformers import Mask2FormerForUniversalSegmentation
from transformers import AutoImageProcessor
import evaluate
import time
import json 
import numpy as np


class Mask2FormerFinetuner(pl.LightningModule):

    def __init__(self, model_config, train_config, output_dir):
        super(Mask2FormerFinetuner, self).__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.output_dir = output_dir
        self.id2label = self.train_config["ID2LABEL"]
        self.lr = self.train_config["LR"]
        self.num_classes = len(self.id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_config['PRETRAIN'],
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(model_config['PRETRAIN'])
        # evaluate.load
        self.test_mean_iou = evaluate.load("mean_iou")

        self.metrics_cache = []

    def lr_lambda(self, epoch):
        return max(
            (1 - epoch / self.train_config['EPOCH']) ** self.train_config['POLYNOMIAL_POWER'],
            self.train_config['MIN_LR'])
    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        # Your model's forward method
        return self.model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        batch['pixel_values'] = batch['pixel_values'].to(device)
        batch['mask_labels'] = [label.to(device) for label in batch['mask_labels']]
        batch['class_labels'] = [label.to(device) for label in batch['class_labels']]
        return batch
        
    def on_train_start(self):
        self.start_time = time.time()


    def on_train_end(self):
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            for epoch_metrics in self.metrics_cache:
                json.dump(epoch_metrics, f)
                f.write('\n')

    def training_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )
        loss = outputs.loss
        self.log("trainLoss", loss, sync_dist=self.trainer.num_devices > 1,  batch_size=self.train_config['BATCH_SIZE'])
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        metrics = self.get_metrics(outputs, batch)

        self.log("valLoss", loss, sync_dist=self.trainer.num_devices > 1, batch_size=self.train_config['BATCH_SIZE'])
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, sync_dist=self.trainer.num_devices > 1,
                 batch_size=self.train_config['BATCH_SIZE'], on_epoch=True, logger=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(k, v, sync_dist=self.trainer.num_devices > 1, batch_size=self.train_config['BATCH_SIZE'])
        return loss

    def on_epoch_end(self):

        total_time = time.time() - self.start_time
        epoch_metrics = {'epoch': self.current_epoch, **self.metrics}
        self.metrics_cache.append(epoch_metrics)

        # 在终端打印当前 epoch 的所有日志
        print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}, 'training_time': {total_time} :")
        for key, value in self.metrics.items():
            print(f"{key}: {value}")

    def test_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        metrics = self.get_metrics(outputs, batch)
        self.log("testLoss", loss, sync_dist=self.trainer.num_devices > 1, batch_size=self.train_config['BATCH_SIZE'])
        for k, v in metrics.items():
            self.log(k, v, sync_dist=self.trainer.num_devices > 1, batch_size=self.train_config['BATCH_SIZE'])
        return metrics

    def get_metrics(self,outputs, batch):
        original_images = batch["original_images"]
        ground_truth = batch["original_segmentation_maps"]
        target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]

        # predict segmentation maps
        predicted_segmentation_maps = self.processor.post_process_semantic_segmentation(outputs,
                                                                                        target_sizes=target_sizes)
        predictions = predicted_segmentation_maps[0].cpu().numpy()

        # Calculate FN and FP
        # false_negatives = np.sum((predictions == 0) & (ground_truth[0] == 1))
        # false_positives = np.sum((predictions == 1) & (ground_truth[0] == 0))

        # Total number of instances
        # total_instances = np.prod(predictions.shape)

        # # Calculate percentages
        # percentage_fn = (false_negatives / total_instances)
        # percentage_fp = (false_positives / total_instances)

        # Calculate IoU and accuracy metrics
        metrics = self.test_mean_iou._compute(
            predictions=predictions,
            references=ground_truth[0],
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        # Extract per-category metrics
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        # dice
        # Compute per-category Dice coefficients
        per_category_dice = []
        for i in range(self.num_classes):
            # Get binary masks for the current class
            pred_class = (predictions == i).astype(np.uint8)
            gt_class = (ground_truth[0] == i).astype(np.uint8)

            # Calculate intersection and union
            intersection = np.sum(pred_class * gt_class)
            union = np.sum(pred_class) + np.sum(gt_class)

            # Avoid division by zero
            dice = (2 * intersection) / union if union > 0 else 0.0
            per_category_dice.append(dice)
        # per_category_dice = [2 * iou / (iou + 1) if iou + 1 > 0 else 0 for iou in per_category_iou]

        # Re-define metrics dict to include per-category metrics
        metrics = {
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"],
            # "False Negative": percentage_fn,
            # "False Positive": percentage_fp,
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)},
            **{f"Dice_{self.id2label[i]}": v for i, v in enumerate(per_category_dice)}

        }

        return metrics

    def configure_optimizers(self):
        # AdamW optimizer with specified learning rate
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)

        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         lr_lambda=self.lr_lambda),
        #     'interval': 'epoch',  # Update every epoch
        #     'frequency': 1,  # Every epoch
        # }
        # ReduceLROnPlateau scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=self.train_config['FACTOR'],
                                                                    patience=self.train_config['PATIENCE']),
            'reduce_on_plateau': True,  # Necessary for ReduceLROnPlateau
            'monitor': 'valLoss'  # Metric to monitor for reducing learning rate
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
