import os
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torchmetrics import MeanMetric
from transformers import Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerImageProcessor
from torch.optim.lr_scheduler import PolynomialLR
import evaluate
import json 
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class Mask2FormerFinetuner(pl.LightningModule):

    def __init__(self, model_config, train_config, output_dir):
        super(Mask2FormerFinetuner, self).__init__()
        self.train_loss_metric = MeanMetric(sync_on_compute=True)
        self.val_loss_metric = MeanMetric(sync_on_compute=True)
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
        self.processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False,
                                                   do_rescale=False, do_normalize=False,
                                                   do_reduce_labels=True)

        self.metrics_cache = []
        self.validation_outputs = []
        self.metrics = evaluate.load("mean_iou")

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

        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
                json.dump(self.metrics_cache, f, indent=4)
        print("****  training end  ****")

    def training_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )
        loss = outputs.loss
        self.train_loss_metric.update(loss.detach())

        return loss

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, sync_dist=self.trainer.num_devices > 1,
                 on_epoch=True, logger=True, prog_bar=True)
        avg_loss = self.train_loss_metric.compute()
        self.log("trainLoss", avg_loss, sync_dist=self.trainer.num_devices > 1, on_epoch=True,
                 logger=True, prog_bar=True)
        self.train_loss_metric.reset()


    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        metrics = self.get_metrics(outputs, batch)
        self.val_loss_metric.update(loss.detach())
        self.validation_outputs.append(metrics)
        return {"valLoss": loss, "metrics": metrics}

    def on_validation_epoch_end(self):
        print(f"Running validation for epoch {self.current_epoch}")
        avg_loss = self.val_loss_metric.compute()
        self.log("valLoss", avg_loss, sync_dist=self.trainer.num_devices > 1, on_epoch=True,
                 logger=True, prog_bar=True)
        self.val_loss_metric.reset()

        epoch_metrics = {"epoch": self.current_epoch + 1}

        all_metrics = {}
        for output in self.validation_outputs:
            for k, v in output.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)

        # Now average the metrics
        for k, v_list in all_metrics.items():
            if k != "trainLoss" and k != "valLoss" and k != "learning_rate":
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    # Synchronize the values across all GPUs
                    v_list = torch.tensor(v_list).to(self.device)
                    torch.distributed.all_reduce(v_list)
                    all_metrics[k] = v_list.mean().item()
                else:
                    # If not using distributed, fall back to simple mean
                    all_metrics[k] = np.mean(v_list)

        # Log metrics and add them to epoch metrics
        for metric_name, metric_value in all_metrics.items():
            self.log(metric_name, metric_value, sync_dist=self.trainer.num_devices > 1, on_epoch=True,
                     logger=True, prog_bar=True)
            epoch_metrics[metric_name] = metric_value

        if self.trainer.is_global_zero:
            if not hasattr(self, "metrics_cache"):
                self.metrics_cache = []
            self.metrics_cache.append(epoch_metrics)
        self.validation_outputs.clear()
        print(f"##### epoch {self.current_epoch + 1} metrics saved #####")

    def test_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        metrics = self.get_metrics(outputs, batch)

        self.log("testLoss", loss, sync_dist=self.trainer.num_devices > 1, batch_size=self.train_config['BATCH_SIZE'],
                 logger=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(k, v, sync_dist=self.trainer.num_devices > 1, batch_size=self.train_config['BATCH_SIZE'],
                     logger=True, prog_bar=True)
        return metrics

    def get_metrics(self,outputs, batch):
        original_images = batch["original_images"]
        ground_truth = batch["original_segmentation_maps"]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]

        # predict segmentation maps
        predicted_segmentation_maps = self.processor.post_process_semantic_segmentation(outputs,
                                                                                        target_sizes=target_sizes)
        # print("Predicted Segmentation Maps Type:", len(predicted_segmentation_maps))
        # print("Predicted Segmentation ", predicted_segmentation_maps[0].shape)
        # print("Original Segmentation Maps Type:", type(ground_truth), "length: ", len(ground_truth))
        # print("Original Segmentation ", type(ground_truth[0]))
        metrics = self.metrics.compute(
            predictions=predicted_segmentation_maps,
            references=ground_truth,
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        # Extract per-category metrics
        # per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        # dice
        # Compute Skin Dice coefficients
        for pred_map, gt_map in zip(predicted_segmentation_maps, ground_truth):
            # Convert predicted tensor to numpy and ensure it's a binary mask for the current class (class 1)
            pred_class = (pred_map == 1).cpu().numpy()
            gt_class = (gt_map == 1).astype(np.uint8)

            # Ensure that we ignore class 255
            pred_class = pred_class * (gt_map != 255)
            gt_class = gt_class * (gt_map != 255).astype(np.uint8)

            intersection = np.sum(pred_class * gt_class)
            union = np.sum(pred_class) + np.sum(gt_class)

            dice = (2 * intersection) / (union + 1e-6) if union > 0 else 0.0


        # per_category_dice = [2 * iou / (iou + 1) if iou + 1 > 0 else 0 for iou in per_category_iou]

        # Re-define metrics dict to include per-category metrics
        all_metrics = {
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"],
            # "False Negative": percentage_fn,
            # "False Positive": percentage_fp,
            #**{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[1]}": per_category_iou[1]},
            **{f"Dice_{self.id2label[1]}": dice}
        }

        return all_metrics

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



