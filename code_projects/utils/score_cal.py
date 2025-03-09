import cv2
import numpy as np
import torch
import torch.nn as nn
from medpy.metric import assd
from scipy.spatial import cKDTree
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex
from skimage import measure

def accuracy(model_name, pred, gth: torch.Tensor, classes: int, ignore_index: int, device):

    mask = (gth != ignore_index).float().to(device)
    total_pixels = mask.sum().float()
    if classes > 2:
        output = torch.softmax(pred, dim=1)  # [batch_size, num_classes, H, W]
        output = output.argmax(1)
        # count the number of correctly classified pixels
        correct_pixels = ((output == gth) * mask).sum().float()
        acc = correct_pixels / total_pixels


        # [batch_size,len(skin_classes), H, W] ->[batch_size, H, W]
        # correct_pred_skin = (output == 0) & (gth == 0)
        # correct_pred_nose = (output == 1) & (gth == 1)
        # # merge skin classes
        # correct_pred_skin_nose = (correct_pred_skin | correct_pred_nose).float()
        # correct_skin_pixels = (correct_pred_skin_nose * mask).sum()
        # gth_skin = ((gth == 0) | (gth == 1)).float()
        # total_skin = (gth_skin * mask).sum()
        # acc_skin = correct_skin_pixels / total_skin


    else:
        if model_name == "EfficientNetb0_UNet3Plus":
           pred = pred[0]
        output = (torch.sigmoid(pred) > 0.5).int().squeeze(1)
        # correct = ((output == gth) * mask).sum().float()
        # acc = correct / total_pixels
        #total_skin = ((gth == 1) * mask).sum().float()
        #correct_skin_pixels = ((output == gth) * (gth == 1) * mask).sum().float()
        correct_pixels = ((output == gth) * mask).sum().float()
        acc = correct_pixels / total_pixels

    return acc


def loss_cal(model_name, pred, gth: torch.Tensor, classes: int, ignore: int, device):

    if classes > 2:
        num_classes = pred.shape[1]
        weights = torch.ones(num_classes).to(pred.device)  # By default, all categories have a weight of 1.
        weights[0] = weights[1] = 10.0
        weights[2] = weights[3] = weights[4] = weights[5] = \
            weights[8] = weights[9] = weights[10] = 5.0
        criterion = nn.CrossEntropyLoss(ignore_index=ignore, weight=weights)
        score = criterion(pred, gth)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        score = torch.tensor(0.0, dtype=torch.float32).to(device)
        if model_name == "EfficientNetb0_UNet3Plus":
            loss_list=[]
            w = [0.3, 0.2, 0.2, 0.2, 0.1]
            for i in range(len(pred)):
                loss = criterion(pred[i].squeeze(1), gth.float())
                mask = (gth != ignore).float()
                valid_loss = loss * mask
                score = valid_loss.sum() / mask.sum()
                loss_list.append(score)
            for i in range(len(loss_list)):
                score += loss_list[i] * w[i]
        else:
            loss = criterion(pred.squeeze(1), gth.float())
            mask = (gth != ignore).float()
            valid_loss = loss * mask
            score = valid_loss.sum() / mask.sum()

    return score


def miou_cal(model_name, pred, gth: torch.Tensor, classes: int, ignore: int, device):

    # miou = torch.tensor(0.0, dtype=torch.float32).to(device)
    # skin_miou = torch.tensor(0.0, dtype=torch.float32).to(device)

    if classes > 2:
       class_miou = MulticlassJaccardIndex(num_classes=classes, ignore_index=ignore, average='none').to(device)
       miou = class_miou(pred, gth)
       skin_miou = (miou[0]+miou[1])/2
       # miou = miou.mean()

    else:
       if model_name == "EfficientNetb0_UNet3Plus":
            pred = pred[0]
       binary_miou = BinaryJaccardIndex(ignore_index=ignore).to(device)
       predictions = (torch.sigmoid(pred) > 0.5).float().squeeze(1)
       skin_miou = binary_miou(predictions, gth)
       # skin_miou = miou

    return skin_miou


def Dice_cal(model_name, pred, gth: torch.Tensor, classes: int, ignore_index: int, device, smooth=1e-6):

    mask = (gth != ignore_index).float().to(device)
    if classes > 2:
        output = torch.softmax(pred, dim=1)
        output = output.argmax(1)
        pred_skin = ((output == 0) | (output ==1)).float()
        gth_skin = ((gth == 0) | (gth == 1)).float()
    else:
        if model_name == "EfficientNetb0_UNet3Plus":
            pred = pred[0]
        pred = (torch.sigmoid(pred) > 0.5).int().squeeze(1)
        pred_skin = (pred == 1).float()
        gth_skin = (gth == 1).float()
    intersection = torch.sum(pred_skin * gth_skin * mask, dim=(1, 2))
    dice = (2. * intersection + smooth) / (torch.sum(pred_skin * mask, dim=(1, 2)) +
                                           torch.sum(gth_skin * mask, dim=(1, 2)) + smooth)
    return torch.mean(dice)


def compute_assd(gt, pred, model_name, classes):
    """
    Calculate Average Symmetric Surface Distance (ASSD)

    Parameters:
    - ground truth: batch tensor of ground truth labels
    - pred_mask: batch tensor of predicted masks

    Returns:
    - ASSD value
    """

    if model_name == "EfficientNetb0_UNet3Plus":
        pred = pred[0]

    assd_list = []
    for i in range(pred.size(0)):
        gt_np = gt[i].cpu().numpy().astype(np.uint8)
        # print("Unique values in gt_np:", torch.unique(label[i]))
        if classes > 2:
            output = pred[i].argmax(0).cpu().numpy().astype(np.uint8)
            pred_mask = (((output == 0) | (output == 1)) & (gt_np != 255)).astype(np.uint8)
            gt_mask = (((gt_np == 0) | (gt_np == 1)) & (gt_np != 255)).astype(np.uint8)
        else:
            binary_pred = (torch.sigmoid(pred[i]) > 0.5).int().squeeze(0)
            pred_np = binary_pred.cpu().numpy().astype(np.uint8)
            # print("Unique values in pred_np:", np.unique(pred_np))
            pred_mask = ((pred_np == 1) & (gt_np != 255)).astype(np.uint8)
            gt_mask = ((gt_np == 1) & (gt_np != 255)).astype(np.uint8)
        # calculate ASSD
        if pred_mask.sum() > 0 and gt_mask.sum() > 0:
            assd_value = assd(pred_mask, gt_mask)
            if not np.isnan(assd_value) and not np.isinf(assd_value):
               assd_list.append(assd_value)
        else:
            assd_list.append(0)
        #print(f"the {i + 1}th image: assd = {assd_value}")
    avg_assd = np.mean(assd_list)
    return avg_assd

