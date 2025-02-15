import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex
from skimage import measure

def accuracy(model_name, pred, gth: torch.Tensor, classes: int, ignore_index: int, device):

    mask = (gth != ignore_index).float().to(device)
    total_pixels = mask.sum().float()
    if classes > 2:
        output = torch.softmax(pred, dim=1)  # [batch_size, num_classes, H, W]
        # count the number of correctly classified pixels
        correct_pixels = ((output.argmax(1) == gth) * mask).sum().float()
        acc = correct_pixels / total_pixels

        # probability of merging skin classes
        skin_classes = [1, 2]
        # [batch_size,len(skin_classes), H, W] ->[batch_size, H, W]
        skin_prob = output[:, skin_classes].sum(dim=1) # merge skin classes
        skin_pre = (skin_prob > 0.5).float()  # [batch_size, H, W]
        gth_skin = ((gth == 1) | (gth == 2)).float()
        total_skin = (gth_skin * mask).sum().float()
        correct_pred_skin = ((skin_pre == 1) & (gth_skin == 1)).float()
        correct_skin_pixels = (correct_pred_skin * mask).sum().float()
        acc_skin = correct_skin_pixels / total_skin

    else:
        if model_name == "EfficientNetb0_UNet3Plus":
           pred = pred[0]
        output = (torch.sigmoid(pred) > 0.5).int().squeeze(1)
        correct = ((output == gth) * mask).sum().float()
        acc = correct / total_pixels
        total_skin = ((gth == 1) * mask).sum().float()
        correct_skin_pixels = ((output == gth) * (gth == 1) * mask).sum().float()
        acc_skin = correct_skin_pixels / total_skin

    return acc, acc_skin


def loss_cal(model_name, pred, gth: torch.Tensor, classes: int, ignore: int, device):

    if classes > 2:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore)
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
       skin_miou = (miou[1]+miou[2])/2
       miou = miou.mean()

    else:
       if model_name == "EfficientNetb0_UNet3Plus":
            pred = pred[0]
       binary_miou = BinaryJaccardIndex(ignore_index=ignore).to(device)
       predictions = (torch.sigmoid(pred) > 0.5).float().squeeze(1)
       miou = binary_miou(predictions, gth)
       skin_miou = miou

    return miou, skin_miou


def Dice_cal(model_name, pred, gth: torch.Tensor, ignore_index: int, device, smooth=1e-6):
    if model_name == "EfficientNetb0_UNet3Plus":
        pred = pred[0]
    mask = (gth != ignore_index).float().to(device)
    pred = (torch.sigmoid(pred) > 0.5).int().squeeze(1)
    pred_skin = (pred == 1).float()
    gth_skin = (gth == 1).float()
    intersection = torch.sum(pred_skin * gth_skin * mask, dim=(1, 2))
    dice = (2. * intersection + smooth) / (torch.sum(pred_skin * mask, dim=(1, 2)) +
                                           torch.sum(gth_skin * mask, dim=(1, 2)) + smooth)
    return torch.mean(dice)


def get_boundary_points(gt, pred):
    """
    Extract border points from mask.

    Parameters:
    - pred_tensor: (b, h, w) prediction mask tensor, each element is 0, 1 or 255.
    - gt_tensor: (b, h, w) true label mask tensor, each element is 0, 1 or 255.

    Returns:
    - pred_boundary_points: The coordinates of the predicted boundary points, shape (N, 2).
    - gt_boundary_points: The coordinates of the actual label boundary points, shape (N, 2).
    """
    pred_boundary_points_list = []
    gt_boundary_points_list = []

    for i in range(gt.shape[0]):
        pred_mask = pred[i].cpu().numpy()
        gt_mask = gt[i].cpu().numpy()
        valid_mask = (gt_mask != 255).astype(np.uint8)

        pred_contours, _ = cv2.findContours((pred_mask * valid_mask).astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(pred_contours) > 0:
            pred_boundary = np.vstack(pred_contours[0]).squeeze()  # (N, 2)
            pred_boundary_points_list.append(pred_boundary)
        else:
            pred_boundary_points_list.append(np.array([]))

        gt_contours, _ = cv2.findContours((gt_mask * valid_mask).astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(gt_contours) > 0:
            gt_boundary = np.vstack(gt_contours[0]).squeeze()  # (N, 2)
            gt_boundary_points_list.append(gt_boundary)
        else:
            gt_boundary_points_list.append(np.array([]))

    return pred_boundary_points_list, gt_boundary_points_list


def compute_assd(gt, pred):
    """
    Calculate Average Symmetric Surface Distance (ASSD)

    Parameters:
    - pred_boundaries: List[np.ndarray], each element (N, 2)
    - gt_boundaries: List[np.ndarray], each element (M, 2)

    Returns:
    - ASSD value
    """
    pred_boundaries, gt_boundaries = get_boundary_points(gt,pred)
    batch_size = len(pred_boundaries)
    assd_list = []

    for i in range(batch_size):
        pred_boundary = pred_boundaries[i]
        gt_boundary = gt_boundaries[i]
        if pred_boundary.shape[0] == 2 and len(pred_boundary.shape) == 1:
            pred_boundary = pred_boundary.reshape(1, 2)  # transform to shape(1, 2)
        if gt_boundary.shape[0] == 2 and len(gt_boundary.shape) == 1:
            gt_boundary = gt_boundary.reshape(1, 2)

        if pred_boundary.size == 0 or gt_boundary.size == 0:
            assd_list.append(np.inf)  # If is empty, ASSD is set to infinity.
            continue

        pred_tree = cKDTree(pred_boundary)
        gt_tree = cKDTree(gt_boundary)

        d_pred_to_gt, _ = pred_tree.query(gt_boundary)
        d_gt_to_pred, _ = gt_tree.query(pred_boundary)

        assd = (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)) / 2.0
        assd_list.append(assd)

    avg_assd = np.mean([x for x in assd_list if x != np.inf])
    return avg_assd

