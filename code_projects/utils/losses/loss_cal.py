import torch
import torch.nn as nn
from code_projects.utils.losses.lovasz_loss import lovasz_softmax, lovasz_hinge


def binary_loss(pred, gth: torch.Tensor, ignore: int, bce_weight, lovasz_weight):
    """
        Calculate BCEWithLogitsLoss + lovasz_loss for binary labels
    """
    mask = (gth != ignore).float()
    bce = nn.BCEWithLogitsLoss(reduction='none')
    valid_loss = bce(pred.squeeze(1), gth.float()) * mask
    bce_loss = valid_loss.sum() / mask.sum()
    lovasz_score = lovasz_hinge(pred, gth, ignore=ignore)
    loss = bce_loss * bce_weight + lovasz_score * lovasz_weight
    return loss


def multiclass_loss(pred, gth: torch.Tensor, ignore: int, ce_weight, lovasz_weight):
    """
          Calculate CrossEntropyLoss + lovasz_loss for multi labels
    """
    ce = nn.CrossEntropyLoss(ignore_index=ignore)
    ce_loss = ce(pred, gth)
    lovasz_score = lovasz_softmax(pred, gth, ignore=ignore)
    loss = ce_loss * ce_weight + lovasz_score * lovasz_weight
    return loss


def final_loss(model_name, pred, gth: torch.Tensor, classes: int, ignore: int,
             weight1, lovasz_weight, device):

    score = torch.tensor(0.0, dtype=torch.float32).to(device)
    if classes == 2:
        if model_name == "EfficientNetb0_UNet3Plus":
            loss_list = []
            w = [0.3, 0.2, 0.2, 0.2, 0.1]
            for i in range(len(pred)):
                loss = binary_loss(pred[i], gth, ignore, weight1, lovasz_weight)
                loss_list.append(loss)
            for i in range(len(loss_list)):
                score += loss_list[i] * w[i]
        else:
            score = binary_loss(pred, gth, ignore, weight1, lovasz_weight)
        return score

    elif classes > 2:
        if model_name == "EfficientNetb0_UNet3Plus":
            loss_list = []
            w = [0.3, 0.2, 0.2, 0.2, 0.1]
            for i in range(len(pred)):
                loss = multiclass_loss(pred[i], gth, ignore, weight1, lovasz_weight)
                loss_list.append(loss)
            for i in range(len(loss_list)):
                score += loss_list[i] * w[i]
        else:
            score = multiclass_loss(pred, gth, ignore, weight1, lovasz_weight)
        return score
    else:
        raise ValueError("class_num should be >= 2")