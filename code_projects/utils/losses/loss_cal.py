import torch
import torch.nn as nn
from code_projects.utils.losses.lovasz_loss import lovasz_softmax, lovasz_hinge
from code_projects.utils.losses.dice_loss import DiceLoss, make_one_hot, BinaryDiceLoss


def binary_loss(pred, gth: torch.Tensor, ignore: int, bce_weight, dice_weight):
    """
        Calculate BCEWithLogitsLoss + lovasz_loss for binary labels
    """

    mask = (gth != ignore).float()
    pos_weight = torch.tensor([6.5]).cuda()
    bce = nn.BCEWithLogitsLoss(reduction='none',pos_weight = pos_weight)
    valid_loss = bce(pred.squeeze(1), gth.float()) * mask
    bce_loss = valid_loss.sum() / mask.sum()
    dice_score = 0.0
    if dice_weight > 0.0:
        binary_dice_loss = BinaryDiceLoss(smooth=1.0, p=2, reduction='mean',ignore_index=255)
        dice_score = binary_dice_loss(pred, gth)
    loss = bce_loss * bce_weight + dice_score * dice_weight
    return loss


def multiclass_loss(pred, gth: torch.Tensor, cls, ignore: int, ce_weight, lovasz_weight, dice_weight):
    """
          Calculate CrossEntropyLoss + lovasz_loss for multi labels
    """

    class_weights = torch.tensor([0.25, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2,
                                  1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 0.8, 0.8, 0.8, 0.8], dtype=torch.float32).cuda()
    ce = nn.CrossEntropyLoss(ignore_index=ignore,weight=class_weights)
    ce_loss = ce(pred, gth)
    lovasz_score = 0.0
    if lovasz_weight > 0.0:
       lovasz_score = lovasz_softmax(pred, gth, ignore=ignore)
    dice_score = 0.0
    if dice_weight > 0.0:
        dice_loss_fn = DiceLoss(weight=class_weights, reduction='mean')
        gth_one_hot = make_one_hot(gth, cls)
        dice_score = dice_loss_fn(pred, gth_one_hot)
    loss = ce_loss * ce_weight + lovasz_score * lovasz_weight + dice_score * dice_weight
    return loss


def final_loss(model_name, pred, gth: torch.Tensor, classes: int, ignore: int,
             ce_bce_weight, lovasz_weight, dice_weight, device):

    score = torch.tensor(0.0, dtype=torch.float32).to(device)
    if classes == 2:
        if model_name == "EfficientNetb0_UNet3Plus":
            loss_list = []
            w = [0.3, 0.2, 0.2, 0.2, 0.1]
            for i in range(len(pred)):
                loss = binary_loss(pred[i], gth, ignore, ce_bce_weight, dice_weight)
                loss_list.append(loss)
            for i in range(len(loss_list)):
                score += loss_list[i] * w[i]
        else:
            score = binary_loss(pred, gth, ignore, ce_bce_weight, dice_weight)
        return score

    elif classes > 2:
        if model_name == "EfficientNetb0_UNet3Plus":
            loss_list = []
            w = [0.3, 0.2, 0.2, 0.2, 0.1]
            for i in range(len(pred)):
                loss = multiclass_loss(pred[i], gth, ignore, classes,
                                       ce_bce_weight, lovasz_weight, dice_weight)
                loss_list.append(loss)
            for i in range(len(loss_list)):
                score += loss_list[i] * w[i]
        else:
            score = multiclass_loss(pred, gth, ignore, classes,
                                    ce_bce_weight, lovasz_weight, dice_weight)
        return score
    else:
        raise ValueError("class_num should be >= 2")