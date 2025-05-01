import torch
import torch.nn as nn
from code_projects.utils.losses.lovasz_loss import lovasz_softmax, lovasz_hinge
from code_projects.utils.losses.dice_loss import DiceLoss, make_one_hot, BinaryDiceLoss


def binary_loss(val_phase, pred, gth: torch.Tensor, ignore: int, bce_weight, dice_weight):
    """
        Calculate BCEWithLogitsLoss + lovasz_loss for binary labels
    """

    mask = (gth != ignore).float()
    pos_weight = torch.tensor([2]).cuda()
    if not val_phase:
      bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    else:
      bce = nn.BCEWithLogitsLoss(reduction='none')
    valid_loss = bce(pred.squeeze(1), gth.float()) * mask
    bce_loss = valid_loss.sum() / mask.sum()
    binary_dice_loss = BinaryDiceLoss(smooth=1.0, p=2, reduction='mean',ignore_index=ignore)
    dice_score = binary_dice_loss(pred, gth)
    loss = bce_loss * bce_weight + dice_score * dice_weight
    return loss


def multiclass_loss(val_phase, pred, gth: torch.Tensor, ignore: int, cls, ce_weight, dice_weight):
    """
          Calculate CrossEntropyLoss + lovasz_loss for multi labels
    """
    # processed_weights = [
    #     0.00247,  # BACKGROUND
    #     0.01841,  # SKIN
    #     0.41939,  # NOSE
    #     1.00000,  # RIGHT_EYE
    #     0.99825,  # LEFT_EYE
    #     0.48091,  # RIGHT_BROW
    #     0.47597,  # LEFT_BROW
    #     0.28038,  # RIGHT_EAR
    #     0.27088,  # LEFT_EAR
    #     0.48305,  # MOUTH_INTERIOR
    #     0.28722,  # TOP_LIP
    #     0.25130,  # BOTTOM_LIP
    #     0.03672,  # NECK
    #     0.00654,  # HAIR
    #     0.08685,  # BEARD
    #     0.01589,  # CLOTHING
    #     0.32039,  # GLASSES
    #     0.04314,  # HEADWEAR
    #     0.00000   # FACEWEAR
    # ]
    class_weights = torch.tensor([0.0025, 0.0184, 0.4194, 1.0, 1.0, 0.4810, 0.4760,
                                  0.2804, 0.2709, 0.4831, 0.2872, 0.2513, 0.0367, 0.0066,
                                  0.0868, 0.0169, 0.3204, 0.0432, 0.0], dtype=torch.float32).to(pred.device)
    class_weights_dice = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5,
                                  0.5, 0.5, 0.5, 0.5, 0.0], dtype=torch.float32).to(pred.device)
    if not val_phase:
        ce = nn.CrossEntropyLoss(weight=class_weights,ignore_index=ignore)
        dice_loss_fn = DiceLoss(weight=class_weights_dice, reduction='mean', ignore_index=cls)
    else:
        ce = nn.CrossEntropyLoss(ignore_index=ignore)
        dice_loss_fn = DiceLoss(reduction='mean', ignore_index=cls)

    ce_loss = ce(pred, gth)
    # lovasz_score = 0.0
    # if lovasz_weight > 0.0:
    #lovasz_score = lovasz_softmax(pred, gth, ignore=ignore)
    #dice_score = 0.0
    # if dice_weight > 0.0:
    if ignore is not None:
       gth_one_hot = make_one_hot(gth, cls, 255)
    else:
       gth_one_hot = make_one_hot(gth, cls)
    dice_score = dice_loss_fn(pred, gth_one_hot)
    loss = ce_loss * ce_weight + dice_score * dice_weight
    return loss


def final_loss(val_phase, model_name, pred, gth: torch.Tensor, classes: int, ce_weight,
             bce_weight, lovasz_weight, dice_weight, device, ignore: int = 255):

    score = torch.tensor(0.0, dtype=torch.float32).to(device)
    if classes == 2:
        if model_name == "EfficientNetb0_UNet3Plus":
            loss_list = []
            w = [0.3, 0.2, 0.2, 0.2, 0.1]
            for i in range(len(pred)):
                loss = binary_loss(val_phase, pred[i], gth, ignore, bce_weight, dice_weight)
                loss_list.append(loss)
            for i in range(len(loss_list)):
                score += loss_list[i] * w[i]
        else:
            score = binary_loss(val_phase, pred, gth, ignore, bce_weight, dice_weight)
        return score

    elif classes > 2:
        if model_name == "EfficientNetb0_UNet3Plus":
            loss_list = []
            w = [0.3, 0.2, 0.2, 0.2, 0.1]
            for i in range(len(pred)):
                loss = multiclass_loss(val_phase, pred[i], gth, ignore, classes,
                                       ce_weight, dice_weight)
                loss_list.append(loss)
            for i in range(len(loss_list)):
                score += loss_list[i] * w[i]
        else:
            score = multiclass_loss(val_phase, pred, gth, ignore, classes,
                                    ce_weight, dice_weight)
        return score
    else:
        raise ValueError("class_num should be >= 2")