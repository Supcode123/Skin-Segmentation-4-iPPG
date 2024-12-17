import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex


def accuracy(pred: torch.Tensor, gth: torch.Tensor, classes: int, device):

    mask = (gth != 255).float().to(device)
    total_pixels = mask.sum().float()
    if classes == 18:
        output = torch.softmax(pred, dim=1)  # [batch_size, num_classes, H, W]
        # count the number of correctly classified pixels
        correct_pixels = ((output.argmax(1) == gth) * mask).sum().float()
        acc = correct_pixels / total_pixels

        # probability of merging skin classes
        skin_classes = [1, 2]
        # [batch_size,len(skin_classes), H, W] ->[batch_size, H, W]
        skin_prob = output[:, skin_classes].sum(dim=1) # merge skin classes
        skin_pre = (skin_prob > 0.5).float()  # [batch_size, H, W]
        total_skin = ((gth == 1)|(gth == 2) * mask).sum().float()
        gth_skin = ((gth == 1) | (gth == 2)).float()
        correct_skin_pixels = ((skin_pre == gth_skin) * mask).sum().float()
        acc_skin = correct_skin_pixels / total_skin

    elif classes == 2:
        output = (torch.sigmoid(pred) > 0.5).int().squeeze(1)
        correct = ((output == gth) * mask).sum().float()
        acc = correct / total_pixels
        total_skin = ((gth == 1) * mask).sum().float()
        correct_skin_pixels = ((output == gth) * (gth == 1) * mask).sum().float()
        acc_skin = correct_skin_pixels / total_skin
    else:
        raise ValueError

    return acc, acc_skin


def loss_cal(pred: torch.Tensor, gth: torch.Tensor, classes: int, ignore: int):

    if classes == 18:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore)
        score = criterion(pred, gth)
    elif classes == 2:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss = criterion(pred.squeeze(1), gth.float())
        mask = (gth != ignore).float()
        valid_loss = loss * mask
        score = valid_loss.sum() / mask.sum()

    else:
        raise ValueError

    return score


def miou_cal(pred: torch.Tensor, gth: torch.Tensor, classes: int, device):

    skin_miou = torch.tensor(0.0, dtype=torch.float32).to(device)
    class_miou = MulticlassJaccardIndex(num_classes=classes, ignore_index=255, average='none').to(device)
    if classes == 18:
       miou = class_miou(pred, gth)
       skin_miou = (miou[1]+miou[2])/2
    if classes == 2:
       predictions = (torch.sigmoid(pred) > 0.5).int().squeeze(1)
       miou = class_miou(predictions, gth)
       skin_miou = miou[1]

    return miou, skin_miou


