import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex


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
        gth_skin = ((gth == 1) | (gth == 2)).float()
        total_skin = (gth_skin * mask).sum().float()
        correct_pred_skin = ((skin_pre == 1) & (gth_skin == 1)).float()
        correct_skin_pixels = (correct_pred_skin * mask).sum().float()
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

    miou = torch.tensor(0.0, dtype=torch.float32).to(device)
    skin_miou = torch.tensor(0.0, dtype=torch.float32).to(device)

    if classes == 18:
       class_miou = MulticlassJaccardIndex(num_classes=classes, ignore_index=255, average='none').to(device)
       miou = class_miou(pred, gth)
       skin_miou = (miou[1]+miou[2])/2
       miou = miou.mean()

    if classes == 2:
       binary_miou = BinaryJaccardIndex(ignore_index=255).to(device)
       predictions = (torch.sigmoid(pred) > 0.5).float().squeeze(1)
       miou = binary_miou(predictions, gth)
       skin_miou = miou

    return miou, skin_miou




