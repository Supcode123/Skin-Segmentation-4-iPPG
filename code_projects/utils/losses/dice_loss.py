import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes, ignore_index: int = None):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, H, W] containing class indices.
         num_classes: Number of classes.
         ignore_index: Optional integer. Class index to ignore.
    Returns:
        A tensor of shape [N, num_classes, H, W] as one-hot encoded.
    """
    if ignore_index is not None:
        input = input.clone()
        input[input == ignore_index] = int(num_classes)
    shape = list(input.shape)
    shape.insert(1, num_classes+1)  # Insert num_classes dimension at position 1
    one_hot = torch.zeros(shape, dtype=torch.float32, device=input.device)
    one_hot = one_hot.scatter_(1, input.unsqueeze(1), 1)
    return one_hot


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean', ignore_index=None, foreground_weight=2.5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.foreground_weight = foreground_weight

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        if predict.shape[1] == 1:
            predict = predict.squeeze(1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            predict = predict[mask]
            target = target[mask]

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Calculate foreground-weighted Dice Loss
        weight_map = torch.ones_like(target)
        weight_map[target == 1] = self.foreground_weight

        num = torch.sum(weight_map * predict * target, dim=1) + self.smooth
        den = torch.sum(weight_map * (predict.pow(self.p) + target.pow(self.p)), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index(cls) to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape[1] == target.shape[1] - 1, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1] - 1):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1] - 1, \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]