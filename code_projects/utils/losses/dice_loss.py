import torch


def dice_coeff(prediction: torch.Tensor, target: torch.Tensor, ignore_index: int = 255,
               smooth: float = 1.0) -> torch.Tensor:
    """
    Calculate the Dice coefficient, excluding ignored pixels (such as background).
    :param prediction: predicted value, Tensor type.
    :param target: true value, Tensor type.
    :param ignore_index: category to be ignored, default is 0 (background).
    :param smooth: smoothing constant, to prevent division by zero error, default is 1.
    :return: Return the Dice coefficient.
    """
    prediction = torch.sigmoid(prediction)

    target_flat = target.view(-1)
    prediction_flat = prediction.view(-1)

    # Exclude pixels with ignore_index
    mask = target_flat != ignore_index
    target_flat = target_flat[mask]
    prediction_flat = prediction_flat[mask]
    intersection = torch.sum(target_flat * prediction_flat)
    return (2. * intersection + smooth) / (torch.sum(target_flat) + torch.sum(prediction_flat) + smooth)


def dice_loss(prediction: torch.Tensor, target: torch.Tensor, ignore_index: int = 0,
              smooth: float = 1.0) -> torch.Tensor:
    """
    Calculate Dice Loss, excluding ignored pixels.
    :param prediction: predicted value, Tensor type.
    :param target: true value, Tensor type.
    :param ignore_index: category to be ignored, default is 0 (background).
    :param smooth: smoothing constant, prevent division by zero error, default is 1.
    :return: Return Dice Loss (1 - Dice coefficient).
    """
    return 1 - dice_coeff(prediction, target, ignore_index, smooth)