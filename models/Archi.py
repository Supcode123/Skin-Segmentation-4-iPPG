from models.EfficientNetb0_UNet import model_create
from models.unet3plus.UNet_3Plus import UNet_3Plus


def model_select(model_cfg: dict, num_classes: int = 2):
    if model_cfg['NAME'] == "EfficientNetb0_UNet":
        return model_create.efficientnetb0_unet(model_cfg, num_classes)

    if model_cfg['NAME'] == "Unet3plus":
        # if num_classes == 2:
        #     cls = 1
        return UNet_3Plus(in_channels=3, n_classes=1)