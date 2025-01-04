from models.EfficientNetb0_UNet.model_create import efficientnetb0_unet
from models.UNet2Plus.model_create import unet2plus


def model_select(model_cfg: dict, num_classes: int = 2):
    if model_cfg['NAME'] == "EfficientNetb0_UNet":
        return efficientnetb0_unet(model_cfg, num_classes)

    if model_cfg['NAME'] == "Unet++":

        return unet2plus(model_cfg)
