from models.EfficientNetb0_UNet.model_create import efficientnetb0_unet
from models.UNet2Plus.model_create import unet2plus


def model_select(model_cfg: dict, data_cfg: dict):
    if model_cfg['NAME'] == "EfficientNetb0_UNet":
        return efficientnetb0_unet(model_cfg, data_cfg['CLASSES'])

    if model_cfg['NAME'] == "UnetPlusPlus":
        return unet2plus(model_cfg)

