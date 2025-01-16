from models.EfficientNetb0_UNet.model_create import efficientnetb0_unet
from models.UNet2Plus.model_create import unet2plus
from models.Swin_Unet.vision_transformer import SwinUnet
from models.UNet3Plus.unet3plus import UNet_3Plus_DeepSup
#from models.UNet3Plus.UNet_3Plus import UNet_3Plus


def model_select(model_cfg: dict, data_cfg: dict):
    if model_cfg['NAME'] == "EfficientNetb0_UNet":
        return efficientnetb0_unet(model_cfg, data_cfg['CLASSES'])

    elif model_cfg['NAME'] == "UnetPlusPlus":
        return unet2plus(model_cfg)

    elif model_cfg['NAME'] == "EfficientNetb0_UNet3Plus":
        return UNet_3Plus_DeepSup(model_cfg)

    # elif model_cfg['NAME'] == "UNet3Plus":
    #     return UNet_3Plus(in_channels=model_cfg['IN_CHANNELS'], n_classes=model_cfg['N_CLASSES'])

    elif model_cfg['NAME'] == "Swin_Unet":
        net = SwinUnet(model_cfg, num_classes=1)
        net.load_from(model_cfg)
        return net
