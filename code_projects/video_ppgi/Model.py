import segmentation_models_pytorch as smp
import torch


def model_load():
    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights=None,
        classes=1,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16]
    ).to('cuda')
    model.load_state_dict(
        torch.load(r'D:\Skin-Segmentation-4-iPPG\log\EfficientNetb0_UNet_synthetic\model_checkpoint.pt',
                   map_location='cuda')) #C:\kshi\Skin-Segmentation-4-iPPG\log\model_checkpoint_2.pt
    return model