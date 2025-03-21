import segmentation_models_pytorch as smp

""" here used the pre-trained Model,which  
reference to https://github.com/qubvel-org/segmentation_models.pytorch/tree/main"""


def efficientnetb0_unet(model_cfg, num_classes):
    if num_classes > 2:
        cls = 19
    elif num_classes == 2:
        cls = 1
    else:
        raise ValueError
    backbone = model_cfg['BACKBONE']
    head = model_cfg['HEAD']
    model = getattr(smp, head)(
        encoder_name=backbone,
        encoder_weights=model_cfg['ENCODER_WEIGHTS'],
        classes=cls,
        activation=None,
        encoder_depth=model_cfg['ENCODER_DEPTH'],
        decoder_channels=model_cfg['DECODER_CHANNELS']
    )
    return model
