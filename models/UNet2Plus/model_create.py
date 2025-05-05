import segmentation_models_pytorch as smp

""" here used the pre-trained Model,which  
reference to https://github.com/qubvel-org/segmentation_models.pytorch/tree/main"""


def unet2plus(model_cfg,num_classes):
    backbone = model_cfg['BACKBONE']
    head = model_cfg['HEAD']
    model = getattr(smp, head)(
        encoder_name=backbone,
        encoder_weights=model_cfg['ENCODER_WEIGHTS'],
        classes=num_classes,
        activation=None,
        encoder_depth=model_cfg['ENCODER_DEPTH'],
        decoder_channels=model_cfg['DECODER_CHANNELS']
    )
    return model
