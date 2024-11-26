import segmentation_models_pytorch as smp

""" here used the pre-trained Model,which  
reference to https://github.com/qubvel-org/segmentation_models.pytorch/tree/main"""


def model_create(model_cfg: dict, data_cfg: dict):
    backbone = model_cfg['BACKBONE']
    head = model_cfg['HEAD']
    cls = data_cfg['CLASSES']
    model = getattr(smp, head)(
        encoder_name=backbone,
        encoder_weights=model_cfg['ENCODER_WEIGHTS'],
        classes=cls,
        activation=None,
        encoder_depth=model_cfg['ENCODER_DEPTH'],
        decoder_channels=model_cfg['DECODER_CHANNELS']
    )
    return model
