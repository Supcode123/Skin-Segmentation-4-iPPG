import segmentation_models_pytorch as smp

""" here used the pre-trained Model,which  
reference to https://github.com/qubvel-org/segmentation_models.pytorch/tree/main"""


def model_create(model_cfg: dict, data_cfg: dict):
    backbone = model_cfg['BACKBONE']
    head = model_cfg['HEAD']
    cls = data_cfg['CLASSES']
    model = getattr(smp, head)(
        encoder_name=backbone,
        encoder_weights='imagenet',
        classes=cls,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16]
    )
    return model
