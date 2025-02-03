from mmseg.models.segmentors import EncoderDecoder


def seg_next(model_config):
    model = EncoderDecoder(backbone=model_config["BACKBONE"],
                           decode_head=model_config["DECODER_HEAD"])
    return model._init_decode_head(model_config["DECODER_HEAD"])