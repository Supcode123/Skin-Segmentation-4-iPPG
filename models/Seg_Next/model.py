import os
import sys

import yaml

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'mmsegmentation')))
from mmseg.models.segmentors import EncoderDecoder


def seg_next(model_config):
    model = EncoderDecoder(backbone=model_config["BACKBONE"],
                           decode_head=model_config["DECODER_HEAD"])
    return model


# if __name__ == "__main__":
#
#   path = r'C:\kshi\Skin-Segmentation-4-iPPG\code_projects\configs\model\SegNext.yaml'
#   with open(path, "r") as doc:
#        model_config = yaml.load(doc, Loader=yaml.Loader)
#   model = seg_next(model_config)
#   print(model)