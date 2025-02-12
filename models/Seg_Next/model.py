
from mmseg.models import EncoderDecoder
from mmseg.models.utils.wrappers import resize

from torch import nn


class seg_next(nn.Module):

      def __init__(self, model_config):
          super(seg_next, self).__init__()
          self.model_config = model_config

          self.model = EncoderDecoder(backbone=model_config['BACKBONE'],
                                      decode_head=model_config['DECODER_HEAD'],
                                      init_cfg=model_config['init_cfg']
                                        )

      def forward(self, x):
          output = self.model(x)
          output = resize(
              input=output,
              size=(256,256),
              mode="bilinear",
              align_corners=False
          )

          return output

# if __name__ == "__main__":
#
#   path = r'C:\kshi\Skin-Segmentation-4-iPPG\code_projects\configs\model\SegNext.yaml'
#   with open(path, "r") as doc:
#        model_config = yaml.load(doc, Loader=yaml.Loader)
#   model = seg_next(model_config)
#   print(model)