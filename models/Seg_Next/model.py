
from mmseg.models import MSCAN, LightHamHead
from mmseg.models.utils.wrappers import resize

from torch import nn


class seg_next(nn.Module):

      def __init__(self, model_config):
          super(seg_next, self).__init__()
          self.model_config = model_config
          self.backbone = MSCAN(
                                embed_dims=model_config['BACKBONE']["embed_dims"],
                                mlp_ratios=model_config['BACKBONE']["mlp_ratios"],
                                drop_path_rate=model_config['BACKBONE']['drop_path_rate'],
                                depths=model_config['BACKBONE']["depths"],
                                norm_cfg=model_config['BACKBONE']['norm_cfg'],
                                init_cfg=model_config['BACKBONE']['init_cfg'], # pretrain
                                )
          self.decoder = LightHamHead(
                                num_classes=model_config['DECODER_HEAD']['num_classes'],
                                channels=model_config['DECODER_HEAD']['channels'],
                                ham_channels=model_config['DECODER_HEAD']['ham_channels'],
                                ham_kwargs=model_config['DECODER_HEAD']['ham_kwargs'],
                                in_channels=model_config['DECODER_HEAD']['in_channels'],
                                in_index=model_config['DECODER_HEAD']['in_index'],
                                norm_cfg=model_config['DECODER_HEAD']['norm_cfg']
          )

      def forward(self, x):
          feature = self.backbone(x)
          # feature = feature[0]
          output = self.decoder(feature)
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