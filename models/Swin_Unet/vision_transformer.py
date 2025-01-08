# coding=utf-8
import copy
import logging
import torch
import torch.nn as nn


from models.Swin_Unet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, model_cfg, num_classes=1, zero_head=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.model_cfg = model_cfg

        self.swin_unet = SwinTransformerSys(img_size=224,
                                patch_size=model_cfg['SWIN']['PATCH_SIZE'],
                                in_chans=3,
                                num_classes=self.num_classes,
                                embed_dim=model_cfg['SWIN']['EMBED_DIM'],
                                depths=model_cfg['SWIN']['DEPTHS'],
                                num_heads=model_cfg['SWIN']['NUM_HEADS'],
                                window_size=model_cfg['SWIN']['WINDOW_SIZE'],
                                mlp_ratio=model_cfg['SWIN']['MLP_RATIO'],
                                qkv_bias=model_cfg['SWIN']['QKV_BIAS'],
                                qk_scale=model_cfg['SWIN']['QK_SCALE'],
                                drop_rate=model_cfg['DROP_RATE'],
                                drop_path_rate=model_cfg['DROP_PATH_RATE'],
                                ape=model_cfg['SWIN']['APE'],
                                patch_norm=model_cfg['SWIN']['PATCH_NORM'],
                                use_checkpoint=model_cfg['USE_CHECKPOINT'])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, model_cfg):
        pretrained_path = model_cfg['PRETRAIN_CKPT']
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
 