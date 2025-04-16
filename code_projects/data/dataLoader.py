import albumentations as A
import cv2
from torch.utils.data import DataLoader

from code_projects.data.dataset_class import Dataset
from code_projects.data.transfrom_pipeline import pipeline


class Dataload():
    def __init__(self,
                 root: str,
                 train_info: dict,
                 data_info: dict,
               ):
        self.root = root
        self.train_info = train_info
        self.data_info = data_info

    def get_dataloaders(self):
        self.train_dataset = Dataset(root=self.root,
                                     num_classes=self.data_info['CLASSES'],
                                     exp=self.data_info['EXP'],
                                     transform=pipeline(),
                                     swin_unet=self.data_info['SWIN_UNET'],
                                     img_normalization=A.Normalize(mean=self.data_info['MEAN'],
                                                                   std=self.data_info['STD']),
                                     mode='train',
                                     filter_mislabeled=self.data_info['FILTER_MISLABELED']
                                     )

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.train_info["BATCH_SIZE"],
                                           shuffle=True,
                                           drop_last=True,
                                           pin_memory=True,
                                           num_workers=self.train_info["WORKERS"],
                                           #prefetch_factor=None
                                           )

        self.val_dataset = Dataset(root=self.root,
                                   num_classes=self.data_info['CLASSES'],
                                   exp=self.data_info['EXP'],
                                   # transform= None,
                                   # transform=A.CenterCrop(width=224, height=224), # for Swin_Unet, synthetic dataset
                                   img_normalization=A.Normalize(mean=self.data_info['MEAN'],
                                                                 std=self.data_info['STD']),
                                   swin_unet=self.data_info['SWIN_UNET'],
                                   mode='val',
                                   filter_mislabeled=self.data_info['FILTER_MISLABELED'])

        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.train_info["BATCH_SIZE"],
                                         shuffle=False,
                                         drop_last=False,
                                         pin_memory=True,
                                         num_workers=self.train_info["WORKERS"],
                                         #prefetch_factor=None
                                         )

        return self.train_dataset, self.train_dataloader, self.val_dataset, self.val_dataloader

    def get_test_dataloaders(self):
        self.test_dataset = Dataset(root=self.root,
                                    num_classes=self.data_info['CLASSES'],
                                    exp=self.data_info['EXP'],
                                    transform=None,
                                    img_normalization=A.Normalize(mean=self.data_info['MEAN'],
                                                                  std=self.data_info['STD']),
                                    swin_unet=self.data_info['SWIN_UNET'],
                                    mode='test',
                                    filter_mislabeled=self.data_info['FILTER_MISLABELED'])
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.train_info["BATCH_SIZE"],
                                          shuffle=False,
                                          drop_last=False,
                                          pin_memory=True,
                                          num_workers=self.train_info["WORKERS"]
                                          )
        return self.test_dataset, self.test_dataloader

# if __name__ == "__main__":
#     train_config = {
#         "BATCH_SIZE": 6,
#         "WORKERS": 8,
#     }
#     data_config = {
#         "MEAN": [0.485, 0.456, 0.406],
#         "STD": [0.229, 0.224, 0.225],
#         "EXP": "EXP1",  # 你的实验设置
#         "CLASSES": 18,  # 可能是二分类还是多分类的选择
#         "SWIN_UNET": False,  # 你的原配置里是 "Ture"，修正拼写错误
#         'FILTER_MISLABELED': False
# }
#     train_dataset, train_dataloader, val_dataset, val_dataloader = Dataload(
#         root=r"D:/MA_DATA/sythetic_data/dataset_100/256x256",
#         train_info=train_config,
#         data_info=data_config).get_dataloaders()
#     for i, (img, mask, _) in enumerate(train_dataloader):
#         if i < 2:  # 只打印前5个，防止输出过多
#             print(f"Batch {i}: img shape = {img.shape}, mask shape = {mask.shape}")