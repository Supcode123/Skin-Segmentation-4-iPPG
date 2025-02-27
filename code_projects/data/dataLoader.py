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
                                     exp=self.data_info['EXP'],
                                     transform=pipeline(),
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
                                   exp=self.data_info['EXP'],
                                   # transform= None,
                                   # transform=A.CenterCrop(width=224, height=224), # for Swin_Unet, synthetic dataset
                                   img_normalization=A.Normalize(mean=self.data_info['MEAN'],
                                                                 std=self.data_info['STD']),
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
                                    exp=self.data_info['EXP'],
                                    transform=None,
                                    img_normalization=A.Normalize(mean=self.data_info['MEAN'],
                                                                  std=self.data_info['STD']),
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
#     data_path = "D:/sythetic_data/dataset_100/256,256"
#     img_id = "000096"
#     img_path = os.path.join(data_path, "test", "images", img_id + ".png")
#     mask_path = os.path.join(data_path, "test", "labels", img_id + "_seg.png")
#     mask2_path = os.path.join(data_path, "masks", img_id + "_seg.png")
#     # plot_simple(img_path, mask2_path)
#     plot_simple(img_path, mask_path)