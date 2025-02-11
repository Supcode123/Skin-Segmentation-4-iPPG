import albumentations as A
from torch.utils.data import DataLoader

from code_projects.data.dataset_class import Dataset


class Dataload():
    def __init__(self,
                 root: str,
                 train_info: dict,
                 data_info: dict,
               ):
        self.root = root
        self.train_info = train_info
        self.data_info = data_info
        self.train_dataset = Dataset(root=root,
                                exp=data_info['EXP'],
                                transform=A.Compose([
                                    # for SwinTransformer Windowsize, synthetic dataset
                                    # A.RandomCrop(width=224, height=224),

                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    # Randomly shift,zoom,rotate
                                    A.ShiftScaleRotate(
                                        shift_limit=0.1,
                                        scale_limit=0.2,
                                        rotate_limit=30,
                                        border_mode=0,
                                        p=0.8  # 执行的概率
                                    ),
                                    # Randomly adjust brightness and contrast
                                    # A.RandomBrightnessContrast(
                                    #     brightness_limit=0.2,
                                    #     contrast_limit=0.2,
                                    #     p=0.8),  # Probability of execution

                                    #Randomly adjust color saturation and hue
                                    A.ColorJitter(
                                        brightness=(0.8, 1.2),
                                        contrast=(0.8, 1.2),
                                        saturation=(0.8, 1.2),
                                        hue=(-0.1, 0.1),
                                        p=0.8),
                                    # Randomly apply Gaussian noise (to simulate ambient light interference)
                                    A.GaussNoise(
                                        var_limit=50.0,
                                        p=0.3),
                                ]),
                                img_normalization=A.Normalize(mean=data_info['MEAN'], std=data_info['STD']),
                                mode= 'train',
                                filter_mislabeled=data_info['FILTER_MISLABELED']
                                     )

        self.train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=train_info["BATCH_SIZE"],
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True,
                                      num_workers=train_info["WORKERS"],
                                      persistent_workers=True,
                                      prefetch_factor=None
                                     )

        self.val_dataset = Dataset(root=root,
                              exp=data_info['EXP'],
                              # transform= None,
                              # transform=A.CenterCrop(width=224, height=224), # for Swin_Unet, synthetic dataset
                              img_normalization=A.Normalize(mean=data_info['MEAN'], std=data_info['STD']),
                              mode='val',
                              filter_mislabeled=data_info['FILTER_MISLABELED'])

        self.val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=train_info["BATCH_SIZE"],
                                    shuffle=False,
                                    drop_last=True,
                                    pin_memory=True,
                                    num_workers=train_info["WORKERS"],
                                    persistent_workers=True,
                                    prefetch_factor=None
                                    )

        self.test_dataset = Dataset(root=root,
                               exp=data_info['EXP'],
                               transform=None,
                               img_normalization=A.Normalize(mean=data_info['MEAN'], std=data_info['STD']),
                               mode='test',
                               filter_mislabeled=data_info['FILTER_MISLABELED'])
        self.test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=train_info["BATCH_SIZE"],
                                     shuffle=False,
                                     drop_last=True,
                                     pin_memory=True,
                                     num_workers=train_info["WORKERS"]
                                    )

    def get_dataloaders(self):

       return self.train_dataset, self.train_dataloader, self.val_dataset, \
           self.val_dataloader, self.test_dataset, self.test_dataloader

# if __name__ == "__main__":
#     data_path = "D:/sythetic_data/dataset_100/256,256"
#     img_id = "000096"
#     img_path = os.path.join(data_path, "test", "images", img_id + ".png")
#     mask_path = os.path.join(data_path, "test", "labels", img_id + "_seg.png")
#     mask2_path = os.path.join(data_path, "masks", img_id + "_seg.png")
#     # plot_simple(img_path, mask2_path)
#     plot_simple(img_path, mask_path)