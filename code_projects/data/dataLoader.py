import albumentations as A
from torch.utils.data import DataLoader
from code_projects.data.dataset_class import Dataset


def data_load(root: str, train_info: dict, data_info: dict):
    train_dataset = Dataset(root=root,
                            classes=data_info['CLASSES'],
                            transform=A.Compose([
                                A.RandomCrop(width=224, height=224), # for SwinTransformer Windowsize
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
                                    var_limit=(10.0, 50.0),
                                    p=0.3),
                            ]),
                            img_normalization=A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            mode= 'train',
                            filter_mislabeled=data_info['FILTER_MISLABELED'])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_info["BATCH_SIZE"],
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=train_info["WORKERS"])

    val_dataset = Dataset(root=root,
                          classes=data_info['CLASSES'],
                          transform=None,
                          img_normalization=A.Normalize(mean=data_info['MEAN'], std=data_info['STD']),
                          mode='val',
                          filter_mislabeled=data_info['FILTER_MISLABELED'])

    val_dataloader = DataLoader(val_dataset,
                                batch_size=train_info["BATCH_SIZE"],
                                shuffle=False,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=train_info["WORKERS"])

    test_dataset = Dataset(root=root,
                           classes=data_info['CLASSES'],
                           transform=None,
                           img_normalization=A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           mode='test',
                           filter_mislabeled=data_info['FILTER_MISLABELED'])
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=train_info["BATCH_SIZE"],
                                 shuffle=False,
                                 drop_last=True,
                                 pin_memory=True,
                                 num_workers=train_info["WORKERS"])

    return train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
