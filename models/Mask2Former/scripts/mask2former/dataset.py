import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader 
from transformers import AutoImageProcessor
import albumentations as A

from code_projects.data.experiments import EXP2
from models.Mask2Former.scripts.mask2former.config import _args, _config


def remap_mask(mask: np.ndarray, exp_dict: dict, ignore_label: int = 255) -> np.ndarray:

    if not hasattr(remap_mask, "remap_array"):
        class_remapping = exp_dict["LABEL"]
        remap_array = np.full(256, ignore_label, dtype=np.uint8)
        for key, val in class_remapping.items():
            for v in val:
                remap_array[v] = key
        remap_mask.remap_array = remap_array
    else:
        remap_array = remap_mask.remap_array

    # Ensure mask values are within the valid range
    assert mask.min() >= 0 and mask.max() <= 255, "Mask values must be in the range [0, 255]"

    # Apply the remapping using the remap_array
    remapped_mask = remap_array[mask]

    return remapped_mask

class ImageSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if not f.startswith('.')]
        self.EXP = EXP2
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.filenames[idx] + '.png')
        mask_path = os.path.join(self.masks_dir, self.filenames[idx] + '_seg.png')
        image = Image.open(img_path).convert("RGB")
        np_image=np.array(image)
        # convert to C, H, W
        np_image = np_image.transpose(2,0,1)
        mask = Image.open(mask_path).convert("L")
        mask=np.array(mask)
        np_mask = remap_mask(mask, self.EXP)
        np_mask[np_mask == 255] = 1

        return np_image, np_mask


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, model_conf, train_conf):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.model_conf = model_conf
        self.train_conf = train_conf
        self.batch_size = train_conf['BATCH_SIZE']
        self.num_workers = train_conf['WORKERS']
        self.processor = AutoImageProcessor.from_pretrained(model_conf['PRETRAIN'], use_fast=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'train', 'images'),
                                                          masks_dir=os.path.join(self.dataset_dir, 'train', 'labels'),
                                                          transform=A.Compose([
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
                                    A.ColorJitter(
                                        brightness=(0.8, 1.2),
                                        contrast=(0.8, 1.2),
                                        saturation=(0.8, 1.2),
                                        hue=(-0.1, 0.1),
                                        p=0.8),
                                    A.GaussNoise(
                                        var_limit=50.0,
                                        p=0.3),
                                ])
            ) # Add your transforms here
            self.val_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir,  'val', 'images'),
                                                        masks_dir=os.path.join(self.dataset_dir, 'val', 'labels'),
                                                        transform=None) # Add your transforms here
            print(f"{len(self.train_dataset)} training samples.")
            print(f"{len(self.val_dataset)} validation samples.")
        if stage == 'test' or stage is None:
            self.test_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'test', 'images'),
                                                         masks_dir=os.path.join(self.dataset_dir, 'test', 'labels'),
                                                         transform=None) # Add your transforms here
            print(f"{len(self.test_dataset)} validation samples.")

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True,
                          persistent_workers=False, prefetch_factor=None, collate_fn=self.collate_fn)
        print(f"load train dataloader.")
        return train_loader
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True,
                          persistent_workers=False, prefetch_factor=None, collate_fn=self.collate_fn)
        print(f"load val dataloader.")
        return val_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True,
                          collate_fn=self.collate_fn)

    def collate_fn(self,batch):
      inputs = list(zip(*batch))
      images=inputs[0]
      segmentation_maps=inputs[1]
      batch = self.processor(
          images,
          segmentation_maps=segmentation_maps,
          size=(256,256),
          return_tensors="pt",
      )
      batch["original_images"] = images
      batch["original_segmentation_maps"] = segmentation_maps

      return batch


# if __name__=="__main__":
#    args = _args()
#    print(args)
#    output_dir, data_config, model_config, train_config = _config(args)
#    data = SegmentationDataModule(dataset_dir=args.data_path, model_conf=model_config, train_conf=train_config)
#    data.setup('fit')
#    dataset = data.train_dataset
#    print(dataset)
#    batch = next(iter(dataloader))
#    for k, v in batch.items():
#       if isinstance(v, torch.Tensor):
#           print(k, v.shape)
#       else:
#           print(k, v[0].shape)