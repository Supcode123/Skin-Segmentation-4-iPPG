import os

import torch
from PIL import Image
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader 
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import albumentations as A

from models.Mask2Former.scripts.mask2former.experiments import EXP
from models.Mask2Former.scripts.mask2former.config import _args, _config



train_transform = A.Compose([
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


# def remap_mask(mask: torch.Tensor, exp_dict: dict, ignore_label: int = 255):
#
#     class_remapping = exp_dict["LABEL"]
#     remap_array = np.full(256, ignore_label, dtype=np.uint8)
#     for key, val in class_remapping.items():
#         for v in val:
#             remap_array[v] = key
#     mask = mask.int()
#     remap_mask = remap_array[mask]
#     remap_mask_tensor = torch.from_numpy(remap_mask)
#     return remap_mask_tensor

class ImageSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, normalization=None, transform=None ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.normalization = normalization
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if not f.startswith('.')]
        self.EXP = EXP
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.filenames[idx] + '.png')
        mask_path = os.path.join(self.masks_dir, self.filenames[idx] + '_seg.png')

        original_image = np.array(Image.open(img_path).convert("RGB"))
        original_mask = np.array(Image.open(mask_path).convert("L"))
        if self.transform is not None:
            # To ensure the same transformation is applied to img + mask
            data = self.transform(image=original_image, mask=original_mask)
            img = data['image']
            mask = data['mask']
            #img = self.normalization(image=img)['image']
        else:
            #img = self.normalization(image=original_image)['image']
            img = original_image
            mask = original_mask
        # convert to C, H, W
        image = img.transpose(2, 0, 1)


        return image, mask, original_image, original_mask


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, train_conf, model_conf):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = train_conf['BATCH_SIZE']
        self.num_workers = train_conf['WORKERS']
        self.processor = Mask2FormerImageProcessor(ignore_index=254, do_resize=False,
                                                   do_rescale=False, do_normalize=True,
                                                   do_reduce_labels=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'train', 'images'),
                                                          masks_dir=os.path.join(self.dataset_dir, 'train', 'new_labels'),
                                                          transform=train_transform,
                                                          )
            # Add your transforms here
            self.val_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir,  'val', 'images'),
                                                        masks_dir=os.path.join(self.dataset_dir, 'val', 'new_labels'),
                                                        ) # Add your transforms here

            print(f"{len(self.train_dataset)} training samples.")
            print(f"{len(self.val_dataset)} validation samples.")
        if stage == 'test' or stage is None:
            self.test_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'test', 'images'),
                                                         masks_dir=os.path.join(self.dataset_dir, 'test', 'new_labels'),
) # Add your transforms here
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
                          num_workers=self.num_workers, drop_last=True, pin_memory=False,
                          persistent_workers=False, collate_fn=self.collate_fn)

    def collate_fn(self,batch):
        inputs = list(zip(*batch))
        images=inputs[0]
        segmentation_maps=inputs[1]
        batch = self.processor(
            images=images,
            segmentation_maps=segmentation_maps,
            size=(256,256),
            return_tensors="pt",
        )
        batch["original_images"] = inputs[2]
        batch["original_segmentation_maps"] = inputs[3]

        return batch


# if __name__=="__main__":
#
#    args = _args()
#    print(args)
#    output_dir, data_config, model_config, train_config = _config(args)
#    data = SegmentationDataModule(dataset_dir=args.data_path, model_conf=model_config, train_conf=train_config)
#    data.setup('test')
#    dataloader = data.test_dataloader()
#    #print(dataset)
#    batch = next(iter(dataloader))
#    for k, v in batch.items():
#       if isinstance(v, torch.Tensor):
#           print(k, v.shape)
#       else:
#           print(k, v[0].shape)
   # pixel_values torch.Size([1, 3, 256, 256])
   # pixel_mask torch.Size([1, 256, 256])
   # mask_labels torch.Size([15, 256, 256])
   # class_labels torch.Size([15])
   # original_images (256, 256, 3)
   # original_segmentation_maps (256, 256)
