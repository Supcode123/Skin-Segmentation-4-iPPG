import os
import torch
import torch.utils.data as data
import numpy as np
from torchvision import transforms
from albumentations.pytorch.functional import img_to_tensor
from PIL import Image
TT = transforms.ToTensor()


class Dataset(data.Dataset):
    def __init__(self,
                 root: str,
                 spatial_transform=None,
                 img_normalization=None,
                 mode: str = "train",
                 filter_mislabeled: bool = False):

        """ Creates class instance.

        :param root: Path to dataset
        :param spatial_transform: Transformations applied to images AND label masks
        :param img_normalization: Normalization transformations, only applied to labels
        :param mode: train/val/test
        """
        assert os.path.isdir(root)
        self.sample_list = []
        self.mode = mode
        self.normalization = img_normalization
        self.spatial_transform = spatial_transform

        self.filter_mislabeled = filter_mislabeled
        self.mislabeled_samples = []
        mislabeled_file_path = './mislabeled_files.txt'
        if filter_mislabeled:
            if os.path.exists(mislabeled_file_path):
                with open(mislabeled_file_path, 'r') as f:
                    for line in f.readlines():
                        self.mislabeled_samples.append(line.replace('\n', ''))
            else:
                print("file not exist")
        img_file_list = []
        label_file_list = []

        if mode in ['train', 'val', 'test']:
            _path = os.path.join(root, mode)
        else:
            raise ValueError(f"invalid mode: {mode}, valid values should be ['train', 'val', 'test']")

        for _, _dir, _file in os.walk(_path):
            for d in _dir:
                if d == "images":
                    for f in _file:
                        img_file_list.append(os.path.join(_, d, f))
                if d == "labels":
                    for f in _file:
                        label_file_list.append(os.path.join(_, d, f))

        # Filter list of image file paths
        if filter_mislabeled:
            _img_file_list = []
            _label_file_list = []
            for img_file, label_file in zip(img_file_list, label_file_list):
                clean = True
                for mislabeled_sample in self.mislabeled_samples:
                    if mislabeled_sample in img_file:
                        clean = False
                if clean:
                    _img_file_list.append(img_file)
                    _label_file_list.append(label_file)
            img_file_list = _img_file_list
            label_file_list = _label_file_list

        assert len(img_file_list) == len(label_file_list)

        for i in range(len(img_file_list)):
            img_name = img_file_list[i].split('/')[-1]
            tmp_dict = {"img": img_file_list[i],
                        "mask": label_file_list[i],
                        "name": img_name}
            self.sample_list.append(tmp_dict)

        def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, str):
            sample = self.sample_list[index]
            img = np.array(Image.open(sample["img"]))
            mask = np.array(Image.open(sample["mask"]))
            file_name = sample["name"]

            if self.spatial_transform is not None:
                # To ensure the same transformation is applied to img + mask
                data = self.spatial_transform(image=img, mask=mask)
                img = data['image']
                mask = data['mask']

            if self.normalization is not None:
                img = self.normalization(image=img)['image']

            img = img_to_tensor(img)
            mask = torch.from_numpy(mask)

            mask = self.remap_mask(mask)
            mask = mask.long().squeeze(0)

            return img, mask, file_name

        def __len__(self) -> int:
            return len(self.sample_list)

        def img_size(self) -> tuple:
            return self[0][0].shape

