import random

import albumentations as A
import cv2

from data.occlusion_light import StripeShadow, RandomSkinLight,\
    GradientLighting, PerlinTexture, RandomGrayScale


def pipeline():

    transform = A.Compose([
        # for SwinTransformer Windowsize, synthetic dataset
        # A.RandomCrop(width=224, height=224),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Randomly shift,zoom,rotate
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8  # 执行的概率
        ),
        # Randomly adjust brightness and contrast
        # A.RandomBrightnessContrast(
        #     brightness_limit=0.2,
        #     contrast_limit=0.2,
        #     p=0.8),  # Probability of execution

        RandomSkinLight(p=0.5),
        GradientLighting(alpha_range=(0.2, 2.0), p=0.5),
        # ** Occlusion enhancement **
        StripeShadow(num_stripes=4, p=0.3),  # stripes of light

        # Randomly adjust color saturation and hue
        A.ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.7, 1.3),
            saturation=(0.7, 1.3),
            hue=(-0.1, 0.1),
            p=0.8
        ),

        # A.ColorJitter(
        #     brightness=(0.8, 1.2),
        #     contrast=(0.8, 1.2),
        #     saturation=(0.8, 1.2),
        #     hue=(-0.1, 0.1),
        #     p=0.8),

        # ** Gamma transformation (simulating underexposure and overexposure) **
        # A.RandomGamma(gamma_limit=(0.9, 1.1), p=1.0),
        PerlinTexture(p=0.5),
        RandomGrayScale(p=0.3),
        # Randomly apply Gaussian noise (to simulate ambient light interference)
        A.GaussNoise(
            var_limit=(10.0, 30.0),
            p=0.3),
    ], additional_targets={'mask': 'mask'})
    return transform
