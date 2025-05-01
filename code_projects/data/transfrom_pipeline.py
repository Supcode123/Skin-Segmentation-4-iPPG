import random
from functools import partial

import albumentations as A
import cv2

from code_projects.data.occlusion_light import RandomGrayScale, Wrinkles


def pipeline():

    transform = A.Compose([
        # for SwinTransformer Windowsize, synthetic dataset
        # A.RandomCrop(width=224, height=224),
        #
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

        #RandomSkinLight(radius_range=(50, 80), alpha_range=(0.3, 0.6), p=0.3),
        #GradientLighting(alpha_range=(0.2, 3.0), p=1),

        RandomGrayScale(p=0.2),

        Wrinkles(contrast_factor=(1.0, 1.2),p=0.3),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        # # Randomly adjust brightness and contrast

        #A.ColorJitter(brightness=(0.5, 2.0), contrast=0, saturation=0, hue=0, p=1.0),
        #A.ColorJitter(contrast=(0.5, 2.0), brightness=0, saturation=0, hue=0, p=1.0),
        A.ColorJitter(saturation=(0, 3.0), contrast=0, brightness=0, hue=0, p=0.8),
        A.ColorJitter(hue=(-0.05, 0.05),saturation=0, contrast=0, brightness=0, p=0.5),

        # A.ColorJitter(
        #     brightness=(0.5, 1.5),
        #     contrast=(0.5, 1.5),
        #     saturation=(0.8, 1.2),
        #     hue=(-0.1, 0.1),
        #     p=0.8),

        # # ** Gamma transformation (simulating underexposure and overexposure) **
        A.RandomGamma(gamma_limit=(10, 150), p=0.8),

        # # Randomly apply Gaussian noise (to simulate ambient light interference)
        A.GaussNoise(
            var_limit=(20.0, 50.0),
            p=0.5),
    ], additional_targets={'mask': 'mask'})
    return transform
