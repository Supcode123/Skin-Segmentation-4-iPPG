import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A

from data.transfrom_pipeline import pipeline

img = cv2.imread(r"D:\MA_DATA\sythetic_data\dataset_100\image\000050.png")[:, :, ::-1]
mask = np.array(Image.open(r"D:\MA_DATA\sythetic_data\dataset_100\label\000001_seg.png"))

transform=A.CenterCrop(width=350, height=350)
augmented = transform(image=img, mask=mask)
aug_img = augmented['image']
aug_mask = augmented['mask']

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(img); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(aug_img); plt.title("Augmented")
plt.show()