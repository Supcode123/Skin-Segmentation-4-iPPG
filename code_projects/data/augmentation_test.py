import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data.transfrom_pipeline import pipeline

img = cv2.imread(r"D:\sythetic_data\dataset_100\256x256\train\images\000061.png")[:, :, ::-1]
mask = np.array(Image.open(r"D:\sythetic_data\dataset_100\256x256\train\labels\000061_seg.png"))

transform = pipeline()
augmented = transform(image=img, mask=mask)
aug_img = augmented['image']
aug_mask = augmented['mask']

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(img); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(aug_img); plt.title("Augmented")
plt.show()