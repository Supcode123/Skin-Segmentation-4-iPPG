import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A
from data.transfrom_pipeline import pipeline


img = cv2.imread(r"D:\MA_DATA\sythetic_data\dataset_100\image\000081.png")[:, :, ::-1]
mask = np.array(Image.open(r"D:\MA_DATA\sythetic_data\dataset_100\label\000081_seg.png"))
# x, y = 250, 100
# label = mask[y, x]
# print(f"labels: {np.unique(mask)}")
# print(f"掩码中位置 [100, 250] 的标签值是: {label}")

transform=A.CenterCrop(width=350, height=350)
transform2=A.CenterCrop(width=224, height=224)

#mask = center_crop(mask)
augmented = transform(image=img, mask=mask)
aug_img = augmented['image']
aug_mask = augmented['mask']
d_img = cv2.resize(aug_img, (256, 256), interpolation=cv2.INTER_AREA)
d_mask = cv2.resize(aug_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
augmented2 = transform2(image=d_img, mask=d_mask)
aug_img2 = augmented2['image']


plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.imshow(img); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(aug_img2); plt.title("Augmented")
plt.show()