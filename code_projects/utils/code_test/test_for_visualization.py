import torch
import numpy as np

from matplotlib import pyplot as plt

from code_projects.utils._plot import create_fig

torch.manual_seed(42)
np.random.seed(42)


batch_size =4
height, width = 128, 128
num_classes = 2

img_batch = torch.rand(batch_size, 3, height, width)

gt_mask_batch = torch.randint(0, num_classes, (batch_size, height, width))

prob_map = np.random.rand(256, 256)
pred_mask_batch = torch.randint(0, num_classes, (batch_size, height, width))

print("Input Image Batch Shape:", img_batch.shape)
print("Ground Truth Mask Batch Shape:", gt_mask_batch.shape)
print("Predicted Mask Batch Shape:", pred_mask_batch.shape)

fig = create_fig(pred_mask_batch, gt_mask_batch, img_batch, num_classes, prob_map)

plt.show()