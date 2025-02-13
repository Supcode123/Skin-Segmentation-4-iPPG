import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_images(img, remapped_mask, remapped_colormap, classes_exp):
    """
    Generates plot of Image and RGB mask with class colorbar
    :param img: 3D ndarray of input image
    :param remapped_mask: 2D/3D ndarray of input segmentation mask with class ids
    :param remapped_colormap: dictionary that indicates color corresponding to each class
    :param classes_exp: dictionary of classes names and corresponding class ids
    :param experiment: experimental setup
    :return: plot of image and rgb mask with class colorbar
    """
    mask_rgb = mask_to_colormap(remapped_mask, colormap=remapped_colormap)

    fig, axs = plt.subplots(1, 2, figsize=(26, 7))
    plt.subplots_adjust(left=1 / 16.0, right=1 - 1 / 16.0, bottom=1 / 8.0, top=1 - 1 / 8.0)
    axs[0].imshow(img)
    axs[0].axis("off")

    img_u_labels = np.unique(remapped_mask)
    c_map = []
    cl = []
    for i_label in img_u_labels:
        if i_label == 255:  # Skip ignore_label (255)
            continue
        for i_key, i_color in remapped_colormap.items():
            if i_label == i_key:
                c_map.append(i_color)
        for i_key, i_class in classes_exp.items():
            if i_label == i_key:
                cl.append(i_class)
    cl = np.asarray(cl)
    cmp = np.asarray(c_map) / 255
    cmap_mask = LinearSegmentedColormap.from_list("seg_mask_colormap", cmp, N=len(cmp))
    im = axs[1].imshow(mask_rgb, cmap=cmap_mask)
    intervals = np.linspace(0, 255, num=len(c_map) + 1 )
    ticks = intervals[:-1] + int((intervals[1] - intervals[0]) / 2)
    divider = make_axes_locatable(axs[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(mappable=im, cax=cax1, ticks=ticks, orientation="vertical")
    cbar1.ax.set_yticklabels(cl)
    axs[1].axis("off")
    fig.tight_layout()
    plt.show()

def plot_simple(img_path, mask_path):
    """
    Function for plotting of mask with smplified classes
    :param img: 3D ndarray of input image
    :param mask: 2D/3D ndarray of input segmentation mask with class ids
    :return: plot of image and rgb mask with class colorbar
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
    remapped_mask, classes_exp, colormap = remap_simple(mask)
    return plot_images(img, remapped_mask, colormap, classes_exp)

def remap_original(mask):
    """Remap mask for original image&mask"""
    class_remapping_exp = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [4],
        5: [5],
        6: [6],
        7: [7],
        8: [8],
        9: [9],
        10: [10],
        11: [11],
        12: [12],
        13: [13],
        14: [14],
        15: [15],
        16: [16],
        17: [17],
        # 18: [18],
    }
    classes_exp = {
        0: 'Background',
        1: 'Skin',
        2: 'Nose',
        3: 'Right_Eye',
        4: 'Left_Eye',
        5: 'Right_Brow',
        6: 'Left_Brow',
        7: 'Right_Ear',
        8: 'Left_Ear',
        9: 'Mouth_Interior',
        10: 'Top_Lip',
        11: 'Bottom_Lip',
        12: 'Neck',
        13: 'Hair',
        14: 'Beard',
        15: 'Clothing',
        16: 'Glasses',
        17: 'Headwear',
       # 18: 'Facewear',
    }
    colormap = get_remapped_colormap(class_remapping_exp)
    remapped_mask = remap_mask(mask, class_remapping=class_remapping_exp)
    return remapped_mask, classes_exp, colormap


def remap_simple(mask):
    """Remap mask which smplified classes"""
    class_remapping_exp = {
         0: [0],
         1: [1,2],
         2: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],

    }

    classes_exp = {
        0: "Background",
        1: "Skin",
        2: "Non-Skin",
    }

    colormap = get_remapped_colormap(class_remapping_exp)
    remapped_mask = remap_mask(mask, class_remapping=class_remapping_exp)
    return remapped_mask, classes_exp, colormap

def get_colormap():
    """
    Returns colormap
    :return: ndarray of rgb colors
    """
    return np.asarray(
        [
            [58, 0, 82],
            [253, 234, 39],
            [255, 156, 201],
            [99, 0, 255],
            [255, 0, 0],
            [255, 0, 165],
            [255, 255, 255],
            [141, 141, 141],
            [255, 218, 0],
            [173, 156, 255],
            [73, 73, 73],
            [250, 213, 255],
            [255, 156, 156],
            [99, 255, 0],
            [157, 225, 255],
            [255, 89, 124],
            [173, 255, 156],
            [255, 60, 0],
            [40, 0, 255],
        ]
    )


def remap_mask(mask, class_remapping, ignore_label=255):
    """
    Remaps mask class ids
    :param mask: 2D/3D ndarray of input segmentation mask
    :param class_remapping: dictionary that indicates class remapping
    :param ignore_label: class ids to be ignored
    :return: 2D/3D ndarray of remapped segmentation mask
    """
    classes = []
    for key, val in class_remapping.items():
        for cls in val:
            classes.append(cls)
    assert len(classes) == len(set(classes))

    N = max(len(classes), mask.max() + 1)
    remap_array = np.full(N, ignore_label, dtype=np.uint8)
    for key, val in class_remapping.items():
        for v in val:
            remap_array[v] = key
    return remap_array[mask]


def get_remapped_colormap(class_remapping):
    """
    Generated colormap of remapped classes
    Classes that are not remapped are indicated by the same color across all experiments
    :param class_remapping: dictionary that indicates class remapping
    :return: 2D ndarray of rgb colors for remapped colormap
    """
    colormap = get_colormap()
    remapped_colormap = {}
    for key, val in class_remapping.items():
        if key == 255:
            remapped_colormap.update({key: [0, 0, 0]})
        else:
            remapped_colormap.update({key: colormap[val[0]]})
    return remapped_colormap


def mask_to_colormap(mask, colormap):
    """
    Genarates RGB mask colormap from mask with class ids
    :param mask: 2D/3D ndarray of input segmentation mask
    :param colormap: dictionary that indicates color corresponding to each class
    :return: 3D ndarray Generated RGB mask
    """
    rgb = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    for label, color in colormap.items():
        rgb[mask == label] = color
    return rgb


def subplot(fig, position: list, remapped_mask, mask_rgb, colormap, classes_exp):
    """
    Generates plot of Image and RGB mask with class colorbar
    :param fig: matplotlib.figure.Figure object
    :param position: subgraph position index, for example [row number, column number, index]
    :param remapped_mask: 2D/3D ndarray of input segmentation mask with class ids
    :param mask_rgb: 3D ndarray Generated RGB mask
    :param colormap: dictionary that indicates color corresponding to each class
    :param classes_exp: corresponding class numbers
    :return: plot of image and rgb mask with class colorbar
    """
    N = position[0]
    sample_count = position[1]
    pos = position[2]
    ax = fig.add_subplot(N, sample_count, pos, xticks=[], yticks=[])
    img_u_labels = np.unique(remapped_mask)  # 获取唯一标签
    c_map = []
    cl = []
    if len(img_u_labels) == 1:
        label = img_u_labels[0]
        c_map = [colormap[label]]
        cl = [classes_exp[label]]
    else:
        for i_label in img_u_labels:
            if i_label == 255:  # Skip ignore_label (255)
                continue
            for i_key, i_color in colormap.items():
                if i_label == i_key:
                    c_map.append(i_color)
            for i_key, i_class in classes_exp.items():
                if i_label == i_key:
                    cl.append(i_class)

    cl = np.asarray(cl)
    cmp = np.asarray(c_map) / 255
    cmap_mask = LinearSegmentedColormap.from_list("seg_mask_colormap", cmp, N=len(cmp))
    im = ax.imshow(mask_rgb, cmap=cmap_mask)

    if len(c_map) > 1:
        intervals = np.linspace(0, 255, num=len(cmp) + 1)
        ticks = intervals[:-1] + int(intervals[1] - intervals[0]) / 2
        divider = make_axes_locatable(ax)
        caxl = divider.append_axes("right", size="5%", pad=0.05)
        cbarl = fig.colorbar(mappable=im, cax=caxl, ticks=ticks, orientation="vertical")
        cbarl.ax.set_yticklabels(cl)

    ax.axis("off")

def plot(path, remapped_mask, mask_rgb, colormap, classes_exp, name: str, probability_map=None):
    """
    Generates plot of Image and RGB mask with class colorbar
    :param path: save path
    :param remapped_mask: 2D/3D ndarray of input segmentation mask with class ids
    :param mask_rgb: 3D ndarray Generated RGB mask
    :param colormap: dictionary that indicates color corresponding to each class
    :param classes_exp: corresponding class numbers
    :param name: str name of img
    :param probability_map: 2D ndarray of class probabilities
    :return: plot of image and rgb mask with class colorbar
    """
    img_u_labels = np.unique(remapped_mask)  # 获取唯一标签
    c_map = []
    cl = []
    for i_label in img_u_labels:
        if i_label == 255:  # Skip ignore_label (255)
            continue
        for i_key, i_color in colormap.items():
            if i_label == i_key:
                c_map.append(i_color)
        for i_key, i_class in classes_exp.items():
            if i_label == i_key:
                cl.append(i_class)
    # cl = np.asarray(cl)
    cmp = np.asarray(c_map) / 255
    cmap_mask = LinearSegmentedColormap.from_list("seg_mask_colormap", cmp, N=len(cmp))

    plt.figure(figsize=(5, 5), dpi=100)
    plt.xticks([])
    plt.yticks([])
    im = plt.imshow(mask_rgb)
    im.set_cmap(cmap_mask)

    # Optionally add probability heatmap overlay
    if probability_map is not None:
        # Ensure values are within [0, 1]
        prob_norm = np.clip(probability_map.cpu().numpy(), 0, 1)
        colors = [(0.0, (58/255, 0/255, 82/255)),
                  (1.0, (253/255, 234/255, 39/255))]  # define the start and end colors of the gradient
        n_bins = 100  # set the subdivision level of the gradient
        cmap_name = "purple_yellow"
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        plt.imshow(prob_norm, cmap=custom_cmap, alpha=0.5)  # Overlay with alpha transparency
        # ax_color = divider.append_axes("left", size="5%", pad=0.05)
        # colorbar = plt.colorbar(cax, cax=ax_color, orientation="vertical")
        # colorbar.ax.yaxis.set_ticks_position('left')

    plt.axis('off')
    save_path = os.path.join(path, name)
    plt.savefig(save_path)


def create_fig(pred_mask_batch: torch.Tensor, gt_mask_batch: torch.Tensor,
               img_batch: torch.Tensor, cls: int):
    """
    Given a batch of predicted masks, gt masks and input images,
    return a matplotlib figure.

    Args:
        pred_mask_batch(tensor): batch of predicted masks, (B, H, W)
        gt_mask_batch(tensor): batch of gt masks, (B, H, W)
        img_batch(tensor): batch of input images, (B, C, H, W)
        prob: batch of 2D ndarray of class probabilities

    Returns:
        fig: plot object
    """

    N, _, _ = pred_mask_batch.size()
    assert pred_mask_batch.shape == gt_mask_batch.shape
    assert N == img_batch.shape[0]
    assert pred_mask_batch.shape[-2:] == img_batch.shape[-2:]

    fig = plt.figure(figsize=(N * 5, 4 * 2.5), dpi=100)
    # col_titles = ['Input Image', 'Ground Truth', 'Predicted (19 classes)', 'Predicted (2 classes)']
    # for i, title in enumerate(col_titles):
    #     fig.text(0.1 + i * 0.25, 1.0, title, ha='center', va='center', fontsize=12)
    for n in range(N):
        if cls == 18 :
            # with 19 classes
            remapped_pred_mask_18, classes_18, colormap_18 = remap_original(pred_mask_batch[n, ...].cpu().numpy())
            remapped_gt_mask, _, _ = remap_original(gt_mask_batch[n, ...].cpu().numpy())
            # with 2 classes
            remapped_pred_mask_2, classes_2, colormap_2 = remap_simple(pred_mask_batch[n, ...].cpu().numpy())

            pred_mask_rgb_18 = mask_to_colormap(remapped_pred_mask_18, colormap=colormap_18)
            pred_mask_rgb_2 = mask_to_colormap(remapped_pred_mask_2, colormap=colormap_2)
            gt_mask_rgb = mask_to_colormap(remapped_gt_mask, colormap=colormap_18)
            img = img_batch[n, ...].permute(1, 2, 0).cpu().numpy()
            # image
            ax = fig.add_subplot(N, 4, n * 4 + 1, xticks=[], yticks=[])
            ax.imshow(img)
            # ground truth mask
            subplot(fig, [N, 4, n * 4 + 2], remapped_gt_mask, gt_mask_rgb, colormap_18, classes_18)
            # predicted mask with 19 classes
            subplot(fig, [N, 4, n * 4 + 3], remapped_pred_mask_18, pred_mask_rgb_18, colormap_18, classes_18)
            # remapped the predicted mask with 2 classes
            subplot(fig, [N, 4, n * 4 + 4], remapped_pred_mask_2, pred_mask_rgb_2, colormap_2, classes_2)

        elif cls == 2 :
            remapped_pred_mask, classes, colormap = remap_simple(pred_mask_batch[n, ...].cpu().numpy())

            remapped_gt_mask, _, _ = remap_simple(gt_mask_batch[n, ...].cpu().numpy())
            pred_mask_rgb = mask_to_colormap(remapped_pred_mask, colormap=colormap)
            gt_mask_rgb = mask_to_colormap(remapped_gt_mask, colormap=colormap)
            img = img_batch[n, ...].permute(1, 2, 0).cpu().numpy()
            # image
            ax = fig.add_subplot(N, 3, n * 3 + 1, xticks=[], yticks=[])
            ax.imshow(img)
            # ground truth mask
            subplot(fig, [N, 3, n * 3 + 2], remapped_gt_mask, gt_mask_rgb, colormap, classes)
            # remapped the predicted mask with 2 classes
            subplot(fig, [N, 3, n * 3 + 3], remapped_pred_mask, pred_mask_rgb, colormap, classes)
            # subplot(fig, [N, 4, n * 4 + 4], remapped_pred_mask, pred_mask_rgb, colormap, classes, prob[n, ...])
        else:
            raise ValueError

        ax.axis("off")
        fig.tight_layout()
    return fig


def create_fig_test(samples: list, save_path: str):

    for n in range(4):
        remapped_pred_mask, classes, colormap = remap_simple(samples[n][1].cpu().numpy())
        remapped_gt_mask, _, _ = remap_simple(samples[n][2].cpu().numpy())
        pred_mask_rgb = mask_to_colormap(remapped_pred_mask, colormap=colormap)
        gt_mask_rgb = mask_to_colormap(remapped_gt_mask, colormap=colormap)
        sample = denormalize(samples[n][3], [123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
        img = sample.permute(1, 2, 0).cpu().numpy()
        # image
        img_name = samples[n][0][0].split('.')[0]
        dir_path = os.path.join(save_path, img_name)
        os.makedirs(dir_path)
        plt.imsave(os.path.join(dir_path, 'img.png'), img)

        # ground truth mask
        plot(dir_path, remapped_gt_mask, gt_mask_rgb, colormap, classes, 'gt.png')
        # predicted mask with 2 classes
        plot(dir_path, remapped_pred_mask, pred_mask_rgb, colormap, classes, 'pred.png')
        # predicted mask with 2 classes and probability heatmap
        plot(dir_path, remapped_pred_mask, pred_mask_rgb, colormap, classes, 'hotmap.png', samples[n][4])


def denormalize(img_tensor: torch.Tensor, mean: list , std: list) -> torch.Tensor:

    mean = torch.tensor(mean, device=img_tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img_tensor.device).view(1, -1, 1, 1)

    return img_tensor * std + mean


# if __name__ == "__main__":
#     data_path = "D:/sythetic_data/dataset_100/256,256"
#     img_id = "000096"
#     img_path = os.path.join(data_path, "test", "images", img_id + ".png")
#     mask_path = os.path.join(data_path, "test", "labels", img_id + "_seg.png")
#     mask2_path = os.path.join(data_path, "masks", img_id + "_seg.png")
#     # plot_simple(img_path, mask2_path)
#     plot_simple(img_path, mask_path)