from typing import List, Tuple

import cv2
import numpy as np
import torch


def overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,


    Returns:
        image_combined: The combined image. np.ndarray

    """
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def extract_roi(frames: List[np.ndarray], pred_masks: List[torch.Tensor], label_classes):

    rois = list()
    for frame, pred_mask in zip(frames,pred_masks):
        if label_classes == 2:
            pred = (torch.sigmoid(pred_mask) > 0.5).int().squeeze().cpu().numpy()
            kernel_erode = np.ones((3, 3), np.uint8)
            erode = cv2.erode(pred.astype(np.uint8), kernel_erode, iterations=2)  # dilation
            # mask_face = pred * 255
            mask_face = erode * 255

        elif label_classes > 2:
            pred = pred_mask.argmax(0).squeeze().cpu().numpy()  # [num_classes, H, W]
            skin = np.isin(pred, [1, 2]).astype(np.uint8)
            mask_eyes = np.isin(pred, [3, 4]).astype(np.uint8)
            mask_eyebrows = np.isin(pred, [5, 6]).astype(np.uint8)
            #
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin = cv2.erode(skin, kernel, iterations=1)  # erode
            mask_eyes = 1 - mask_eyes  # Invert the mask
            mask_eyebrows = 1 - mask_eyebrows
            # combine
            mask_face = (skin * mask_eyes * mask_eyebrows) * 255

        rois.append(mask_face)

    return rois


def apply_masks(frames: List[np.ndarray],masks:List[np.ndarray])->np.ndarray:
    """Computes the average over all color channels per frame based on the masked regions and return a RGB signal

    Args:
        frames (List[np.ndarray]): List of T Frames with NxMx3 dimension
        masks (List[np.ndarray]): List of T Masks with NxMx3 dimension

    Returns:
        np.ndarray: Tx3 RGB signals
    """
    rgbt = np.zeros((len(frames),3))
    for idx,(frame,mask) in enumerate(zip(frames,masks)):
        if mask is None or mask.size == 0:
            print(f"Warning: mask at index {idx} is empty")
            continue
        mask = np.uint8(mask)
        assert mask.shape[:2] == frame.shape[:2],(
            f"Error: mask and frame shape mismatch at index {idx}!\n"
        )
        rgbt[idx,:] = np.array(cv2.mean(frame,mask)[:-1])
    return rgbt