from typing import List

import cv2
import numpy as np
import torch


def extract_roi(frames: List[np.ndarray], pred_masks: List[torch.Tensor]):

    rois = list()
    for f_i, (frame, pred_mask) in enumerate(zip(frames,pred_masks)):

        # pred = pred_mask.argmax(0).squeeze().cpu().numpy() # [num_classes, H, W]
        pred = (torch.sigmoid(pred_mask) > 0.5).int().squeeze().cpu().numpy()
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # opened = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        # skin = np.isin(pred, [1, 2]).astype(np.uint8)
        # kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # kernel_erode = np.ones((3, 3), np.uint8)
        # erode = cv2.erode(pred.astype(np.uint8), kernel_erode, iterations=2)  # dilation
        #mask_face = erode

        # mask_face = np.isin(erode, 1).astype(np.uint8)

        # mask_eyes = np.isin(face_area, [2, 3]).astype(np.uint8)
        # mask_eyebrows = np.isin(face_area, [4, 5]).astype(np.uint8)
        #
        # kernel_eyes = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        # mask_eyes = cv2.dilate(mask_eyes, kernel_eyes, iterations=1)  # dilation
        # kernel_eyebrows = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))
        # mask_eyebrows = cv2.dilate(mask_eyebrows, kernel_eyebrows, iterations=1)  # dilation
        # mask_eyes = 1 - mask_eyes  # Invert the mask
        # mask_eyebrows = 1 - mask_eyebrows

        # combine
        # mask_face = mask_face * mask_eyes * mask_eyebrows * 255
        mask_face = pred * 255
        #mask_face = erode * 255
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