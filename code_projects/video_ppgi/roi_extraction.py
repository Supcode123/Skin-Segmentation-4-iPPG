from typing import List

import cv2
import numpy as np
import torch

from video_ppgi.video_processing import detect_face, apply_bounding_box_mask


def extract_roi(frames: List[np.ndarray], pred_masks: List[torch.Tensor]):

    rois = list()
    for f_i, (frame, pred_mask) in enumerate(zip(frames,pred_masks)):
        face_locate = detect_face(frame)
        pred = torch.softmax(pred_mask, dim=1)  # [batch_size, num_classes, H, W]
        pred = pred.argmax(1).squeeze().cpu().numpy()
        face_area = apply_bounding_box_mask(pred, face_locate)

        mask_face = np.isin(face_area, [0, 1]).astype(np.uint8)
        mask_eyes = np.isin(face_area, [2, 3]).astype(np.uint8)
        mask_eyebrows = np.isin(face_area, [4, 5]).astype(np.uint8)

        kernel_eyes = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        mask_eyes = cv2.dilate(mask_eyes, kernel_eyes, iterations=1)  # dilation
        kernel_eyebrows = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))
        mask_eyebrows = cv2.dilate(mask_eyebrows, kernel_eyebrows, iterations=1)  # dilation
        mask_eyes = 1 - mask_eyes  # Invert the mask
        mask_eyebrows = 1 - mask_eyebrows

        # combine
        mask_face = mask_face * mask_eyes * mask_eyebrows * 255
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
        rgbt[idx,:] = np.array(cv2.mean(frame,mask)[:-1])
    return rgbt