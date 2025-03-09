from typing import Tuple

import cv2
import numpy as np
import torch
from albumentations.pytorch.functional import img_to_tensor
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
import mediapipe as mp
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, images: list[np.ndarray], transform=None):
        """
        Args:
            images (list[np.ndarray]): raw image list (H, W, C) format
            transform: torchvision preprocessing (such as ToTensor, Normalize)
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Make sure the image dimensions are correct (H, W, C)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image


transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])


def load_video(video_path):
    video_file = video_path     #VIDEO PATH (e.g. "*.mp4")
    T = 20 # seconds

    frames = list()
    cap = cv2.VideoCapture(video_file)
    fs = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(f"frame_width is {frame_width} \n frame_height is {frame_height}")

    n_frames = int(fs*T)
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Video too short! Could only load {i}/{n_frames} frames")
            break
        # h, w, _ = frame.shape
        # crop_size = min(h, w)
        # start_x = (w - crop_size) // 2
        # start_y = (h - crop_size) // 2
        # cropped_frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
        rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    print(f"Loaded {i+1} frames.")
    return frames


def detect_face(frame: np.ndarray):
    """Use BlazeFace to perform face detection and return the cropped face area"""
    mp_face_detection = mp.solutions.face_detection


    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Make sure the cropped area does not extend beyond the image boundaries
                x, y, w, h = max(0, x), max(0, y - int(0.07 * h)) , min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
                # face_crop = frame[y:y + h, x:x + w]
                return (x, y, w, h)

    print("Warning: No face detected in the frame!")
    return None  # No face detected

def segment_skin(face_img: list[np.ndarray], batch_size: int = 1):
    """Using U-Net + EfficientNetB0 for skin segmentation"""
    pred_list = []
    dataset = ImageDataset(face_img, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights=None,
        classes=18,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16]
    ).to('cuda')
    model.load_state_dict(
        torch.load(r'D:\Skin-Segmentation-4-iPPG\log\EfficientNetb0_UNet_synthetic\model_checkpoint_18.pt',
                   map_location='cuda'))
    model.eval()

    with torch.no_grad():
        pbar = tqdm(dataloader)
        for i, sample in enumerate(pbar, start=1):
            sample = sample.to("cuda" if torch.cuda.is_available() else "cpu")
            pred = model(sample)
            pred_list.append(pred)
        print(f"**** inference of {i} imgs done ")
    return pred_list


def apply_bounding_box_mask(output: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Sets the output values outside the bounding box to 0, keeping only the area inside the bounding box.

    Args:
        output (np.ndarray): The segmentation output where non-zero values represent the skin region.
        bbox (tuple): The bounding box coordinates (x, y, w, h).

    Returns:
        np.ndarray: Modified output with values outside the bounding box set to 0.
    """
    # Unpack the bounding box
    x, y, w, h = bbox

    # Set the values outside the bounding box to 0
    output_with_bbox = output.copy()
    output_with_bbox[:y, :] = 255 # Top region
    output_with_bbox[y + h:, :] = 255  # Bottom region
    output_with_bbox[:, :x] = 255  # Left region
    output_with_bbox[:, x + w:] = 255  # Right region

    return output_with_bbox

def extract_roi(frames: list[np.ndarray], pred_masks: list[torch]):

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


if __name__ == "__main__":
    data_path = r"D:\MA_DATA\video\vid.avi"
    frames = load_video(data_path)
    test_frame = frames[:2]
    #resized_frame = cv2.resize(frames[0], (256, 256), interpolation=cv2.INTER_LINEAR)
    pred = segment_skin(test_frame)
    rois = extract_roi(test_frame, pred)
    overlayed_roi = overlay(test_frame[0].astype(np.uint8), rois[0], (0, 255, 0), 0.3)
    plt.figure()
    plt.imshow(overlayed_roi)
    plt.show()










    # # overlay
    # mask = frames[10].copy()
    # # expanded_output = output.copy()
    #
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # # eyes_dilated = cv2.dilate(mask_eyes, kernel, iterations=1)
    #
    # # mask_dilated = cv2.morphologyEx(mask_2_4, cv2.MORPH_OPEN, kernel)
    # # expanded_output[mask_dilated == 1] = 2
    # #
    # # mask_0_1 = np.isin(expanded_output, 2)
    # #green_overlay[..., 1] = pred_mask * 255
    # mask_face = extract_roi(outputs)
    # mask[..., 1][mask_face.astype(bool)] = 255
    #
    # alpha = 0.4
    # overlay = cv2.addWeighted(frames[10], 1 - alpha, mask, alpha, 0)
    # #
    # #model = model.to('cuda')
    # pred_mask = pred_mask.astype(np.uint8) * 255
    # plt.figure()
    # plt.imshow(overlay)
    # plt.show()
    #
    # # openning (remove small noise dots)
    # def openning_operation(img, interration: int=1):
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    #     #kernel = np.ones((6, 6), np.uint8)
    #     kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     for i in range(interration):
    #         img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #         img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    #     print(f"{i+1} times done")
    #     return img
    #
    #
    # open_mask = openning_operation(pred_mask, 1)
    # green_overlay2 = np.zeros_like(overlay)
    # green_overlay2[..., 1] = open_mask
    # morphology = (overlay * (1 - alpha) + green_overlay2 * alpha).astype(np.uint8)
    #
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #
    # axes[0].imshow(frames[10])
    # axes[0].set_title("Origin")
    #
    # axes[1].imshow(overlay)
    # axes[1].set_title(f"Processed Mask (Closing -> Opening)")
    # plt.show()