from typing import List
from torchvision import transforms
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, images: List[np.ndarray], transform=None):
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


def center_crop_faces(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Perform center crop (256x256) on each frame. Skip frames that are too small."""
    cropped_faces = []
    crop_size = 256
    half_crop = crop_size // 2

    for i, frame in enumerate(frames):
        img_h, img_w, _ = frame.shape

        # Skip frames that are too small
        if img_h < crop_size or img_w < crop_size:
            print(f"Warning: Frame {i} too small ({img_w}x{img_h}), skipped.")
            continue

        # Calculate center coordinates
        x_center = img_w // 2
        y_center = img_h // 2

        # Calculate crop boundaries
        x1 = x_center - half_crop
        y1 = y_center - half_crop
        x2 = x_center + half_crop
        y2 = y_center + half_crop

        crop = frame[y1:y2, x1:x2]

        if crop.shape[0] == crop_size and crop.shape[1] == crop_size:
            cropped_faces.append(crop)
        else:
            print(f"Warning: Frame {i} crop size mismatch ({crop.shape}), skipped.")

    return cropped_faces


def detect_and_crop_faces(frames: List[np.ndarray]):
    """Use BlazeFace to perform face detection and return the cropped face area"""

    cropped_faces = []
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        for i, frame in enumerate(frames):
            img_h, img_w, _ = frame.shape
            crop_size = 256
            half_crop = crop_size // 2

            # Skip frames that are too small to crop 256×256
            if img_h < crop_size or img_w < crop_size:
                print("Warning: Frame too small, skipped.")
                continue

            # Face detection
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                detection = results.detections[0]  # Only use the first detected face
                bboxC = detection.location_data.relative_bounding_box

                x_center = int((bboxC.xmin + bboxC.width / 2) * img_w)
                y_center = int((bboxC.ymin + bboxC.height / 2) * img_h) - 30
            else:
            # No detection, fall back to center crop
                x_center = img_w // 2
                y_center = img_h // 2
               # print(f"Warning: Frame {i} no face detected, performing center crop.")

            # Initial crop box
            x1 = x_center - half_crop
            y1 = y_center - half_crop
            x2 = x_center + half_crop
            y2 = y_center + half_crop

            # Adjust crop box if it goes beyond image borders
            if x1 < 0:
                x2 += -x1
                x1 = 0
            if x2 > img_w:
                diff = x2 - img_w
                x1 -= diff
                x2 = img_w
                x1 = max(0, x1)

            if y1 < 0:
                y2 += -y1
                y1 = 0
            if y2 > img_h:
                diff = y2 - img_h
                y1 -= diff
                y2 = img_h
                y1 = max(0, y1)

            crop = frame[y1:y2, x1:x2]

            if crop.shape[0] == 256 and crop.shape[1] == 256:
                cropped_faces.append(crop)
            else:
                print("Warning: Cropped region not 256x256 after adjustment, skipped.")

        return cropped_faces


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


def segment_skin(face_img: List[np.ndarray], model, batch_size: int = 1):
    """sing U-Net + EfficientNetB0U for skin segmentation"""
    pred_list = []
    dataset = ImageDataset(face_img, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                            drop_last=False)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Processing Batchs")
        for i, sample in enumerate(pbar, start=1):
            sample = sample.to("cuda" if torch.cuda.is_available() else "cpu")
            pred = model(sample)
            pred = list(pred)
            pred_list.extend(pred)
        print(f"**** inference of {len(pred_list)} imgs done ")
    return pred_list