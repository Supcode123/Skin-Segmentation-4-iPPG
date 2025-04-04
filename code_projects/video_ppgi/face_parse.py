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