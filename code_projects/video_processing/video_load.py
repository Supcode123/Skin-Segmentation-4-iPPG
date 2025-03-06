import cv2
import numpy as np
import torch
from albumentations.pytorch.functional import img_to_tensor
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
import mediapipe as mp

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
                x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
                face_crop = frame[y:y + h, x:x + w]
                return face_crop, (x, y, w, h)

    return None, None  # No face detected

def segment_skin(face_img: np.ndarray):
    """Using U-Net + EfficientNetB0 for skin segmentation"""

    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(image=face_img)['image']
    img = img_to_tensor(img).unsqueeze(0)
    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights=None,
        classes=18,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16]
    )
    model.load_state_dict(
        torch.load(r'D:\Skin-Segmentation-4-iPPG\log\EfficientNetb0_UNet_synthetic\model_checkpoint_18.pt',
                   map_location='cuda'))
    model.eval()

    with torch.no_grad():
        pred = model(img)

    return pred


if __name__ == "__main__":
    data_path = r"D:\MA_DATA\video\vid.avi"
    frames = load_video(data_path)
    #resized_frame = cv2.resize(frames[0], (256, 256), interpolation=cv2.INTER_LINEAR)
    img,_ = detect_face(frames[0])

    face_resized = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)
    pred = segment_skin(face_resized)
    output = torch.softmax(pred, dim=1)  # [batch_size, num_classes, H, W]
    output = output.argmax(1).cpu().numpy()
    #
    #
    # pred_mask = (torch.sigmoid(pred) > 0.5).float().squeeze()
    # pred_mask = pred_mask.cpu().numpy()
    # # overlay
    overlay = face_resized.copy()
    expanded_output = output.copy()
    mask_2_4 = np.isin(output,2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.morphologyEx(mask_2_4, cv2.MORPH_OPEN, kernel)
    expanded_output[mask_dilated == 1] = 2

    mask_0_1 = np.isin(expanded_output, 2)

    green_overlay = np.zeros_like(overlay)
    #green_overlay[..., 1] = pred_mask * 255
    green_overlay[..., 1] = mask_0_1 * 255

    alpha = 0.4
    overlay1 = (overlay * (1 - alpha) + green_overlay * alpha).astype(np.uint8)
    #
    # #model = model.to('cuda')
    # pred_mask = pred_mask.astype(np.uint8) * 255
    plt.figure()
    plt.imshow(overlay1)
    plt.show()
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
    # axes[0].imshow(overlay1)
    # axes[0].set_title("Overlay")
    #
    # axes[1].imshow(morphology)
    # axes[1].set_title(f"Processed Mask (Closing -> Opening)")
    # plt.show()