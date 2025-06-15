import json
import os
import shutil
import time

import cv2
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from code_projects.data.dataLoader import Dataload
from models.Archi import model_select
from code_projects.utils.before_train import parse_eval_args
from code_projects.utils._plot import denormalize


def plot(pred_mask, img=None, save_path=None):
    id2color = {
        0: [58, 0, 82],
        1: [253, 234, 39],
        2: [255, 156, 201],
        3: [99, 0, 255],
        4: [255, 0, 0],
        5: [255, 0, 165],
        6: [141, 141, 141],
        7: [255, 218, 0],
        8: [173, 156, 255],
        9: [73, 73, 73],
        10: [250, 213, 255],
        11: [255, 156, 156],
        12: [99, 255, 0],
        13: [157, 225, 255],
        14: [255, 89, 124],
        15: [173, 255, 156],
        16: [255, 60, 0],
        17: [40, 0, 255],
        18: [255, 255, 255]
    }
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Map each category to the corresponding color
    for class_id, color in id2color.items():
        color_mask[pred_mask == class_id] = color

    plt.figure(figsize=(8, 8))
    if img is not None:
        overlay = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        plt.imshow(overlay,cmap='gray')  # BGR to RGB
        plt.imshow(color_mask, alpha=0.3)
    else:
        plt.imshow(color_mask)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def main():
    args = parse_eval_args()
    assert os.path.isdir(args.data_path), "Invalid data path."
    assert os.path.isdir(args.chkpt_path), "Invalid checkpoint path."
    cur_time = int(round(time.time() * 1000))
    cur_time = time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime(cur_time / 1000))

    save_path = os.path.join(args.save_path, f"plots/{cur_time}")
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=False)

    print("##### Load config")

    with open(args.test_conf, "r") as doc:
        cfg_info = yaml.load(doc, Loader=yaml.Loader)
    model_info = cfg_info["MODEL"]
    train_info = cfg_info["TRAIN"]
    data_info = cfg_info["DATA"]

    print("##### Load data")

    test_dataset, test_dataloader = Dataload(
        root=args.data_path,
        train_info=train_info,
        data_info=data_info,
    ).get_test_dataloaders()

    print("##### Load model etc.")

    model = model_select(model_info, data_info).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.chkpt_path, "model_checkpoint.pt"), map_location=args.device))
    model.eval()

    #save_path = r"D:\MA_DATA\test_pics\test\temp"
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        for i, (sample, label, name) in enumerate(pbar, start=1):

            sample = sample.to(args.device)
            img = denormalize(sample, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
            img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            label = label.squeeze().cpu().numpy()

            pred = model(sample)
            if model_info['NAME'] == "EfficientNetb0_UNet3Plus":
                pred = pred[0]
            if data_info['CLASSES'] == 2:
                #pred = pred[0]
                pred_mask = (torch.sigmoid(pred) > 0.5).float().squeeze()
                pred_mask = pred_mask.cpu().numpy()
                plot(label, img=img, save_path=os.path.join(save_path, f"{data_info['CLASSES']}_gt_" + name[0]))
                plot(pred_mask, img=img, save_path=os.path.join(save_path, f"{data_info['CLASSES']}_pred_" + name[0]))

            elif data_info['CLASSES'] > 2:
                pred_mask = pred[0].argmax(0).cpu().numpy().squeeze()
                plot(label, img=img, save_path=os.path.join(save_path,f"{data_info['CLASSES']}_gt_"+name[0]))
                plot(pred_mask, img=img, save_path=os.path.join(save_path, f"{data_info['CLASSES']}_pred_" + name[0]))

if __name__ == "__main__":
    main()