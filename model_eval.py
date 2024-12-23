import os
import shutil

import numpy as np

import yaml
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from code_projects.data.dataLoader import data_load
from models.smp_model import model_create
from code_projects.unitls.before_train import parse_eval_args
from code_projects.unitls.visualization_plot import create_fig, denormalize
from code_projects.unitls.score_cal import miou_cal


def main():

    args = parse_eval_args()
    data_path = os.path.abspath(args.data_path)
    chkpt_path = os.path.abspath(args.chkpt_path)
    assert os.path.isdir(args.data_path), "Invalid data path."
    assert os.path.isdir(args.chkpt_path), "Invalid checkpoint path."

    eval_path = os.path.join(args.chkpt_path, "eval")
    if os.path.isdir(eval_path):
        shutil.rmtree(eval_path)
    os.makedirs(eval_path, exist_ok=False)

    print("##### Load config")

    with open(args.test_conf, "r") as doc:
        cfg_info = yaml.load(doc, Loader=yaml.Loader)
    model_info = cfg_info["MODEL"]
    train_info = cfg_info["TRAIN"]
    data_info = cfg_info["DATA"]

    print("##### Load data")

    _, _, _, _, test_dataset, test_dataloader = data_load(
        root=args.data_path,
        train_info=train_info,
        data_info=data_info
    )

    print("##### Load model etc.")

    model = model_create(model_info,data_info).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.chkpt_path, "model_checkpoint.pt"), map_location=args.device))
    model.eval()

    # mIoU = JaccardIndex(task='multiclass', num_classes=test_dataset.num_classes, average=None).to(args.device)

    # miou_scores_per_sample = torch.empty((0,), device=args.device)

    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        for i, (sample, label, name) in enumerate(pbar):
            sample, label = sample.to(args.device), label.to(args.device)
            pred = model(sample)
            prob = torch.sigmoid(pred).squeeze(1)
            # pred_label = torch.argmax(pred, dim=1).cpu().numpy()
            pred = (torch.sigmoid(pred)> 0.5).int().squeeze(1)
            fig = create_fig(pred[:4, ...],
                             label[:4, ...],
                             denormalize(sample[:4, ...], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                             test_dataset.num_classes,
                             prob[:4, ...])
            plt.savefig(os.path.join(args.chkpt_path, "eval", f"prediction_{i}batch.png"))
            plt.close(fig)


if __name__ == "__main__":
    main()