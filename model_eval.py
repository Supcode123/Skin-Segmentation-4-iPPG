import os
import shutil
import random

import yaml
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from code_projects.data.dataLoader import data_load
from models.Archi import model_select
from code_projects.unitls.before_train import parse_eval_args
from code_projects.unitls.visualization_plot import  create_fig_test



def main():

    args = parse_eval_args()
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

    model = model_select(model_info,data_info).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.chkpt_path, "model_checkpoint.pt"), map_location=args.device))
    model.eval()

    # mIoU = JaccardIndex(task='multiclass', num_classes=test_dataset.num_classes, average=None).to(args.device)
    results = []
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        for i, (sample, label, name) in enumerate(pbar):
            sample, label = sample.to(args.device), label.to(args.device)
            pred = model(sample)
            # iou_score, _ = miou_cal(pred, label, test_dataset.num_classes, args.device)
            # iou = iou_score.item()
            prob = torch.sigmoid(pred).squeeze(1)
            pred = (torch.sigmoid(pred) > 0.5).int().squeeze(1)
            results.append((name, pred.squeeze(0), label.squeeze(0), sample.squeeze(0), prob.squeeze(0)))

    # sorted_ious = sorted(ious, key=lambda x: x[0], reverse=True)
    random.seed(42)
    samples = random.sample(results, 8)

    save_path = os.path.join(args.chkpt_path, "eval")
    os.makedirs(save_path, exist_ok=True)
    create_fig_test(samples, save_path)

    # fig = create_fig(pred[:4, ...],
    #                  label[:4, ...],
    #                  denormalize(sample[:4, ...], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                  test_dataset.num_classes,
    #                  prob[:4, ...])


if __name__ == "__main__":
    main()