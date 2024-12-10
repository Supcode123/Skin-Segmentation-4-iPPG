import os
import shutil

import numpy as np

import yaml
import torch
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex
from tqdm import tqdm
from code_projects.data.dataLoader import data_load
from models.smp_model import model_create
from code_projects.unitls.before_train import parse_eval_args
from code_projects.unitls.visualization_plot import create_fig, denormalize

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
model.load_state_dict(torch.load(args.chkpt_path + "model_checkpoint.pt", map_location=args.device))
model.eval()

mIoU = JaccardIndex(task='multiclass', num_classes=test_dataset.num_classes, average=None).to(args.device)

miou_scores_per_sample = None
acc_per_sample = None
with torch.no_grad():
    pbar = enumerate(tqdm(test_dataloader))
    for i, (sample, label, name) in pbar:
        pred = model(sample.to(args.device))
        pred_label = torch.argmax(pred, dim=1).cpu().numpy()

        for n in range(sample.shape[0]):
            fig = create_fig(pred[n:n+1].argmax(dim=1),
                             label[n:n+1],
                             denormalize(sample[n:n+1], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                             test_dataset.num_classes)
            plt.savefig(args.chkpt_path + f"eval/prediction_{name[n]}")
            plt.close(fig)

        miou_score = mIoU(pred, label.to(args.device))
        miou_scores_per_sample = miou_score.unsqueeze(0) if miou_scores_per_sample is None \
            else torch.cat([miou_scores_per_sample, miou_score.unsqueeze(0)])

        acc_score = np.sum(pred.argmax(dim=1).cpu().numpy() == label.cpu().numpy()) / (sample.shape[-1] * sample.shape[-2])

miou_score_mean = torch.mean(miou_scores_per_sample, dim=0)
miou_score_std = torch.std(miou_scores_per_sample, dim=0)

fig, ax = plt.subplots()
ax.bar(np.arange(0, test_dataset.num_classes),
       miou_score_mean.cpu().numpy(),
       yerr=miou_score_std.cpu().numpy(),
       alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Mean IoU')
ax.yaxis.grid(True)
ax.set_xticks(np.arange(0, test_dataset.num_classes))
ax.set_xticklabels(list(test_dataset.EXP['CLASS'].values()), rotation=45, ha="right")
ax.set_xlabel('Class')

plt.tight_layout()
plt.savefig(args.chkpt_path + f"eval/mIoU_per_class.png")
plt.show()