import os
import time
import shutil
import numpy as np
import yaml
import torch
from tqdm import tqdm
from code_projects.data.dataLoader import Dataload
from models.Archi import model_select
from code_projects.utils.before_train import parse_eval_args, create_logger
from code_projects.utils._plot import create_fig_test, create_fig, denormalize
from code_projects.utils.metrics_cal import miou_cal, Dice_cal, accuracy


def main():

    args = parse_eval_args()
    assert os.path.isdir(args.data_path), "Invalid data path."
    assert os.path.isdir(args.chkpt_path), "Invalid checkpoint path."
    cur_time = int(round(time.time() * 1000))
    cur_time = time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime(cur_time / 1000))

    eval_path = os.path.join(args.save_path, f"eval/{cur_time}")
    if os.path.isdir(eval_path):
        shutil.rmtree(eval_path)
    os.makedirs(eval_path, exist_ok=False)

    print("##### Load config")

    with open(args.test_conf, "r") as doc:
        cfg_info = yaml.load(doc, Loader=yaml.Loader)
    model_info = cfg_info["MODEL"]
    train_info = cfg_info["TRAIN"]
    data_info = cfg_info["DATA"]

    log_file_path = os.path.join(eval_path, 'training.log')
    logger = create_logger(log_file_path)
    logger.info(args)

    print("##### Load data")

    test_dataset, test_dataloader = Dataload(
        root=args.data_path,
        train_info=train_info,
        data_info=data_info,
    ).get_test_dataloaders()

    print("##### Load model etc.")

    model = model_select(model_info,data_info).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.chkpt_path, "model_checkpoint.pt"), map_location=args.device))
    model.eval()

    iou = []
    dice = []
    acc=[]

    with torch.no_grad():

        pbar = tqdm(test_dataloader)
        for i, (sample, label, name) in enumerate(pbar,start=1):

            sample, label = sample.to(args.device), label.to(args.device)
            pred = model(sample)

            iou_score = miou_cal(model_info["NAME"], pred, label, data_info['CLASSES'], 255, args.device)
            iou.append(iou_score.item())
            dice_score = Dice_cal(model_info['NAME'], pred, label, data_info['CLASSES'], 255, args.device)
            dice.append(dice_score.item())
            acc_score = accuracy(model_info['NAME'], pred, label, data_info['CLASSES'], 255, args.device)
            acc.append(acc_score.item())

        miou=np.mean(iou)
        iou_std = np.std(iou)
        mdice=np.mean(dice)
        dice_std = np.std(dice)
        macc = np.mean(acc)

        message = 'PA: %3.6f | mIoU(skin): %3.6f, Std: %3.6f | Dice(skin): %3.6f, Std: %3.6f|'\
                  % (macc, miou, iou_std, mdice, dice_std)
        print(message)
        logger.info(message)


if __name__ == "__main__":
    main()