import json
import os
import shutil
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from code_projects.data.dataLoader import Dataload
from models.Archi import model_select
from code_projects.utils.before_train import parse_eval_args, create_logger
from code_projects.utils.visualization_plot import create_fig_test, create_fig, denormalize
from code_projects.utils.score_cal import miou_cal, Dice_cal, compute_assd


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


    # mIoU = JaccardIndex(task='multiclass', num_classes=test_dataset.num_classes, average=None).to(args.device)
    results = []
    iou = []
    dice = []
    assd=[]

    with torch.no_grad():

        pbar = tqdm(test_dataloader)
        for i, (sample, label, name) in enumerate(pbar,start=1):

            sample, label = sample.to(args.device), label.to(args.device)
            pred = model(sample)
            iou_score = miou_cal(model_info["NAME"], pred, label, data_info['CLASSES'], 255, args.device)
            iou.append(iou_score.item())
            dice_score = Dice_cal(model_info['NAME'], pred, label, data_info['CLASSES'], 255, args.device)
            dice.append(dice_score.item())
            #prob = torch.sigmoid(pred).squeeze(1)
            assd_score = compute_assd(label, pred, model_info['NAME'],data_info['CLASSES'])
            assd.append(assd_score)
            #results.append((name, pred.squeeze(0), label.squeeze(0), sample.squeeze(0), prob.squeeze(0)))
            if train_info['BATCH_SIZE'] == 1:
                results.append((iou_score.item(), dice_score.item(), name))
               #results.append((iou_score.item(), dice_score.item(), name, pred.squeeze(0), sample.squeeze(0), label.squeeze(0)))
        miou=np.mean(iou)
        iou_std = np.std(iou)
        mdice=np.mean(dice)
        dice_std = np.std(dice)
        massd = np.mean(assd)
        assd_std = np.std(assd)
        print("the len of assd_list: ", len(assd))
        message = '| mIoU(skin): %3.6f, Std: %3.6f | Dice(skin): %3.6f, Std: %3.6f' \
                  '| mASSD: % 3.6f, Std: % 3.6f' % (miou, iou_std, mdice, dice_std, massd, assd_std)
        print(message)
        logger.info(message)
        if train_info['BATCH_SIZE'] == 1:
            sorted_ious = sorted(results, key=lambda x: x[0])
            lowest_path = os.path.join(args.chkpt_path, "eval", "lowest")
            highest_path = os.path.join(args.chkpt_path, "eval", "highest")
            os.makedirs(lowest_path, exist_ok=True)
            os.makedirs(highest_path, exist_ok=True)
            lowest_results = sorted_ious[:10]
            highest_results = sorted_ious[-10:]
            save_file = os.path.join(lowest_path, "lowest_100_results.json")
            with open(save_file, "w") as f:
                json.dump(lowest_results, f, indent=4)
            highest_file = os.path.join(highest_path, "highest_100_results.json")
            with open(highest_file, "w") as f:
                json.dump(highest_results, f, indent=4)
            print(f"Highest 100 results saved to: {highest_file}")
            print(f"lowest 100 results saved to: {save_file}")
            lowest_results_to_show = lowest_results[:10]
            highest_results_to_show = highest_results[:10]
            pre_to_show(args, model, test_dataloader, lowest_results_to_show, lowest_path)
            pre_to_show(args, model, test_dataloader, highest_results_to_show, highest_path)
            print(f"Visualiztion Pics saved")
def pre_to_show(args,model,test_dataloader,results,path):

    for i, (sample, label, name) in enumerate(test_dataloader):
        for j, result in enumerate(results):
            if name[0] in result[2]:
                sample, label = sample.to(args.device), label.to(args.device)
                pred = model(sample)
                new_result = result + (pred.squeeze(0), sample.squeeze(0), label.squeeze(0))
                results[j] = new_result
    create_fig_test(results, path)


if __name__ == "__main__":
    main()