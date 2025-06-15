import os
import time

import torch
import yaml

from data.dataLoader import Dataload
from models.Archi import model_select
from utils.before_train import parse_eval_args


def main():
    num_warmup = 5
    pure_inf_time = 0
    total_iters = 20

    args = parse_eval_args()

    with open(args.test_conf, "r") as doc:
        cfg_info = yaml.load(doc, Loader=yaml.Loader)
    model_info = cfg_info["MODEL"]
    train_info = cfg_info["TRAIN"]
    data_info = cfg_info["DATA"]
    train_info["BATCH_SIZE"] = 1

    print("##### Load data")

    print("##### Load model etc.")

    model = model_select(model_info, data_info).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.chkpt_path, "model_checkpoint.pt"), map_location=args.device))
    model.eval()

    test_dataset, test_dataloader = Dataload(
        root=args.data_path,
        train_info=train_info,
        data_info=data_info,
    ).get_test_dataloaders()

    for i, (sample, label, name) in enumerate(test_dataloader):

        sample, label = sample.to(args.device), label.to(args.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()  # record time

        with torch.no_grad():
            _ = model(sample)  # reference

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time  # time calculate

        if i >= num_warmup:
            pure_inf_time += elapsed  # cumulative time

        if (i + 1) == total_iters:  # FPS
            fps = (total_iters - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.2f} img / s')
            break


if __name__ == "__main__":
    main()