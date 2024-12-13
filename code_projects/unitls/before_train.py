import argparse
import os
import shutil
import time
import logging
from pathlib import Path

import yaml
import torch
from timm import optim

from torch.utils.tensorboard import SummaryWriter


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            step_size,
            iter_warmup,
            iter_max,
            power,
            min_lr=0,
            last_epoch=-1,
    ):
        self.step_size = step_size
        self.iter_warmup = int(iter_warmup)
        self.iter_max = int(iter_max)
        self.power = power
        self.min_lr = min_lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        iter_cur = float(self.last_epoch)
        if iter_cur < self.iter_warmup:
            coef = iter_cur / self.iter_warmup
            coef *= (1 - self.iter_warmup / self.iter_max) ** self.power
        else:
            coef = (1 - iter_cur / self.iter_max) ** self.power
        return (lr - self.min_lr) * coef + self.min_lr

    def get_lr(self):
        if (
                (self.last_epoch == 0)
                or (self.last_epoch % self.step_size != 0)
                or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]

    def step_update(self, num_updates):
        self.step()


def parse_train_args():
    """
    Parse input arguments for training scripts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', help='path to dataset root', type=str)
    parser.add_argument('--data_conf', dest='data_conf', help='path to the data config file', type=str)
    parser.add_argument('--model_conf', dest='model_conf', help='path to the models config file', type=str)
    parser.add_argument('--train_conf', dest='train_conf', help='path to the training config file', type=str)
    parser.add_argument('--log_path', dest='log_path', help='path to the logging destination', type=str)
    parser.add_argument('--warmstart', dest='warmstart', help='whether log_path is an extisting checkpoint',
                        default=False, type=bool)
    parser.add_argument('--warmstart_epoch', dest='warmstart_epoch', help='the epoch numer to warmstart',
                        default=None, type=int)
    parser.add_argument('--warmstart_dir', dest='warmstart_dir', help='the path of the dir for saved uncompleted model',
                        default=None, type=str)
    parser.add_argument('--device', dest='device', default='cuda', help='Device to use', type=str)

    args = parser.parse_args()
    return args

def parse_eval_args():
    """
    Parse input arguments for test scripts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', help='path to dataset root', type=str)
    parser.add_argument('--test_conf', dest='test_conf', help='path to the test config file', type=str)
    parser.add_argument('--chkpt_path', dest='chkpt_path', help='path to model checkpoint DIR', type=str)
    parser.add_argument('--device', dest='device', default='cuda', help='Device to use', type=str)

    args = parser.parse_args()
    return args

def create_optimizer(model: torch.nn.Module, train_info: dict):
    """ Creates optimizer instance from config."""

    optimizer = optim.create_optimizer_v2(model,
                                          opt=train_info["OPTIMIZER"],
                                          lr=train_info["BASE_LR"],
                                          #momentum=train_info["MOMENTUM"],
                                          weight_decay=train_info["WEIGHT_DECAY"],
                                          betas=eval(train_info["OPTIM_BETAS"]))
    return optimizer


def create_scheduler(optimizer, train_info: dict, step_size: int):
    """ Creates scheduler instance from config. """

    name = train_info["SCHEDULER"]
    min_lr = train_info["MIN_LR"]

    if name.upper() == "POLYNOMIALLR":
        scheduler = PolynomialLR(
            optimizer=optimizer,
            step_size=step_size,
            iter_warmup=train_info["WARMUP_LR"],
            iter_max=train_info["MAX_EPOCH"],
            power=train_info["POWER"],
            min_lr=min_lr,
        )
    else:
        raise ValueError("Currently we only support POLYNOMIALLR")

    return scheduler


def create_logger(log_path: str):
    """Create a logger"""
    logger = logging.getLogger("file_logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def get_train_info(args):
    """
    Return input info for training
    """
    cur_time = int(round(time.time() * 1000))
    cur_time = time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime(cur_time / 1000))

    with open(args.data_conf, "r") as doc:
        data_config = yaml.load(doc, Loader=yaml.Loader)
        data_name = Path(args.data_conf).stem
        # print (f"data_name: {data_name}")
        # data_name = args.data_conf.split("/")[-1].split(".")[0]
    with open(args.model_conf, "r") as doc:
        model_config = yaml.load(doc, Loader=yaml.Loader)
        model_name = model_config["NAME"]
    with open(args.train_conf, "r") as doc:
        train_config = yaml.load(doc, Loader=yaml.Loader)

    name = model_name + "_" + data_name

    device = args.device
    warmstart_dir = args.warmstart_dir
    output_dir = os.path.join(args.log_path, f"{name}/{cur_time}")
    tb_dir = os.path.join(output_dir, "tensorboard")

    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir, exist_ok=False)

    if args.warmstart:
        start_epoch = args.warmstart_epoch
    else:
        start_epoch = 0
    log_file_path = os.path.join(output_dir, 'training.log')
    logger = create_logger(log_file_path)
    logger.info(args)

    shutil.copy(args.data_conf, os.path.join(output_dir, "data_config.yaml"))
    shutil.copy(args.model_conf, os.path.join(output_dir, "model_config.yaml"))
    shutil.copy(args.train_conf, os.path.join(output_dir, "train_config.yaml"))

    tb_writer = SummaryWriter(log_dir=tb_dir)

    return device, output_dir, logger, data_config, model_config, train_config,\
           args.warmstart, start_epoch, warmstart_dir, tb_writer
