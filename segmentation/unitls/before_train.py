import argparse
import os
import shutil
import time
import logging
import yaml
from torch.utils.tensorboard import SummaryWriter

def parse_train_args():
    """
    Parse input arguments for training scripts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', help='path to dataset root', type=str)
    parser.add_argument('--data_config', dest='data_config', help='path to the data config file', type=str)
    parser.add_argument('--model_config', dest='model_config', help='path to the model config file', type=str)
    parser.add_argument('--train_config', dest='train_config', help='path to the training config file', type=str)
    parser.add_argument('--log_path', dest='log_path', help='path to the logging destination', type=str)
    parser.add_argument('--warmstart', dest='warmstart', help='whether log_path is an extisting checkpoint',
                        default=False, type=bool)
    parser.add_argument('--device', dest='device', default='cuda', help='Device to use', type=str)

    args = parser.parse_args()
    return args


def create_logger(log_path: str):
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
        data_name = args.data_conf.split("/")[-1].split(".")[0]
    with open(args.model_conf, "r") as doc:
        model_config = yaml.load(doc, Loader=yaml.Loader)
        model_name = model_config["NAME"]
    with open(args.train_conf, "r") as doc:
        train_config = yaml.load(doc, Loader=yaml.Loader)

    name = model_name + "_" + data_name

    device = args.device

    output_dir = os.path.join(args.log_path, f"{name}/{cur_time}")
    tb_dir = os.path.join(output_dir, "tensorboard")

    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir, exist_ok=False)

    if args.warmstart:
        raise NotImplementedError
    else:
        start_epoch = 0
        log_file_path = os.path.join(output_dir, 'training.log')
        logger = create_logger(log_file_path)
        logger.info(args)

    shutil.copy(args.data_conf, os.path.join(output_dir, "data_config.yaml"))
    shutil.copy(args.model_conf, os.path.join(output_dir, "model_config.yaml"))
    shutil.copy(args.train_conf, os.path.join(output_dir, "train_config.yaml"))

    tb_writer = SummaryWriter(log_dir=tb_dir)

    return device, output_dir, logger, data_config, model_config, train_config, args.warmstart, start_epoch, tb_writer