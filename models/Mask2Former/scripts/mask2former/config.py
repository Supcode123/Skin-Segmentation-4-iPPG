import argparse
import shutil
import time
from pathlib import Path

import yaml
import os


def _args():
    """
    Parse input arguments for training scripts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', help='path to dataset root', type=str)
    parser.add_argument('--data_conf', dest='data_conf', help='path to the data config file', type=str)
    parser.add_argument('--model_conf', dest='model_conf', help='path to the models config file', type=str)
    parser.add_argument('--train_conf', dest='train_conf', help='path to the training config file', type=str)
    parser.add_argument('--log_path', dest='log_path', help='path to the logging destination', type=str)

    args = parser.parse_args()
    return args


def _config(args):
    cur_time = int(round(time.time() * 1000))
    cur_time = time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime(cur_time / 1000))
    with open(args.data_conf, "r") as doc:
        data_config = yaml.load(doc, Loader=yaml.Loader)
        data_name = Path(args.data_conf).stem
    with open(args.model_conf, "r") as doc:
        model_config = yaml.load(doc, Loader=yaml.Loader)
        model_name = model_config["NAME"]
    with open(args.train_conf, "r") as doc:
        train_config = yaml.load(doc, Loader=yaml.Loader)

    name = model_name + "_" + data_name
    output_dir = os.path.join(args.log_path, f"{name}/{cur_time}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.data_conf, os.path.join(output_dir, "data_config.yaml"))
    shutil.copy(args.model_conf, os.path.join(output_dir, "model_config.yaml"))
    shutil.copy(args.train_conf, os.path.join(output_dir, "train_config.yaml"))

    return output_dir, data_config, model_config, train_config
