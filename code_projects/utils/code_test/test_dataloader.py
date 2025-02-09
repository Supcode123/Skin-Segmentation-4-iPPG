from code_projects.utils.before_train import parse_train_args, get_train_info
from code_projects.data.dataLoader import Dataload

args = parse_train_args()
print(args)
device, output_dir, logger, data_config, model_config, train_config, \
warmstart, start_epoch, warmstart_dir, tb_writer = get_train_info(args)

print("##### Load data ... #####")

train_dataset, train_dataloader, val_dataset, val_dataloader, _, _ = Dataload(
    root=args.data_path,
    train_info=train_config,
    data_info=data_config,
    pretrain=model_config.get('PRETRAIN')).get_dataloaders()
print(f"{len(train_dataset)} training samples.")
print(f"{len(val_dataset)} validation samples.")
