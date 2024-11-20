
from data.processed.DataLoader import data_create
from unitls.before_train import parse_train_args, get_train_info

print("##### config Load ... #####")

args = parse_train_args()
device, output_dir, logger, data_config, model_config, train_config,\
    warmstart, start_epoch, tb_writer = get_train_info(args)

print("##### Load data ... #####")

train_dataset, train_dataloader, val_dataset, val_dataloader, _, _ = data_create(
    root=args.data_path,
    train_info=train_config,
    data_info=data_config
)
print(f"{len(train_dataset)} training samples.")
print(f"{len(val_dataset)} validation samples.")