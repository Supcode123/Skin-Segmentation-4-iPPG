import os
import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from code_projects.data.dataLoader import Dataload
from code_projects.data.experiments import remap_label
from models.Archi import model_select
from code_projects.unitls.before_train import parse_train_args, get_train_info, \
    create_optimizer, create_scheduler
from code_projects.unitls.visualization_plot import create_fig, denormalize
from code_projects.unitls.early_stopping import EarlyStopping
from code_projects.unitls.score_cal import accuracy, loss_cal, miou_cal, Dice_cal
from code_projects.unitls.csv_create import csv_file



def main():
    print("##### config Load ... #####")

    args = parse_train_args()
    print(args)
    device, output_dir, logger, data_config, model_config, train_config, \
    warmstart, start_epoch, warmstart_dir, tb_writer = get_train_info(args)

    print("##### Load data ... #####")

    train_dataset, train_dataloader, val_dataset, val_dataloader, _, _ = Dataload(
        root=args.data_path,
        train_info=train_config,
        data_info=data_config).get_dataloaders()
    print(f"{len(train_dataset)} training samples.")
    print(f"{len(val_dataset)} validation samples.")
    print("##### Load models ...###")

    model = model_select(model_config, data_config).to(device)

    optimizer = create_optimizer(model, train_config)

    # criterion = nn.CrossEntropyLoss(ignore_index=train_config['IGNORE_LABEL'])
    # tv_loss = FocalTverskyLoss(num_classes=train_dataset.num_classes).to(device)

    # acc = MulticlassAccuracy(num_classes=train_dataset.num_classes,
                                #ignore_index=train_config['IGNORE_LABEL']).to(device)
    # class_miou = MulticlassJaccardIndex(num_classes=train_dataset.num_classes,
                                        # ignore_index=255, average='none').to(device)
    # if train_dataset.num_classes  == 2:
    #    acc = BinaryAccuracy(ignore_index=train_config['IGNORE_LABEL']).to(device)
    scheduler = create_scheduler(optimizer, train_config)

    # Log graph of models
    # im = torch.zeros((1, 3, 256, 256), device=device)
    # model.eval()
    # tb_writer.add_graph(model, im)
    # model.train()

    if warmstart:
        model.load_state_dict(torch.load(os.path.join(warmstart_dir, 'model_checkpoint.pt'), map_location=device))
        optimizer.load_state_dict(torch.load(os.path.join(warmstart_dir, 'optim_checkpoint.pt'), map_location=device))
        scheduler.load_state_dict(
            torch.load(os.path.join(warmstart_dir, 'scheduler_checkpoint.pt'), map_location=device))

    print("##### Training")

    num_epochs = train_config["MAX_EPOCH"]
    best_val = 0.

    early_stopping = EarlyStopping(patience=train_config["PATIENCE"], min_delta=train_config["THRESHOLD"], verbose=True)

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        train_acc = 0.
        train_loss = 0.
        val_miou = 0.
        val_acc = 0.
        val_skin_miou = 0.
        val_loss = 0.
        val_skin_acc = 0.
        val_dice = 0.

        model.train()
        pbar = tqdm(train_dataloader)

        for train_step, (sample, label, _) in enumerate(pbar, start=1):
            pbar.set_description(f"epoch: {epoch + 1}/{num_epochs}")
            sample, label = sample.to(device), label.to(device)
            optimizer.zero_grad()
            train_pred = model(sample)
            print("******pred: ", train_pred.shape)
            batch_loss = loss_cal(model_config['NAME'], train_pred, label,
                                  train_dataset.num_classes, train_config["IGNORE_LABEL"],
                                  device)
            batch_loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_accuracy, _ = accuracy(model_config['NAME'], train_pred, label,
                                             train_dataset.num_classes, 255, device)
                train_acc += train_accuracy.item()
            train_loss += batch_loss.item()

        # learn rate scheduler
        # if train_config["SCHEDULER"].upper() == "REDUCELRONPLATEAU":
        #     scheduler.step(train_loss)
        # else:
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for val_step, (sample, label, _) in enumerate(val_dataloader, start=1):
                # print(f"Validation step: {val_step}")
                sample, label = sample.to(device), label.to(device)
                val_pred = model(sample)
                print("Segmentation logits unique values:", torch.unique(val_pred.argmax(dim=1)))
                print("Ground truth unique values:", torch.unique(label))

                batch_loss = loss_cal(model_config['NAME'], val_pred, label,
                                      val_dataset.num_classes, train_config["IGNORE_LABEL"], device)
                val_accuracy, val_skin_accuracy = accuracy(model_config['NAME'], val_pred,
                                                           label, val_dataset.num_classes, 255, device)
                val_acc += val_accuracy.item()
                val_skin_acc += val_skin_accuracy.item()
                val_loss += batch_loss.item()
                miou_score, skin_miou = miou_cal(model_config['NAME'], val_pred,
                                                 label, val_dataset.num_classes, 255, device)
                if val_dataset.num_classes == 2:
                    dice = Dice_cal(model_config['NAME'], val_pred, label, 255, device)
                    val_dice += dice.item()
                val_miou += miou_score.item()
                val_skin_miou += skin_miou.item()

            message = '[%03d/%03d] %2.2f sec(s) lr: %f Train Acc: %3.6f Loss: %3.6f |' \
                      ' Val Acc: %3.6f Loss: %3.6f M-IoU: %3.6f Dice(skin): %3.6f'% \
                      (epoch + 1, num_epochs, time.time() - epoch_start_time, optimizer.param_groups[0]['lr'],
                       train_acc / len(train_dataloader),
                       train_loss / len(train_dataloader),
                       val_acc / len(val_dataloader),
                       val_loss / len(val_dataloader),
                       val_miou / len(val_dataloader),
                       val_dice / len(val_dataloader))
            print(message)
            logger.info(message)

            early_stopping(val_miou / len(val_dataloader))
            if early_stopping.early_stop:
                print(f"Training stopped early at epoch {epoch + 1}")
                break
            current_score = val_miou / len(val_dataloader)
            if current_score > best_val:  # improve,better
                best_val = current_score

                tb_writer.add_scalar('train/loss', train_loss / len(train_dataloader), epoch + 1)
                # tb_writer.add_scalar('train/cross_entropy(last step)', ce_score.item(), epoch)
                tb_writer.add_scalar('train/accuracy', train_acc / len(train_dataloader), epoch + 1)
                tb_writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
                tb_writer.add_scalar('val/loss', val_loss / len(val_dataloader), epoch + 1)
                tb_writer.add_scalar('val/accuracy', val_acc / len(val_dataloader), epoch + 1)
                tb_writer.add_scalar('val/mIoU', val_miou / len(val_dataloader), epoch + 1)
                tb_writer.add_scalar('val/Dice', val_dice / len(val_dataloader), epoch + 1)

                # save the best models
                print("save models: Processing...")
                torch.save(model.state_dict(), os.path.join(output_dir, 'model_checkpoint.pt'))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optim_checkpoint.pt'))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler_checkpoint.pt'))
                print(f"save models at {epoch+1} epoch: done!")

                # Make example plot
                if val_dataset.num_classes == 18:
                    val_pred = torch.argmax(val_pred, dim=1)
                if val_dataset.num_classes == 2:
                    if model_config['NAME'] == "EfficientNetb0_UNet3Plus":
                        val_pred = val_pred[0]
                    val_pred = (torch.sigmoid(val_pred) > 0.5).int().squeeze(1)
                # unique_labels = torch.unique(val_pred)
                # print("标签值:", unique_labels)
                # Only saving max. 4 examples
                max_ind = min(4, val_pred.shape[0])
                # max_ind = val_pred.shape[0]
                create_fig(val_pred[:max_ind, ...],
                           label[:max_ind, ...],
                           denormalize(sample[:max_ind, ...], data_config['MEAN'], data_config['STD']),
                           data_config["CLASSES"])
                plt.savefig(os.path.join(output_dir, f'example_plot.png'))
                plt.close()
                # save final acc, m_iou data to compare
                # csv_file(args.log_path, val_skin_acc / len(val_dataloader),
                #          val_skin_miou / len(val_dataloader), val_dataset.num_classes)


if __name__ == '__main__':
    main()
