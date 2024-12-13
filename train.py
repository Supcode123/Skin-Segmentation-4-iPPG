import os
import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from code_projects.data.dataLoader import data_load
from code_projects.data.experiments import remap_label
from models.smp_model import model_create
from code_projects.unitls.before_train import parse_train_args, get_train_info, \
    create_optimizer, create_scheduler
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from code_projects.unitls.visualization_plot import create_fig, denormalize
from unitls.early_stopping import EarlyStopping


def main():
    print("##### config Load ... #####")

    args = parse_train_args()
    print(args)
    device, output_dir, logger, data_config, model_config, train_config, \
    warmstart, start_epoch, warmstart_dir, tb_writer = get_train_info(args)

    print("##### Load data ... #####")

    train_dataset, train_dataloader, val_dataset, val_dataloader, _, _ = data_load(
        root=args.data_path,
        train_info=train_config,
        data_info=data_config
    )
    print(f"{len(train_dataset)} training samples.")
    print(f"{len(val_dataset)} validation samples.")

    print("##### Load models ...###")

    model = model_create(model_config, data_config).to(device)

    optimizer = create_optimizer(model, train_config)

    criterion = nn.CrossEntropyLoss(ignore_index=train_config['IGNORE_LABEL'])
    # elif train_dataset.num_classes == 2:
    #     criterion = nn.BCEWithLogitsLoss(reduction='none')
    # tv_loss = FocalTverskyLoss(num_classes=train_dataset.num_classes).to(device)

    acc = MulticlassAccuracy(num_classes=train_dataset.num_classes,
                                ignore_index=train_config['IGNORE_LABEL']).to(device)
    class_miou = MulticlassJaccardIndex(num_classes=train_dataset.num_classes,
                                        ignore_index=255, average='none').to(device)
    # if train_dataset.num_classes  == 2:
    #    acc = BinaryAccuracy(ignore_index=train_config['IGNORE_LABEL']).to(device)

    scheduler = create_scheduler(optimizer, train_config, len(train_dataloader))

    # Log graph of models
    im = torch.zeros((1, 3, 256, 256), device=device)
    model.eval()
    tb_writer.add_graph(model, im)
    model.train()

    if warmstart:
        model.load_state_dict(torch.load(os.path.join(warmstart_dir, 'model_checkpoint.pt'), map_location=device))
        optimizer.load_state_dict(torch.load(os.path.join(warmstart_dir, 'optim_checkpoint.pt'), map_location=device))
        scheduler.load_state_dict(
            torch.load(os.path.join(warmstart_dir, 'scheduler_checkpoint.pt'), map_location=device))

    print("##### Training")

    num_epochs = train_config["MAX_EPOCH"]
    best_val = 0

    early_stopping = EarlyStopping(patience=10, min_delta=0.0001, verbose=True)

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        train_acc = 0.
        train_loss = 0.
        val_miou = 0.
        val_class_miou = torch.zeros(size=(val_dataset.num_classes,)).to(device)
        val_loss = 0.
        val_acc = 0.

        model.train()
        pbar = tqdm(train_dataloader)
        
        for train_step, (sample, label, _) in enumerate(pbar, start=1):
            pbar.set_description(f"epoch: {epoch + 1}/{num_epochs}")
            sample, label = sample.to(device), label.to(device)
            optimizer.zero_grad()
            train_pred = model(sample)

            # create a mask, ignoring the 255 category
            ce_score = criterion(train_pred, label)  # * ce_weighting
                # f_tv_score = tv_loss(train_pred, label) * f_tv_weighting
                # lovasz_score = lovasz_softmax(train_pred, label, ignore=train_config['IGNORE_LABEL']) * lovasz_weighting
            batch_loss = ce_score  # + lovasz_score  + f_tv_score
            batch_loss.backward()
            optimizer.step()
            with torch.no_grad():
                # probab = torch.sigmoid(train_pred)
                # predict = probab >= 0.5
                #train_acc += acc(predict.float(), label).item()
                train_acc += acc(train_pred, label).item()
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
                val_ce_score = criterion(val_pred, label)  # * ce_weighting
                # val_lovasz_score = lovasz_softmax(val_pred, label, ignore=train_config['IGNORE_LABEL']) * lovasz_weighting
                batch_loss = val_ce_score  # + val_lovasz_score  + val_f_tv_score
                val_loss += batch_loss.item()
                val_acc += acc(val_pred, label).item()
                # val_pred_label = torch.argmax(val_pred, dim=1).cpu()
                miou_score = class_miou(val_pred, label)
                val_miou += miou_score.mean().item()
                val_class_miou += miou_score

            message = '[%03d/%03d] %2.2f sec(s) lr: %f Train Acc: %3.6f Loss: %3.6f | Val Acc(M-IoU): %3.6f loss: %3.6f' % \
                      (epoch + 1, num_epochs, time.time() - epoch_start_time, optimizer.param_groups[0]['lr'],
                       train_acc / len(train_dataloader),
                       train_loss / len(train_dataloader),
                       val_miou / len(val_dataloader),
                       val_loss / len(val_dataloader))

            early_stopping(val_loss / len(val_dataloader))
            if early_stopping.early_stop:
                print(f"Training stopped early at epoch {epoch + 1}")
                break

            print(message)
            logger.info(message)
            tb_writer.add_scalar('train/loss', train_loss / len(train_dataloader), epoch+1)
            # tb_writer.add_scalar('train/cross_entropy(last step)', ce_score.item(), epoch)
            tb_writer.add_scalar('train/accuracy', train_acc / len(train_dataloader), epoch+1)
            tb_writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
            tb_writer.add_scalar('val/loss', val_loss / len(val_dataloader), epoch+1)
            tb_writer.add_scalar('val/accuracy', val_acc / len(val_dataloader), epoch+1)
            # tb_writer.add_scalar('val/cross_entropy(last step)', val_ce_score.item(), epoch)
            tb_writer.add_scalar('val/mIoU', val_miou / len(val_dataloader), epoch+1)
            for k in range(val_dataset.num_classes):
                remapped_label, label_name = remap_label(label=k,classes=val_dataset.num_classes)
                # ,exp=val_dataset.exp_id
                tb_writer.add_scalar(f"val_classes/mIoU_{label_name}",
                                     val_class_miou[k].item() / len(val_dataloader), epoch+1)

            # save the best models
            if val_miou / len(val_dataloader) > best_val:
                best_val = val_miou / len(val_dataloader)

                print("save models: Processing...")
                torch.save(model.state_dict(), os.path.join(output_dir, 'model_checkpoint.pt'))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optim_checkpoint.pt'))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler_checkpoint.pt'))
                print("save models: done!")

                # Make example plot
                val_pred = torch.argmax(val_pred, dim=1)
                # Only saving max. 4 examples
                max_ind = min(4, val_pred.shape[0])
                # max_ind = val_pred.shape[0]
                create_fig(val_pred[:max_ind, ...],
                           label[:max_ind, ...],
                           denormalize(sample[:max_ind, ...], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           data_config["CLASSES"])
                plt.savefig(os.path.join(output_dir, f'example_plot_epoch{epoch + 1}_batch{val_step}.png'))
                plt.close()



if __name__ == '__main__':
    main()
