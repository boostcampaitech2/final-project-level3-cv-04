import os
import argparse
import torch
import numpy as np
import random
import yaml
import time
from importlib import import_module
import utils.utils as utils

from torch.utils.data import DataLoader

from data_set.data_set import HandWashDataset, OneSamplePerVideoDataset
from data_set.data_augmenation import get_transform
from model.loss import create_criterion
from utils.logger import yaml_logger, make_dir, best_logger

# Seed 고정
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))

    header = f"Epoch: [{epoch}]"
    for video, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(video)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc2"].update(acc2.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        for video, target in metric_logger.log_every(data_loader, 100, header):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc2"].update(acc2.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        " * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@2 {top2.global_avg:.3f}".format(
            top1=metric_logger.acc1, top2=metric_logger.acc2
        )
    )
    return metric_logger.acc1.global_avg

def run(args, cfg, device):
    seed_everything(cfg['seed'])
    # wandb_init(cfg['exp_name'])

    # cfg saved 폴더에 저장
    cfg['saved_dir'] = make_dir(cfg['saved_dir'], cfg['exp_name'])
    yaml_logger(args, cfg)

    # Transform 불러오기
    train_transform = get_transform(cfg['transforms']['train'])
    val_transform = get_transform(cfg['transforms']['valid'])

    # # DataSet 설정
    train_dataset = OneSamplePerVideoDataset(cfg['train_path'], cfg['frame_per_clip'], transform=train_transform)
    valid_dataset = OneSamplePerVideoDataset(cfg['valid_path'], cfg['frame_per_clip'], transform=val_transform)

    # # DataLoader 설정
    train_loader = DataLoader(dataset=train_dataset, **cfg['train_dataloader']['params'])
    valid_loader = DataLoader(dataset=valid_dataset, **cfg['valid_dataloader']['params'])

    # Model 불러오기
    model_module = getattr(import_module("model.model"), cfg['model']['name'])
    model = model_module(num_classes = cfg['model']['class']).to(device)
    
    # Loss function 설정
    criterion = create_criterion(cfg['criterion']['name'])

    # Optimizer 설정
    opt_module = getattr(import_module("torch.optim"), cfg['optimizer']['name'])
    optimizer = opt_module(params = model.parameters(), **cfg['optimizer']['params'])

    # Scheduler 설정
    scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfg['scheduler']['name'])
    scheduler=scheduler_module(optimizer, **cfg['scheduler']['params'])

    # 학습 파라미터 설정
    N_EPOCHS = cfg['epochs']
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler() if cfg['fp16'] else None

    # 학습
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_one_epoch(model, criterion, optimizer, scheduler, train_loader, device, epoch, print_freq=100, scaler=scaler)
        accuracy = evaluate(model, criterion, valid_loader, device)
        best_logger(cfg['saved_dir'],epoch, N_EPOCHS, accuracy)

        if cfg['saved_dir']:
                checkpoint = {
                    "model": model.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                utils.save_on_master(checkpoint, os.path.join(cfg['saved_dir'], f"model_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(cfg['saved_dir'], "checkpoint.pth"))
    
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./config/base_test.yaml', help='yaml file path')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Yaml 파일에서 config 가져오기
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(args, cfg, device)