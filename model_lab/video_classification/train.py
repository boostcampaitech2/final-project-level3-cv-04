import os
import argparse
import torch
import numpy as np
import random
import yaml
import time
from importlib import import_module

from torch.utils.data import DataLoader
from data_set.data_set import HandWashDataset, OneSamplePerVideoDataset
from data_set.data_augmenation import get_transform
from model.loss import create_criterion
from utils.utils import yaml_logger, make_dir, best_logger, save_model, save_confusion_matrix, ConfusionMatrix, MetricLogger

# Seed 고정
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, num_classes, scaler=None):
    num_iteration = len(data_loader)
    header = f"Epoch: [{epoch}]"
    metric_logger = MetricLogger(num_iteration, header)
    confusion_matrix = ConfusionMatrix(num_classes)

    model.train()
    for i, (video, target) in enumerate(data_loader):
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

        _, preds = torch.max(output, 1)
        confusion_matrix.update(target, preds)        

        if i % print_freq == 0:
            accuracy, class_accuracy = confusion_matrix.get_accuracy()
            f1, class_f1 = confusion_matrix.get_f1()
            
            metric_logger.update(Loss=loss.item(), F1=f1, Accuracy=accuracy)
            metric_logger.log(i)

        lr_scheduler.step()


def evaluate(model, criterion, data_loader, print_freq, num_classes, device):
    model.eval()
    num_iteration = len(data_loader)
    header = "Test:"
    metric_logger = MetricLogger(num_iteration, header)
    confusion_matrix = ConfusionMatrix(num_classes)

    model.eval()
    with torch.inference_mode():
        for i, (video, target) in enumerate(data_loader):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            _, preds = torch.max(output, 1)
            confusion_matrix.update(target, preds)

            if i % print_freq == 0:
                accuracy, class_accuracy = confusion_matrix.get_accuracy()
                f1, class_f1 = confusion_matrix.get_f1()
                
                metric_logger.update(Loss=loss.item(), F1=f1, Accuracy=accuracy)
                metric_logger.log(i)
            
    accuracy, class_accuracy = confusion_matrix.get_accuracy()
    f1, class_f1 = confusion_matrix.get_f1()
    print(
        f" * Acc {accuracy:.3f} Clip F1 {f1:.3f}".format(
            accuracy=accuracy, f1=f1
        )
    )
    return confusion_matrix.get(), (accuracy, f1)

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
    # train_dataset = OneSamplePerVideoDataset(cfg['train_path'], cfg['frame_per_clip'], transform=train_transform)
    train_dataset = HandWashDataset(cfg['train_path'], cfg['frame_per_clip'], cfg['frame_per_clip'], transform=train_transform)
    valid_dataset = HandWashDataset(cfg['valid_path'], cfg['frame_per_clip'], cfg['frame_per_clip'], transform=val_transform)

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

    classes = valid_dataset.classes
    # 학습
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_one_epoch(model, criterion, optimizer, scheduler, train_loader, device, epoch, print_freq=100, num_classes=cfg['model']['class'], scaler=scaler)
        confusion_matrix, metrics = evaluate(model, criterion, valid_loader, 100, num_classes=cfg['model']['class'], device=device)
        
        save_confusion_matrix(cfg['saved_dir'], epoch, confusion_matrix, classes)
        best_logger(cfg['saved_dir'],epoch, N_EPOCHS, metrics)

        if cfg['saved_dir']:
            checkpoint = model.model.state_dict()
            save_model(checkpoint, os.path.join(cfg['saved_dir'], f"model/model_{epoch}.pth"))
            save_model(checkpoint, os.path.join(cfg['saved_dir'], "model/checkpoint.pth"))
    
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