from collections import defaultdict
import os
import shutil
import torch
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

# 디렉토리 생성
def make_dir(saved_dir, saved_name):
    path = os.path.join(saved_dir, saved_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'model'), exist_ok=True)
    os.makedirs(os.path.join(path, 'matrix'), exist_ok=True)

    return path

# yaml 파일 saved 폴더에 저장
def yaml_logger(args, cfg):
    file_name = f"{cfg['exp_name']}.yaml"
    shutil.copyfile(args.config, os.path.join(cfg['saved_dir'], file_name))

def best_logger(saved_dir, epoch, num_epochs, metrics):
    accuracy, f1 = metrics
    with open(os.path.join(saved_dir, 'best_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f"Epoch [{epoch}/{num_epochs}], Accuracy :{accuracy}, F1 Score :{f1}\n")

def save_model(*args, **kwargs):
        torch.save(*args, **kwargs)

def save_confusion_matrix(saved_dir, epoch, confusion_matrix, classes):
    df_cm = pd.DataFrame(confusion_matrix.numpy(), index = classes, columns = classes)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(saved_dir, f'matrix/output_{epoch}.png'))

class MetricLogger:
    def __init__(self, num_iteration, header = ""):
        self.num_iteration = num_iteration
        self.metric  = defaultdict()
        self.header = header

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metric[key] = value

    def log(self, iteration):
        log_message = f"{self.header} [{iteration}/{self.num_iteration}]"
        for key, value in self.metric.items():
            log_message += f" {key} : {value:.3f}"

        print(log_message)

# https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial
class ConfusionMatrix:
    def __init__(self, num_classes=6):
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

    def update(self, target, preds):
        for t, p in zip(target.view(-1), preds.view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1

    def get_accuracy(self):
        epsilon = 1e-7
        correct = 100*self.confusion_matrix.diag().tolist()
        whole = self.confusion_matrix.sum(1).clamp(min=epsilon).tolist()
        class_accuracy = [100 * c / w for c, w in zip(correct, whole)]
        return sum(class_accuracy)/len(class_accuracy), class_accuracy

    def get_f1(self):
        class_f1 = []
        for i in range(len(self.confusion_matrix)):
            tp = self.confusion_matrix[i][i].item()
            fn = sum(self.confusion_matrix[i]).item() - tp
            fp = sum([x[i] for x in self.confusion_matrix]).item() - tp

            precision = tp / (tp+fp) if fp>0 else 0.0
            recall = tp / (tp+fn) if fn>0 else 0.0
            f1 = 2 / (1/precision + 1/recall) if precision>0 and recall >0 else 0.0
            
            class_f1.append(f1)

        return sum(class_f1)/len(class_f1), class_f1
    
    def get(self):
        return self.confusion_matrix