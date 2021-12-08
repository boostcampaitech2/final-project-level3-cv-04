import os
import shutil

# 디렉토리 생성
def make_dir(saved_dir, saved_name):
    path = os.path.join(saved_dir, saved_name)
    os.makedirs(path, exist_ok=True)

    return path

# yaml 파일 saved 폴더에 저장
def yaml_logger(args, cfg):
    file_name = f"{cfg['exp_name']}.yaml"
    shutil.copyfile(args.config, os.path.join(cfg['saved_dir'], file_name))

def best_logger(saved_dir, epoch, num_epochs, accuracy):
    with open(os.path.join(saved_dir, 'best_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f"Epoch [{epoch}/{num_epochs}], Accuracy :{accuracy}\n")