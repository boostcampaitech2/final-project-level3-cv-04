import os
import shutil
import argparse

def make_delete_label_folder(target):
    target_num = int(target)
    # 기존 데이터셋 복사
    shutil.copytree("./final_dataset", f"./delete_step{target}_dataset")
    
    # target 경로 지정
    train_img_path = f"./delete_step{target}_dataset/images/train"
    train_ann_path = f"./delete_step{target}_dataset/labels/train"
    valid_img_path = f"./delete_step{target}_dataset/images/valid/golden"
    valid_ann_path = f"./delete_step{target}_dataset/labels/valid/golden"
    
    # train 데이터에서 target label 이미지, 라벨 삭제
    shutil.rmtree(os.path.join(train_img_path, f"Step_{target}"))
    shutil.rmtree(os.path.join(train_ann_path, f"Step_{target}"))
    
    # train 데이터에서 삭제된 라벨 뒤의 라벨들의 기존 라벨을 -1씩 만들기
    for label in os.listdir(train_ann_path):
        label_num = int(label.split("_")[-1])
        if label_num < target_num:
            continue
        
        now_directory = os.path.join(train_ann_path, label)
        for file in os.listdir(now_directory):
            with open(os.path.join(now_directory, file), 'r') as f:
                line = f.readline()
                new_line = str(int(line[0]) - 1) + line[1:]
            
            with open(os.path.join(now_directory, file), 'w') as f:
                f.write(new_line)
    
    # valid 데이터에서 target label의 이미지 삭제 & label 수정
    for file in os.listdir(valid_ann_path):
        with open(os.path.join(valid_ann_path, file), 'r') as f:
            line = f.readline()
        class_num = int(line[0])
        # class < target label인 경우, 넘기기
        if class_num < target_num:
            continue
        # class = target label인 경우, 해당 txt 파일과 이미지 삭제
        elif class_num == target_num:
            image = file.split(".")[0] + ".jpeg"
            # 이미지 파일 지우기
            os.remove(os.path.join(valid_img_path, image))
            # 라벨 텍스트 파일 지우기
            os.remove(os.path.join(valid_ann_path, file))
        # class > target label 인경우, label -1씩 수정
        elif class_num > target_num:
            new_line = str(class_num - 1) + line[1:]
            with open(os.path.join(valid_ann_path, file), 'w') as f:
                f.write(new_line)
                
    # train의 이미지 폴더명 수정
    for folder in sorted(os.listdir(train_img_path)):
        if int(folder[-1]) > target_num:
            folder_split = folder.split("_")
            new_folder = folder_split[0] + "_" + str(int(folder_split[1]) - 1)
            os.rename(os.path.join(train_img_path, folder), os.path.join(train_img_path, new_folder))
    
    # train의 라벨 폴더명 수정
    for folder in sorted(os.listdir(train_ann_path)):
        if int(folder[-1]) > target_num:
            folder_split = folder.split("_")
            new_folder = folder_split[0] + "_" + str(int(folder_split[1]) - 1)
            os.rename(os.path.join(train_ann_path, folder), os.path.join(train_ann_path, new_folder))
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True, help="delete label number")
    
    args = parser.parse_args()
    make_delete_label_folder(args.label)
    