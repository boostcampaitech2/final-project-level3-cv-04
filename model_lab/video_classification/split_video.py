import os
import pandas as pd

from torchvision.datasets.utils import list_dir
from torchvision.io import read_video, write_video

from collections import Counter
from tqdm import tqdm

# 라벨 설정
def get_label(is_washing, movement_code):
    if movement_code != 0:
        return movement_code
    elif is_washing == 1:
        return 0
    elif is_washing == 0:
        return 8

# 저장할 디렉토리 생성:
def make_directory(to_path):
    # 디렉토리 생성
    for i in range(0, 9):
        os.makedirs(os.path.join(to_path, f"Class_{i}"), exist_ok=True)

def split_video(data_path, to_path):
    # Dataset1 ~ Dataset12
    for i in range (1, 12):
        dataset_path = os.path.join(data_path, f"DataSet{i}")
        annotators = list(sorted(list_dir(os.path.join(dataset_path, "Annotations"))))
        filenames = pd.read_csv(os.path.join(dataset_path, "statistics.csv"))['filename'].tolist()

        # Datset에 들어있는 파일마다
        for filename in tqdm(filenames):
            filename = filename.split('.')[0]
            annotations = []

            # Annotator별 annotation 가져오기
            for annotator in annotators:
                if os.path.isfile(os.path.join(dataset_path, f"Annotations/{annotator}/{filename}.csv")):
                    video_anno = pd.read_csv(os.path.join(dataset_path, f"Annotations/{annotator}/{filename}.csv"))
                    is_washings = video_anno['is_washing'].tolist()
                    movement_codes = video_anno['movement_code'].tolist()

                    annotation = [get_label(is_washing, movement_code) for is_washing, movement_code, in zip(is_washings, movement_codes)]
                    annotations.append(annotation)
            
            start_index = 0
            subclip = 0
            temp_video, _, _ = read_video(os.path.join(dataset_path, f"Videos/{filename}.mp4")) # 비디오 불러오기
            labels = [Counter(anno).most_common(1)[0][0] for anno in zip(*annotations)] # Annotator들이 가장 많이 라벨링한 라벨로 매핑
            now_label = labels[0]

            # 라벨별 동영상 분리
            for j in range(len(labels)):
                if now_label == labels[j]:
                    continue
                elif j - start_index < 2:
                    now_label = labels[j]
                    start_index = j
                else:
                    write_video(f"{to_path}/Class_{now_label}/{filename}_{subclip}.mp4", temp_video[start_index:j-1], 30)
                    now_label = labels[j]
                    start_index = j
                    subclip += 1

            if len(labels) - start_index > 1:
                write_video(f"{to_path}/Class_{now_label}/{filename}_{subclip}.mp4", temp_video[start_index:len(labels)-1], 30)
                
        print(f"split Dataset{i}")


if __name__ == '__main__':
    # Dataset이 들어있는 경로로 설정
    data_path = "/opt/ml/data/hand_wash/data/original"
    # 분리한 동영상 저장할 경로로 설정
    to_path = '/opt/ml/data/hand_wash/data/split'

    make_directory(to_path)
    split_video(data_path, to_path)