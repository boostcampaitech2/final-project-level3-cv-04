"""
병원 데이터셋 라벨별 Subclip으로 분리
"""
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
    for i in range(1, 7):
        os.makedirs(os.path.join(to_path, f"Class_{i}"), exist_ok=True)

def split_video(data_path, to_path):
    # Dataset1 ~ Dataset12
    for i in range(1, 12):
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

            if len(annotations) < 2:
                continue

            temp_video, _, _ = read_video(os.path.join(dataset_path, f"Videos/{filename}.mp4")) # 비디오 불러오기
            labels = []
            # Annotator들이 가장 많이 라벨링한 라벨로 매핑
            for anno in zip(*annotations):
                most = Counter(anno).most_common(1)
                if most[0][1] == len(annotations):
                    labels.append(most[0][0])
                else:
                    labels.append(-1)

            remove_label = (-1, 0, 7, 8)
            # 라벨별 동영상 분리
            start = -1
            now_label = -1
            subclips = []

            for j, label in enumerate(labels):
                if label not in remove_label:
                    if now_label in remove_label:
                        start = j
                        now_label = label
                    elif now_label == label:
                        continue
                    else:
                        subclips.append((start, j-1, now_label))
                        start = j
                        now_label = label
                else:
                    if now_label in remove_label:
                        continue
                    else:
                        subclips.append((start, j-1, now_label))
                        start = j
                        now_label = label

            if now_label not in remove_label:
                subclips.append((start, len(labels)-1, now_label))

            for subclip in subclips:
                start, end, now_label = subclip
                if end - 10 <= start:
                    continue
                write_video(f"{to_path}/Class_{now_label}/{filename}_{start}_{end-10}.mp4", temp_video[start:end-10], 30)
        print(f"split Dataset{i}")


if __name__ == '__main__':
    # Dataset이 들어있는 경로로 설정
    data_path = "/opt/ml/data/hand_wash/data/original"
    # 분리한 동영상 저장할 경로로 설정
    to_path = '/opt/ml/data/hand_wash/data/inspect_data'

    make_directory(to_path)
    split_video(data_path, to_path)
