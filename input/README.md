## 병원 데이터셋 받기

1. `bash download.sh`				# 2~30분정도 소요
2. `python3 make_input.py`	# tmux 세션에서 하는거 추천(1~2시간정도 소요)

## 캐글 데이터셋 받기

1. https://www.kaggle.com/realtimear/hand-wash-dataset 
2. 다운로드 후 `python3 make_kaggle_input.py`

## 통합 데이터 라벨링

병원 데이터셋과 캐글 데이터셋을 합친 데이터 제작 코드입니다.  
위의 두 데이터셋의 라벨을 기반으로 실행됩니다.  

- `python3 make_full_input.py`

## Bounding Box Pseudo Labeling for Object Detection

`handtracking` 디렉토리 `README.md` 참고
