## Real-Time Object Detection Based on <a href="https://github.com/ultralytics/yolov5">YOLOv5</a>

```
├──datasets
|   └──handwash
|       ├──images
|       └──labels
└──yolov5
    ├──train.py
    └──detect.py
```

### 1. Prepare images and labels
YOLOv5에서 학습할 수 있도록 데이터를 준비합니다. 

1. Bounding Box가 없는 Data라면, `input/handtracking` 에서 Pseudo-Labeling을 진행합니다. 

2. [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)를 참고하여 디렉토리를 구성합니다. 

### 2. Train
- train
```
# base model
python train.py
# multiscale model
python train.py --multiscale
# pretrained model
python train.py --weights [PATH_TO_WEIGHT_FILE]

```
model 실험 결과 best.pt는 `yolov5/output`에 저장됩니다. 
- detect
```
python detect.py --weigts [PATH_TO_WEIGHT_FILE] --name [DIR_TO_SAVE_RESULT] --source [VIDEO_PATH_TO_INFERENCE]
```