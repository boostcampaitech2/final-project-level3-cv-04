## Video Classification

Our train, utils code is based on [torchvision video classification reference](https://github.com/pytorch/vision/tree/main/references/video_classification)

## Train
- **[Config 예시](https://github.com/boostcampaitech2/final-project-level3-cv-04/blob/model/classification/video-classification/model_lab/video_classification/config/base_test.yaml)**

- **Train with config file**
  ```bash
  $ python train.py --config './config/base_test.yaml' # config 경로
  ````
- **Train 결과**
  ```
  saved
    └──exp_name                       # config에서 설정한 exp_name
        ├──checkpoint.pth             # best_model
        ├──exp_name{1}.pth            # epoch 당 저장된 모델
        |   ~ exp_name{epochs}.pth
        ├──exp_name.yaml              # 실험에 사용한 yaml 파일    
        └──best_log.txt               # best validation score 갱신 시 Accuracy 기록
  ```
