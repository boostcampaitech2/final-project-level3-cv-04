## Video Classification

Our train code is based on [torchvision video classification reference](https://github.com/pytorch/vision/tree/main/references/video_classification)

## Train
- **[Config 예시](https://github.com/boostcampaitech2/final-project-level3-cv-04/blob/model/classification/video-classification/model_lab/video_classification/config/base_test.yaml)**

- **Train with config file**
  ```bash
  $ python train.py --config './config/base_test.yaml' # config 경로
  ````
- **Train 결과**
  ```
  saved
    └──exp_name                            # config에서 설정한 exp_name
        ├──model                           # model 저장 폴더
        |    ├──checkpoint.pth             # latest 모델
        |    └──exp_name{1}~{epochs}.pth   # epoch마다 저장된 모델
        ├──matrix                          # confusion matrix 저장 폴더
        |    └──output_{1}~{epochs}.png    # epoch마다 저장된 valid confusion matrix
        ├──exp_name.yaml                   # 실험에 사용한 yaml 파일    
        └──best_log.txt                    # best validation score 갱신 시 Accuracy 기록
  ```
