# San Francisco Crime Classification

## Data

https://www.kaggle.com/c/sf-crime/data

## Description

### 샌프란시스코 범죄 예측

샌프란시스코에서 2003.1.6.부터 2015.5.13. 까지의 발생한 범죄에 대한 데이터가 주어집니다. 훈련 데이터와 테스트 데이터는 매주 로테이션됩니다. 즉 1, 3, 5, 7,... 주는 훈련 데이터, 2, 4, 6, 8,..... 주는 테스트 데이터로 쓰입니다. 총 훈련 데이터는 약 870,000개입니다. 

모델은 LGBM을 사용했습니다. GBM은 하이퍼 파라미터가 굉장히 중요합니다. 하이퍼 파라미터에 따라 모델 성능이 달라지기 때문입니다. 하지만 어떤 하이퍼 파라미터로 세팅을 해주어야 하는지 결정하는 것은 쉽지 않은 작업입니다. 이를 해결하기 위한 방법 중 하나가 Bayesian Optimization입니다. 적절한 하이퍼 파라미터를 골라주는 기법입니다. 제 노트북에선 Bayesian Optimization을 하는 부분까진 다루지 않았습니다. 이는 추후 과제로 남기고 Yannis Pappas가 구한 하이퍼 파라미터를 그대로 활용했습니다.

## Reference

 Yannis Pappas 커널: https://www.kaggle.com/yannisp/sf-crime-analysis-prediction