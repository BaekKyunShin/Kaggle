# Porto Seguro's Safe Driver Prediction

## Data

https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

## Description

### 안전 운전자 예측

Porto Seguro는 브라질의 자동차 보험 회사입니다. 본 경진 대회의 목적은 어떤 차주가 내년에 보험 청구를 할 확률을 예측하는 겁니다.

데이터는 59만개의 훈련 데이터와 89만개의 테스트 데이터로 구성되어 있습니다. 테스트 데이터가 훈련 데이터보다 많습니다. 그리고 Null값이 np.NaN이 아닌 -1로 되어있습니다. 또한, Feature가 무엇을 뜻하는지 제시하지 않았다는 것이 특징입니다. 데이터를 안내한 부분에도 나와있지 않고, Feature의 이름으로 유추하기도 어렵습니다. 다만, Feature가 binary인지, categorical인지, oridnal인지, nominal인지만 구분할 수 있을 뿐입니다. 

아래 Top Ranker들의 커널을 필사하며 공부했습니다.

#### 1st Kernel: Data Preparation & Exploration

Bert Carremans 커널로 EDA, Feature Engineering(Dummification, Interaction), Feature Selection(Zero and Low Variance 제거, SelectFromModel) 등을 했습니다. 본 커널의 특징은 데이터 관리를 위해 Meta Data를 만들어 활용했다는 점입니다. 

#### 2nd Kernel:  Interactive Porto Insights - A Plot.ly Tutorial 

Anisotropic의 커널로 [Plot.ly](https://plot.ly/)에 대해 배울 수 있습니다. Plot.ly는 정적인 일반 Plot과 대조되게 동적입니다. 마우스 커서를 움직임에 따라 variable 이름을 보여주며, 드래그를 하면 해당 부분을 확대시켜 줍니다.

## Reference

Bert Carremans 커널: https://www.kaggle.com/bertcarremans/data-preparation-exploration

Anisotropic 커널: https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial