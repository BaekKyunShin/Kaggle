# Costa Rican Household Poverty Level Prediction

https://www.kaggle.com/c/costa-rican-household-poverty-prediction

## Description

한 가정이 가지고 있는 140여 개의 속성을 기반으로 그 가정이 가난한 가정인지 부유한 가정인지 판단하는 대회입니다. 훈련 데이터 9,500개, 테스트 데이터 23,000개로 테스트 데이터가 더 많습니다. feature는 총 142개로 이루어져 있습니다.



## 1st Kernel: A Complete Introduction Walkthrough

본 커널에서는 EDA를 많이 해주어 데이터에 대해 전반적으로 이해할 수 있는 커널입니다. 새로 알게된 아래 3가지 기능에 대해 기록해봤습니다.

**Custom Aggregation Function**

Feature Engineering시 Aggregation을 자주하는데 주로 count, min, max, sum 등의 내장 함수를 사용합니다. custom 함수를 사용하고 싶다면 아래와 같이 하면 됩니다. ` __name__`을 통해 이름도 정해줄 수 있습니다.

```python
# Define custom function
range_ = lambda x: x.max() - x.min()
range_.__name__ = 'range_'

# Group and aggregate
ind_agg = ind.drop(columns='Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std', range_])
```

**Pipeline을 활용한 Imputing, Scaling**

Imputing (Null값을 채워주는 것)과 Scaling (Feature들의 크기, 범위를 정규화시켜주는 것)을 각각 할 수도 있지만 Pipeline 라이브러리를 활용하여 아래와 같이 한번에 처리해줄 수도 있습니다. 간단하쥬?

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler

pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'median')), 
                      ('scaler', MinMaxScaler())])

# Fit and transform training data
train_set = pipeline.fit_transform(train_set)
test_set = pipeline.transform(test_set)
```

**RFECV**

RFECV는 Recursive Feature Elimintion with Cross Validation의 약자입니다. Feature Importance가 작은 순으로 Feature를 제거하면서 Cross Validation score (교차 검정 성능)를  측정합니다. feature 갯수가 `min_features_to_select` *(default=1)* 에 도달할 때까지 계속 제거하면서 성능을 측정합니다. [(Reference1)](https://scikit-learn.org/stable/modules/feature_selection.html#rfe)

```python
from sklearn.feature_selection import RFECV

# Create a model for freature selection
estimator = RandomForestClassifier(random_state=10, n_estimators=100)

# Create the object
selector = RFECV(estimator, step=1, cv=3, scoring=scorer)

selector.fit(train.set, train_labels)
```

fit을 해주면 성능이 가장 높을 때의 feature 갯수를 최적의 feature 갯수로 저장합니다. `selector.ranking_`을 통해 feature의 중요도를 알아볼 수 있습니다. 최적의 feature에 해당하는 feature는 모두 ranking이 1임을 유의하시기 바랍니다.

## 2nd Kernel: 3250 feats -> 532 feats using SHAP

캐글코리아의 이유한님의 커널입니다. 몇 천개의 feature를 만들어준 뒤 shap 라이브러리를 활용해 차원 축소를 해줬습니다.



캐글을 하다보면 apply함수를 자주 씁니다. 아래 두 식은 같은 의미입니다. 

```python
# In [29]
# lambda식과 함께 쓴 apply 함수
df_train['edjefa'] = df_train['edjefa'].apply(lambda x: replace_edjef(x)).astype(float)

# lmabda식 없이 쓴 apply 함수
df_test['edjefa'] = df_test['edjefa'].apply(replace_edjef).astype(float)
```

이는 In [32]에도 동일하게 적용됩니다.

```python
# In [32]
# lambda식 없이 쓴 apply 함수
df_train['roof_waste_material'] = df_train.apply(fill_roof_exception, axis=1)

# lmbda식과 함께 쓴 apply 함수
df_train['roof_waste_material'] = df_train.apply(lambda x: fill_roof_exception(x), axis=1)
```

lambda는 별도의 함수를 생성하지 않은 상태에서 apply 함수 내에서 간단한 함수를 만들어 사용할 때 쓰면 좋습니다. 따라서 위의 두 사례 모두 lambda 없이 쓰는 것이 더 좋을 것이라 생각 됩니다.

**Feature Engineering**

feature engineering을 할 때는 상당히 많은 feature를 추가해줬습니다. 우선, 1~2개 혹은 그 이상의 feature들 간의 사칙연산을 통해 새로운 feature를 만들어주었습니다. feature끼리 더하거나 나누거나 곱할 때는 무작정하기 보다는 어느정도의 추측을 하면서 연관된 feature끼리 해주어야 합니다. 또한, 나누기를 할때는 무한대가 나올 수 있으므로 무한대는 0으로 바꾸어주었습니다. np.inf를 np.nan으로 바꾸고, fillna(0)를 해주면 됩니다. 이렇게 사칙연산만으로 feature engineering을 한 결과 730여개의 feature가 생겼습니다. 처음에 140여 개의 feature가 있었는데 5배 이상 늘었군요.

추가로, 오직 하나의 값을 갖는 feature는 제거했습니다. feature는 Target을 구분할 수 있게 하는 역할을 하는데 특정 feature값이 하나로 똑같다면 Target을 구분할 수 없기 때문입니다.

aggregation을 통한 feature engineering도 해주었습니다. 여기서 총 3,200개 가량의 굉장히 많은 feature를 생성해주었습니다.

**SHAP**

SHAP는 Feature importance 순으로 추출해주는 것과 유사한 개념입니다. A, B, C Feature가 있다고 할 때, A, B, C가 Target 값에 미치는 영향도를 100이라고 합시다. 이때 B를 제거했을 때, A와 C가 Target 값에 미치는 영향도가 80이라면, B의 영향도 (다른 말로하면 중요도)는 20이 됩니다. 이런 식으로 각 feature마다의 영향도를 측정해주는 것이 SHAP입니다. [(Reference2)](https://shap.readthedocs.io/en/latest/)

참고로, SHAP에서 구한 중요도와 `clf.feature_importances_`에서 구한 중요도는 서로 다릅니다. 중요도를 구하는 식이 각기 다르기 때문입니다. 따라서 SHAP로 구한 TOP 500개의 feature와 `clf.feature_importances_`로 구한 TOP 500개의 feature를 합한 뒤 unique 값만 추출했습니다. 그러면 538개의 feature가 구해집니다. 3,200개 가량의 feature를 538개로 줄인 겁니다.

## 3rd Kernel: XGBoost

본 커널에서는 여러 개의 classifier를 만들어준 뒤 voting방식을 활용했습니다.

여러 개의 XGB를 합친 votingClassifier와 여러 개의 Random Forest를 합친 votingClassifier가 예측한 결과를 결합해 최종 예측을 했습니다. 

**전처리**

이미 One-Hot Encoding되어있는 feature를 Label Encoding 형식으로 바꾸어주었습니다. Tree model에서는 Label Encoding을 해주어도 괜찮기 때문입니다. Null 처리는 왠만하면 다 0으로 해주었습니다. 다른 커널과 마찬가지로 각 종 비율 및 Aggregation을 적용하여 새로운 Feature를 만들어주는 Feature Engineering도 해주었습니다.

본 경진대회의 Target 데이터 값은 불균형합니다. 사실 대부분의 데이터셋은 불균형하긴 합니다. `class_weight`라이브러리를 사용하면 불균형한 Target값에 서로 다른 가중치(weight)를 부여할 수 있습니다. 아래와 같이 하면 `y_train_weights`에 서로 다른 가중치 값이 저장됩니다.

```python
from sklearn.utils import class_weight

# figure out the class weights for training with unbalanced classes
y_train_weights = class_weight.compute_sample_weight('balanced', y_train)
```

**joblib을 활용한 멀티프로세싱**

joblib은 loop를 돌 때 멀티 프로세싱 하게 해주는 라이브러리입니다.

```python
from math import sqrt
[sqrt(i ** 2) for i in range(10)]
>>> [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
```

위 loop를 아래와 같이 해주면 2개의 CPU를 사용해서 더 빠르게 loop를 처리합니다. [(Reference3)](https://joblib.readthedocs.io/en/latest/parallel.html)

```python
from math import sqrt
from joblib import Parallel, delayed
Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
>>> [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
```

## Reference Kernels

1st Kernel: https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough

2nd Kernel: https://www.kaggle.com/youhanlee/3250feats-532-feats-using-shap-lb-0-436

3rd Kernel: https://www.kaggle.com/skooch/xgboost