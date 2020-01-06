## Home Credit Default Risk

https://www.kaggle.com/c/home-credit-default-risk

## Description

### A Gentle Introduction

Home Credit Default Risk에 대해 전체적으로 훑어보는 커널입니다. EDA, 간단한 Feature Engineering을 통해 예측을 해봅니다. 우선, Categorical Data를 인코딩합니다. 2개의 값으로만 구성되어 있는 feature는 Label Encoding을 하고, 2개 이상의 값으로 구성되어 있는 feature는 One-Hot Encoding을 합니다.

간단한 Feature Engineering으로는 Polynomial feature와 Domain Knowledge feature를 사용합니다. Polynomial feature는 특정 feature를 제곱, 세제곱하거나 서로의 곱(interaction)을 통해 새로운 다항식으로 구성된 feature를 의미합니다. 중요한 feature인 `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`를 기반으로 Polynomial feature를 만들었습니다. 예를 들어, `EXT_SOURCE_1^2`, `EXT_SOURCE_2^2` 같이 하나의 feature를 제곱, 세제곱 등을 해주거나 `EXT_SOURCE_1` x `EXT_SOURCE_2`, `EXT_SOURCE_1` x `EXT_SOURCE_2^2`, `EXT_SOURCE_1^2` x `EXT_SOURCE_2^2`와 같이 서로를 곱해준 feature를 만들 수 있습니다.

그 다음은 아래와 같이 Domain Knowledge feature를 사용했습니다.

- `CREDIT_INCOME_PERCENT`: the percentage of the credit amount relative to a client's income
- `ANNUITY_INCOME_PERCENT`: the percentage of the loan annuity relative to a client's income
- `CREDIT_TERM`: the length of the payment in months (since the annuity is the monthly amount due
- `DAYS_EMPLOYED_PERCENT`: the percentage of the days employed relative to the client's age

최종적으로 예측은 Logistic Regression과 Random Forest를 사용했습니다.

### Introduction to Manual Feature Engineering (Part I)

본 커널에서는 각종 Feature Engineering을 해주어 `A Gentle Introduction`보다 더 성능을 향상시켰습니다.

우선, Numeric Feature에 대해 Aggregation한 feature를 추가해줬습니다. `SK_ID_CURR`을 기준으로 Numerica feature의 `mean`, `max`, `min`, `sum`을 구했습니다. 이렇게 구한 Aggregation Feature를 기존의 train, test DataFrame에 추가해줬습니다. Categorical Feature는 One-Hot-Encoding을 해준 뒤 `sum`, `mean` Aggregation을 해주었습니다. 

결측값 처리도 해주었습니다. 각 Feature별로 결측치가 90%가 넘어가는 Feature는 drop해주려고 했으나 결측치가 90% 넘는 Feature는 없었습니다.

각 Feature와 Target 간의 상관계수도 구했습니다. 이 상관계수를 기반으로 가장 중요시되는 feature를 도출했습니다. 중요한 feature(즉, 상관계수의 절대값이 높은 feature)에는 Feature Engineering으로 만든 feature들도 상당수 있었습니다.

또, 각 Feature간의 다중공선성도 살펴봤습니다. 각 feature간의 상관계수가 0.8이 넘어가면 둘 중 하나는 drop해주었습니다.

최종적으로 모델은 LightGBM을 사용했습니다. `A Gentle Introduction`에서 간단한 feature engineering한 것보다 성능이 크게 향상되었습니다.

### Stacking Test-Sklearn, XGBoost, CatBoost, LightGBM

본 커널의 코드는 간단하지만 성능면에서는 뛰어납니다. 

최적의 하이퍼파라미터가 이미 정해진 SKlearn, CatBoost, XGBoost, LightGBM를 Stacking하여 최종 예측값을 구했습니다. Feature Engineering을 크게 없었으며, 모델의 하이퍼파라미터 및 Stacking에 중점을 둔 커널입니다. 하이퍼파라미터는 이미 다른 커널에서 구해놓은 것을 그대로 가져다가 썼기때문에 코드가 간단해질 수 있었습니다.