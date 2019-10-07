# Predict Future Sales

## Data

https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data

## Description

### Predict Future Sales

 훈련데이터는 2013년 1월부터 2015년 10월까지의 물품 정보 및 판매량이며, 테스트 데이터는 2015년 11월 물품 정보입니다. 이를 기반으로 11월 물품의 판매량을 예측하는 프로젝트입니다. 특히, Feature Engineering에 집중한 코드입니다. 

본 프로젝트의 Feature Engineering에서 주로 다뤘던 부분은 Lag 판매량입니다. Lag 판매량이란 이전 판매량을 뜻합니다. 어떤 가게의 어떤 물품이 이번달에는 몇개가 팔렸는데, 1달 전, 2달 전, 6달 전..등엔 몇개가 팔렸는지에 대한 수치입니다. 당장 이번달만의 데이터가 아니라 그 이전의 데이터까지 종합적으로 고려해 현재 상태를 파악한다는 개념입니다.