---
title: [Hands on ML] Chapter 1
layout: post
categories: [ML, Python]
image: /assets/img/hands_on_ml/idea_sketch.jpg
description: "[Hands on ML] Chapter 1. 한눈에 보는 머신러닝"
---

## 1. 머신러닝이란?

- 명시적인 프로그래밍 없이 컴퓨터가 학습하도록 프로그래밍하는 과학(기술)

## 2. 머신러닝을 사용하는 이유

- 기존 솔루션으로는 많은 수동 조정과 규칙이 필요한 문제
- 전통적인 방식으로는 해결 방법이 없는 문제
- 유동적인 환경: 시스템이 새로운 데이터에 적응
- 복잡한 문제와 대량의 데이터에서 통찰력 얻기

## 3. 머신러닝 시스템의 종류

### 3.1지도학습, 비지도학습과 준지도학습

- 지도학습: 머신러닝 알고리즘에 사용되는 훈련데이터에 레이블 존재
    - 분류(classification)
        1. k-nearest neighbors
        2. SVM
        3.  Decision tree / Random forest
        4. Neural networks
        5.  Logistic regression  
    - 회귀(regression): feature를 이용하여 target 수치 예측
        1. Linear regression 
- 비지도 학습: 훈련 데이터에 레이블이 없음
    - 군집(clustering)
        1. k-means
        2. DBSCAN
        3. Hierarchical clustering analysis
        4. Outlier detection / novelty detection
        5. One-class SVM
        6. Isolation Forest
    - 시각화(Visualiztion)와 차원축소(dimensionality reduction)
        1. PCA
        2. Kernel PCA
        3. Locally-linear embedding
        4. t-SNE
    - 연관규칙 학습(association rule learning)
        1. Apriori
        2. Eclat
- 준지도 학습
    - 레이블이 있는 샘플과 레이블이 없는 샘플이 섞여있는 데이터를 학습 알고리즘
        -> Deep belief networks: 비지도 학습 방식으로 RBM을 훈련 후, 지도학습으로 세부조정

## 3.2 강화학습

- agent가 environment를 관찰,  action을 취함에 따라 reward가 주어지는 학습방식

## 3.3 배치학습과 온라인학습

- 배치학습
    1. 점진적으로 학습하지 않고 가용한 데이터를 모두 사용하여 한번에 학습 
    2. 주로 많은 시간과 자원을 투자하여 오프라인에서 학습이 이루어짐 (offline learning)
- 온라인 학습
    1. 데이터를 순차적으로 미니배치(mini-batch)라는 작은 묶음단위로 순차적으로 학습
    2. 매 학습이 빠르고 비용이 적게들어 데이터가 도착하는대로 즉시 학습이 이루어짐
    3. learning rate가 크면 예전 학습 내용이 빨리 사라짐  
    4. learning rate가 작으면  시스템이 변화에 빠르게 대응할 수 없음

## 3.4 사례 기반 학습과 모델기반 학습

- 사례기반 학습
    1. 시스템이 훈련 샘플을 기억하고, 새로운 데이터와의 유사도를 측정하여 분류
- 모델기반 학습
    1. 샘플들의 모델을 만들어 예측에 사용하는 방법
    2. 모델이 얼마나 좋은지 측정하는 utility function 또는 fitness function 사용
    3. 모델이 얼마나 나쁜지 측정하는 비용함수(cost function)  사용
    4. 훈련데이터를 공급하여 모델의 파라미터를 찾는 과정을 학습(training)이라고 정의

# 4. 머신러닝의 주요 도전 과제

- 충분하지 않은 양의 훈련데이터
- 대표성 없는 훈련데이터
    1. smapling noise: 우연에 의한 대표성 없는 데이터
    2. sampling bias: 잘못된 표본 추출로 대표성 없는 데이터
- 낮은 품질의 데이터
    1. 에러, 이상치(outlier), 잡음으로 가득찬 데이터
- 관련없는 특성(feature)
    1. feature selection: 훈련에 가장 유용한 특성을 선택하여 해결
    2. feature extraction: 특성을 결합하여 더 유용한 특성을 만듬
    3. 새로운 데이터 수집
- 과대적합: 데이터의 잡음(noise) 양에 비해 모델이 너무 복잡할 때 발생
    1. parameter의 수가 적은 모델을 선택
    2. feature를 적게 사용
    3. 모델에 regularizaition 적용
    4. 훈련데이터 추가수집
    5. 오류데이터 수정 및 이상치제거 
- 과소적합: 모델이 데이터의 내재구조를 학습하지 못할 때 발생
    1. parameter가 더 많은 모델 사용
    2. feature engineering
    3. regularizaition을 감소시킴

## 5. 테스트와 검증

- 일반화 오차(generalization error): 훈련된 시스템에 주어진 test data에 대한 오류
- test set에서 좋은 결과를 가져온 모델도 실제 결과가 좋지 않을 수 있음

    → 방지 위해 validation set(dev set)에서 모델 평가/선택 후 test set에 적용하여 일반화 오차 측정

- 실제 시스템에서 사용될 데이터와 훈련 데이터가 불일치
    - 일반화 오차의 원인 파악이 어려움 (데이터 불일치 또는 과대적합)
    - 이를 구분하기 위한 방법으로 훈련 데이터에서 train-dev set을 분리하여 모델 평가
        - train-dev set에서 성능이 좋다면 → 데이터 불일치에 의한 영향이 있음
        - train-dev set에서 성능이 나쁘다면 → 과대적합 문제 먼저 해결
