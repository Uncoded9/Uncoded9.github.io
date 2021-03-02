---
title: (Hands on ML) Chapter 3
layout: post
categories: [Hands on ML, ML, Python]
image: /assets/img/hands_on_ml/ch3/OkamuraYuta — The Design Kids.jpg
description: "[Hands on ML] Chapter 3. 분류"
---
## 1. MNIST

- sklearn의 MNIST 데이터셋 확인

```python
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784', version=1)

X, y =mnist["data"], mnist["target"]

# 데이터의 형태(size) 확인
print(X.shape)
print(y.shape)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/111630A4-3064-465B-AFCA-C997401D83F2_4_5005_c.jpeg)

- X에 저장된 첫번째 데이터를 불러와서 28 x 28 형태로 변환, 이미지 확인  code

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit=X[0]
some_digit_image=some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_03.jpg)

- label의 데이터 타입이 문자형이므로 숫자형으로 변환

```python
# label의 데이터 타입이 문자형이므로 숫자형으로 변환
import numpy as np

print('before:', type(y[0]))
y=y.astype(np.int)
print('after:', type(y[0]))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/ADBC2FBF-1304-4982-94EF-D6AA02CF55EE_4_5005_c.jpeg)

- 훈련세트와 테스트셋 구분: 데이터의 학습 순서에 민감한 알고리즘이나 비슷한 샘플이 연이어 나타나는 데이터셋의 경우 반드시 데이터의 순서를 무작위로 선택하여 훈련세트와 테스트셋으로 구분

```python
X_train, X_test, y_train, y_test= X[:60000],X[60000:],y[:60000],y[60000:]
```

## 2. 이진분류기

- '5' 여부만을 구분하는 이진분류기 생성
- SGDClassifier: 확률적 경사하강법(Stochastic Gradient Descent)은 한번에 훈련 샘플을 하나씩 독립적으로 처리하므로 매우 큰 데이터 셋을 효율적으로 학습
- 앞서 추출한 some_digit을 이용하여 test

```python
y_train_5=(y_train==5)
y_test_5=(y_test==5)

from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([some_digit]))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/52C78DCC-2294-47F4-BB4C-EC66C2C4086E_4_5005_c.jpeg)

## 3. 성능측정

### 3.1 교차 검증을 사용한 정확도 측정

- SGDClassifier를 사용한 분류기를 3-fold cv 실행한 뒤 accuracy 평가

```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_06.jpg)

- 이미지의 10%만이 5이므로 무조건 5 아님으로 예측하는 분류기(Never5Classifier)도 정확도 90%에 도달
→ accuracy는 모델 평가 기준으로 부족

```python
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self,X,y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X),1) ,dtype=bool)
    
never_5_clf=Never5Classifier()
cross_val_score(never_5_clf,X_train, y_train_5, cv=3, scoring='accuracy')
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_07.jpg)

### 3.2 오차행렬

- 오차행렬(confusion matrix): 샘플의 실제 클래스와 예측된 클래스를 카운트하여 비교
- negative class(첫번째 행)
    - true negative: 53,892개를 '5 아님'으로 분류
    - false negative: 687개를 '5'로 잘못 분류
- positive class(두번째 행)
    - true positive: 3,530개를 '5'로 분류
    - false positive: 1,891개를 '5 아님'으로 잘못 분류

```python
from sklearn.model_selection import cross_val_predict
y_train_pred= cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# 오차행렬(confusion_matrix) 생성

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_train_pred))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/79D72045-A68F-4F30-A8A3-EDA7A6B2F01E_4_5005_c.jpeg)

- 정밀도(precision): 양성으로 예측된 샘플 중 실제 양성 샘플의 비율

$$Precision:  \frac{TP}{TP+FP}$$

- 재현도: 양성 샘플 중 분류기가 정확히 예측한 양성샘플의 비율 
→ 민감도(sensitivity) 또는 진짜양성비율(True Positive Rate: TPR)로도 표현

$$Recall:  \frac{TP}{TP+FN} $$

### 3.3 정밀도와 재현율

- sklearn.metrics을 이용한 정밀도와 재현율 계산

```python
# sklearn에서의 정밀도와 재현율
from sklearn.metrics import precision_score, recall_score
print("precision_score:",precision_score(y_train_5, y_train_pred))
print("recall_score:",recall_score(y_train_5, y_train_pred))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/E516029D-BDB5-4741-B878-F775FD1E2A94_4_5005_c.jpeg)

- F1 score: 정밀도와 재현율 조화평균
    1. 정밀도와 재현율이 비슷한 분류기에서 F1 score가 높음
    2. 정밀도 또는 재현율이 더 높은 것이 중요할 때는 F1 score만으로 모델을 평가할 수 없음
       
$$F_{1}=\frac{2}{\frac{1}{Precision} + \frac{1}{Recall}} = 2 \times \frac{Precision \times Recall}{Precision + Recall}  = \frac{TP}{TP + \frac{FN+FP}{2}}$$


```python
from sklearn.metrics import f1_score
print("f1_score:",f1_score(y_train_5, y_train_pred))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_10.jpg)

### 3.4 정밀도/재현율 trade-off

- SGDClassifier는 결정함수(decision function)를 사용하여 각 샘플의 점수를 계산한 뒤, decision threshold를 기준으로 분류
- decision threshold가 증가하면 FP가 감소하여 정밀도가 증가하지만, FN이 증가하여 재현율 감소
- sklearn에서는 임계값을 직접 지정할수 없지만 결정함수 점수를 기반으로 원하는 입계값을 정해 예측을 만들 수 있음

```python
y_scores=sgd_clf.decision_function([some_digit])
print('score of decision_function:', y_scores)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/87E4FF14-C474-4E3C-B566-DAC1E1B31E3F_4_5005_c.jpeg)

- 적절한 임계값을 지정을 위한 과정
    1. cross_val_predict를 이용해 훈련 샘플에 있는 모든 decision score를 반환
    2. precision_recall_curve 함수를 이용해서 가능한 모든 임계값에 대한 정밀도와 재현율 계산

```python
y_scores=cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
 
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds=precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label='precisions')
    plt.plot(thresholds, recalls[:-1], "g-", label='recalls')
    
plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
plt.legend()
plt.grid()
plt.show()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_12.jpg)

- 재현율/정밀도 곡선을 사용하여 thresholds 결정: 정밀도가 급격히 감소하는 시점을 thresholds로 설정

```python
# 재현율/정밀도 곡선을 사용하여 thresholds 결정
# 정밀도가 급격히 감소하는 시점을 thresholds로 설정
plt.plot(precisions[:-1], recalls[:-1],"r--" )
plt.xlabel('recalls')
plt.ylabel('precisions')
plt.grid()
plt.show()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_13.jpg)

- 만약 정밀도 90% 달성이 목표라면 np.argmax를 사용해 thresholds를 설정한 뒤 이를 이용해 샘플의 클래스 예측

    → 정밀도와 재현율은 trade-off 관계이므로 목표 설정시 정밀도와 재현율의 목표치를 함께 고려

```python
# 정밀도 90% 달성을 위한 thresholds 찾기
thresholds_90_prcision=thresholds[np.argmax(precisions >= 0.90)]

# predict()를 사용하지않고 thresholds를 이용하여 분류
y_train_pred_90=(y_scores >= thresholds_90_prcision)

#정밀도와 재현율 확인 
print('precision_score:',precision_score(y_train_5, y_train_pred_90))
print('recall_score:',recall_score(y_train_5, y_train_pred_90))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_13_1.jpg)

### 3.5. ROC 곡선

- ROC curve는 False positive rate(FPR)에 대한 True positiv rate(TPR)의 관계를 나타냄

    → True positiv rate(TPR) = 재현율(Recall)

    → True negative rate = 특이도(specificity)

$$FPR=\frac{FP}{FP+TN} = \frac{FP+TN-TN}{FP+TN} = 1- \frac{TN}{FP+TN} = 1-TNR$$

- TPR과 FPR은 함께 증가: trade-off 관계

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    
plot_roc_curve(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.grid()
plt.show()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_14.jpg)

- 곡선 아래의 면적 (Area under Curve: AUC)을 계산하여 각 분류기를 비교

    → 완전한 분류기: 1 

    → 완전한 랜덤분류기: 0.5

```python
from sklearn.metrics import roc_auc_score
print('roc_auc_score:', roc_auc_score(y_train_5, y_scores))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_15.jpg)

- RandomForestClassifier로 생성한 분류기와 SGDClassifier로 생성한 분류기 비교
    1. RandomForestClassifier는 method로 decision_function 대신 predict_proba 사용
    2. cross_val_predict의 결과에서 행은 sample,열은 class를 의미
    3. positive class에 대한 확률을 점수로 사용해서 roc curve 생성한 뒤 SGDClassifier와 비교

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf=RandomForestClassifier(random_state=42)

#  1. RandomForestClassifier는 method로 decision_function 대신 predict_proba 사용
y_probas_forest=cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
print('predict_proba.shape:',y_probas_forest.shape)

# 2. cross_val_predict의 결과에서 행은 sample,열은 class를 의미
print(y_probas_forest)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_16.jpg)

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_17.jpg)

```python
# 3. positive class에 대한 확률을 점수로 사용해서 roc curve 생성한 뒤 SGDClassifier와 비교
y_scores_forest=y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
    
plot_roc_curve(fpr_forest, tpr_forest,"RandmForest")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.grid()
plt.show()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_18.jpg)

```python
print('roc_auc_score:', roc_auc_score(y_train_5, y_scores_forest))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_19.jpg)

## 4. 다중분류(Multiclass classifier)

- OvR 전략: 특정 숫자하나만 분류하는 이진 분류기 10개(0~9)를 훈련시킨 후 각 분류기의 결정점수 중 가장 높은 클래스 선택
- OvO 전략: 가능한 모든 숫자의 조합(0과 1, 1과 2 등)을 구분하는 이진 분류기들을 사용하여 가장 많이 양성으로 분류된 클래스 선택
- 대부분의 이진분류 알고리즘에서는 OvR 선호하지만 SVM은 훈련세트 크기에 민감하므로 OvO 선호

```python
from sklearn.svm import SVC 
svm_clf=SVC()
svm_clf.fit(X_train,y_train)
svm_clf.predict([some_digit])
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_20.jpg)

- decision_function을 호출하여 decision score 확인

```python
# decision_function()을 호출하여 decision score 확인
some_digit_scores=svm_clf.decision_function([some_digit])
print(some_digit_scores)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_21.jpg)

- 가장 점수가 높은 class 확인

```python
print('arg max:',np.argmax(some_digit_scores))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch3/handson_ml_ch03_22.jpg)