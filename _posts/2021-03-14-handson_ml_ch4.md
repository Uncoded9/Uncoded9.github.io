---
title: (Hands on ML) Chapter 4
layout: post
categories: [Hands on ML, ML, Python]
image: /assets/img/hands_on_ml/ch4/HarukaFukushi_illust on Instagram.jpg
description: "[Hands on ML] Chapter 4. 모델훈련"
---


# 1. 선형회귀

$\hat{y}=h_\theta(\mathbf{x})=\theta \cdot \mathbf{x}$

- $$\theta$$ : $$\theta_{0}$$ 부터  $$\theta_{n}$$ 까지 담고있는 파라미터 벡터($$\theta_{0}$$ 는 편향)
- $$\mathbf{x}$$ : $$x_{0}$$ 부터 $$x_{n}$$ 까지 담고있는 샘플의 특성 벡터( $$x_{0}$$은 항상 1)
- $$h_{\theta}$$: 모델 파라미터 $$\theta$$를 사용한 가설(hypothesis) 함수
- 선형회귀모델의 MSE 비용함수: $$MSE(\mathbf{X}, h_{\theta})=\frac{1}{m} \sum_{i=1}^{m}(\theta^{T} \mathbf{x}^{(i)}-y^{(i)})^{2}$$

### 1.1 정규방정식

- 정규방정식: 비용함수를 최소화하는 $\theta$값을 찾기위한 해석적인 공식

$\hat{\theta}=(\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}$

- 정규방정식을 이용한 선형회귀 code

```python
import numpy as np
X=2*np.random.rand(100,1)
y=4+3*X+np.random.rand(100,1)
X_b=np.c_[np.ones((100,1)),X]
theta_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('theta_best:', theta_best)

X_new=np.array([[0],[2]])
X_new_b=np.c_[np.ones((2,1)),X_new]
y_predict=X_new_b.dot(theta_best)
print('y_predict:',y_predict)

import matplotlib.pyplot as plt
plt.plot(X_new, y_predict,"r-")
plt.plot(X, y,"b.")
plt.axis([0,2,0,15])
plt.show()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/41F831B2-984C-4ACC-9DE5-0D037E1A5299_4_5005_c.jpg)

- sklearn을 이용한 선형회귀 code

```python
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
print('lin_reg.intercept_:',lin_reg.intercept_)
print('lin_reg.coef_:',lin_reg.coef_)
print('lin_reg.predict:',lin_reg.predict(X_new))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/2FC2DC43-AF87-4B50-AD42-D8597FBD0E0F_4_5005_c.jpg)

- np.linalg.lstsq를 이용한 $\hat{\theta} = \mathbf{X}^{+} \mathbf{y}$ 계산 →$(\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T} = \mathbf{X}^{+}$
    1. $$X=\mathbf{U} \Sigma \mathbf{V}^{T}$$ 로 SVD 될 때 $$X^{+}=\mathbf{V} \Sigma^{+} \mathbf{U}^{T}$$ 
    2. $$\Sigma^{+}$$: $$\Sigma$$에서 일정 임계치보다 낮은 모든 원소를 0으로 바꾼 뒤 0이 아닌 모든 원소의 역수를 취한 행렬 

```python
theta_best_svd, residuals, rank, s= np.linalg.lstsq(X_b,y,rcond=1e-6)
print('theta_best_svd:',theta_best_svd)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4//65548FAC-1024-418D-8042-551508C37502_4_5005_c.jpg)

- np.linalg.pinv를 이용한 pseudo inverse 계산

```python
np.linalg.pinv(X_b)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/1C3664F4-F3DB-4442-B368-68821E9E6B56.jpg)

### 1.2 계산복잡도

- 정규방정식: $$\mathbf{X}^{T} \mathbf{X}$$의 역행렬을 계산하므로 $$O(n^{2.4})$에서 $O(n^{3})$$ 사이의 계산복잡도(computational complexity)를 가짐
- sklearn의 LinearRegression: SVD를 사용하므로 계산복잡도가 $$O(n^{2})$$ → 정규방정식 보다 상대적으로 낮음

## 2. 경사 하강법

- 파라미터 벡터 $$\theta$$에 대해 비용함수의 현재 그래디언트를 계산한 뒤 그래디언트가 감소하는 방향으로 파라미터 벡터  
$$\theta$$ 이동
- 학습률(learning rate): 파라미터 벡터 $$\theta$$가 그래디언트 감소 방향으로 이동하는 크기

    → 학습률이 너무 작을 때: 알고리즘 수렴에 많은 시간 소요

    → 학습률이 너무 클 때: 최솟값을 가지는 $$\theta$$를 지나칠 수 있으며 이 경우 비용함수가 오히려 증가

- 선형회귀의 MSE 비용함수는 볼록함수(convex function)으로서 지역 최솟값(local minimum) 없이 전역 최솟값(global minimum)만 가짐
- 비용함수가 convex function이지만 특성들의 스케일이 크게 달라 특정 파라미터 축에서 길쭉한 형태를 가질 경우  최솟값으로 수렴하는 시간 증가

    →  모든 특성이 같은 스케일을 가지도록 sklearn의 StandardScaler 사용  

  

### 2.1 배치경사하강법

- 비용함수의 편도함수(partial derivative): $$\theta_{j}$$가 매우 조금 변화할 때 비용함수의 변화량

$\frac{\partial}{\partial \theta_{j}} \mathbf{MSE}(\theta)=\frac{2}{m}\sum^{m}_{i=1} (\theta^{T} \mathbf{x}^{(i)} - y^{(i)})x_{j}^{(i)} $

- 비용함수의 그래디언트 벡터

$\nabla_{\theta} \mathbf{MSE}(\theta)=\left( \begin{array}{c} \frac{\partial} {\partial \theta_{0}} \mathbf{MSE}(\theta) \\\frac{\partial} {\partial \theta_{1}} \mathbf{MSE}(\theta) \\ \vdots \\ \frac{\partial} {\partial \theta_{n}} \mathbf{MSE}(\theta) \end{array} \right) = \frac{2}{m} \mathbf{X}^{T} (\mathbf{X} \theta  - \mathbf{y})$

- 경사하강법의 스텝: 학습률 $\eta$ 만큼 $\theta$ 이동

$\theta^{( \mathbf{next \; step} )} = \theta - \eta \nabla_{\theta} \mathbf{MSE} (\theta)  $

- 배치 경사하강법 예제 code

 

```python
eta=0.1
n_iteration=1000
m=100

theta=np.random.randn(2,1)

for iteration in range(n_iteration):
    gradients= 2/m * X_b.T.dot(X_b.dot(theta)-y)
    theta=theta-eta*gradients
    
print('theta:', theta)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/E30112EE-7F9D-4CD5-AB38-0A3701B8DD48_4_5005_c.jpg)

- 적절한 학습률을 찾기 위한 그리드 탐색: 그리드 탐색의 반복횟수를 매우 크게 하고 그래디언트의 norm이 일정기준 보다 작아지면 종료하도록 설정

### 2.2 확률적 경사하강법

- 배치경사하강법은 매 스텝에서 전체 훈련세트를 사용하므로 학습에 오랜시간이 걸림
- 확률정 경사하강법: 매 스텝에서 한개의 샘플을 무작위로 선택 후 그래디언트 계산
    1. 배치경사하강법에 비해 계산속도가 매우 빠름
    2. 비용함수의 감소 트렌드가 매우 불안정 
    3. 무작위성으로 인해 지역최솟값에 수렴하지 않을 수 있지만 전역 최솟값에도 수렴하지 못할 수 있음
    4. 학습률을 점진적으로 감소시켜 전역 최솟값에 도달하도록 유도   

        → 학습률 스케쥴(learning rate schedule): 매 반복에서 학습률을 결정하는 함수

- 정규방정식을 이용한 확률적 경사하강법 예제 code

```python
n_epochs = 1000
t0, t1 = 5, 50

def learning_schedule(t):
    return t0/(t+t1)

theta=np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index=np.random.randint(m)
        xi=X_b[random_index:random_index+1]
        yi=y[random_index:random_index+1]
        gradients= 2/m * xi.T.dot(xi.dot(theta)-yi)
        eta=learning_schedule(epoch*m+i)
        theta=theta-eta*gradients
        
print('theta:', theta)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/0FCEF1D1-06CE-4B95-A9DA-B7B7921E0F3D_4_5005_c.jpg)

- sklearn의 SGDRegressor를 이용한 예제code

```python
from sklearn.linear_model import SGDRegressor
sgd_reg=SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X,y.ravel())

print('sgd_reg.intercept_:', sgd_reg.intercept_)
print('sgd_reg.coef_:', sgd_reg.coef_)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/AF0AC4AD-8F5F-463E-A2DE-309F7A605701_4_5005_c.jpg)

### 2.3 미니배치 경사하강법

- 각 스텝에서 미니배치라는 임의의 작은 샘플세트에 대해 그래디언트를 계산하는 방법
    - 장점: GPU를 사용한 성능향상과 확률적 경사하강법 보다 안정적으로 비용함수 감소
    - 단점: 확률적 경사하강법 보다 지역 최솟값에서의 탈출이 어려울 수 있음

## 3. 다항회귀

- 2차식으로 생성한 비선형 데이터의 다항회귀모델 적합 예제 code
    - sklearn.preprocessing의 PolynomialFeatures를 사용해 2차항을 데이터에 추가 후  LinearRegression()으로 모형 생성
    - 특성 $$a,b$$에  PolynomialFeatures(dgree=3)를 적용하면 $$a^{2},a^{3},b^{2},b^{3},ab,a^{2}b,b^{2}a$$  특성 추가

     

```python
np.random.seed(42)
m=100
X=6*np.random.randn(m,1)-3
y=0.5 * X**2 + X + 2 + np.random.randn(m,1)

from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2, include_bias=False)
X_poly=poly_features.fit_transform(X)

print('X[0]:',X[0])
print('X_poly[0]:',X_poly[0])

lin_reg=LinearRegression()
lin_reg.fit(X_poly, y)

print('lin_reg.intercept_:', lin_reg.intercept_)
print('lin_reg.coef_:', lin_reg.coef_)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/19D3EF78-9253-4848-8B51-782FB9FDE72E_4_5005_c.jpg)

## 4.  학습곡선

- 훈련세트와 검증세트의 성능을 훈련세트 크기 (또는 훈련반복)의 함수로 나타낸 것으로 모델의 과소적합 또는 과대적합 여부 테스트
- 학습곡선 예제 code

 

```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors=[],[]
    
    for m in range(1,len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict=model.predict(X_train[:m])
        y_val_predict=model.predict(X_val[:m])
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val[:m],y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set size", fontsize=14) 
    plt.ylabel("RMSE", fontsize=14)              
```

- 단순선형회귀의 학습곡선: 과소적합 예시
    1. 데이터가 추가되어도 훈련세트의 RMSE가 비교적 큼
    2. 검증세트의 RMSE와 훈련세트의 RMSE 사이에 큰 차이가 없음 

```python
line_reg=LinearRegression()
plot_learning_curves(line_reg, X,y)
plt.axis([0, 80, 0, 40])                         
plt.show()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/321BB7E8-274A-4E71-B1A0-178705308192_4_5005_c.jpg)

- 10차 다항회귀의 학습곡선: 과대적합 예시
    1. 훈련데이터의 오차가 훨씬 낮음
    2. 훈련세트에서의 성능이 검증세트에서의 성능보다 훨씬 좋음

 

```python
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])

plot_learning_curves(polynomial_regression, X, y)

plt.axis([0, 80, 0, 5])                         
plt.show()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/454AD51A-58EF-4A67-AA5E-3A74D93C6D3B_4_5005_c.jpg)

- 편향/분산 트레이드오프
    1. 편향: 잘못된 모델 가정으로 인한 것으로 편향이 큰 모델은 과소적합 발생
    2. 분산: 훈련 데이터의 작은 변동에 모델이 과도하게 민감하기 때문에 나타나며 자유도가 높은 모델이 높은 분산을 가지기 쉬어 과대적합 발생
    3. 줄일 수 없는 오차:  데이터 자체에 있는 noise

## 5. 규제가 있는 선형모형

### 5.1 릿지회귀

- 규제항 $$\alpha \sum_{i=1}^{n} \theta^{2}$$  를 비용함수에 추가 (가중치 벡터의 $l^{2}$ norm)
- $$\alpha=0$$이면 선형회귀와 같아지며, $\alpha$가 아주 커지면 모든 가중치가 0에 가까워짐 (variance $$\uparrow$$, bias $$\downarrow$$)
- bias 항인 $$\theta_{0}$$는 규제항에 포함되지 않음

$J(\theta) =  MSE(\theta) + \alpha \frac{1}{2}\sum_{i=1}^{n} \theta_{i}^{2}$

- 릿지회귀의 정규방정식: A는 맨 좌측상단의 원소가 0인 (n+1) $\times$(n+1)인 단위행렬

$\mathbf{\theta}= (X^{T}X + \alpha A)^{-1}X^{T}y$

- sklearn에서 Cholesky decomposition을 사용한 릿지회귀 예제 code

    → matrix A가 대칭이고 positive definite일 때, $$A=L L^{T}$$  (L은 lower triangular matrix)임을 이용

    $(X^{T}X + \alpha A) \hat{\theta}= (\mathbf{L}\mathbf{L}^{T})\hat{\theta}=\mathbf{X}^{T} \mathbf{y}$

```python
from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X,y)
print("ridge_reg.predict([[1.5]]):", ridge_reg.predict([[1.5]]))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/ECC4BDCA-66AD-4C96-A697-368230A9B308_4_5005_c.jpg)

- sklearn에서 확률적 경사하강법을 사용한 릿지회귀

```python
sgd_reg=SGDRegressor(penalty='l2')
sgd_reg.fit(X,y.ravel())
print("sgd_reg([[1.5]]):", sgd_reg.predict([[1.5]]))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/F382D99C-7F0D-4AC0-BA94-F0D607B530C6_4_5005_c.jpg)

### 5.2 라쏘회귀

- 라쏘회귀의 비용함수

$J(\theta) =  MSE(\theta) + \alpha \sum_{i=1}^{n}  | \theta_{i}|$

- 라쏘회귀는 자동으로 특성을 선택하고 sparse model을 만듬
- 릿지회귀와 라쏘회귀의 차이점
    1. 릿지회귀는 손실함수가 global minimum에 가까워질수록 그래디언트가 작아져 수렴에 도움이 됨
    2. 릿지회귀는 $\alpha$를 증가시킬 때 최적 파라미터가 0에 가까워지지만 완전히0이 되지는 않음 
- 라쏘의 비용함수는 0일때 미분가능하지 않으므로 subgradient vectot를 이용하여 경사하강법 적용

$g(\theta, J)=\nabla_{\theta} MSE(\theta) + \alpha \begin{pmatrix}
\mathsf{sign} (\theta_{1})\\
\mathsf{sign}(\theta_{2})\\ \vdots \\ \mathsf{sign}(\theta_{n})
\end{pmatrix}$

```python
from sklearn.linear_model import Lasso
lasso_reg=Lasso(alpha=0.1)
lasso_reg.fit(X,y)
lasso_reg.predict([[1.5]])
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/07F9BD63-E6CA-40FE-9D6E-E05B4613B1D4_4_5005_c.jpg)

### 5.3 엘라스틱 넷

- 릿지와 라쏘의 절충모델로 각각의 규제항을 단순히 더해서 사용하며 혼합비율 $r$을 이용해서 조절
- 릿지가 기본이 되지만 쓰이는 특성이 몇개뿐이라고 의심이 되면 라쏘나 엘라스틱넷을 사용
- 특성 수가 훈련샘플수보다 많거나 특성 몇 개가 강하게 연결되어 있으면 라쏘가 문제를 일으킬수 있으므로 엘라스틱넷을 선호

    → 라쏘는 특성수가 샘플 수(n)보다 많으면 최대 n개의 특성을 선택하고, 여러 개의 특성이 서로 강하게 연관되어 있으면 이들 중 임의의 특성 하나를 선택  

$J(\theta) = MSE(\theta) + r\alpha\sum_{i=1}^{n} |\alpha| + \frac{1-r}{2} \alpha \sum_{i=1}^{n}\theta^{2}$

```python
from sklearn.linear_model import ElasticNet
elastic_net=ElasticNet(alpha=0.1,l1_ratio=0.5)
elastic_net.fit(X,y)
print("elastic_net.predict([[1.5]]):", elastic_net.predict([[1.5]]))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/71CB00A9-38A5-44BB-8D59-57675383DD20_4_5005_c.jpg)

### 5.4 조기종료

- 알고리즘 학습이 진행됨에 따라서 훈련세트의 에러와 검증세트의 에러가 함께 감소하다가 검증에러만 다시 증가할 수 있음 → 과대적합
- 조기종료는 검증에러가 최소에 도달하는 즉시 트레이닝을 멈추는 테크닉
- 조기종료 예제 code: warm_start=True로 설정하면 fit()이 호출될 때 이전 파라메터에서 훈련을 이어감

    

```python
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

poly_scaler=Pipeline([("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
("std_sclaer", StandardScaler())])

X_train_poly_scaled=poly_scaler.fit_transform(X_train)
X_val_poly_scaled=poly_scaler.fit_transform(X_val)

sgd_reg=SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
minimum_val_error=float("inf")
best_epoch=None
best_model=None

for epoch in range(10):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_val_predict=sgd_reg.predict(X_val_poly_scaled)
    val_error=mean_squared_error(y_val, y_val_predict)
    
    if val_error < minimum_val_error:
        minimum_val_error=val_error
        best_epoch=epoch
        best_model=clone(sgd_reg)
```

## 6. 로지스틱 회귀

- 로지스틱 회귀모델은 입력특성의 가중치 합을 계산하고 편향을 더한 뒤 결괏값의 logistic을 출력

$\hat{p}=h_{\theta}(\mathbf{x})=\sigma(\theta^{T} \mathbf{x}) \\ \sigma(t)=\frac{1}{1+\exp (-t)}$

- 샘플 $$\mathbf{x}$$ 가 양성 클래스에 속할 확률 $$\hat{p}=h_{\theta}(\mathbf{x})$$를 다음과 같이 추정

$\hat{y}= \left\{ \begin{array}{lcl} 0 \;\; \mathsf{when} \;\; \hat{p} < 0.5 \\  1 \;\; \mathsf{when} \;\; \hat{p} >= 0.5 \end{array}\right.$

## 6.1 훈련과 비용함수

- cost function으로 log loss function 사용하는데, 최솟값을 계산할수 있는 closed form은 없음
- 하지만 convex function이므로 gradient descent를 사용하면 global minimum을 찾을 수 있음

$c(\theta)= \left\{ \begin{array}{lcl} -\log (\hat{p}) \;\; \mathsf{when} \;\; y=1 \\  -\log (\hat{1-p}) \;\; \mathsf{when} \;\;y = 0                 \end{array}\right. \\ J(\theta) = - \frac{1}{m}\sum_{i=1}^{m} [ y^{(i)} \log (\hat{p}^{(i)}) + (1-y^{(i)}) \log (1 - \hat{p}^{(i)}) ] \\ \frac{\partial}{\partial \theta_{j}} J(\theta)= \frac{1}{m} \sum_{i=1}^{m} ( \sigma(\theta^{T} \mathbf{x^{(i)}} )  - y^{(i)}) \mathbf{x^{(i)}_{j}}$

## 6.2 결정경계

- iris 데이터를 훈련시킨 로지스틱 회귀모델의 결정경계 확인 예제 code

```python
from sklearn import datasets
iris=datasets.load_iris()
list(iris.keys())
X=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X,y)

X_new=np.linspace(0,3,1000).reshape(-1,1)
y_proba=log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], "g-", label="Iris virginica")
plt.plot(X_new, y_proba[:,0], "b--", label="Not Iris virginica")
plt.legend(loc="upper right") 

# 로지스틱 회귀모형을 이용한 분류예측
print("log_reg.predict([[1.7],[1.5]]):", log_reg.predict([[1.7],[1.5]]))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/3983DC01-B65C-48EF-A1D6-BCF0DB671817_4_5005_c.jpg)

## 6.3 소프트맥스 회귀

- 샘플 x가 주어지면 소프트맥스 회귀 모델이 각 클래스 k에 해당하는 점수 $$s_k(x)$$를 계산하고 그 점수에 소프트맥스 함수를 적용하여 각 클래스의 확률 추정
- 각 클래스는 자신만의 파라미터 벡터 $$\theta^{(k)}$$가 있으며 이 값들은 파라미터 행렬  $\Theta$에 행으로 저장
- $$s_{k}(\mathbf{x})$$는 샘플 x에 대한 각 클래스의 점수이고, $$\sigma(s(\mathbf{x}))_{k}$$는 샘플이 클래스 k에 속할 추정 확률
- 소프트맥스 회귀는 한번에 하나의 클래스만 예측(not multioutput)

$s_{k}(\mathbf{x})= (\theta^{(k)})^{T}\mathbf{x} \\ \hat{p}_{k} = \sigma(s(\mathbf{x}))_{k}=\frac{\exp(s_{k}(\mathbf{x}))}{\sum_{j=1}^{K} \exp(s_{j}(\mathbf{x}))} \\ \hat{y}=\argmax_{k} \sigma(s(\mathbf{x}))_{k} = \argmax_{k}s_{k}(\mathbf{x}) = \argmax_{k} ((\theta^{(k)})^{T} \mathbf{x})$

- 크로스 엔트로피 비용함수로 추정된 클래스의 확률이 타겟 클래스에 얼마나 잘맞는지 측정

$J(\Theta) = - \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y^{(i)}_{k} \log(\hat{p}^{(i)}_{k}) \\ \nabla_{\theta^{(k)}} J(\Theta) = \frac{1}{m} \sum_{i=1}^{m}(\hat{p}^{(i)}_{k} - y^{(i)}_{k} ) \mathbf{x}^{(i)}$

- $$y^{(i)}_{k}$$는 i 번째 샘플이 클래스 k에 속하는지 여부로 0 또는 1의 값을 가짐
- LogisticRegression에서 multi_class="multinomial"로 설정하여 소프트맥스 회귀를 적용하는 예제code

```python
X=iris["data"][:,(2,3)]
y=iris["target"]

softmax_reg=LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X,y)

print(softmax_reg.predict([[5,2]]))
print(softmax_reg.predict_proba([[5,2]]))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch4/62062BED-A4F9-4F06-8D8D-EB402D3BCB67_4_5005_c.jpg)