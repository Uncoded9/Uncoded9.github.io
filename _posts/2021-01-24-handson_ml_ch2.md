---
title: (Hands on ML) Chapter 2
layout: post
categories: [ML, Python]
image: /assets/img/hands_on_ml/ch2/SlimLim _ Line Art.jpg
description: "[Hands on ML] Chapter 2. 머신러닝 프로젝트 처음부터 끝까지"
---

## 1. 데이터 다운로드
- tgz 파일을 다운로드 후 압축을 푸는 code 작성

```python
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)  # housing_path가 없다면 directory 생성 (/datasets/housing)
    tgz_path = os.path.join(housing_path, "housing.tgz") 
    urllib.request.urlretrieve(housing_url, tgz_path) # tgz_path에 파일다운로드
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()
```
- pandas를 이용해서 csv 파일 열기
```python
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path) # housing_path에 있는 csv file을 dataframe으로 읽어오기

housing = load_housing_data()
housing.head() # 데이터의 처음 5행 확인
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_01.jpg)


- info()를 이용해 데이터 특성 파악 
```python
housing.info()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_02.jpg)

- Data type이 object인 경우 데이터 타입이 불명확
- 이 데이터에서는 categorical value이기 때문에 value_counts()로 값 확인 

```python
housing['ocean_proximity'].value_counts() 
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_03.jpg)

- 숫자형 데이터의 정보요약
```python
housing.describe()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_04.jpg)

- jupyter notebook안에서 plot이 그려지도록 설정  
```python
%matplotlib inline   
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15));
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_05.jpg)



## 2. Test set 만들기
- 전체 데이터 중 20%를 random하게 추출하여 test set으로 설정하는 code
- 단점: random하게 추출이므로 매번 다른 test set이 생성 
```python
import numpy as np
def split_train_set(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_set(housing, 0.2)
print(len(train_set))
print(len(test_set))
```
- 각 샘플 index의 해시 값을 계산하여 해시 최댓값의 20%보다 작거나 같은 샘플만 test set으로 설정
    -> 새로운 데이터가 추가되도 기존의 테스트 세트는 유지되고 새로운 데이터의 20%만 추가
```python
housing_with_id=housing.reset_index() # 데이터 셋에 index 추가 

from zlib import crc32
def test_set_check(identifier, test_ratio): # 해시값 생성 함수           
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

train_set, test_set=split_train_test_by_id(housing_with_id,0.2,"index")
```

- sklearn의 random_state를 사용하여 난수 초기값 설정     

```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

- 예제 데이터에서 중위소득의 비율을 히스토그램을 이용하여 시각화 
    1. 전체 데이터에서 각 카테고리가 차지하는 비율이 다름 
    2. 무작위 샘플링으로 train set과 test set을 설정할 경우 편향을 발생시킬 수 있음 

```python 
housing["income_cat"]=pd.cut(housing["median_income"],bins=[0. ,1.5 ,3.0 ,4.5 ,6. ,np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_06.jpg)

- 샘플링 편향을 방지하기 위한 계층적 샘플링(stratified sampling) code
```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
```


## 3. 데이터 이해를 위한 탐색과 시각화
- 원본을 손상시키지 않고 데이터를 탐색하기 위한 훈련 세트의 복사본 생성 후 산점도로 시각화
```python
housing=strat_train_set.copy()
housing.plot(kind="scatter", x="longitude",y="latitude");
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_07.jpg)

- 산점도에서 겹쳐진 point들의 밀집도를 확인하기 위해 alpha 조정
```python
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1);
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_08.jpg)

- 컬러 맵(cmap)을 이용하여 산점도에 중간주택가격을 함께 표시 
```python
housing.plot(kind='scatter', x='longitude', y='latitude', s=housing['population']/100, alpha=0.4,
             label='population',figsize=(10,7),c='median_house_value',cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend();
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_09.jpg)

## 4. 상관관계 조사
- 중간 주택가격과 다른 feature 사이의 상관관계 분석
```python
corr_matrix=housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_10.jpg)

- scatter_matrix를 이용한 상관관계 시각화 
```python
from pandas.plotting import scatter_matrix
attribute=['median_house_value','median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attribute],figsize=(12,8));
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_11.jpg)


## 5. 특성조합으로 실험
- 가구당 방갯수, 방당 침실갯수, 가구당 인원과 같은 새로운 특성(feature) 생성 
```python
housing['rooms_per_household']=housing['total_rooms']/housing['households']
housing['bedrooms_per_room']=housing['total_rooms']/housing['total_rooms']
housing['population']=housing['population']/housing['households']
```

## 6. 머신러닝 알고리즘을 위한 데이터 준비
- strat_train_set에서 housing으로 median_house_value(target)를 제외한 데이터 복사
- strat_train_set에서 housing_lables로 median_house_value(target) 복사
```python 
housing=strat_train_set.drop('median_house_value', axis=1)
housing_labels=strat_train_set['median_house_value'].copy()
```

## 7. 데이터 정제
- 결측치 처리를 위한 3가지 방법
    1. 결측치가 있는 row만 제거
    2. 결측치가 존재하는 feature 전부 제거
    3. 특정값으로 대체(중위수 등): test set에도 train set에서 사용한 값으로 결측치 대체

```python
housing.dropna(subset=['total_bedrooms']) # 1. 결측치가 있는 row만 제거
housing.drop('total_bedrooms',axis=1) # 2. 결측치가 존재하는 feature 전부 제거
median=housing['total_bedrooms'].median() 
housing['total_bedrooms'].fillna(median, inplace=True) # 3. 특정값으로 대체(중위수 등) 
```

```python
# 사이킷런을 이용한 결측치 처리
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median') # 결측치를 중위수로 대체하는 객체생성
housing_num=housing.drop('ocean_proximity',axis=1) # 수치형 데이터만 따로 저장 
imputer.fit(housing_num) # train set의 중위수 계산
X=imputer.transform(housing_num) # 계산된 값을 train set에 적용
housing_tr=pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index) # DataFrame으로 변환
```

# 9. 텍스트와 범주형 특성 다루기

```python
# 텍스트 데이터인 'ocean_proximity'의 샘플확인
housing_cat=housing[['ocean_proximity']]
housing_cat.head()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_12.jpg)


```python
# 'ocean_proximity'는 문자형 카테고리 feture이므로 숫자형 카테고리로 변환 
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_13.jpg)

```python
# 변환에 사용된 문자형 카테고리 확인
ordinal_encoder.categories_
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_14.jpg)

```python
# 숫자형 카테고리로 반환할 경우, 머신러닝 알고리즘이 가까운 숫자끼리 더 비슷한 것으로 인식할 수 있음
# 이를 방지하기 위해 one-hot encoding을 사용 
from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_15.jpg)


# 10. 나만의 변환기 사용

```python
# 특성조합을 만드는 class 예제

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, househlods_ix= 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:,househlods_ix]
        population_per_househlod = X[:, population_ix] / X[:, househlods_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,population_ix] / X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_househlod, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_househlod]
    
attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)
```

# 11. 변환 파이프라인

```python
# 데이터 전처리를 순서대로 처리하는 Pipeline 클래스
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline=Pipeline([
                       ('imputer', SimpleImputer(strategy='median')),
                       ('attr_adder', CombinedAttributesAdder()),
                       ('std_scaler', StandardScaler())
                       ])

housing_num_tr=num_pipeline.fit_transform(housing_num)
```

```python
# column의 특성에 따라 데이터를 전처리하는 ColumnTransformer 클래스
from sklearn.compose import ColumnTransformer

num_attribs=list(housing_num)
cat_attribs=['ocean_proximity']

full_pipeline = ColumnTransformer([
                                   ('num', num_pipeline, num_attribs),
                                   ('cat', OneHotEncoder(), cat_attribs)
                                   ])

housing_prepared=full_pipeline.fit_transform(housing)
```

# 12. 모델 선택과 훈련

```python
# 선형 회귀모형 적합
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 선형 회귀모형 예측치 계산
from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)

# 선형 회귀모형 RMSE 계산
lin_mse=mean_squared_error(housing_labels, housing_predictions)
lin_rmse=np.sqrt(lin_mse)
print(lin_rmse)
```

```python
# Decision tree 모형 적합
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# Decision tree 모형 예측치 계산 및 평가
housing_predictions=tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels, housing_predictions)
tree_rmse=np.sqrt(tree_mse)
print(tree_rmse)
```

# 13. 교차검증을 사용한 평가

k-fold cross validation는 training data set을 k개의 subset으로 분할하여 k-1개의 training data set으로 모델을 훈련시킨 뒤, 나머지 1개의 set으로 모델을 평가하는 과정을 k번 반복하여 모델을 평가하는 방법이다.   

```python
# sklearn의 cross validation (k=10)from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)

# cross_val_score에서 scoring은 neg_mean_squared_error는 mse의 음수(-)값으로 측정
# rmse를 계산하기 위해서 scores를 -scores로 변환
tree_rmse_scores=np.sqrt(-scores)

# descision tree regression의 k-cross validation 결과확인
def display_scores(scores):
    print("점수:", scores)
    print("평균:", scores.mean())
    print("표준편차:", scores.std())

display_scores(tree_rmse_scores)
```

# 14. 모델 세부튜닝

```python
# grid search는 일정범위의 hyper parameter의 모든 조합을 탐색
# 1차 search: 'n_estimators':[3,10,30], 'max_features':[2,4,6,8]의 조합(3*4 = 12번) 탐색
# 2차 search: 'n_estimators':[3,10], 'max_features':[2,3,4]의 조합(2*3 = 6번) 탐색
# 1차와 2차 각각 5-fold cross validaion 실행(12*5 + 6*5 = 총 90번의 training 실행)  

from sklearn.model_selection import GridSearchCV
param_grid=[
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
]

forest_reg=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg,param_grid, cv=5,scoring='neg_mean_squared_error',
                         return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)  

# 선택된 최적의 parameter 확인 
print(grid_search.best_params_)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_15_2.jpg)

```python
# grid search의 cross validation 결과확인
cvres=grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_16.jpg)

# 15. 특성(feature)의 중요도 분석

```python
# 특성 중요도를 feature_importances에 저장
feature_importances=grid_search.best_estimator_.feature_importances_

extra_attribs=["rooms_per_hhold", "pop_per_hhold",  "bedrooms_per_hhold"]
cat_encoder=full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs=list(cat_encoder.categories_[0])
attributes=num_attribs + extra_attribs + cat_one_hot_attribs

# 특성 중요도와 특성 명칭 결합
sorted(zip(feature_importances, attributes), reverse=True)
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_17.jpg)

# 16. 테스트 세트로 시스템 평가하기

```python
# 최종모델 선택
final_model=grid_search.best_estimator_

# test set에서 target 분리
X_test=strat_test_set.drop("median_house_value", axis=1)
y_test=strat_test_set["median_house_value"].copy()

# training에서 사용했던 것과 동일한 전처리 pipeline 적용
X_test_prepared=full_pipeline.transform(X_test)

# test set에서의 성능 평가
final_predictions=final_model.predict(X_test_prepared)
final_mse=mean_squared_error(y_test, final_predictions)
final_rmse=np.sqrt(final_mse)

# 오차의 95% 신뢰구간 계산
from scipy import stats
confidence=0.95
squared_errors=(final_predictions-y_test)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
```
