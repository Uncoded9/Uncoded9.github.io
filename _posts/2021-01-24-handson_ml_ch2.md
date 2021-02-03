---
title: Chapter 2
layout: post
categories: [ML, Python]
image: /assets/img/hands_on_ml/idea_sketch.jpg
description: "Welcome to Hands on ML chapter 2"
---

## 1. 데이터 다운로드


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

import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path) # housing_path에 있는 csv file을 dataframe으로 읽어오기

housing = load_housing_data()
housing.head() # 데이터의 처음 5행 확인
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_01.jpg)


```python
housing.info() #데이터에 대한 간략한 설명
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_02.jpg)



```python
# Dtype이 object인 경우 데이터 타입이 불명확
# 이 데이터에서는 categorical value이기 때문에 value_counts()로 값 확인 

housing['ocean_proximity'].value_counts() 
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_03.jpg)

```python
housing.describe() #숫자형 feature의 요약정보
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_04.jpg)

```python
%matplotlib inline   # jupyter notebook안에서 plot이 그려지도록 설정 
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_05.jpg)


```python
%matplotlib inline   # jupyter notebook안에서 plot이 그려지도록 설정 
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
```

# 3. Test set 만들기

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

# 위에서 정의한 split_train_set을 사용하면 매번 다른 test set이 생성되는 단점이 있음 
```

위에서 정의한 split_train_set을 사용하면 매번 다른 test set이 생성되는 단점이 있다.  이를 방지하기 위해서 각 샘플 index의 해시값을 계산하여 해시 최댓값의 20%보다 작거나 같은 샘플만 test set으로 설정할수 있다. 이 방법을 사용하면 새로운 데이터가 추가되도 기존의 테스트 세트는 유지되고 새로운 데이터의 20%만 추가된다.

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

sklearn에서는 random_state를 사용하여 난수 초기값 설정가능 하다.     

```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```
  

```python
# 무작위 샘플링은 샘플링 편향을 발생시킬 수 있음
# 예제 데이터에서 중위소득의 비율을 히스토그램을 이용하여 시각화   
housing["income_cat"]=pd.cut(housing["median_income"],bins=[0. ,1.5 ,3.0 ,4.5 ,6. ,np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()
```

![](https://uncoded9.github.io/assets/img/hands_on_ml/ch2/handson_ml_ch02_06.jpg)


```python
# 샘플링 편향을 방지하기 위한 계층적 샘플링(stratified sampling)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
```
