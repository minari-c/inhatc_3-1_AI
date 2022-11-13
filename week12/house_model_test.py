from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

# 데이터를 불러 옵니다.
df = pd.read_csv("../csv/house_train.csv")

# 카테고리형 변수를 0과 1로 이루어진 변수로 바꾸어 줍니다.(12장 3절)
df = pd.get_dummies(df)

# 결측치를 전체 칼럼의 평균으로 대체하여 채워줍니다.
df = df.fillna(df.mean())

# 업데이트된 데이터프레임을 출력해 봅니다.
# df
# 집 값을 제외한 나머지 열을 저장합니다.
cols_train = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
X_train_pre = df[cols_train]

# 집 값을 저장합니다.
y = df['SalePrice'].values
X_train_pre = X_train_pre.astype(float)
y = y.astype(float)
# 전체의 80%를 학습셋으로, 20%를 테스트셋으로 지정합니다.
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2, shuffle=False)

model = load_model('../week11/code/data/model/house1.hdf5')
print('테스트 loss1', model.evaluate(X_test, y_test) / 10000000)

model = load_model('../week11/code/data/model/house2.hdf5')
print('테스트 loss2', model.evaluate(X_test, y_test) / 10000000)

'''
1. 테이블을 주면 코드로 변환하는 문제

2. 파라미터 수 ( W + b )
'''
