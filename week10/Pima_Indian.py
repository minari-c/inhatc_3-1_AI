import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 불러올 때 각 컬럼에 해당하는 이름을 지정합니다.
df = pd.read_csv(
	'../csv/pima-indians-diabetes.csv',
	names=[
		"pregnant", "plasma", "pressure",
		"thickness", "insulin", "BMI",
		"pedigree", "age", "class"
	]
)

# 처음 5줄을 봅니다.
print(df.head(5))

# 데이터의 전반적인 정보를 확인해 봅니다.
print(df.info())

# 각 정보별 특징을 좀더 자세히 출력합니다.
print(df.describe())

# 데이터 중 임신 정보와 클래스 만을 출력해 봅니다.
print(df[['plasma', 'class']])

# 데이터 간의 상관관계를 그래프로 표현해 봅니다.

colormap = plt.cm.gist_heat  # 그래프의 색상 구성을 정합니다.
plt.figure(figsize=(12, 12))  # 그래프의 크기를 정합니다.

# 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시되게 합니다.
sns.heatmap(
	df.corr(),
	linewidth=0.1,
	vmax=0.5,
	cmap=colormap,
	linecolor='white',
	# annot=True
)

plt.show()

grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()

# ==============================================================================================


# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러 옵니다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 필요한 라이브러리를 불러 옵니다.
import numpy as np

# 실행할 때마다 같은 결과를 출력 하기 위해 설정 하는 부분 입니다.
np.random.seed(3)
tf.random.set_seed(3)

# data 를 불러 옵니다.
dataset = np.loadtxt("../csv/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 모델을 설정 합니다.
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))  # 입력층과 은닉층이 동시에 있다.
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일 합니다.
model.compile(
	loss='binary_crossentropy',
	optimizer='adam',
	metrics=['accuracy']
)

# 모델을 실행 합니다.
model.fit(X, Y, epochs=200, batch_size=10, verbose=0)

# 결과를 출력 합니다.
print("Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
