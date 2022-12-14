from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
df = pd.read_csv(
	'../csv/iris.csv',
	names=[
		"sepal_length", "sepal_width",
		"petal_length", "petal_width",
		"species"
	]
)

# 그래프로 확인
sns.pairplot(df, hue='species')
plt.show()

# 데이터 분류
dataset = df.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]

# 문자열을 숫자로 변환 -> week11에 한번 더 언급 하심
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)  # one hot encoding [ 0, 0, 1 ]

# 모델의 설정
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(
	loss='categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy']
)

# 모델 실행
with tf.device("/device:GPU:0"):
	model.fit(X, Y_encoded, epochs=50, batch_size=1)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))
