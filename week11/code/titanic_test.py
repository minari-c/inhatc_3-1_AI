import numpy as np
import pandas as pd
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)
		tf.config.experimental.set_memory_growth(gpus[0], True)
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

# 데이터 세트를 읽어들인다.
train = pd.read_csv("../csv/train.csv", sep=',')
test = pd.read_csv("../csv/test.csv", sep=',')

# 필요없는 컬럼을 삭제한다.
train.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name',
            'Cabin', 'PassengerId', 'Fare', 'Age'],
           inplace=True,
           axis=1)
test.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name',
           'Cabin', 'PassengerId', 'Fare', 'Age'],
          inplace=True,
          axis=1)

# 결손치가 있는 데이터 행은 삭제한다.
train.dropna(inplace=True)
test.dropna(inplace=True)

# 기호를 수치로 변환한다.
for ix in train.index:
	if train.loc[ix, 'Sex'] == "male":
		train.loc[ix, 'Sex'] = 1
	else:
		train.loc[ix, 'Sex'] = 0

for ix in test.index:
	if test.loc[ix, 'Sex'] == "male":
		test.loc[ix, 'Sex'] = 1
	else:
		test.loc[ix, 'Sex'] = 0

# 2차원 배열을 1차원 배열로 평탄화한다.
target = np.ravel(train.Survived)

# 생존여부를 학습 데이터에서 삭제한다.
train.drop(['Survived'], inplace=True, axis=1)
train = train.astype(float)  # 최근 소스에서는 float형태로 형변환하여야
test = test.astype(float)  # 최근 소스에서는 float형태로 형변환하여야

'''

입력 층 : 변동 없음
은닉층 1  :  16 ( relu)
은닉측 2  :   8  (relu)
은닉측 3  :   8  (relu)
출력 층   변동 없음.

'''

# 케라스 모델을 생성한다.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# 케라스 모델을 컴파일한다.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 케라스 모델을 학습시킨다.
with tf.device("/device:GPU:0"):
	model.fit(train, target, epochs=30, batch_size=1, verbose=1)

idx = test.shape[0]
pred = model.predict(test)
test_np = test.to_numpy()
for i in range(idx):
	pass
# print(test_np[i, :], pred[i])

print(test.head())
