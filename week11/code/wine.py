import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pandas as pd

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
# 	try:
# 		tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)
# 		tf.config.experimental.set_memory_growth(gpus[0], True)
# 	except RuntimeError as e:
# 		# Memory growth must be set before GPUs have been initialized
# 		print(e)

# 데이터를 입력합니다.
df = pd.read_csv('../csv/wine.csv', header=None)

# 와인의 속성을 X로 와인의 분류를 y로 저장합니다.
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]
X = X.astype(float)
y = y.astype(float)

# 학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조를 설정합니다.
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 학습의 자동 중단 및 최적화 모델 저장
# 학습이 언제 자동 중단 될지를 설정합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# 최적화 모델이 저장될 폴더와 모델의 이름을 정합니다.
modelpath = "./data/model/Ch14-4-bestmodel.hdf5"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 모델을 실행합니다.
with tf.device("/device:GPU:1"):
	history = model.fit(X_train, y_train, epochs=2000, batch_size=500,
	                    validation_split=0.25, verbose=1,
	                    callbacks=[early_stopping_callback, checkpointer])

# 테스트 결과를 출력합니다.
score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
