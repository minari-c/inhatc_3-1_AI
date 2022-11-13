# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다.
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)
		tf.config.experimental.set_memory_growth(gpus[0], True)
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

tf.random.set_seed(42)
# tf.config.experimental.enable_op_determinism()

from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = \
	keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled,
                                                                      train_target,
                                                                      test_size=0.2,
                                                                      random_state=42)

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                              padding='same', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)
with tf.device("/device:GPU:0"):
	history = model.fit(train_scaled, train_target, epochs=20,
	                    validation_data=(val_scaled, val_target),
	                    callbacks=[checkpoint_cb, early_stopping_cb])

model.evaluate(val_scaled, val_target)
