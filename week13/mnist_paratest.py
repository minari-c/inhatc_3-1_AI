import tensorflow as tf
from tensorflow.keras import datasets, layers, models

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)
		tf.config.experimental.set_memory_growth(gpus[0], True)
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images = train_images[:100, :, :, :]
train_labels = train_labels[:100]
test_images = test_images[:100, :, :, :]
test_labels = test_labels[:100]

# cv2_imshow(train_images[0, :, :, :])

# 픽셀 값을 0~1 사이로 정규화한다.
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()

input_shape = (28, 28, 1)

model.add(layers.Input(shape=input_shape))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with tf.device("/device:GPU:1"):
	model.fit(train_images, train_labels, epochs=5)
model.save('../model_file/mnist_model2.hdf5')
