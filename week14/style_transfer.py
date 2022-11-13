import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None)
		tf.config.experimental.set_memory_growth(gpus[0], True)
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

with tf.device("/device:GPU:0"):
	def tensor_to_image(tensor):
		tensor = tensor * 255
		tensor = np.array(tensor, dtype=np.uint8)
		if np.ndim(tensor) > 3:
			assert tensor.shape[0] == 1
			tensor = tensor[0]
		return PIL.Image.fromarray(tensor)
	
	
	def load_img(path_to_img):
		max_dim = 256
		img = tf.io.read_file(path_to_img)
		img = tf.image.decode_image(img, channels=3)
		img = tf.image.convert_image_dtype(img, tf.float32)
		
		shape = tf.cast(tf.shape(img)[:-1], tf.float32)
		long_dim = max(shape)
		scale = max_dim / long_dim
		
		new_shape = tf.cast(shape * scale, tf.int32)
		
		img = tf.image.resize(img, new_shape)
		img = img[tf.newaxis, :]
		return img
	
	
	def imshow(image, title=None):
		if len(image.shape) > 3:
			image = tf.squeeze(image, axis=0)
		
		plt.imshow(image)
		if title:
			plt.title(title)
	
	
	content_image = load_img('../resource/week14/Ang.jpg')
	style_image = load_img('../resource/week14/style5.jpg')
	
	plt.subplot(1, 2, 1)
	imshow(content_image, 'Content Image')
	
	plt.subplot(1, 2, 2)
	imshow(style_image, 'Style Image')
	
	import tensorflow_hub as hub
	
	hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
	stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
	stylized_image_file = tensor_to_image(stylized_image)
	
	stylized_image_file.show()
	stylized_image_file.save('../resource/week14/Ang-e.png', format='png')
