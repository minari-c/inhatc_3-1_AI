import tensorflow as tf
import keras
from tensorflow.python.client import device_lib

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], False)

print(device_lib.list_local_devices())

print(tf.__version__)

print(keras)
