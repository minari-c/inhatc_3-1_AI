import numpy as np
import tensorflow as tf

epoch_500 = [
    [0.08245987],
    [0.9338578],
    [0.94681925],
    [0.97675204]
]
epoch_400 = [
    [0.09270042],
    [0.9330059],
    [0.9340155],
    [0.9719212]
]
epoch_300 = [
    [0.11526483],
    [0.92341316],
    [0.9252143],
    [0.9802184]
]

y_true = [[0], [1], [1], [1]]
y_pred = epoch_300

bce = tf.keras.losses.BinaryCrossentropy()
print(bce(y_true=y_true, y_pred=y_pred).numpy())


'''
epochs == 500:
    0.05816477
    
epochs == 400:
    0.065842144

epochs == 300:
    0.074963674
'''