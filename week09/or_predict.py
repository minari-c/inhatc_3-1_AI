import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

model = tf.keras.models.Sequential()


# add layer
model.add(layers.Dense(units=2, input_shape=(2,), activation='sigmoid'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=0.3), metrics=['accuracy'])

print(model.summary())

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

model.fit(X, y, batch_size=1, epochs=300)

print(model.predict(X))

'''
epochs == 500:
    1/4 [======>.......................] - ETA: 0s - loss: 0.0068 - accuracy: 1.0000
    4/4 [==============================] - 0s 500us/sample - loss: 0.0037 - accuracy: 1.0000
    [[0.08245987]
     [0.9338578 ]
     [0.94681925]
     [0.97675204]]

epochs == 400:
    1/4 [======>.......................] - ETA: 0s - loss: 0.0045 - accuracy: 1.0000
    4/4 [==============================] - 0s 250us/sample - loss: 0.0046 - accuracy: 1.0000
    [[0.09270042]
     [0.9330059 ]
     [0.9340155 ]
     [0.9719212 ]]

epochs == 300:
    1/4 [======>.......................] - ETA: 0s - loss: 0.0134 - accuracy: 1.0000
    4/4 [==============================] - 0s 500us/sample - loss: 0.0064 - accuracy: 1.0000
    [[0.11526483]
     [0.92341316]
     [0.9252143 ]
     [0.9802184 ]]
'''