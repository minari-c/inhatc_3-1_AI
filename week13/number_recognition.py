import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('../model_file/mnist_model1.hdf5')

img = cv2.imread('../resource/7.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.astype('float32')

# cv2.imshow('white', img)
img = 255 - img
# cv2.imshow('gray', img)
# cv2.waitKeyEx(0)
# cv2.destroyWindow(0)

img = img / 255.0
img = img[np.newaxis, :, :, np.newaxis]

test_pred = model.predict(img)

print(np.round(test_pred, 2))
