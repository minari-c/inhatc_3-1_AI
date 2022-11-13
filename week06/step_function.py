import numpy as np
import matplotlib.pyplot as plt


def relu(x):
	return np.maximum(0, x)


x = np.arange(-10.0, 10.0, 0.1)
# y = step(x)
# y = sigmoid(x)
# y = tanh(x)
y = relu(x)
plt.plot(x, y)
plt.show()
