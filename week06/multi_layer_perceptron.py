import numpy as np
import matplotlib.pyplot as plt

w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1


def MLP(x, w, b):
	y = np.sum(w * x) + b
	
	if y <= 0:
		return 0
	else:
		return 1


def NAND(x1, x2):
	return MLP(np.array([x1, x2], w11, b1))


def OR(x1, x2):
	return MLP(np.array([x1, x2], w12, b2))


def AND(x1, x2):
	return MLP(np.array([x1, x2], w2, b3))


def XOR(x1, x2):
	return AND(NAND(x1, x2), OR(x1, x2))


