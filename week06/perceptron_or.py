import numpy
from typing import Type

epsilon = 0.0000001


def perceptron(x1: float, x2: float) -> int:
	'''
:param x1:
	input1
:param x2:
	input1
:return:
	0 or 1 integer value
	'''
	w1, w2, b = 1.0, 1.0, -1.5
	sum = (x1 * w1) + (x2 * w2) + b
	
	if sum > epsilon:
		return 1
	else:
		return 0


print(perceptron(0, 0))
print(perceptron(1, 0))
print(perceptron(0, 1))
print(perceptron(1, 1))
