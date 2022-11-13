import cv2
import numpy as np

gimg = cv2.imread('../resize_img/test1_halfgray.png')
kernal_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gimge = cv2.filter2D(gimg, -1, kernal_x)

cv2.imwrite('../resize_img/test_halfgray_filter.png', gimge)
