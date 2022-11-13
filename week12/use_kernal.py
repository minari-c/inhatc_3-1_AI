import cv2
import numpy as np

gimg = cv2.imread('../resize_img/test1_halfgray.png')
kernal1 = np.ones((3, 3), dtype=np.float64) / 9
kernal2 = np.ones((5, 5), dtype=np.float64) / 25
kernal3 = np.ones((7, 7), dtype=np.float64) / 49

gimg1 = cv2.filter2D(gimg, -1, kernal1)
gimg2 = cv2.filter2D(gimg, -1, kernal2)
gimg3 = cv2.filter2D(gimg, -1, kernal3)

cv2.imwrite('../resize_img/test_halfgray_filt1.png', gimg1)
cv2.imwrite('../resize_img/test_halfgray_filt2.png', gimg2)
cv2.imwrite('../resize_img/test_halfgray_filt3.png', gimg3)
