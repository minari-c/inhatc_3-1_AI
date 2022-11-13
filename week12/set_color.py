import cv2

img = cv2.imread('../resize_img/half.png')

img[100:300, 100:300, :] = 255
img[100:300, 300:500, :] = 0
img[100:300, 500:700, :] = 127

img[300:500, 100:300, 0] = 255
img[300:500, 300:500, 1] = 255
img[300:500, 500:700, 2] = 255

cv2.imwrite('../resize_img/test1_half_changed.png', img)
