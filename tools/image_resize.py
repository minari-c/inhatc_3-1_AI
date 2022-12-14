from time import sleep
import cv2

img = cv2.imread('../resource/cat.jpg')
cv2.imwrite('../resize_img/cat.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 100])

img2 = cv2.imread('../resize_img/cat.png')
img2 = cv2.resize(img2, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('../resize_img/cat.png', img2)

print(img2.shape)
cv2.imshow('Color', img2)
