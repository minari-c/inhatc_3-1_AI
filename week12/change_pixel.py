from time import sleep
import cv2

img = cv2.imread('../resize_img/half.png')

print(f'img.ndim = {img.ndim}')
print(f'img.ndim = {img.shape}')
print(f'img.ndim = {img.dtype}')

img = cv2.resize(img, (0, 0), interpolation=cv2.INTER_LANCZOS4, fx=0.5, fy=0.5)
cv2.imwrite('../resize_img/half.png', img)
print(img.shape)
cv2.imshow('Color', img)
