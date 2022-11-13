import cv2

img = cv2.imread('../resize_img/half.png')
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('../resize_img/test1_halfgray.png', gimg)
print(gimg.shape)
