import cv2
import numpy as np
import skimage.io as sio


# Resizing and Cropping 改变大小和剪切
img = sio.imread("./resources/lambo1.png")
print("skimage:", img.shape)

img = cv2.imread("./resources/lambo1.png")

print("CV2:", img.shape)  # (288, 658, 3) 高:宽(y:x)

imgResize = cv2.resize(img, (300, 200))  # (300, 200) 宽:高(x:y)
print(imgResize.shape)

# 图像裁剪
imgCropped = img[0:200, 200:500]

imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgGrb = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
print(imgGrb.shape)
print(imgGrb)
cv2.imshow("GRB", imgGrb)
cv2.imshow("Image", img)
cv2.imshow("Image Resize", imgResize)
cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)
