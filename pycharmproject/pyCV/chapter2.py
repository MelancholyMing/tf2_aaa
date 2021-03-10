import cv2
import numpy as np


img = cv2.imread("./resources/len_std.png")
kernel = np.ones((5, 5), np.uint8)
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)  # 高斯模糊（3 5 7 9奇数）
# imgCanny = cv2.Canny(img, 100, 100)
imgCanny1 = cv2.Canny(img, 150, 200)  # 边缘检测
imgDilation = cv2.dilate(imgCanny1, kernel, iterations=1)  # 形态学膨胀（iterations=2）
imgEroded = cv2.erode(imgDilation, kernel, iterations=1)  # 图片腐蚀

cv2.imshow('Gray image', imgGray)
cv2.imshow('Blur image', imgBlur)
# cv2.imshow('Canny image', imgCanny)
cv2.imshow('Canny1 image', imgCanny1)
cv2.imshow('Dilation image', imgDilation)
cv2.imshow('Eroded image', imgEroded)

cv2.waitKey(0)
