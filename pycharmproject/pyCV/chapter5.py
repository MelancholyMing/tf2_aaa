import cv2
import numpy as np

# Warp Perspective 透视变换

img = cv2.imread("resources/cards.jpg")
print(img.shape)

width, height = 250, 350
pts1 = np.float32([[137, 151], [257, 129], [169, 324], [298, 294]])  # 左上，右上，左下，右下
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Image", img)
cv2.imshow("Output", imgOutput)
cv2.waitKey(0)
