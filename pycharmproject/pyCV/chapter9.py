import cv2
import numpy as np

# Face detection
# viola & jones

faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

img = cv2.imread("resources/lena.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# minNeighbors=3：匹配成功所需要的周围矩形框的数目，每一个特征匹配到的区域都是一个矩形框，只有多个矩形框同时存在的时候，才认为是匹配成功，比如人脸，这个默认值是 3
faces = faceCascade.detectMultiScale(imgGray, 1.1, 3)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (250, 0, 0), 2)

cv2.imshow("Result",img)
cv2.waitKey(0)
