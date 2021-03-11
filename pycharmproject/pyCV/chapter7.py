import cv2
import numpy as np

from pycharmproject.pyCV.chapter6 import stackImages

"""HSV 颜色模型
Hue:色调0-360'
Saturation:饱和度，色彩纯净度 100%
Value: 明度 100%
HSV 颜色分量范围:
H:  0— 180
S:  0— 255
V:  0— 255
"""  # 寻找色点(color detection)


def empty(a):
    pass


if __name__ == '__main__':

    path = "resources/lambo_std.PNG"
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    # cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    # cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    # cv2.createTrackbar("Saturation Min", "TrackBars", 0, 255, empty)
    # cv2.createTrackbar("Saturation Max", "TrackBars", 255, 255, empty)
    # cv2.createTrackbar("Value Min", "TrackBars", 0, 255, empty)
    # cv2.createTrackbar("Value Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
    cv2.createTrackbar("Saturation Min", "TrackBars", 110, 255, empty)
    cv2.createTrackbar("Saturation Max", "TrackBars", 240, 255, empty)
    cv2.createTrackbar("Value Min", "TrackBars", 153, 255, empty)
    cv2.createTrackbar("Value Max", "TrackBars", 255, 255, empty)

    while True:
        img = cv2.imread(path)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # imgHSV1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # print(imgHSV.shape)

        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Saturation Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Saturation Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Value Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Value Max", "TrackBars")
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("Original", img)
        # cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
        # cv2.imshow("HSV", imgHSV)
        # cv2.imshow("HSV1", imgHSV1)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Result", imgResult)

        imagStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
        cv2.imshow("imageStack", imagStack)
        cv2.waitKey(1)
