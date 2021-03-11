import cv2
import numpy as np
from pycharmproject.pyCV.chapter6 import stackImages


# contours/shape detection
def getContours(img):
    # https://blog.csdn.net/hjxu2016/article/details/77833336
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)  # 计算轮廓的面积
        print(area)
        # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
        if area > 500:
            # 画出图片中的轮廓值，也可以用来画轮廓的近似值 参数说明:img 表示输入的需要画的图片， contours 表示轮廓值，-1 表示轮廓的索引, 如果是 - 1，则绘制其中的所有轮廓，(0, 0, 255) 表示颜色， 2 表示线条粗细
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)  # 计算轮廓的周长
            print(peri)
            # 用于获得轮廓的近似值，把一个连续光滑曲线折线化.使用 cv2.drawCountors 进行画图操作.cnt 为输入的轮廓值， epsilon 为阈值 T，通常使用轮廓的周长作为阈值，True 表示的是轮廓是闭合的
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(approx)
            objCor = len(approx)
            # 获得外接矩形. x，y, w, h 分别表示外接矩形的 x 轴和 y 轴的坐标，以及矩形的宽和高， cnt 表示输入的轮廓值
            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 3:
                objectType = "Triangle"
            elif objCor == 4:
                aspRatio = w / float(h)  # 宽高比
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4:
                objectType = "Circles"
            else:
                objectType = "None"

            # 根据坐标在图像上画出矩形  img 表示传入的图片， (x, y) 表示左上角的位置，（x+w， y+h）表示加上右下角的位置，（0, 255, 0) 表示颜色，2 表示线条的粗细
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 55, 100), 2)


path = "resources/shapes.png"
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)  # 边缘化
getContours(imgCanny)

imgBlank = np.zeros_like(img)
imgStack = stackImages(0.6, ([img, imgGray, imgBlur],
                             [imgCanny, imgContour, imgBlank]))

# cv2.imshow("Original", img)
# cv2.imshow("Gray", imgGray)
# cv2.imshow("Blur", imgBlur)
cv2.imshow("stack", imgStack)
cv2.waitKey(0)
