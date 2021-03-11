import cv2
# read images-videos-webcam
# img
# img = cv2.imread("./resources/len_std.png")
#
# cv2.imshow("output", img)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

# video
cap = cv2.VideoCapture("./resources/test_video.mp4")

while True:
    success, img = cap.read()
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(10, 100)
#
# while True:
#     success, img = cap.read()
#     cv2.imshow('Video', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
