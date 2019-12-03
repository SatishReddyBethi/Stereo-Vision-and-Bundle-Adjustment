import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('ImageData/img_00000.jpg',0)
imgR = cv2.imread('ImageData/img_00001.jpg',0)

cap = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)
ret, imgL = cap.read()
imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
while(1):
    ret, imgR = cap.read()
    imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    imgL = imgR.copy()
    # plt.imshow(disparity,'gray')
    # plt.show()

    max = np.max(disparity)
    disparity = disparity/max
    disparity = 255*disparity

    cv2.namedWindow('Original Image')
    cv2.namedWindow('Disparity')
    cv2.imshow('Original Image', imgL)
    cv2.imshow('Disparity', disparity)
    k = cv2.waitKey(20) & 0xFF  # (20) - window stays open for 20ms | 0xFF - 255
    if k == 27 or k == ord('e'):
        break
cv2.destroyAllWindows()
