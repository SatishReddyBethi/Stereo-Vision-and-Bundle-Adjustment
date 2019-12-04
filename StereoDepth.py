import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('Captures/0.jpg',0)
imgR = cv2.imread('Captures/1.jpg',0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)



disparity = disparity + 16
disparity = disparity/1024
#disparity = 255*disparity
cv2.namedWindow('Original Image')
cv2.namedWindow('Disparity')
cv2.imshow('Original Image', imgL)
cv2.imshow('Disparity', disparity)
# plt.imshow(disparity,'gray')
# plt.show()
k = cv2.waitKey(0) & 0xFF  # (20) - window stays open for 20ms | 0xFF - 255
cv2.destroyAllWindows()
