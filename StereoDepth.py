import numpy as np
import cv2

imgL = cv2.imread('Data/0.png',0)
imgR = cv2.imread('Data/1.png',0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)

max = np.max(disparity)
disparity = disparity/max
disparity = 255*disparity

cv2.namedWindow('Original Image')
cv2.namedWindow('Disparity')
cv2.imshow('Original Image', imgL)
cv2.imshow('Disparity', disparity)
k = cv2.waitKey(0) & 0xFF  # (20) - window stays open for 20ms | 0xFF - 255

cv2.destroyAllWindows()
