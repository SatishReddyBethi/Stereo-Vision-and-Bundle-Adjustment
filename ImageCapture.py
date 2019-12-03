import cv2
import numpy as np

np.set_printoptions(suppress=True)

ProjPath = 'C:\\Users\\satis\\OneDrive\\Documents\\MS\\Sem 3\\Robot Perception\\assignment1\\'

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('Camera Footage')
cap = cv2.VideoCapture(1)
imageNo = 8
while(1):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('Camera Footage',frame)
    k = cv2.waitKey(20) & 0xFF # (20) - window stays open for 20ms | 0xFF - 255
    if k == 27 or k == ord('e') :
        break
    elif k == ord('c'):
        cv2.imwrite(ProjPath+'AR_Demo.png',frame) # save image as png
        imageNo +=1
cv2.destroyAllWindows()