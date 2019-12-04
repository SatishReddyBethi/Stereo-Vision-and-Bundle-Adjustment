import cv2
import numpy as np

np.set_printoptions(suppress=True)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('Camera Footage')
cap = cv2.VideoCapture(0)
imageNo = 0#np.load("Data/ImageNo.npy")
while(1):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('Camera Footage',frame)
    k = cv2.waitKey(20) & 0xFF # (20) - window stays open for 20ms | 0xFF - 255
    if k == 27 or k == ord('e') :
        break
    elif k == ord('c'):
        cv2.imwrite('Captures/' + str(imageNo) +'.jpg',frame) # save image as png
        imageNo +=1
        np.save("Data/ImageNo", imageNo)
cv2.destroyAllWindows()