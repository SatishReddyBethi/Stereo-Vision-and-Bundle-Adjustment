import cv2
import numpy as np

path = "ImageData/img_00000.jpg"
image1  =  cv2.imread(path)

for i in range(40):

    path = "ImageData/img_" + str(i + 1).zfill(5) + ".jpg"
    print(i)
    image2 = cv2.imread(path)
    stitcher = cv2.createStitcher() #if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch([image1,image2])
    if status == 0:
        # write the output stitched image to disk
        #cv2.imwrite(args["output"], stitched)
        pass

        # display the output stitched image to our screen


    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
       print("[INFO] image stitching failed ({})".format(status))
    image1 = stitched

cv2.imshow("Stitched", image1)
cv2.waitKey(0)