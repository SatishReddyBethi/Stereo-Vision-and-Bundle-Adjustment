import cv2
import numpy as np

path = "ImageData/img_00000.jpg"
print(path)
image1  =  cv2.imread(path)

for i in range(10):

    path = "ImageData/img_0000" + str(i+1) + ".jpg"
    image2 = cv2.imread(path)
    stitcher = cv2.createStitcher() #if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch([image1,image2])
    if status == 0:
        # write the output stitched image to disk
        #cv2.imwrite(args["output"], stitched)

        # display the output stitched image to our screen
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)

    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
       print("[INFO] image stitching failed ({})".format(status))
    image1 = stitched