import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt
# from scipy.optimize import least_squares
# import tensorflow as tf
from scipy import optimize

COUNT = 10


def getImageandplot():
    global COUNT
    print("Getting Images")

    number_of_images = 41
    images = []

    for i in range(0,41):
        imgDefine = "ImageData/img_" + "%05d" % i + ".jpg"
        img = cv.imread(imgDefine,1)
        images.append(img)

    #
    size = 2500
    center = (2000,2000)

    initial_center = np.array([[1.0000000e+00,0.0,center[0]-320],[0.0,1.0000000e+00,center[1]-240],[0.0,0.0,1.0000000e+00]])

    keypoints =[]
    descriptors =[]

    for img in images :
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()

        kp, des = sift.detectAndCompute(gray,None)
        keypoints.append(kp)
        descriptors.append(des)

    T = initial_center
    f0 = cv.warpPerspective(images[0],T,(3100,3100)) # placing image 0 with its center at 2000,2000 in a 3100x3100 image
    b0 = cv.warpPerspective(images[0],T,(3100,3100))
    # print(f0.shape)
    # cv.imshow("f0", f0)
    # cv.imshow("b0", b0)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    for k in range (1,25):
        f1 = warpI(k,images,f0,keypoints,descriptors)
        f0 = f1

    # cv.imshow("f1", f1)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    for k in range(1,18):

        b1 = warpI(41-k,images,b0,keypoints,descriptors)
        b0 = b1

    # cv.imshow("b1", b1)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    print ("Success!")

    pre_opt_img = warpP(f1,b1)
    # cv.imshow("Pre_Opt",pre_opt_img)
    pre_opt_img[1800:,2000:,:]=0
    # cv.imshow("Blackedned",pre_opt_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    cv.imwrite("Output_Images/pre_optimised_image.png", pre_opt_img)
    for i in range (0,5):
        k_fwd = 4-i
        K_bkwd = 40 - k_fwd
        final = warpP(images[k_fwd],pre_opt_img)
        pre_opt_img = final
        final = warpP(images[K_bkwd],pre_opt_img)
        pre_opt_img = final


    cv.imwrite("Output_Images/optimised_image.png",final)

def warpP(imageone, imagetwo):
    global saving
    color0 = cv.cvtColor(imageone,cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp_0,des_0 = sift.detectAndCompute(color0,None)

    color1 = cv.cvtColor(imagetwo, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(color1, None)

    kp1,kp2 = kp_0,kp_1
    des1,des2 = des_0,des_1

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    if len(good) > COUNT:
        srcpoints = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        distpoints = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M,mask = cv.findHomography(srcpoints,distpoints,cv.RANSAC,5.0)
        matchMask = [mask.ravel().tolist()]

        h,w = 3100,3100
        points = np.float32([ [0,6],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dist = cv.perspectiveTransform(points,M)

    else:
        print ("Not enough matches found " % (len(good),COUNT))
        matchMask = None

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchMask=matchMask, flags=2)

    T =cv.getPerspectiveTransform(points,dist)

    final = cv.warpPerspective(imageone,T,(3100,3100))
    final_OGM = final[:,:,0]+final[:,:,1]+final[:,:,2]
    mask = np.where(final_OGM == 0)
    final[mask[0],mask[1]] = imagetwo[mask[0],mask[1]]
    cv.imwrite("Output_Images/Optimize"+str(saving)+".png", final)
    saving+=1
    return final

saving = 0
def warpI (k,images,prev,keypoints,descriptors):
    global saving
    gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    # Using SIFT again because the images go from 0 to 24 then 41 to 25.
    # It can be optimized using the already known descriptors.
    kp_P0, des_P0 = sift.detectAndCompute(gray, None)

    image_no = k
    kp1, kp2 = keypoints[image_no], kp_P0
    des1, des2 = descriptors[image_no], des_P0

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    # COUNT is 10
    if len(good) > COUNT:
        srcpoints = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        distpoints = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # Homography between two sets of corresponding points
        M,mask = cv.findHomography(srcpoints,distpoints,cv.RANSAC,5.0)
        matchMask = [mask.ravel().tolist()]

        h,w = 640,480
        points = np.float32([ [0,6],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dist = cv.perspectiveTransform(points,M)

    else:
        print ("Not enough matches found - %d/%d" % (len(good),COUNT))
        matchMask = None

    # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchMask=matchMask, flags=2)

    T = cv.getPerspectiveTransform(points, dist)

    print("Optimzing images" + str(k))

    #print("Homography:",M)
    final = cv.warpPerspective(images[image_no], T, (3100, 3100))
    final_OGM = final[:, :, 0] + final[:, :, 1] + final[:, :, 2]
    mask = np.where(final_OGM == 0)
    #print(mask)
    final[mask[0], mask[1]] = prev[mask[0], mask[1]]
    cv.imwrite("Output_Images/"+str(saving)+".png", final)
    saving +=1
    return final

if __name__ == '__main__':
    getImageandplot()

