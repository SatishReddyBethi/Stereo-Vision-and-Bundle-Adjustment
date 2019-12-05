import cv2
import numpy as np

def NewPixels(pts1,pts2,stichedImg,Img2):
    AvgPt1 = np.average(pts1, axis=0)
    AvgPt2 = np.average(pts2, axis=0)
    Theta = np.arccos(np.dot(AvgPt1,AvgPt2)/(np.linalg.norm(AvgPt1)*np.linalg.norm(AvgPt2)))
    Delta = AvgPt2 - AvgPt1

    (h, w) = stichedImg.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -Theta, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    NewstichedImg = cv2.warpAffine(stichedImg, M, (nW, nH))

    # perform translation on stiched img
    M = np.float32([[1, 0, Delta[0]], [0, 1, Delta[1]]])
    #print(nW,nH,stichedImg.shape[0],stichedImg.shape[1])
    NewstichedImg = cv2.warpAffine(NewstichedImg, M, (NewstichedImg.shape[1]+int(np.abs(Delta[0])+5), NewstichedImg.shape[0]+int(np.abs(Delta[1]))+5))
    #TNewstichedImg = cv2.warpAffine(stichedImg, M, (nW + int(np.abs(Delta[0])), nH + int(np.abs(Delta[1]))))
    NewstichedImg[0:Img2.shape[0], 0:Img2.shape[1]] = Img2
    #TNewstichedImg[0:NewImg1.shape[0], 0:NewImg1.shape[1]] = NewImg1
    #cv2.imshow('Rotated Image2', NewImg1)
    cv2.imshow('Img2', Img2)
    cv2.imshow('Prev Stitched Img', stichedImg)
    cv2.imshow('New Stitched Img', NewstichedImg)
    cv2.waitKey(0)

    return Img2,NewstichedImg



path = "ImageData/img_00000.jpg"
image1  =  cv2.imread(path)
StitchedImg = image1.copy()
N = 40
sift = cv2.xfeatures2d.SIFT_create()

for i in range(N):
    print(i)
    path = "ImageData/img_" + str(i + 1).zfill(5) + ".jpg"
    image2 = cv2.imread(path)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # print(pts1)
    # print(pts2)
    image1,StitchedImg = NewPixels(pts1,pts2,StitchedImg,image2)

cv2.imshow('New Stitched Img', StitchedImg)
cv2.imwrite('Output_Images/Out.jpg',StitchedImg)
cv2.waitKey(0)

