
import glob
import cv2
import numpy as np
import cv2 
import matplotlib.pyplot as plt


shark_img_1 = cv2.imread('homework_1/Q4.SIFT/Q4_Image/Shark1.jpg')  
shark_img_2 = cv2.imread('homework_1/Q4.SIFT/Q4_Image/Shark2.jpg')  
gray_shark_img_1 = cv2.cvtColor(shark_img_1, cv2.COLOR_BGR2GRAY)
gray_shark_img_2 = cv2.cvtColor(shark_img_2, cv2.COLOR_BGR2GRAY)

kp1_sort = []
kp2_sort = []
matches = []


def Detect_Keypoints ():

    global img_1, img_2, kp1_sorted, kp2_sorted, descriptors_1, descriptors_2 

    gray1 = cv2.cvtColor(shark_img_1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(shark_img_2, cv2.COLOR_BGR2GRAY)

    #  detect and compute keypoints
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(shark_img_1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(shark_img_2,None)

    # count the kp size
    for num in range(len(keypoints_1)):
        kp1_size = keypoints_1[num]
        kp1_sort.append(kp1_size)
        
    for num in range(len(keypoints_2)):
        kp2_size = keypoints_2[num]
        kp2_sort.append(kp2_size)

    kp1_sorted = sorted(kp1_sort, key=lambda kp1_sort: kp1_sort.size , reverse=True)[:200]
    kp2_sorted = sorted(kp2_sort, key=lambda kp2_sort: kp2_sort.size , reverse=True)[:200]

    img_1 = cv2.drawKeypoints(gray1,kp1_sorted,shark_img_1,\
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    img_2 = cv2.drawKeypoints(gray2,kp2_sorted,shark_img_2,\
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    return img_1, img_2, kp1_sorted, kp2_sorted, descriptors_1, descriptors_2

def Show_Keypoints():

    Detect_Keypoints()
    print()

    plt.subplots(figsize = (9, 6))
    plt.subplot(121),plt.imshow(img_1)
    plt.title('Shark_1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_2)
    plt.title('Shark_2'), plt.xticks([]), plt.yticks([])
    plt.show()

def ROOTSIFT(grayIMG, kpsData):

    extractor = cv2.SIFT_create()
    (kps, descs) = extractor.compute(grayIMG, kpsData)
    if len(kps) > 0:
        #L1-正規化
        eps=1e-7
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        #取平方根
        descs = np.sqrt(descs)
        return (kps, descs)
    else:
        return ([], None)
def Matched_Keypoints ():

    global img_out, matches, kpsA, kpsB
    detector = cv2.SIFT_create()
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    kpsA = detector.detect(gray_shark_img_1)
    kpsB = detector.detect(gray_shark_img_2)
    (kpsA, featuresA) = ROOTSIFT(gray_shark_img_1, kpsA)
    (kpsB, featuresB) = ROOTSIFT(gray_shark_img_2, kpsB)

    for num in range(len(kpsA)):
        kp1_size = kpsB[num]
        kp1_sort.append(kp1_size)
        
    for num in range(len(kpsB)):
        kp2_size = kpsB[num]
        kp2_sort.append(kp2_size)

    kp1_sorted = sorted(kp1_sort, key=lambda kp1_sort: kp1_sort.size , reverse=True)[:200]
    kp2_sorted = sorted(kp2_sort, key=lambda kp2_sort: kp2_sort.size , reverse=True)[:200]


    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        #print ("#1:{} , #2:{}".format(m[0].distance, m[1].distance))
        if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    matches_sorted = sorted(matches,reverse=True)[:200]
    (hA, wA) = gray_shark_img_1.shape[:2]
    (hB, wB) = gray_shark_img_2.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB), dtype="uint8")
    vis[0:hA, 0:wA] = gray_shark_img_1
    vis[0:hB, wA:] = gray_shark_img_2
    for (trainIdx, queryIdx) in matches_sorted:
        color = (0, 255, 0)
        ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
        ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
        #print(type(ptA),type(ptB))
        img_out = cv2.line(vis, ptA, ptB, color, 1)

    return img_out, matches, kpsA, kpsB

def Show_Matches_Img ():

    
    Matched_Keypoints()

    cv2.namedWindow('img', 0)
    cv2.resizeWindow('img', 1400, 600)
    cv2.imshow('img', img_out)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def Warp_image():

    matcher = cv2.DescriptorMatcher_create("BruteForce")
    detector = cv2.SIFT_create()
    kpsA = detector.detect(gray_shark_img_1)
    kpsB = detector.detect(gray_shark_img_2)

    (kpsA, featuresA) = ROOTSIFT(gray_shark_img_1, kpsA)
    (kpsB, featuresB) = ROOTSIFT(gray_shark_img_2, kpsB)
    
    matches = []
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    for m in rawMatches:
        #print ("#1:{} , #2:{}".format(m[0].distance, m[1].distance))
        if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    matches_sorted = sorted(matches,reverse=True)[:200]

    image_1_points = np.zeros((len(matches_sorted), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches_sorted), 1, 2), dtype=np.float32)
 
    for i in range(0,len(matches_sorted)):
        #print(matches_sorted[i])
        image_1_points[i] = kpsA[matches_sorted[i][1]].pt
        image_2_points[i] = kpsB[matches_sorted[i][0]].pt
    outimages = []
    homography, mask = cv2.findHomography(image_2_points, image_1_points, cv2.RANSAC , ransacReprojThreshold=8, maxIters= 1500, confidence=0.9)
    newimage = cv2.warpPerspective(gray_shark_img_2, homography, (2*gray_shark_img_2.shape[1], gray_shark_img_2.shape[0]), flags=cv2.INTER_LINEAR)

    outimages.append(newimage)

    cv2.namedWindow('img', 0)
    cv2.resizeWindow('img', 1400, 600)
    cv2.imshow('img', newimage)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
 
    return homography

if __name__ == '__main__':
    #print()
    Detect_Keypoints()
    #Show_Keypoints()
    #Matched_Keypoints()
    #Show_Matches_Img()
    Warp_image()
    