import cv2
import numpy as np
from pathlib import Path
import glob

from numpy.core.fromnumeric import resize

image_path = ("homework_1/Q1.Camera_Calibration/Q1_Image/")
chessboardSize = (11,8)

def CameraMatrix(image):
    
    global ret, intrinsic_matrix, dist, rvecs, tvecs, img, corners, dst

    
    frameSize = (2048,2048)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    img = cv2.imread(image)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        objpoints = np.array(objpoints)
        imgpoints.append(corners)
        imgpoints = np.array(imgpoints)

    
    ret, intrinsic_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    return ret, intrinsic_matrix, dist, rvecs, tvecs, img, corners
    
def Find_Corner():
    for image_num in range(1,16,1):
        images = glob.glob("homework_1/Q1.Camera_Calibration/Q1_Image/{}.bmp".format(image_num))
        CameraMatrix(images[0])
        cv2.drawChessboardCorners(img,patternSize=(chessboardSize), corners=(corners),patternWasFound=(True)) 
        cv2.namedWindow('img', 0)
        cv2.resizeWindow('img', 1200, 800)
        cv2.imshow('img', img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

def Find_The_Intrinsic_Matrix():

    for image_num in range(1,16,1):
        images = glob.glob("homework_1/Q1.Camera_Calibration/Q1_Image/{}.bmp".format(image_num))

        CameraMatrix(images[0])
        print('########################################')
        print('The Intrinsic Matrix of Image"{}" is:'.format(image_num))
        print(intrinsic_matrix)
        print('########################################')


def Find_The_Extrinsic_Matrix(select_image):
    images = glob.glob("homework_1/Q1.Camera_Calibration/Q1_Image/{}.bmp".format(select_image))
    #print('The Extrinsic Matrix of Image"{}" is:'.format(select_image))
    CameraMatrix(images[0])
    Extrinsic_Matrix = cv2.Rodrigues(np.asarray(rvecs))
    print('########################################')
    print('The Extrinsic Matrix of Image.{} is:'.format(select_image))
    print(Extrinsic_Matrix[0],tvecs)
    print('########################################')

def Find_The_Distortion_Matrix(image_path):

    for image_num in range(1,16,1):
        images = glob.glob("homework_1/Q1.Camera_Calibration/Q1_Image/{}.bmp".format(image_num))
        print('The Intrinsic Matrix of Image"{}" is:'.format(image_num))
        CameraMatrix(images[0])
        print('########################################')
        print(dist)
        print('########################################')

def Show_The_undistorted_result():
    for image_num in range(1,16,1):
        images = glob.glob("homework_1/Q1.Camera_Calibration/Q1_Image/{}.bmp".format(image_num))
        CameraMatrix(images[0])
        h,  w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, dist, (w,h), 1, (w,h))
        # Undistort
        dst = cv2.undistort(img, intrinsic_matrix, dist, None, newCameraMatrix)

        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        Undistort = cv2.resize(dst,(2048,2048))
        distort = img
        merge = np.concatenate((distort,Undistort),axis=1)
        
        cv2.namedWindow('img', 0)
        cv2.resizeWindow('img', 1200, 800)
        cv2.imshow('img', merge)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print()

    #1.1
    Find_Corner()

    #1.2 
    Find_The_Intrinsic_Matrix()

    #1.3
    which_one_slect = input() # remeber to edit when you finished to bulit a GUI
    Find_The_Extrinsic_Matrix(which_one_slect)

    #1.4
    Find_The_Distortion_Matrix(image_path)

    #1.5
    Show_The_undistorted_result()

