import cv2
import numpy as np
from pathlib import Path
import glob
from numpy.core.fromnumeric import resize


chessboardSize = (11,8)
image_path = ("homework_1/Q2.Augmented_Reality/Q2_Image")
words = ['C','V','D','L','H','K']
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def CameraMatrix(image):
    
    global ret, intrinsic_matrix, dist, rvecs, tvecs, img, corners, dst, Extrinsic_Matrix, imgpoints, objpoints, objp

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
        #imgpoints = np.array(imgpoints)

        ret, intrinsic_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    
        rvec,tvec=cv2.solvePnPRansac(objp,corners,intrinsic_matrix,dist)
        imgpts=cv2.projectPoints(axis,rvec, tvec,intrinsic_matrix,dist)

        img = draw(img,corners,imgpts)

        #print(corners)
        cv2.drawChessboardCorners(img,patternSize=(11,8), corners=(corners),patternWasFound=(True)) 
        cv2.namedWindow('img', 0)
        cv2.resizeWindow('img', 1200, 800)
        cv2.imshow('img', img)
        cv2.waitKey(500)
        
    Extrinsic_Matrix = cv2.Rodrigues(np.asarray(rvecs))
    #np.savez('camera_matrix',mtx=intrinsic_matrix,dist=dist)
    return ret, intrinsic_matrix, dist, rvecs, tvecs, img, corners, Extrinsic_Matrix, objpoints, imgpoints, objp

def read_yaml():
    yaml_file = ('homework_1/Q2.Augmented_Reality/Q2_Image/Q2_lib/alphabet_lib_onboard.yaml')
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    for letter in range(0,5,1):
        fn = fs.getNode("{}".format(words[letter]))
        print('The letter of word is :',words[letter])
        print()
        print(fn.mat())
        print()



def save_img_camera_matrix():
    for image_num in range(1,6,1):
        images = glob.glob('homework_1/Q2.Augmented_Reality/Q2_Image/{}.bmp'.format(image_num))
        CameraMatrix(images[0])
        np.savez('homework_1/Q2.Augmented_Reality/Q2_Image/camera_matrix_{}'.format(image_num),mtx=intrinsic_matrix,dist=dist)

def Show_Word_On_Board():
    for image_num in range(1,6,1):        
        images = glob.glob('homework_1/Q2.Augmented_Reality/Q2_Image/{}.bmp'.format(image_num))
        CameraMatrix(images[0])




if __name__ == '__main__':
    print()
    #read_yaml()
    #save_img_camera_matrix()
    #Show_Word_On_Board()

    for image_num in range(1,6,1):        
        images = glob.glob('homework_1/Q2.Augmented_Reality/Q2_Image/{}.bmp'.format(image_num))
        CameraMatrix(images[0])
    