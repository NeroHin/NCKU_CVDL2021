import numpy as np
import cv2
from matplotlib import pyplot as plt


def Stereo_Disparity_Map():

    # Read Image
    imgL = cv2.imread('/Users/nerohin/NCKU_CVDL2021/homework_1/Q3.Stereo_Disparity_Map/Q3_Image/imL.png',0)
    imgR = cv2.imread('/Users/nerohin/NCKU_CVDL2021/homework_1/Q3.Stereo_Disparity_Map/Q3_Image/imR.png',0)

    # Stereo Disparity Map
    stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
    disparity = stereo.compute(imgL,imgR)

    # Show Imaeg
    plt.imshow(disparity,'gray')
    plt.axis("off")
    plt.show()


