import math
import numpy as np
from scipy.signal import convolve2d
import cv2

from MyConvolution import convolve 
from MyHybridImages import *
np.set_printoptions(threshold=np.inf)

def test_convolution():
    iterTest = 10
    for i in range(10):
        row, col = np.random.randint(10, 100), np.random.randint(10, 100)
        # testImage = np.random.rand(row, col, 3)
        testImage = np.random.rand(row, col)

        trow, tcol = np.random.choice([3, 5, 7]), np.random.choice([3, 5, 7])
        testKernel = np.random.randint(-5, 5, size=(trow, tcol))

        myconv = convolve(testImage, testKernel)

        # myconv = myconv[~np.all(myconv == 0, axis=(1, 2))]
        # myconv = myconv[:, ~np.all(myconv == 0, axis=(0, 2))]
        scipyconv = np.zeros(testImage.shape)
        scipyconv = convolve2d(testImage, testKernel, mode='same')
        # for j in range(3):
            # scipyconv[:, :, j] = convolve2d(testImage[:, :, j], testKernel, mode='same', boundary='fill', fillvalue=0)

        myconv = myconv.astype(np.float32)
        scipyconv = scipyconv.astype(np.float32)

        if (myconv == scipyconv).all():
            print(True)
        else:
            print(False)
    
def test_hybrid():
    image_cat = cv2.imread("E:\comp3204-cwk1_py\cat.bmp")
    image_dog = cv2.imread("E:\comp3204-cwk1_py\dog.bmp")
    image_cd = myHybridImages(image_dog, 6, image_cat, 14)
    
    image_1 = cv2.imread("E:\comp3204-cwk1_py\image1.jpg")
    image_2 = cv2.imread("E:\comp3204-cwk1_py\image2.jpg")
    image_cd = myHybridImages(image_2, 6, image_1, 15)
    # cv2.imshow('image window', image_cd)
    # cv2.waitKey(0)

    cv2.imwrite(r"e:/comp3204-cwk1_py/hybrid_image.jpg", image_cd)
    print("done")

test_hybrid()
# test_convolution()