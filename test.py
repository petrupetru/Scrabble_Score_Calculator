import numpy as np
import cv2 as cv
import os

path_train = 'antrenare'
path_aux = 'imagini_auxiliare'
path_test = 'testare'
path_eval = 'evaluare'
path_debug = 'putine'
path_letters = 'letter_templates'
index_debug = 0

width = 2334
height = 2490
bias_up = 182
bias_left = 198
cell_width = 130
cell_height = 142
num_cells = 15


def getImages(path):
    images = []
    files = os.listdir(path)
    for file in files:
        if file[-3:] == 'jpg':
            img = cv.imread(path + '/' + file)
            images.append(img)
    return images

def imageProcessing(image):
    original_image = image
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image[:, :, 1] #doar un canal de culoare merge mai bine decat grayscale

    #image = cv.GaussianBlur(image, (3, 3), 0) #blur
    _, image = cv.threshold(image, 170, 255, cv.THRESH_BINARY) #thresholding
    kernel = np.ones((3, 3), np.uint8)
    #image = cv.erode(image, kernel, iterations=6) #erode
    #image = cv.dilate(image, kernel, iterations=6) #dilate

    #dst = cv.Laplacian(image, cv.CV_16S, ksize=7) #edges
    #image = cv.convertScaleAbs(dst)

    return original_image, image




if __name__ == '__main__':
    images = getImages(path_debug)
    for img in images:
        original, img = imageProcessing(img)
        img = cv.resize(img, (0, 0), fx=0.3, fy=0.3)
        cv.imshow('img', img)
        cv.waitKey()


