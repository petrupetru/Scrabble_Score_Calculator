import numpy as np
import cv2 as cv

path_train = 'antrenare'
path_aux = 'imagini_auxiliare'
path_test = 'testare'
path_eval = 'evaluare'
path_debug = 'putine'

width = 2334
height = 2490
bias_up = 182
bias_left = 198
cell_width = 130
cell_height = 142
num_cells = 15

letters = cv.imread(path_aux + '/litere_2.jpg')

def imageProcessing(image):
    original_image = image
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image[:, :, 2] #doar un canal de culoare merge mai bine decat grayscale
    #image = cv.GaussianBlur(image, (3, 3), 0) #blur
    _, image = cv.threshold(image, 120, 255, cv.THRESH_BINARY) #thresholding
    kernel = np.ones((3, 3), np.uint8)
    image = cv.erode(image, kernel, iterations=6) #erode
    image = cv.dilate(image, kernel, iterations=6) #dilate

    #dst = cv.Laplacian(image, cv.CV_16S, ksize=7) #edges
    #image = cv.convertScaleAbs(dst)

    return original_image, image

def getBoard(original_image, image, index_debug):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    image_copy = original_image.copy()
    cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)
    #image_copy = cv.resize(image_copy,(0,0), fx=0.2, fy=0.2)

    #map the board to a square using the corners
    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    image = cv.warpPerspective(original_image, M, (width, height))
    #cv.imshow('img', image)
    #cv.waitKey()
    #cv.imwrite('board.jpg', image)
    return image

def findBoardCells(board):
    cells = []
    lines_horizontal = []
    for i in range(bias_up + 2 * cell_height, (bias_up + (num_cells - 2) * cell_height + 1), cell_height):
        l = []
        l.append((bias_up, i))
        l.append(((bias_up + num_cells * cell_height - 1), i))
        lines_horizontal.append(l)

    lines_vertical = []
    for i in range(bias_left + 2 * cell_width, (bias_left + (num_cells - 2) * cell_width + 1), cell_width):
        l = []
        l.append((i, bias_left))
        l.append((i, (bias_left + num_cells * cell_width - 1)))
        lines_vertical.append(l)


    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            if i % 2 == 1 and j % 2 == 1:
                cv.imwrite(f'letter_templates/{i}_{j}.jpg', board[x_min:x_max, y_min:y_max, :])
    return cells

original, img = imageProcessing(letters)
img = getBoard(original, img, 0)
cells = findBoardCells(img)
cv.imshow('img', img)
cv.waitKey()

