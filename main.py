import numpy as np
import cv2 as cv
import os
import imutils
from collections import defaultdict

path_train = 'antrenare'
path_test = 'evaluare/fake_test'
path_debug = 'putine'
path_letters = 'letter_templates'
path_result = 'evaluare/fisiere_solutie/352_Burdusa_Petru'
index_debug = 0

width = 2334
height = 2490
bias_up = 182
bias_left = 198
cell_width = 130
cell_height = 142
num_cells = 15
let = 'ABCDEFGHIJKLMNO' #row notations

# positions of multipliers
triple_word = [(0, 0), (0, 7), (0, 14),
               (7, 0), (7, 14),
               (14, 0), (14, 7), (14, 14)]
double_word = [(1, 1), (1, 13), (2, 2), (2, 12),
               (3, 3), (3, 11), (4, 4), (4, 10),
               (7, 7),
               (10, 4), (10, 10), (11, 3), (11, 11),
               (12, 2), (12, 12), (13, 1), (13, 13)]
triple_letter = [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
                 (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)]
double_letter = [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
                 (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11),
                 (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14),
                 (12, 6), (12, 8), (14, 3), (14, 11)]


score_letters = {'A': 1, 'B': 9, 'C': 1, 'D': 2, 'E': 1, 'F': 8, 'G': 9, 'H': 10, 'I': 1, 'J': 10, 'L': 1, 'M': 4,
                 'N': 1, 'O': 1, 'P': 2, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 8, 'X': 10, 'Z': 10, '?': 0}


def getImages(path):
    """get all jpg files from path"""
    images = []
    names = []
    files = os.listdir(path)
    for file in files:
        if file[-3:] == 'jpg':
            img = cv.imread(path + '/' + file)
            images.append(img)
            names.append(file[:-4])
    return images, names


def imageProcessing(image):
    """preprocess the image of the whole board so it is easier to detect the position of the board"""
    original_image = image
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image[:, :, 2]  # use only last color(Red)

    # image = cv.GaussianBlur(image, (3, 3), 0) # blur
    _, image = cv.threshold(image, 120, 255, cv.THRESH_BINARY) # thresholding (binary image, white board, black background)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.erode(image, kernel, iterations=6)  # erode so the small imperfections disappear
    image = cv.dilate(image, kernel, iterations=6)  # dilate to get the board to the original size

    # dst = cv.Laplacian(image, cv.CV_16S, ksize=7) #edges
    # image = cv.convertScaleAbs(dst)

    return original_image, image


def getBoard(original_image, image, index_debug):
    """crop the board and map it to standard size so all boards are same dimentions"""
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
    # image_copy = cv.resize(image_copy,(0,0), fx=0.2, fy=0.2)

    # map the board to a square using the corners
    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    image = cv.warpPerspective(original_image, M, (width, height))
    # cv.imshow('img', image)
    # cv.waitKey()
    # cv.imwrite('board.jpg', image)
    return image


def findBoardCells():
    """find coordinates of the board cells by splitting the board with horizontal and vertical lines"""
    cells_coordinates = []
    lines_horizontal = []
    for i in range(bias_up, (bias_up + num_cells * cell_height + 1), cell_height):
        l = []
        l.append((bias_up, i))
        l.append(((bias_up + num_cells * cell_height - 1), i))
        lines_horizontal.append(l)

    lines_vertical = []
    for i in range(bias_left, (bias_left + num_cells * cell_width + 1), cell_width):
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
            cells_coordinates.append((x_min, x_max, y_min, y_max))
            # if 0 == 0:
            #    cv.rectangle(board, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=2)
    return cells_coordinates


def find_letter(cell):
    """evaluate if a board cell has a letter or not"""
    cell = cell[:, :, 1]
    _, cell = cv.threshold(cell, 170, 255, cv.THRESH_BINARY)
    cell_value = np.mean(cell)
    if cell_value < 100:
        res = 0
    else:
        res = 1
    return res, cell_value


def classify_cell(cell, templates):
    """for a cell, make template matching with every letter with rotations, compare scores and select the best
    fitting letter """
    scores = defaultdict(int)
    cell = cell[:, :, 1]
    _, cell = cv.threshold(cell, 170, 255, cv.THRESH_BINARY)
    for (letter, noation) in templates:
        # cv.imshow('img', cell)
        # cv.waitKey(0)
        # cv.imshow('img', letter)
        # cv.waitKey(0)
        # letter = letter[:, :, 1]                                     move this line in getTemplates so it gets executed just once per letter
        # _, letter = cv.threshold(letter, 170, 255, cv.THRESH_BINARY) move this line in getTemplates so it gets executed just once per letter
        res = cv.matchTemplate(cell, letter, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if max_val > scores[noation]:
            scores[noation] = max_val
        # scores[noation]+=max_val
    result = max(scores, key=scores.get)
    if result == 'Jo':
        result = '?'

    return result


def getTemplates():
    """for every letter image make a set of rotated templates
        for templates I subtracted blue color and applied a threshold"""
    templates = []
    letter_templates, letter_notations = getImages(path_letters)
    for (letter_template, notation) in zip(letter_templates, letter_notations):
        for degree in range(-4, 5):
            letter_rotated = imutils.rotate(letter_template, degree)
            letter_rotated = letter_rotated[:, :, 1]
            _, letter_rotated = cv.threshold(letter_rotated, 170, 255, cv.THRESH_BINARY)

            templates.append((letter_rotated, notation))

            # cv.imshow('img', letter_rotated)
            # cv.waitKey(0)
    return templates

def score(board, letters_added):
    """first got all new formed words with the letters added at that round
    then calculated the score for every word and made sum"""
    score = 0
    words = []
    if len(letters_added) == 1:
        (i, j) = letters_added[0]
        x, y = i, j
        #caut prima litera
        while x > 0 and board[x-1][y] != '@':
            x -= 1
        word = [(x, y)]
        #merg spre ultima litera si construiesc cuvantul
        while x < 14 and board[x+1][y] != '@':
            x += 1
            word.append((x, y))
        if len(word) > 1:
            words.append(word)

        x, y = i, j
        while y > 0 and board[x][y-1] != '@':
            y -= 1
        word = [(x, y)]
        while y < 14 and board[x][y+1] != '@':
            y += 1
            word.append((x, y))
        if len(word) > 1:
            words.append(word)
    if len(letters_added) > 1:
        (i0, _) = letters_added[0]
        (i1, _) = letters_added[1]
        if i0 == i1:
            vertical_word = False
        else:
            vertical_word = True
        if vertical_word:
            (x, y) = letters_added[0]
            while x > 0 and board[x - 1][y] != '@':
                x -= 1
            main_word = [(x, y)]
            while x < 14 and board[x + 1][y] != '@':
                x += 1
                main_word.append((x, y))
            if len(main_word) > 1:
                words.append(main_word)
            for (x, y) in letters_added:
                while y > 0 and board[x][y - 1] != '@':
                    y -= 1
                aux_word = [(x, y)]
                while y < 14 and board[x][y + 1] != '@':
                    y += 1
                    aux_word.append((x, y))
                if len(aux_word) > 1:
                    words.append(aux_word)
        else: #horizontal main word
            (x, y) = letters_added[0]
            while y > 0 and board[x][y-1] != '@':
                y -= 1
            main_word = [(x, y)]
            while y < 14 and board[x][y+1] != '@':
                y += 1
                main_word.append((x, y))
            if len(main_word) > 1:
                words.append(main_word)
            for (x, y) in letters_added:
                while x > 0 and board[x - 1][y] != '@':
                    x -= 1
                aux_word = [(x, y)]
                while x < 14 and board[x + 1][y] != '@':
                    x += 1
                    aux_word.append((x, y))
                if len(aux_word) > 1:
                    words.append(aux_word)

    for word in words:
        score_word = 0
        for letter in word:
            (i, j) = letter
            score_letter = score_letters[board[i][j]]
            if letter in letters_added:
                if letter in double_letter:
                    score_letter *= 2
                if letter in triple_letter:
                    score_letter *= 3
            score_word += score_letter

        for letter in word:
            if letter in letters_added:
                if letter in double_word:
                    score_word *= 2
                if letter in triple_word:
                    score_word *= 3
        score += score_word

    return score


if __name__ == '__main__':

    templates = getTemplates()
    images, names = getImages(path_test)
    cells_coordinates = findBoardCells()
    letters_list = []
    board = [['@' for _ in range(15)] for _ in range(15)] #empty board
    round_no = 1
    for (img, name) in zip(images, names):
        letters_added = []
        print(name)
        f = open(f'{path_result}/{name}.txt', 'w')
        original, img = imageProcessing(img)
        index_debug += 1
        img = getBoard(original, img, index_debug)
        i = 0
        j = 0
        for cell_coordinates in cells_coordinates:
            (xmin, xmax, ymin, ymax) = cell_coordinates
            cell = img[xmin:xmax, ymin:ymax, :]
            res, value = find_letter(cell)
            if res != 0 and ((i, j) not in letters_list):
                letters_added.append((i, j))
                letters_list.append((i, j))
                res = classify_cell(cell, templates)
                board[i][j] = res
                f.write(f'{i + 1}{let[j]} {res}\n')
            j += 1
            if j > 14:
                i += 1
                j = 0

        score_round = score(board, letters_added)
        f.write(f'{score_round}\n')
        f.close()

        round_no += 1
        if round_no > 20:
            round_no = 1
            board = [['@' for _ in range(15)] for _ in range(15)]
            letters_list = []
