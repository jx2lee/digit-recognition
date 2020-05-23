#!/usr/bin/env python

from core.common import *
from core.model import *

BOX_PATH = "tmp/box/"
SAVE_PATH = "res/"


def rotate_img(_path, _file):
    """
    :param _path: image path
           _file: image name
    :return rotated: rotated image - list
    """
    raw = cv2.imread(_path + _file)
    img = raw.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    col_stk = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(col_stk)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def resize_img(img):
    """
    :param img: resize image object (==list)
    :return thresh: list after resizing image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    ratio_w = int(128 / height * width)
    img_to_gray = cv2.resize(gray, (ratio_w, 128), interpolation=cv2.INTER_LINEAR)
    ret, thresh = cv2.threshold(img_to_gray, 127, 255, 0)

    return thresh


def remove_vertical_line(img):
    """
    :param img: To remove vertical line image
    :return img: list after removing vertical line
    """
    height, width = img.shape

    white_space = width * 255
    end_point = []
    start_point = []

    for i in range(height - 1):
        if (np.sum(img[i, :width]) == white_space) & (np.sum(img[i + 1, :width]) != white_space):
            start_point.append(i)

        if (np.sum(img[i, :width]) != white_space) & (np.sum(img[i + 1, :width]) == white_space):
            end_point.append(i)

    if np.sum(img[0, :width]) != white_space:
        start_point.insert(0, 0)
    if np.sum(img[-1, :width]) != white_space:
        end_point.append(width)

    location = list([a, b] for a, b in zip(start_point, end_point))
    del_location = list(b - a for a, b in zip(start_point, end_point))
    max_idx = np.argmax(del_location)
    del (del_location[max_idx])
    del (location[max_idx])

    if len(location) > 0:
        for st, en in location:
            img[st:en + 1, :] = 255

    return img


def level_img(img, _path, _files):
    """
    :param img: image for leveling
         _path: path for image
        _files: image name
    :return array after leveling image
    """
    thresh = img.copy()
    re_height, re_width = thresh.shape
    sample = []
    white_space = re_height * 255
    end_point = []
    start_point = []

    for i in range(re_width - 1):
        if (np.sum(thresh[:re_height, i]) == white_space) & (np.sum(thresh[:re_height, i + 1]) != white_space):
            start_point.append(i)
        if (np.sum(thresh[:re_height, i]) != white_space) & (np.sum(thresh[:re_height, i + 1]) == white_space):
            end_point.append(i)

    if np.sum(thresh[:re_height, 0]) != white_space:
        start_point.insert(0, 0)
    if np.sum(thresh[:re_height, -1]) != white_space:
        end_point.append(re_width)

    location = list([a, b] for a, b in zip(start_point, end_point))
    del_location = list([b - a] for a, b in zip(start_point, end_point))

    if np.max(del_location) > 128:
        return []

    while len(location) > 17:
        loc_len = []
        for i in range(len(location)):
            loc_len.append(location[i][1] - location[i][0])

        min_idx = np.argmin(loc_len)
        if min_idx != 0:
            len_merge_left = loc_len[min_idx] + loc_len[min_idx - 1]
        else:
            len_merge_left = 999999

        if min_idx != len(loc_len):
            len_merge_right = loc_len[min_idx] + loc_len[min_idx + 1]
        else:
            len_merge_right = 999999

        if len_merge_left < len_merge_right:
            location[min_idx - 1][1] = location[min_idx][1]
        else:
            location[min_idx + 1][0] = location[min_idx][0]

        del location[min_idx]

    for order in range(len(location)):
        start = location[order][0]
        end = location[order][1]
        front_len = int((128 - (end - start)) / 2)
        end_len = int((128 - (end - start)) / 2)
        if front_len < 0 and end_len < 0:
            front_len, end_len = -front_len, -end_len
        if front_len + end_len + end - start != 128:
            front_len = front_len + 1

        center_rec = thresh[0:128, start:end]
        left_rec = np.zeros((128, front_len), np.uint8)
        left_rec[:, :] = 255
        right_rec = np.zeros((128, end_len), np.uint8)
        right_rec[:, :] = 255
        left_center = np.hstack((left_rec, center_rec))
        full = np.hstack((left_center, right_rec))
        sample.append(full)

    return sample


if __name__ == '__main__':
    from core.common import *
    import cv2
    import errno
    import numpy as np
    import os

    FILES = []
    for _, dirs, files in os.walk(BOX_PATH):
        FILES = files

    for file in FILES:
        img_rotate = rotate_img(BOX_PATH, file)
        img_resize = resize_img(img_rotate)
        img_clean = remove_vertical_line(img_resize)
        cv2.imwrite(SAVE_PATH + 'box/' + file[:-4] + '.png', img_clean)
        prep_img = level_img(img_clean, BOX_PATH, file)

        try:
            if not (os.path.isdir(SAVE_PATH + 'char/' + file[:-4])):
                os.makedirs(os.path.join(SAVE_PATH + 'char/' + file[:-4]))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print_error('Failed to create directory, check your path')
                raise

        img_dir = SAVE_PATH + 'char/' + file[:-4] + '/'
        for i in range(len(prep_img)):
            cv2.imwrite(img_dir + file[:-4] + '_' + str(i) + '.png', prep_img[i])
        print_info("%s preprocess finished!"%(file))

