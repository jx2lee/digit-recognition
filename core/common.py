#!/usr/bin/env python3
import cv2
import logging
import numpy as np


def print_info(string):
    print('\033[31m \033[43m' + '[INFO]'+ '\033[0m' + ' ' + string)


def print_error(string):
    print('\033[37m \033[101m' + '[ERROR]'+ '\033[0m' + ' ' + string)


def make_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    logger.addHandler(console)
    return logger


def resize_input(image):
    """
    Resize image
    :param image: image path for resize
    :return: res: resized image
    """
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, 2)
    res = np.expand_dims(img, 0)
    return res


def image_to_float(img_path):
    """
    Image to float function
    :param img_path: Path To float image file
    :return thresh: 2d array after gray-scale image
    """
    raw = cv2.imread(img_path)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, axis=2)

    return gray