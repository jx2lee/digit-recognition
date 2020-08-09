#!/usr/bin/env python
from core.common import *

PATH = "tmp/"
RESULT_PATH = "res/"
RES_NAME = "blackboxed.jpg"


def show_image(img_path):
    """
    Return Image
    :param img_path: Image Path you want to show
    :return: no value, show image
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import platform
    if platform.system() == 'Darwin':
        matplotlib.use("TkAgg")

    dpi = 200  # control parameter
    im_data = plt.imread(img_path)
    _channel = im_data.shape
    fig_size = _channel[0] / float(dpi), _channel[1] / float(dpi)

    plt.figure(figsize=fig_size)
    plt.xticks([]), plt.yticks([])
    plt.imshow(im_data)
    plt.show()


def black_box(input_path):
    """
    Using Black-Box Algorithm (==Double Contours)
    :param input_path: image path
    :return: cv2 object(list) by black-box algorithm
    """
    from core.common import make_logger
    import cv2

    logger = make_logger()
    raw_image = cv2.imread(input_path)
    first_img = raw_image.copy()

    img_to_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_to_gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # first-contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        channel_shape = first_img.shape
        if w < channel_shape[0] * 0.05:
            cv2.rectangle(first_img, (x, y), (x + w + 10, y + h), (0, 0, 0), -1)

    last_img = raw_image.copy()
    img_to_gray_ = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    cv2.Canny(img_to_gray_, 50, 200, apertureSize=3)
    blur_ = cv2.blur(img_to_gray_, (5, 5))
    cv2.threshold(blur_, 127, 255, 0)
    contours, _ = cv2.findContours(blur_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # second-contours
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        channel = last_img.shape
        if (w / h > 8) & (w / h < 15):
            cv2.rectangle(last_img, (x, y), (x + w + 10, y + h), (0, 255, 0), 3)
            cv2.putText(last_img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print("[CONTOUR DETAILS]\t", 'x: ', x, "\t", 'y: ', y, "\t", 'w: ', w, "\t", 'h: ', h, '\tchannel: ', channel, '|', w / channel[0], '|', h / channel[1], '|', w / h)
        '''
        if idx == 529:
            cv2.imwrite(SAVE_PATH + "box/" + input_path[-11:-4] + ".png", raw_image[y : y + h, x : x + w])
        '''

    cv2.imwrite(RESULT_PATH + input_path[-11:-4] + '_' + RES_NAME, last_img)
    #print_info('Saving image finished')
    logger.info("Saving image Finished")

    return last_img


if __name__ == '__main__':
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description='Blackbox Tutorial')
    parser.add_argument('file_name', type=str, 
            help='Blackbox 알고리즘을 테스트 할 사진파일 이름을 넣으세요(example: python blackbox.py sample1.png)')
    args = parser.parse_args()
    black_box(PATH + args.file_name)
