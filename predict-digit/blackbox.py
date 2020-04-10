#!/Users/jj/.virtualenvs/deep/bin/python

PATH = "./tmp/"
RESULT_PATH = "./res/"
RES_NAME = "blackboxed.jpg"


def show_image(img_path):
    """
    Return Image
    :param img_path: Image Path you want to show
    :return: no value, show image
    """
    from matplotlib import pyplot as plt

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
    import cv2
    raw_image = cv2.imread(input_path)
    first_img = raw_image.copy()

    img_to_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_to_gray, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    _, contours, _ = cv2.findContours(blur_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # second-contours
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        channel = last_img.shape
        if (w / h > 8) & (w / h < 15):
            cv2.rectangle(last_img, (x, y), (x + w + 10, y + h), (0, 255, 0), 3)
            cv2.putText(last_img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print("contour details\t", 'x : ', x, 'y : ', y, 'w : ', w, 'h : ', h, channel, w / channel[0]
                  , h / channel[1], w / h)
        '''
        if idx == 529:
            cv2.imwrite(SAVE_PATH + "box/" + input_path[-11:-4] + ".png", raw_image[y : y + h, x : x + w])
        '''

    cv2.imwrite(RESULT_PATH + input_path[-11:-4] + '_' + RES_NAME, last_img)
    print('Saving image finished!! ')

    return last_img


if __name__ == '__main__':
    import matplotlib
    import sys

    matplotlib.use('TkAgg')  # TkAgg line is for Mac.

    file_name = sys.argv[1]
    black_box_return = black_box(PATH + file_name)
    show_image(RESULT_PATH + file_name[:-4] + '_' + RES_NAME)
