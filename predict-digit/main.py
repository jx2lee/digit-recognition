#!/usr/bin/env python
if __name__ == "__main__":
    from core.common import *
    from core.model import *
    import os
    import sys
    import tensorflow

    TRAIN_DATA_PATH = 'tmp/Fnt/'

    try:
        mode = sys.argv[1]
    except:
        print_error('Enter the Mode train or test')
        print(' [USAGE] ./main.py train {epoch} {batch_size}')
        print(' [USAGE] ./main.py test')
        sys.exit(1)
    if mode == "train":
        try:
            EPOCH = int(sys.argv[2])
            BATCH_SIZE = int(sys.argv[3])
        except:
            print_error('Enter the epoch and batch size')
            print(' [USAGE] ./main.py train {epoch} {batch_size}')
            sys.exit(1)
        MODEL_CHECK_POINT = 'res/model_checkpoint/LetterCNN.ckpt'
        all_x, all_y = import_data(TRAIN_DATA_PATH)
        x_train, x_test, y_train, y_test = split_data(all_x, all_y)
        train_model(MODEL_CHECK_POINT, EPOCH, x_train, y_train, BATCH_SIZE)
    elif mode == "test":
        INPUT_FOLDER_NAME = 'sample2'
        prediction(INPUT_FOLDER_NAME)
    else:
        print_error('You enter the Mode train test!')

'''
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import string
import os
import re
from IPython.display import Image
from random import sample
from sklearn import cross_validation
import model


# -------------------------------------------------------------------------
# this code This is the code you use when you're on a Mac. If it is not a Mac, import matplotlib.pyplot as plt is possible with 1 line.
import matplotlib
matplotlib.use('TkAgg')
# -------------------------------------------------------------------------
from matplotlib import pyplot as plt



# ALP
ALP = alphabet()

# restore and predict


#sample = 'sample1'
sample = 'sample2'
#sample = 'sample3'


PWD = "./result/box/{}/".format(sample)

SAMPLE = []
for _, _, files in os.walk(PWD):
    for i in range(len(files)):
            SAMPLE.append(files[i])


X, _, Y_proba, _, _, saver, _ = defineModel()
checkpoint_path = "./result/checkpoint/Saved/LetterCNN.ckpt"

# restore and run
with tf.Session() as sess:

    saver.restore(sess, checkpoint_path)
    img_SAMPLE = []

    for i in range(len(SAMPLE)):
        img_SAMPLE.append(ImgtoFloat(PWD+SAMPLE[i]))
        BATCH = []
        BATCH.append(img_SAMPLE[i])

        test = Y_proba.eval(feed_dict={X: BATCH})
        idx = np.argmax(test)

        print('predict : ', ALP[idx], 'real -> show image')
        cv2.imshow('test', ImgtoFloat(PWD+SAMPLE[i]))
        cv2.waitKey(0)

    cv2.destroyAllWindows()
'''
