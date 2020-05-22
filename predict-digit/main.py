#!/usr/bin/env python

if __name__ == "__main__":
    import sys
    import core.model as model

    TRAIN_DATA_PATH = 'tmp/Fnt/'
    mode = sys.argv[1]
    if mode == "train":
        EPOCH = int(sys.argv[2])
        BATCH_SIZE = int(sys.argv[3])
        MODEL_CHECK_POINT = 'res/model_checkpoint/LetterCNN.ckpt'
        all_x, all_y = model.import_data(TRAIN_DATA_PATH)
        x_train, x_test, y_train, y_test = model.split_data(all_x, all_y)
        model.train_model(MODEL_CHECK_POINT, EPOCH, x_train, y_train, BATCH_SIZE)

    elif mode == "test":
        import model

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
