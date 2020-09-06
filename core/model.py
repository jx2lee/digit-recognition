#!/usr/bin/env python3
from core.common import *
from sklearn.model_selection import train_test_split
import numpy as np
import os
import re
import string
import tensorflow as tf
import zipfile


def reset_graph(seed=42):
    """
    Reset Tensorflow graph function
    """
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)


def alphabet():
    """
    Get alphabet
    """
    char = []
    for i in range(10):
        char.append((i + 1) % 10)
    for a in string.ascii_uppercase[:27]:
        char.append(a)
    return char


def import_data(data_path):
    """
    Import data
    :param data_path: digit data path
    :return: X / y dataset
    """
    zip_path = os.getcwd() + '/tmp'
    if not os.path.isdir(zip_path + '/Fnt'):
        with zipfile.ZipFile(zip_path + '/Fnt.zip') as unzip:
            unzip.extractall(zip_path)
    pwd = data_path
    x, y = [], []

    for path, dirs, files in os.walk(pwd):
        for dr in dirs:
            if int(re.findall('\d+', dr)[0]) < 37:
                sub_pwd = os.path.join(path, dr)
                for sub_path, sub_dirs, sub_files in os.walk(sub_pwd):
                    for sub_file in sub_files:
                        sample = os.path.join(sub_pwd, sub_file)
                        x.append(image_to_float(sample))
                        y.append(int(re.findall("\d+", dr)[0]) - 1)
                    print(sub_pwd, "Done.")

    x = np.array(x)
    y = np.array(y)
    return x, y


def split_data(x, y):
    """
    Split train/test data
    :param x: input dataset
    :param y: output dataset
    :return: X train/test, y train/test data set
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def prepare_batch(x_train, y_train, batch_size):
    """
    Create batch dataset
    :param x_train: x-batch-data
    :param y_train: y-batch-data
    :param batch_size: size
    :return: numpy array - x_batch, y_batch
    """
    all_batch = np.random.randint(0, len(x_train), batch_size)
    x_batch = np.stack([x_train[idx] for idx in all_batch])
    y_batch = np.stack([y_train[idx] for idx in all_batch])

    return x_batch, y_batch


def define_model():
    """
    :model-description
    input shape : (None,128,128)
    1-conv layer : (5,5) kernel size, 8 feature maps, 1 stride, same padding
    2-conv layer : (3,3) kernel size, 4 feature maps, 2 stride, same padding
    pooling layer : (2, 2) kernel size, 4 feature maps, 2 stride, valid padding
    """

    height = 128
    width = 128
    channels = 1

    conv1_fmaps = 8
    conv1_ksize = 5
    conv1_stride = 1
    conv1_pad = "same"

    conv2_fmaps = 4
    conv2_ksize = 3
    conv2_stride = 2
    conv2_pad = "same"

    pool_fmaps = conv2_fmaps
    pool_stride = 2

    n_fc1 = 64
    n_outputs = 36

    reset_graph()

    with tf.name_scope("inputs"):
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
        y = tf.compat.v1.placeholder(tf.int32, shape=[None], name="y")

    conv1 = tf.keras.layers.Conv2D(filters=conv1_fmaps, kernel_size=conv1_ksize, \
                             strides=conv1_stride, padding=conv1_pad, \
                             activation=tf.nn.relu, name="conv1")(x)
    conv2 = tf.keras.layers.Conv2D(filters=conv2_fmaps, kernel_size=conv2_ksize, \
                             strides=conv2_stride, padding=conv2_pad, \
                             activation=tf.nn.relu, name="conv2")(conv1)

    with tf.name_scope("pool"):
        pool3 = tf.nn.max_pool2d(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        pool3_flat = tf.reshape(pool3, shape=[-1, pool_fmaps * int(height / conv1_stride / conv2_stride / pool_stride) * \
                                              int(width / conv1_stride / conv2_stride / pool_stride)])
    with tf.name_scope("fc1"):
        fc1 = tf.keras.layers.Dense(n_fc1, activation='relu')(pool3_flat)

    with tf.name_scope("output"):
        logits = tf.keras.layers.Dense(n_outputs)(fc1)
        y_prob = tf.nn.softmax(logits, name="Y_proba")

    with tf.name_scope("train"):
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(entropy)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.compat.v1.train.AdamOptimizer()
        training_op = optimizer.minimize(loss, global_step=global_step)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope("init_and_save"):
        init = tf.compat.v1.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()

    return x, y, y_prob, training_op, accuracy, saver, init


def train_model(checkpoint_path, epochs, x, y, batch_size):
    """
    training model
    :param checkpoint_path: model checkpoints
    :param epochs: epoch
    :param x: all input (include train/test)
    :param y: all output (include train/test)
    :param batch_size: train/test batch size
    """
    x_train, x_test, y_train, y_test = split_data(x, y)
    x, y, y_prob, training_op, accuracy, saver, init = define_model()
    train_logger = make_logger('train')

    with tf.compat.v1.Session() as sess:
        init.run()
        for epoch in range(epochs):
            train_logger.info(f'train data size: {len(x_train)},\titeration: {len(x_train) // batch_size}')
            for iteration in range(len(x_train) // batch_size):
                x_batch, y_batch = prepare_batch(x_train, y_train, batch_size)
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
                train_logger.info(f'epoch: {epoch + 1},\titeration: {iteration},\ttrain..')
            acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={x: x_test, y: y_test})
            train_logger.info(f'Train accuracy:, {acc_train}, Test accuracy:, {acc_test}')
            saver.save(sess, checkpoint_path)

    print("training finished..")


def prediction(input_folder_name):
    """
    predict VIN using trained model
    :param input_folder_name: folder name for prediction
    """
    path = "res/char/{}".format(input_folder_name)
    sample = []
    for _, _, files in os.walk(path):
        for file in files:
            sample.append(file)

    x, _, y_prob, _, _, saver, _ = define_model()
    checkpoint_path = 'res/model_checkpoint/LetterCNN.ckpt'
    y_list = alphabet()

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, checkpoint_path)
        for i in range(len(sample)):
            test_image = resize_input(path + '/' + sample[i])
            test = y_prob.eval(feed_dict={x: test_image})
            idx = np.argmax(test)
            print('[PREDICT]:', y_list[idx], '[TARGET]:', '%s/%s'%(path, sample[i]))