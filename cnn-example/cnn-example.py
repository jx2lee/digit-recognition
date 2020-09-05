#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import time, os
from tensorflow.examples.tutorials.mnist import input_data

# except for WARNING LOG
old_version = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

mnist = input_data.read_data_sets('./cnn-example/mnist_data', one_hot=True)

LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIT_STEP = 10
DROPOUT = 0.75
N_EPOCH = 10
N_CLASSES = 10

tf.compat.v1.reset_default_graph()

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
dropout = tf.placeholder(tf.float32, name='dropout')

with tf.variable_scope('data'):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='X_placeholder')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y_placeholder')

with tf.variable_scope('conv1') as scope:
    images = tf.reshape(X, shape=[-1, 28, 28, 1])  # None의 num of input filter 크기가 1(=gray scale)인 28x28을 input으로 사용
    kernel = tf.get_variable('kernel', [5, 5, 1, 32],  # 5x5 receptive field, 1개의 filter의 크기를 32로 stack activation map
                             initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [32], initializer=tf.truncated_normal_initializer())
    conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1],
                        padding='SAME')  # [0]-[3] 은 보통 1, [1]-[2]는 fillter의 shape이라고 생각
    conv1 = tf.nn.relu(conv + biases, name=scope.name)

with tf.variable_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable('kernels', [5, 5, 32, 64],
                             initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [64], initializer=tf.truncated_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv + biases, name=scope.name)

with tf.variable_scope('pool2') as scope:
    pool2 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

with tf.variable_scope('fc') as scope:
    input_features = 7 * 7 * 64
    w = tf.get_variable('weights', [input_features, 1024], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [1024], initializer=tf.truncated_normal_initializer())
    pool2 = tf.reshape(pool2, [-1, input_features])
    fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

with tf.variable_scope('softmax_linear') as scope:
    w = tf.get_variable('weights', [1024, N_CLASSES], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES], initializer=tf.truncated_normal_initializer())
    logits = tf.matmul(fc, w) + b

with tf.variable_scope('loss') as scope:
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(entropy, name='loss')

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
is_correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.name_scope('summary'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('histogram_loss', loss)

    summary_op = tf.summary.merge_all()

'''
###### variable_scope vs name_scope
    - 차이점?
        - name_scope : creates namespace for operation
        - variable_scope : creates names for both variables and operation
        즉 name_scope은 get_variable를 무시한다고 생각하면 됨
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('cnn-example/mnist_graph', sess.graph)

    if not os.path.isdir(os.getcwd() + '/cnn-example/checkpoints'):
        os.mkdir(os.getcwd() + '/cnn-example/checkpoints')

    ckpt = tf.train.get_checkpoint_state('cnn-example/checkpoints')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()
    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)
    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCH):
        X_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch, summary = sess.run([optimizer, loss, summary_op],
                                          feed_dict={X: X_batch, y: y_batch, dropout: DROPOUT})
        writer.add_summary(summary, global_step=index)
        total_loss += loss_batch
        if (index + 1) % SKIT_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIT_STEP))
            total_loss = 0.0
            saver.save(sess, 'cnn-example/checkpoints/mnist-convnet', index)
    print("Optimization finished!!!")
    print('Total Time : {0} seconds'.format(time.time() - start_time))

    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, y_batch = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
                                               feed_dict={X: X_batch, y: y_batch, dropout: DROPOUT})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.arg_max(y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print('Accuracy : {0}'.format(total_correct_preds / mnist.test.num_examples))
    print(total_correct_preds)
    print(mnist.test.num_examples, n_batches, sess.run(global_step))