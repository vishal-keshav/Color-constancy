# Testing module for Alexnet weight transfer

import os
import cv2
import numpy as np
import tensorflow as tf
from imagenet_classes import class_names

def conv(x, filter_size, nr_filters, stride, name, groups=1, padding = 'SAME'):
  input_channels = int(x.get_shape()[-1])
  convolve = lambda i, k: tf.nn.conv2d(i, k,
                                       strides = [1, stride, stride, 1],
                                       padding = padding)
  with tf.variable_scope(name) as scope:
    weights = tf.get_variable(name = 'weights', shape = [filter_size, filter_size, input_channels/groups, nr_filters])
    biases = tf.get_variable(name = 'biases', shape = [nr_filters])
    if groups == 1:
      conv = convolve(x, weights)
    else:
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      conv = tf.concat(axis = 3, values = output_groups)
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    relu = tf.nn.relu(bias, name = scope.name)
    return relu

def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    if relu == True:
      relu = tf.nn.relu(act)
      return relu
    else:
      return act


def test_alexnet(input, keep_prob):
    conv1 = conv(input, filter_size = 11, nr_filters = 96, stride = 4,
                    groups=1, padding = 'VALID', name = 'conv1')
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool1')
    norm1 = tf.nn.local_response_normalization(pool1, depth_radius = 2,
                    alpha = 2e-05, beta = 0.75, bias = 1.0, name = 'norm1')

    conv2 = conv(norm1, filter_size = 5, nr_filters = 256, stride = 1,
                    groups=2, padding = 'SAME', name = 'conv2')
    pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool2')
    norm2 = tf.nn.local_response_normalization(pool2, depth_radius = 2,
                    alpha = 2e-05, beta = 0.75, bias = 1.0, name = 'norm2')

    conv3 = conv(norm2, filter_size = 3, nr_filters = 384, stride = 1,
                    groups=1, padding = 'SAME', name = 'conv3')

    conv4 = conv(conv3, filter_size = 3, nr_filters = 384, stride = 1,
                    groups=2, padding = 'SAME', name = 'conv4')

    conv5 = conv(conv4, filter_size = 3, nr_filters = 256, stride = 1,
                    groups=2, padding = 'SAME', name = 'conv5')
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool5')

    flattened = tf.reshape(pool5, [-1, 6*6*256])
    fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
    dropout6 = tf.nn.dropout(fc6, keep_prob)

    fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
    dropout7 = tf.nn.dropout(fc7, keep_prob)

    fc8 = fc(dropout7, 4096, 1000, relu = False, name='fc8')
    return fc8

def test_transfer_weight(sess):
    weight_dict = np.load('bvlc_alexnet.npy', encoding = 'bytes').item()
    #First transfer the weights to model
    transfer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    for op in transfer_list:
        with tf.variable_scope(op, reuse = True):
            for data in weight_dict[op]:
                if len(data.shape) == 1:
                    v = tf.get_variable('biases', trainable = True)
                    sess.run(v.assign(data))
                else:
                    v = tf.get_variable('weights', trainable = True)
                    sess.run(v.assign(data))
    return

def main():
    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32) # BRG mean
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    prediction = tf.nn.softmax(test_alexnet(x,1.0))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # Initializing before weight assignment
        test_transfer_weight(sess)
        input_image = cv2.imread('test_image.jpeg') # Read in BRG
        input_image = cv2.resize(input_image.astype(np.float32), (227,227))
        input_image = input_image - imagenet_mean
        input_image.resize([1,227,227,3])
        out = sess.run(prediction, feed_dict = {x: input_image})
        class_name = class_names[np.argmax(out)]
        print(class_name)


if __name__ == "__main__":
    main()
