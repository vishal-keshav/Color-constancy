# Author: bulletcross@gmail.com (Vishal Keshav)
# Module to construct CNN network
import tensorflow as tf
import numpy as np

def pointconv(input, nr_filters, stride, name):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [1, 1, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        pointconv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = 'SAME', name = name)
        bias = tf.reshape(tf.nn.bias_add(pointconv, biases), pointconv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def conv(input, filter_size, nr_filters, stride, name, padding = 'SAME', dilation = 1):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        conv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = padding, name = name)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

# Reference: https://arxiv.org/pdf/1508.00998.pdf

def cnn_architecture(input):
    layer1 = pointconv(input, nr_filters = 240, stride = 1, name = 'point1_1')

    pool1 = tf.nn.max_pool(layer1, ksize=[1,8,8,1], strides = [1,8,8,1],
                    padding = 'VALID', name = 'pool1')
    flat = tf.reshape(pool1, [-1, 3840])
    dense = tf.layers.dense(inputs=flat, units=40, activation=tf.nn.relu)
    out = tf.layers.dense(inputs = dense, units=3, activation=None)
    return out
