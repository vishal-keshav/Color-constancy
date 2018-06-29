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

def get_model(input):
    conv1 = pointconv(input, nr_filters = 240, stride = 1, name = 'point1')
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,8,8,1], strides = [1,8,8,1],
                    padding = 'VALID', name = 'pool1_1')

    conv2 = pointconv(input, nr_filters = 240, stride = 1, name = 'point2')
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,10,10,1], strides = [1,8,8,1],
                    padding = 'SAME', name = 'pool2_2')

    concat_layer = tf.concat([conv1_pool, conv2_pool], axis = 3)
    concat_pool = tf.nn.max_pool(concat_layer, ksize=[1,4,4,1], strides = [1,4,4,1],
                    padding = 'VALID', name = 'concat_pool')
    conv3 = pointconv(concat_pool, nr_filters = 80, stride = 1, name = 'point3')
    out = pointconv(conv3, nr_filters = 3, stride = 1, name = 'point4')
    out = tf.reshape(out, [-1, 3])
    return out
