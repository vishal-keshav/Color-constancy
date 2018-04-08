# Author: bulletcross@gmail.com (Vishal Keshav)
# Module to construct FC4 network with novel weighted pooling
import tensorflow as tf
import numpy as np


im_mean = [104., 117., 124.] # BGR mean

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

def weighted_pooling(input):
    assert (input.get_shape().as_list()[-1] == 4)
    c, r, g, b = tf.split(input, num_or_size_splits = 4, axis = 3)
    r_weighted = tf.multiply(r,c)
    g_weighted = tf.multiply(g,c)
    b_weighted = tf.multiply(b,c)

    assert (b_weighted.get_shape().as_list()[-1] == 1)
    assert (b_weighted.get_shape().as_list()[-2] == input.get_shape().as_list()[-2])
    assert (b_weighted.get_shape().as_list()[-3] == input.get_shape().as_list()[-3])

    output = tf.concat([r_weighted, g_weighted, b_weighted], axis = 3)
    #print(output.get_shape().as_list())
    return output

def fc4_architecture(input, prob):
    # Process the RGB data to BGR
    r,g,b = tf.split(input, 3, 3)
    bgr = tf.concat([b-im_mean[0], g-im_mean[1], r-im_mean[2]], 3)

    conv1 = conv(bgr, filter_size = 11, nr_filters = 96, stride = 4,
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

    conv6 = conv(pool5, filter_size = 6, nr_filters = 64, stride = 1,
                    groups=1, padding = 'SAME', name = 'conv6')
    conv6_drop = tf.nn.dropout(conv6, keep_prob = prob)

    conv7 = conv(conv6_drop, filter_size = 1, nr_filters = 4, stride = 1,
                    groups=1, padding = 'SAME', name = 'conv7')

    weighted_pool7 = weighted_pooling(conv7)
    summation = tf.reduce_sum(weighted_pool7, [1,2])
    #print(summation.get_shape().as_list())
    normalization = tf.nn.l2_normalize(summation, dim=1)
    assert (normalization.get_shape().as_list()[-1] == 3)
    return normalization
