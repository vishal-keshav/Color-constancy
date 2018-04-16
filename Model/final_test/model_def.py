import tensorflow as tf
import numpy as np

def conv(input, filter_size, nr_filters, stride, name, padding = 'SAME', dilation = 1):
    input_channels = int(input.get_shape())[-1]
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        conv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = padding, name = name)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def depthconv(input, filter_size, stride, name, padding = 'SAME', dilation = 1, multiplier = 1):
    input_channels = int(input.get_shape())[-1]
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, multiplier])
        biases = tf.get_variable(name = name + '_biases', shape = [input_channels])
        depthconv = tf.nn.depthwise_conv2d(input, weights, strides = [1, stride, stride, 1],
                                      padding = padding, rate = dilation, name = name)
        bias = tf.reshape(tf.nn.bias_add(depthconv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def pointconv(input, nr_filters, stride, name):
    input_channels = int(input.get_shape())[-1]
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [1, 1, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        pointconv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = padding, name = name)
        bias = tf.reshape(tf.nn.bias_add(pointconv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def channel_weighted_pooling(weights, channel):
    nr_channel = channel.get_shape().as_list()[-1]
    pool_weights = tf.split(weights, num_or_size_splits = nr_channel, axis = 3)
    channel_outputs = tf.split(channel, num_or_size_splits = nr_channel, axis = 3)
    prod = []
    for i in range(nr_channel):
        prod.append(tf.multiply(pool_weights[i], channel_outputs[i]))
    output = tf.concat(prod, axis = 3)
    return output

def intermediate_residual(depth_in, point_in, name):
    nr_channels = int(depth_in.get_shape())[-1]
    depthconv_inter = depthconv(depth_in, filter_size = 3, stride = 2,
                                    padding = 'SAME', name = name+'_depth1',
                                    dilation = 1, multiplier = 2)
    pointconv_inter = pointconv(point_in, nr_channels*2, stride = 2,
                                    name = name+'_point1')
    tensor_inter = tf.concat([depth_in, point_in], axis = 3)
    conv_inter = conv(tensor_inter, filter_size = 3, nr_filters = 2*nr_channels,
                        stride = 2, name = name+'_conv', padding = 'SAME')
    depth_out_tensor = tf.concat([depthconv_inter, conv_inter], axis = 3)
    depthconv_out = depthconv(depth_out_tensor, filter_size = 3, stride = 2,
                                    padding = 'SAME', name = name+'_depth2',
                                    dilation = 1, multiplier = 1)
    point_out_tensor = tf.concat([pointconv_inter, conv_inter], axis = 3)
    pointconv_out = pointconv(point_out_tensor, nr_channels*2*2, stride = 2,
                                    name = name+'_point2')
    return depthconv_out, pointconv_out



def test_architecture(input):
    depthconv1_1 = depthconv(input, filter_size = 5, stride = 1,
                    padding = 'SAME', name = 'depth1_1', dilation = 2)
    pointconv1_2 = pointconv(input, nr_filters = 3, stride = 1, name = 'point1_1')

    pool1_1 = tf.nn.max_pool(depthconv1_1, ksize=[1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool1_1')
    pool1_2 = tf.nn.max_pool(pointconv1_2, ksize=[1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool1_2')

    depthconv3_1, pointconv3_2 = intermediate_residual(pool1_1, pool1_2, 'inter_res1')

    pool2_1 = tf.nn.max_pool(depthconv3_1, ksize=[1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool2_1')
    pool2_2 = tf.nn.max_pool(pointconv1_1, ksize=[1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool2_2')

    depthconv5_1, pointconv5_2 = intermediate_residual(pool2_1, pool2_2, 'inter_res2')

    ch_pool = channel_weighted_pooling(depthconv5_1, pointconv5_2)
    avg_pool = tf.reduce_mean(ch_pool, [1,2])
    fcn = pointconv(avg_pool, nr_filters = 3, stride = 1, name = 'fcn')
    normalized_out = tf.nn.l2_normalization(fcn, dim=1)
    return normalized_out
