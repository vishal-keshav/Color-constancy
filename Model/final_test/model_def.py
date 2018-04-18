import tensorflow as tf
import numpy as np

def conv(input, filter_size, nr_filters, stride, name, padding = 'SAME', dilation = 1):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        conv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = padding, name = name)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def depthconv(input, filter_size, stride, name, padding = 'SAME', dilation = 1, multiplier = 1):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, multiplier])
        biases = tf.get_variable(name = name + '_biases', shape = [input_channels*multiplier])
        depthconv = tf.nn.depthwise_conv2d(input, weights, strides = [1, stride, stride, 1],
                                      padding = padding, rate = [dilation, dilation], name = name)
        bias = tf.reshape(tf.nn.bias_add(depthconv, biases), depthconv.get_shape().as_list())
        relu = tf.nn.relu(bias, name = scope.name)
        return relu

def pointconv(input, nr_filters, stride, name):
    input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [1, 1, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        pointconv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = 'SAME', name = name)
        bias = tf.reshape(tf.nn.bias_add(pointconv, biases), pointconv.get_shape().as_list())
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
    nr_channels = int(depth_in.get_shape()[-1])
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
    pool2_2 = tf.nn.max_pool(pointconv3_2, ksize=[1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool2_2')

    depthconv5_1, pointconv5_2 = intermediate_residual(pool2_1, pool2_2, 'inter_res2')

    ch_pool = channel_weighted_pooling(depthconv5_1, pointconv5_2)
    avg_pool = tf.reduce_mean(ch_pool, [1,2], keepdims = True)
    fcn = pointconv(avg_pool, nr_filters = 3, stride = 1, name = 'fcn')
    normalized_out = tf.nn.l2_normalize(fcn, dim=3)
    flat_out = tf.squeeze(normalized_out, [1, 2])
    return flat_out

def test_architecture2(input):
    depthconv1_1 = depthconv(input, filter_size = 5, stride = 1,
                    padding = 'SAME', name = 'depth1_1', dilation = 1, multiplier = 4)
    pointconv1_2 = pointconv(input, nr_filters = 12, stride = 1, name = 'point1_1')

    pool1_1 = tf.nn.max_pool(depthconv1_1, ksize=[1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool1_1')
    pool1_2 = tf.nn.max_pool(pointconv1_2, ksize=[1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool1_2')

    depthconv3_1, pointconv3_2 = intermediate_residual(pool1_1, pool1_2, 'inter_res1')
    depthconv5_1, pointconv5_2 = intermediate_residual(depthconv3_1, pointconv3_2, 'inter_res2')
    #tot = tf.concat([depthconv3_1, pointconv3_2], axis = 3)
    """conv_1 = conv(input, filter_size = 5, nr_filters = 24, stride = 1,
                    name = 'conv1', padding = 'SAME')
    tot = tf.nn.max_pool(conv_1, ksize = [1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool1')"""
    #conv_out = conv(tot, filter_size = 3, nr_filters = 48, stride = 1,
    #                name = 'conv_out', padding = 'SAME')
    #pool_out = tf.nn.max_pool(conv_out, ksize = [1,2,2,1], strides = [1,2,2,1],
    #                padding = 'VALID', name = 'pool_out')
    #conv_out2 = conv(pool_out, filter_size = 3, nr_filters = 96, stride = 1,
    #                name = 'conv_out2', padding = 'SAME')
    #pool_out2 = tf.nn.max_pool(conv_out2, ksize = [1,2,2,1], strides = [1,2,2,1],
    #                padding = 'VALID', name = 'pool_out2')

    #pool2_1 = tf.nn.max_pool(depthconv3_1, ksize=[1,2,2,1], strides = [1,2,2,1],
    #                padding = 'VALID', name = 'pool2_1')
    #pool2_2 = tf.nn.max_pool(pointconv3_2, ksize=[1,2,2,1], strides = [1,2,2,1],
    #                padding = 'VALID', name = 'pool2_2')

    #depthconv5_1, pointconv5_2 = intermediate_residual(pool2_1, pool2_2, 'inter_res2')

    ch_pool = channel_weighted_pooling(depthconv5_1, pointconv5_2)
    #ch_pool = tf.concat([depthconv5_1, pointconv5_2], axis = 3)
    avg_pool = tf.reduce_mean(ch_pool, [1,2], keepdims = True)
    fcn = pointconv(avg_pool, nr_filters = 3, stride = 1, name = 'fcn')
    normalized_out = tf.nn.l2_normalize(fcn, dim=3)
    flat_out = tf.squeeze(normalized_out, [1, 2])
    return flat_out

def test_failsafe(input):
    conv1 = conv(input, filter_size = 5, nr_filters = 32, stride = 1,
                    name = 'conv1', padding = 'SAME')
    pool2 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool1')
    conv2 = conv(pool2, filter_size = 3, nr_filters = 64, stride = 1,
                    name = 'conv2', padding = 'SAME')
    pool3 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool3')
    """conv4 = conv(input, filter_size = 3, nr_filters = 64, stride = 1,
                    name = 'conv4', padding = 'SAME')
    pool5 = tf.nn.max_pool(conv4, ksize = [1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool5')
    conv6 = conv(pool5, filter_size = 3, nr_filters = 128, stride = 1,
                    name = 'conv6', padding = 'SAME')
    pool7 = tf.nn.max_pool(conv6, ksize = [1,2,2,1], strides = [1,2,2,1],
                    padding = 'VALID', name = 'pool7')"""

    #weighted_pool7 = conv(pool7, filter_size = 3, nr_filters = 3, stride = 1,
    #                name = 'conv7', padding = 'SAME')
    #summation = tf.reduce_sum(weighted_pool7, [1,2])
    #normalization = tf.nn.l2_normalize(summation, dim=1)
    #return normalization
    avg_pool = tf.reduce_mean(pool3, [1,2], keepdims = True)
    fcn = pointconv(avg_pool, nr_filters = 3, stride = 1, name = 'fcn')
    normalized_out = tf.nn.l2_normalize(fcn, dim=3)
    flat_out = tf.squeeze(normalized_out, [1, 2])
    return flat_out
