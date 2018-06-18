import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def lrelu(x):
    return tf.maximum(x*0.2,x)

# Transposed on x1 is done to make it same shape as x2, then concatenation along 3rd axis
# then proactivily setting the first three shape as none
def upsample_and_concat(x1, x2, output_channels, in_channels, name):
    pool_size = 2
    with tf.name_scope(name) as scope:
        deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        # tf.shape(x2) is the output shape of deconv operation
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )
        # Number of output channels are now doubled
        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])
        return deconv_output

def conv(input_channels, input, filter_size, nr_filters, stride, name, padding = 'SAME', dilation = 1):
    #input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        conv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = padding, name = name)
        bias = tf.nn.bias_add(conv, biases)
        #bias.set_shape(tf.shape(conv))
        #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        #relu = tf.nn.relu(bias, name = scope.name)
        relu = lrelu(bias)
        return relu


"""
Self descriptive
"""
def depthconv(input_channels, input, filter_size, stride, name, padding = 'SAME', dilation = 1, multiplier = 1):
    #input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [filter_size, filter_size, input_channels, multiplier])
        biases = tf.get_variable(name = name + '_biases', shape = [input_channels*multiplier])
        depthconv = tf.nn.depthwise_conv2d(input, weights, strides = [1, stride, stride, 1],
                                      padding = padding, rate = [dilation, dilation], name = name)
        bias = tf.nn.bias_add(depthconv, biases)
        #bias.set_shape(tf.shape(depthconv))
        #bias = tf.reshape(tf.nn.bias_add(depthconv, biases), depthconv.get_shape().as_list())
        #relu = tf.nn.relu(bias, name = scope.name)
        relu = lrelu(bias)
        return relu

"""
Self descriptive
"""
def pointconv(input_channels, input, nr_filters, stride, name):
    #input_channels = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name = name + '_weights', shape = [1, 1, input_channels, nr_filters])
        biases = tf.get_variable(name = name + '_biases', shape = [nr_filters])
        pointconv = tf.nn.conv2d(input, weights, strides = [1, stride, stride, 1], padding = 'SAME', name = name)
        bias = tf.nn.bias_add(pointconv, biases)
        #bias.set_shape(tf.shape(pointconv))
        #bias = tf.reshape(tf.nn.bias_add(pointconv, biases), pointconv.get_shape().as_list())
        #relu = tf.nn.relu(bias, name = scope.name)
        relu = lrelu(bias)
        return relu

"""
Inputs: Tensor1, Tensor2 (same shape)
Output: Tensor of same shape with same number of channels as any input
"""
def channel_weighted_pooling(nr_channel, weights, channel):
    #nr_channel = channel.get_shape().as_list()[-1]
    with tf.name_scope('cwp') as scope:
        pool_weights = tf.split(weights, num_or_size_splits = nr_channel, axis = 3)
        channel_outputs = tf.split(channel, num_or_size_splits = nr_channel, axis = 3)
        prod = []
        for i in range(nr_channel):
            prod.append(tf.multiply(pool_weights[i], channel_outputs[i]))
        output = tf.concat(prod, axis = 3)
        return output

"""
Inputs: Tensor1, Tensor2
Outputs: Tensor1, Tensor2 (H/4, W/4, C*4)
"""
def intermediate_residual(nr_channels, depth_in, point_in, name):
    #nr_channels = int(depth_in.get_shape()[-1])
    with tf.name_scope(name) as scope:
        depthconv_inter = depthconv(nr_channels, depth_in, filter_size = 3, stride = 2,
                                        padding = 'SAME', name = name+'_depth1',
                                        dilation = 1, multiplier = 2)
        pointconv_inter = pointconv(nr_channels, point_in, nr_channels*2, stride = 2,
                                        name = name+'_point1')
        tensor_inter = tf.concat([depth_in, point_in], axis = 3)
        # tensor_inter is of half the spatial dimention and four times the input tensor
        conv_inter = conv(4*nr_channels, tensor_inter, filter_size = 3, nr_filters = 2*nr_channels,
                            stride = 2, name = name+'_conv', padding = 'SAME')
        # By this time, channels are halved, so double on original input
        # Use shuffling and fire module here, instead of conv
        depth_out_tensor = tf.concat([depthconv_inter, conv_inter], axis = 3)
        # Four times the input channel now
        depthconv_out = depthconv(4*nr_channels, depth_out_tensor, filter_size = 3, stride = 2,
                                        padding = 'SAME', name = name+'_depth2',
                                        dilation = 1, multiplier = 1)
        # Again spatial dims halved, so one-fourth of original
        point_out_tensor = tf.concat([pointconv_inter, conv_inter], axis = 3)
        pointconv_out = pointconv(4*nr_channels, point_out_tensor, nr_channels*2*2, stride = 2,
                                        name = name+'_point2')
        return depthconv_out, pointconv_out


def intermediate_residual_modified(nr_channels, depth_in, point_in, name, reduce_dims = True):
    #nr_channels = int(depth_in.get_shape()[-1])
    with tf.name_scope(name) as scope:
        if reduce_dims:
            new_stride = 2
        else:
            new_stride = 1
        depthconv_inter = depthconv(nr_channels, depth_in, filter_size = 3, stride = new_stride,
                                        padding = 'SAME', name = name+'_depth1',
                                        dilation = 1, multiplier = 1)
        pointconv_inter = pointconv(nr_channels, point_in, nr_channels, stride = new_stride,
                                        name = name+'_point1')
        conv_inter = tf.concat([depthconv_inter, pointconv_inter], axis = 3)
        pool = channel_weighted_pooling(nr_channels, depthconv_inter, pointconv_inter)
        # By this time, channels are halved, so double on original input
        # Use shuffling and fire module here, instead of conv
        depth_out_tensor = tf.concat([depthconv_inter, pool], axis = 3)
        # Four times the input channel now
        depthconv_out = depthconv(2*nr_channels, depth_out_tensor, filter_size = 3, stride = 1,
                                        padding = 'SAME', name = name+'_depth2',
                                        dilation = 1, multiplier = 1)
        # Again spatial dims halved, so one-fourth of original
        point_out_tensor = tf.concat([pointconv_inter, pool], axis = 3)
        pointconv_out = pointconv(2*nr_channels, point_out_tensor, nr_channels*2, stride = 1,
                                        name = name+'_point2')
        return depthconv_out, conv_inter, pointconv_out

def acc_network(input):

    # Nowhere we tell the spatial dimentions for other than weight tensors
    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    depth_conv1 = depthconv(32, conv1, filter_size = 3, stride = 1,
                                    padding = 'SAME', name = 'conv1_depth1',
                                    dilation = 1, multiplier = 1)
    depth_conv1_pool=slim.max_pool2d(depth_conv1, [2, 2], padding='SAME')
    point_conv1 = pointconv(32, conv1, 32, stride = 1,
                                    name = 'conv1_point1')
    point_conv1_pool=slim.max_pool2d(point_conv1, [2, 2], padding='SAME')
    #conv1_inter = tf.concat([depth_conv1, point_conv1], axis = 3)
    conv1_inter = channel_weighted_pooling(32, depth_conv1, point_conv1)
    print('conv1_inter: ', conv1_inter.get_shape().as_list())

    depth_conv2, conv2_inter, point_conv2 = intermediate_residual_modified(32, depth_conv1, point_conv1, 'residual1')
    print('conv2_inter: ', conv2_inter.get_shape().as_list())

    depth_conv3, conv3_inter, point_conv3 = intermediate_residual_modified(64, depth_conv2, point_conv2, 'residual2')
    print('conv3_inter: ', conv3_inter.get_shape().as_list())

    depth_conv4, conv4_inter, point_conv4 = intermediate_residual_modified(128, depth_conv3, point_conv3, 'residual3')
    print('conv4_inter: ', conv4_inter.get_shape().as_list())

    depth_conv5, conv5_inter, point_conv5 = intermediate_residual_modified(256, depth_conv4, point_conv4, 'residual4')
    print('conv5_inter: ', conv5_inter.get_shape().as_list())
    # depth_conv5 and point_conv5 are of 512 channels each
    conv5 = channel_weighted_pooling(512, depth_conv5, point_conv5) # Again 512 channels
    print('conv5: ', conv5.get_shape().as_list())

    up6 =  upsample_and_concat( conv5, conv4_inter, 256, 512 , 'up_conv1')
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')
    print('conv6: ', conv6.get_shape().as_list())

    up7 =  upsample_and_concat( conv6, conv3_inter, 128, 256 , 'up_conv2')
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
    print('conv7: ', conv7.get_shape().as_list())

    up8 =  upsample_and_concat( conv7, conv2_inter, 64, 128 , 'up_conv3')
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')
    print('conv8: ', conv8.get_shape().as_list())

    up9 =  upsample_and_concat( conv8, conv1_inter, 32, 64 , 'up_conv4')
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')
    print('conv9: ', conv9.get_shape().as_list())

    conv10=slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    print('conv10: ', conv10.get_shape().as_list())
    out = tf.depth_to_space(conv10,2)
    print('out: ', out.get_shape().as_list())
    return out
