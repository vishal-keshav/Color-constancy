
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os

import os.path as op
#import adb
#from adb import adb_commands
#from adb import sign_m2crypto

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cnn
import dsnet
import fc4
import fc4_squeeze
import fpc
import ours

def get_a_simple_model():
    input = tf.placeholder(tf.float32, [1, 224, 224, 3], name = 'input_tensor')
    #input = tf.placeholder(tf.float32, [1, 47, 47, 3], name = 'input_tensor')
    #conv1 = tf.layers.conv2d(inputs=input, filters=32,
    #            kernel_size=[3, 3], activation=tf.nn.relu)
    #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #output = tf.identity(pool1, name="output_tensor")
    #output = cnn.get_model(input)
    #output = dsnet.hyp_net_inference(input)
    #output = fc4.get_model(input)
    #output = fc4_squeeze.create_convnet(input)
    #output = fpc.get_model(input)
    #output = ours.test_architecture2(input)
    output = ours.test_architecture2_no_param(input)
    output = tf.identity(output, name="output_tensor")
    return input, output

# Parameters
def profile_param():
    """
    Profile with metadata
    """
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    input, output = get_a_simple_model()

    profile_op = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op',
                                    options=profile_op)
    print('PARAMS: ', params.total_parameters)


def print_nodes():
    """
    Print ops in the graph, with feedable information
    """
    tf.reset_default_graph()
    input, output = get_a_simple_model()
    graph = tf.get_default_graph()
    ops_list = graph.get_operations()
    tensor_list = np.array([ops.values() for ops in ops_list])
    print('PRINTING OPS LIST WITH FEED INFORMATION')
    for t in tensor_list:
        print(t)

    """
    Iterate over trainable variables, and compute all dimentions
    """
    total_dims = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape() # of type tf.Dimension
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_dims += variable_parameters
    print('TOTAL DIMS OF TRAINABLE VARIABLES', total_dims)

# Floating point operations
def profile_flops():
    """
    Profiler with metadata
    """
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    input, output = get_a_simple_model()

    profile_op = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op',
                                    options=profile_op)
    print('FLOPS:', flops.total_float_ops)

def create_tflite():
    tf.reset_default_graph()
    input, output = get_a_simple_model()
    graph = tf.get_default_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        # Here, we could have restored the checkpoint and trained weights,
        # but that is not the intention
        graph_def = graph.as_graph_def()
        # Freeze the graph
        output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ["output_tensor"])
        # Convert the model to tflite file directly.
        tflite_model = tf.contrib.lite.toco_convert(
                output_graph, input_tensors=[input], output_tensors=[output])
        with open("model.tflite", "wb") as f:
            f.write(tflite_model)

def profile_file_size():
    file_size = os.path.getsize("model.tflite")
    print("FILE SIZE IN BYTES: ", file_size)


def main():
    print("............Testing the profiler module................")
    profile_flops()
    profile_param()
    print_nodes()
    create_tflite()
    profile_file_size()
    #profile_mobile_exec()
    return

if __name__ == "__main__":
    main()
