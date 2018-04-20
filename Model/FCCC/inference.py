# Author: bulletcross@gmail.com (Vishal Keshav)
# Main module
import tensorflow as tf
import numpy as np
import os
from data_provider import DataProvider
from data_provider import ImageRecord


batch_size = 1
nr_epochs = 1


def load_graph_weights(model_dir, sess, print_ops = False):
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    checkpoint_path = checkpoint.model_checkpoint_path
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta', clear_devices=True)
    saver.restore(sess, checkpoint_path)
    graph_def = tf.GraphDef()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        if print_ops:
            for op in graph.get_operations():
                print(op.name)
    #x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    #y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    return saver

def main():
    """x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
    y = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)
    out = M.fc4_architecture(x, keep_prob)
    angular_loss = N.get_angular_error(out, y)"""
    #dp = DataProvider(True, ['g0'])
    #dp.set_batch_size(batch_size)
    nr_step = 100
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        #saver.restore(sess, "model.ckpt")
        saver = load_graph_weights(model_dir = 'tf_log', sess = sess, print_ops = True)
        """for epoch in range(0, nr_epochs):
            for step in range(0, nr_step):
                batch = dp.get_batch()
                feed_x = batch[0]
                feed_y = batch[2]
                ans, angular_error = sess.run([out, angular_loss], feed_dict = {x: feed_x, y: feed_y, keep_prob: 1.0})

                print(ans)
                print(feed_y)
                print("Angular error:" , angular_error)
        dp.stop()"""

if __name__ == "__main__":
    main()
