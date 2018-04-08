# Author: bulletcross@gmail.com (Vishal Keshav)
# Main module
import tensorflow as tf
import numpy as np
import model_def as M
import net_utility as N
import utility as ut


def main():
    # Declare constants
    lr_rate = 0.0001
    batch = 16
    drop_prob = 0.5
    nr_epochs = 10
    transfer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    init_list ['conv6', 'conv7']
    weight_file = 'bvlc_alexnet.npy'
    # Declare placeholders
    x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
    y = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)
    #Construct computation graph
    out = M.fc4_architecture(x, keep_prob)
    with tf.name_scope("mse loss"):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = out))
    """
    # For training only new layers:
    train_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in init_list]
    with tf.name_scope("mse loss"):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = out))
    with tf.name_scope("train"):
        gradients = tf.gradients(loss, train_var_list)
        grads_var = list(zip(gradients, train_var_list))
        train_op = tf.train.AdamOptimizer(lr_rate).apply_gradients(grads_and_vars=grads_var)
    # For above to work, weight transfer may or may not be trainable mode?
    """
    with tf.name_scope("train"):
        train_op = tf.train.AdamOptimizer(lr_rate).minimize(loss)
    # Here get the train_x and train_y
    train_x, train_y = ut.load_data() # Need implementation since alexnet trained on BRG inputs
    nr_step = int(train_x.shape[0]/batch)
    # Start training in a session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        N.transfer_weight(sess, weight_file, transfer_list, init_list)
        for epoch in range(0, nr_epochs):
            for step in range(0, nr_step):
                feed_x = train_x[step*batch: (step+1)*batch_size, :, :, :]
                feed_y = train_y[step*batch: (step+1)*batch_size, :]
                sess.run(train_op, feed_dict = {x: feed_x, y:feed_y, keep_prob: drop_prob})


if __name__ == "__main__":
    main()
