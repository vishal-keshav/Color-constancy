# Author: bulletcross@gmail.com (Vishal Keshav)
# Main module
import tensorflow as tf
import numpy as np
import model_def as M
import net_utility as N
import utilities as ut
import os
# Thanks to data provide: yuanming-hu
from data_provider import DataProvider
from data_provider import ImageRecord

init_transfer = False
lr_rate = 0.0001
batch_size = 16
drop_prob = 0.5
nr_epochs = 10
weight_decay = 0.00005
transfer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
init_list = ['conv6', 'conv7']
weight_file = 'bvlc_alexnet.npy'

def main():
    print(os.getcwd())
    logs_path = os.path.join(os.getcwd(), 'tf_log')
    print(logs_path)
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    # Declare placeholders
    x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3]) # will change with new dataset
    y = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)
    #Construct computation graph
    out = M.fc4_architecture(x, keep_prob)
    dp = DataProvider(True, ['g0', 'g1', 'g2'])
    dp.set_batch_size(batch_size)
    with tf.name_scope("mse_loss"):
        var = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var if 'biases' not in v.name ])*weight_decay
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = out)) + l2_loss
    tf.summary.scalar('mse_loss', loss)
    with tf.name_scope("angular_loss"):
        angular_loss = N.get_angular_error(out, y)
    tf.summary.scalar('angular_loss', angular_loss)
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
    #train_x, train_y = ut.load_dataset() # RGB->BGR will be done on the fly
    #batch = dp.get_batch()
    #nr_step = int(train_x.shape[0]/batch_size)
    nr_step = 100
    # Logging through tensorboard
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path)
    saver = tf.train.Saver()
    # Start training in a session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        if init_transfer:
            N.transfer_weight(sess, weight_file, transfer_list, init_list)
        for epoch in range(0, nr_epochs):
            for step in range(0, nr_step):
                #feed_x = train_x[step*batch_size: (step+1)*batch_size]
                #feed_y = train_y[step*batch_size: (step+1)*batch_size]
                batch = dp.get_batch()
                feed_x = batch[0]
                feed_y = batch[2]
                _, step_loss, ang_loss = sess.run([train_op,loss,angular_loss], feed_dict = {x: feed_x, y:feed_y, keep_prob: drop_prob})
                if step%1 == 0:
                    summary = sess.run(merged_summary, feed_dict = {x: feed_x, y:feed_y, keep_prob: 1.0})
                    writer.add_summary(summary, epoch*batch_size+step)
                    print('Epoch= %d, step= %d,loss= %.4f, avg_angular_loss= %.4f' % (epoch, step, step_loss, ang_loss))
        dp.stop()
        chk_name = os.path.join(logs_path, 'model.ckpt')
        save_path = saver.save(sess, chk_name)


if __name__ == "__main__":
    main()
