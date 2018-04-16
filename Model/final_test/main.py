
import tensorflow as tf
import numpy as np
import model_def as M
import utilities as ut
import os

# Constants (hyperparameter)
lr_rate = 0.0001
batch_size = 16
drop_prob = 0.5
nr_epochs = 1
weight_decay = 0.00005

def angular_error(predicted,label):
    norm_ab = tf.multiply(tf.linalg.norm(predicted,axis=1),tf.linalg.norm(label,axis=1))
    product_ab=tf.reduce_sum(tf.multiply(predicted,label),axis=1)
    divide_ab=tf.divide(product_ab,norm_ab)
    angu_error_ab=tf.acos(divide_ab)
    return tf.reduce_mean(angu_error_ab)*180/np.pi

def main():
    logs_path = os.path.join(os.getcwd(), 'tf_log')
    print("Printing logs and graphs into: " + logs_path)
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    # Declare placeholders
    x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
    y = tf.placeholder(tf.float32, [None, 3])
    #Construct computation graph
    out = M.test_architecture(x)
    # Will test with a different loss function, maybe with angular loss directly
    with tf.name_scope("mse_loss"):
        var = tf.trainable_variables()
        # Do not l2 regularize for now
        """l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var if 'biases' not in v.name ])*weight_decay
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = out)) + l2_loss"""
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = out))
    tf.summary.scalar('mse_loss', loss)
    with tf.name_scope("angular_loss"):
        angular_loss = angular_error(out, y)
    tf.summary.scalar('angular_loss', angular_loss)
    with tf.name_scope("train"):
        train_op = tf.train.AdamOptimizer(lr_rate).minimize(loss)
    # Here get the train_x and train_y
    train_x, train_y = ut.load_dataset()
    nr_step = int(train_x.shape[0]/batch_size)
    # Logging through tensorboard
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path)
    saver = tf.train.Saver()
    # Start training in a session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        N.transfer_weight(sess, weight_file, transfer_list, init_list)
        for epoch in range(0, nr_epochs):
            for step in range(0, nr_step):
                feed_x = train_x[step*batch_size: (step+1)*batch_size]
                feed_y = train_y[step*batch_size: (step+1)*batch_size]
                _, step_loss, ang_loss = sess.run([train_op,loss,angular_loss], feed_dict = {x: feed_x, y:feed_y})
                if step%5 == 0:
                    summary = sess.run(merged_summary, feed_dict = {x: feed_x, y:feed_y})
                    writer.add_summary(summary, epoch*batch_size+step)
                    print('Epoch= %d, step= %d,loss= %.4f, avg_angular_loss= %.4f' % (epoch, step, step_loss, ang_loss))
        chk_name = os.path.join(logs_path, 'model_epoch'+str(epoch)+'.ckpt')
        save_path = saver.save(sess, chk_name)

if __name__ == "__main__":
    main()
