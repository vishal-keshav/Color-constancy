
import tensorflow as tf
import numpy as np
import model_def as M
import utilities as ut
import os
# Thanks to data provide: yuanming-hu
from data_provider import DataProvider
from data_provider import ImageRecord

batch_size = 1
nr_epochs = 1

def angular_error_fn(predicted,label):
    norm_ab = tf.multiply(tf.linalg.norm(predicted,axis=1),tf.linalg.norm(label,axis=1))
    product_ab=tf.reduce_sum(tf.multiply(predicted,label),axis=1)
    divide_ab=tf.divide(product_ab,norm_ab)
    angu_error_ab=tf.acos(divide_ab)
    return tf.reduce_mean(angu_error_ab)*180/np.pi

def main():
    x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
    y = tf.placeholder(tf.float32, [None, 3])
    out = M.test_architecture2(x)
    dp = DataProvider(True, ['g0'])
    dp.set_batch_size(batch_size)
    angular_loss = angular_error_fn(out, y)
    nr_step = 100
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        for epoch in range(0, nr_epochs):
            for step in range(0, nr_step):
                batch = dp.get_batch()
                feed_x = batch[0]
                feed_y = batch[2]
                ans, angular_error = sess.run([out,angular_loss], feed_dict = {x: feed_x, y:feed_y})
                print(ans)
                print(feed_y)
                print("Angular error:" , angular_error)
        dp.stop()

if __name__ == "__main__":
    main()
