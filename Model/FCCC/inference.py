# Author: bulletcross@gmail.com (Vishal Keshav)
# Main module
import tensorflow as tf
import numpy as np
import os
from data_provider import DataProvider
from data_provider import ImageRecord
import model_def as M
import net_utility as N
import show_patches as sp
import sys
import cv2
import os

batch_size = 1
nr_epochs = 1

def main():
    x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
    y = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)
    out = M.fc4_architecture(x, keep_prob)
    angular_loss = N.get_angular_error(out, y)
    dp = DataProvider(True, ['g0'])
    dp.set_batch_size(batch_size)
    nr_step = 30
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "tf_log/model.ckpt")
        for epoch in range(0, nr_epochs):
            for step in range(0, nr_step):
                batch = dp.get_batch()
                feed_x = batch[0]
                feed_y = batch[2]
                ans, angular_error = sess.run([out, angular_loss], feed_dict = {x: feed_x, y: feed_y, keep_prob: 1.0})
                print(str(step) + " Angular_error: " + str(angular_error))
                print(ans[0])
                print(feed_y[0])
                img = feed_x[0] / feed_x[0].max()
                #cv2.imshow("Input", np.power(img, 1 / 2.2))
                #cv2.waitKey(0)
                cv2.imwrite("data/inference/" + str(step) + "_img_input.png", 255*np.power(img, 1 / 2.2))
                img_gt = sp.apply_gt(img, feed_y[0])
                cv2.imwrite("data/inference/" + str(step) + "_img_gt.png", 255*np.power(img_gt, 1 / 2.2))
                img_pred = sp.apply_gt(img, ans[0])
                cv2.imwrite("data/inference/" + str(step) + "_img_pred.png", 255*np.power(img_pred, 1 / 2.2))
        dp.stop()

if __name__ == "__main__":
    main()
