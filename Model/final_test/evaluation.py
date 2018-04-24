
import tensorflow as tf
import numpy as np
from data_provider import DataProvider
from data_provider import ImageRecord

import model_def as model_def_final
import model_def_fc4 as model_def_fc4

import show_patches as sp
import cv2


FC4_MODEL_PATH = '../FCCC/tf_log/model.ckpt'
FINAL_MODEL_PATH = 'tf_log/model.ckpt'
OUT_TARGET = 'data/evaluation'
NR_IMAGE = 100

def angular_error_fn(predicted,label):
    norm_ab = tf.multiply(tf.linalg.norm(predicted,axis=1),tf.linalg.norm(label,axis=1))
    product_ab=tf.reduce_sum(tf.multiply(predicted,label),axis=1)
    divide_ab=tf.divide(product_ab,norm_ab)
    angu_error_ab=tf.acos(divide_ab)
    return tf.reduce_mean(angu_error_ab)*180/np.pi

def main():
    dp = DataProvider(True, ['g0'])
    dp.set_batch_size(1)
    for i in range(NR_IMAGE):
        feed = dp.get_batch()
        feed_x = feed[0]
        feed_y = feed[2]
        # Construct FC4 graph
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [1, 512, 512, 3])
        y = tf.placeholder(tf.float32, [1, 3])
        keep_prob = tf.placeholder(tf.float32)
        out = model_def_fc4.fc4_architecture(x, keep_prob)
        angular_loss = angular_error_fn(out, y)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, FC4_MODEL_PATH)

            ans, angular_error = sess.run([out, angular_loss], feed_dict = {x: feed_x, y: feed_y, keep_prob: 1.0})
            print("FC4: " + str(i) + " Angular_error: " + str(angular_error))
            print(ans[0])
            print(feed_y[0])
            img = feed_x[0] / feed_x[0].max()

            cv2.imwrite("data/evaluation/" + str(i) + "_img_input.png", 255*np.power(img, 1 / 2.2))
            img_gt = sp.apply_gt(img, feed_y[0])
            cv2.imwrite("data/evaluation/" + str(i) + "_img_gt.png", 255*np.power(img_gt, 1 / 2.2))
            img_pred = sp.apply_gt(img, ans[0])
            cv2.imwrite("data/evaluation/" + str(i) + "_img_pred_fc4.png", 255*np.power(img_pred, 1 / 2.2))
        # Construct final graph first
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [1, 512, 512, 3])
        y = tf.placeholder(tf.float32, [1, 3])
        out = model_def_final.test_architecture2(x)
        angular_loss = angular_error_fn(out, y)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, FINAL_MODEL_PATH)

            ans, angular_error = sess.run([out, angular_loss], feed_dict = {x: feed_x, y: feed_y})
            print("FINAL: " + str(i) + " Angular_error: " + str(angular_error))
            print(ans[0])
            print(feed_y[0])
            img = feed_x[0] / feed_x[0].max()
            img_pred = sp.apply_gt(img, ans[0])
            cv2.imwrite("data/evaluation/" + str(i) + "_img_pred_final.png", 255*np.power(img_pred, 1 / 2.2))
    dp.stop()

if __name__ == "__main__":
    main()
