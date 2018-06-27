
import tensorflow as tf
import numpy as np
from data_provider import DataProvider
from data_provider import ImageRecord

import model_def as model_def_final
import model_def_fc4 as model_def_fc4

import show_patches as sp
import cv2

# This has to be readjusted, depending on data points read.
NR_IMAGE = 10

FC4_MODEL_PATH = '../FCCC/tf_log/model.ckpt'
FINAL_MODEL_PATH = 'tf_log/model.ckpt'
OUT_TARGET = 'data/evaluation'

def angular_error_fn(predicted,label):
    norm_ab = tf.multiply(tf.linalg.norm(predicted,axis=1),tf.linalg.norm(label,axis=1))
    product_ab=tf.reduce_sum(tf.multiply(predicted,label),axis=1)
    divide_ab=tf.divide(product_ab,norm_ab)
    angu_error_ab=tf.acos(divide_ab)
    return tf.reduce_mean(angu_error_ab)*180/np.pi

# This is what we should use!
def angular_error(estimation, ground_truth):
    return acos(np.clip(np.dot(estimation, ground_truth) /
        np.linalg.norm(estimation) /
        np.linalg.norm(ground_truth), -1, 1))
# These are from utils
def summary_angular_errors(errors):
    errors = sorted(errors)

    def g(f):
        return np.percentile(errors, f * 100)

    median = g(0.5)
    mean = np.mean(errors)
    trimean = 0.25 * (g(0.25) + 2 * g(0.5) + g(0.75))
    results = {
      '25': np.mean(errors[:int(0.25 * len(errors))]),
      '75': np.mean(errors[int(0.75 * len(errors)):]),
      '95': g(0.95),
      'tri': trimean,
      'med': median,
      'mean': mean
    }
    return results


def just_print_angular_errors(results):
    print("25: %5.3f," % results['25'])
    print("med: %5.3f" % results['med'])
    print("tri: %5.3f" % results['tri'])
    print("avg: %5.3f" % results['mean'])
    print("75: %5.3f" % results['75'])
    print("95: %5.3f" % results['95'])


def print_angular_errors(errors):
    print(str(len(errors)) + " images tested. Results:")
    results = summary_angular_errors(errors)
    just_print_angular_errors(results)
    return results

def main():
    # We dont want augmentation
    dp = DataProvider(False, ['g0'])
    dp.set_batch_size(1)
    # For all the test images
    fc4_test = []
    ours_test = []
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
        # This is what we want to track
        angular_loss = angular_error_fn(out, y)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # restore the checkpoint, and run the model
            saver.restore(sess, FC4_MODEL_PATH)
            ans, angular_error = sess.run([out, angular_loss], feed_dict = {x: feed_x, y: feed_y, keep_prob: 1.0})
            fc4_test.append(angular_error)
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
            ours_test.append(angular_error)
            print(ans[0])
            print(feed_y[0])
            img = feed_x[0] / feed_x[0].max()
            img_pred = sp.apply_gt(img, ans[0])
            cv2.imwrite("data/evaluation/" + str(i) + "_img_pred_final.png", 255*np.power(img_pred, 1 / 2.2))
    dp.stop()
    # print out the stats
    print_angular_errors(fc4_test)
    print_angular_errors(ours_test)


if __name__ == "__main__":
    main()
