import cPickle as pickle
import sys
import cv2
import os
from data_provider import ImageRecord

import numpy as np

def apply_gt(img, gt_rgb):
    (B, G, R) = cv2.split(img)
    min_rgb = min(min(gt_rgb[0],gt_rgb[1]),gt_rgb[2])
    r = min_rgb/gt_rgb[0]
    g = min_rgb/gt_rgb[1]
    b = min_rgb/gt_rgb[2]
    r_ch = np.multiply(R, r)
    g_ch = np.multiply(G, g)
    b_ch = np.multiply(B, b)
    out = cv2.merge([b_ch, g_ch, r_ch])
    return out

def show_patches():
  from data_provider import DataProvider
  dp = DataProvider(True, ['g0'])
  dp.set_batch_size(1)
  while True:
    batch = dp.get_batch()
    images = batch[0]
    labels = batch[2]
    for i in range(len(images)):
      img = images[i]
      gt = labels[i]
      #img = img / np.mean(img, axis=(0, 1))[None, None, :]
      img = img / img.max()
      cv2.imshow("Input", np.power(img, 1 / 2.2))
      cv2.waitKey(0)
      img = apply_gt(img, gt)
      cv2.imshow("Corrected", np.power(img, 1 / 2.2))
      cv2.waitKey(0)

if __name__ == '__main__':
  show_patches()
