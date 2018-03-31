"""
Author: bulletcross@gmail.com (Vishal Keshav)
"""
import os
import utilities as ut
import test_model as tm
import pandas as pd
import numpy as np
from PIL import Image
import sys

def from_pil(pimg):
    pimg = pimg.convert(mode='RGB')
    nimg = np.asarray(pimg)
    nimg.flags.writeable = True
    return nimg

def to_pil(nimg):
    return Image.fromarray(np.uint8(nimg))

def white_balance(pimg, gt_rgb):
    pimg = pimg.convert('RGB')
    min_rgb = min(min(gt_rgb[0],gt_rgb[1]),gt_rgb[2])
    R = min_rgb/gt_rgb[0]
    G = min_rgb/gt_rgb[1]
    B = min_rgb/gt_rgb[2]
    r_ch, g_ch, b_ch = pimg.split()
    r_ch = r_ch.point(lambda i: i * R)
    g_ch = g_ch.point(lambda i: i * G)
    b_ch = b_ch.point(lambda i: i * B)
    out = Image.merge('RGB', (r_ch, g_ch, b_ch))
    return out

def main():
    (x_train, y_train) = ut.load_dataset()
    print("Dataset loaded...")
    model_def = tm.load_model('trained_model')
    y_predicted = model_def.predict(x_train)
    np.savetxt("gt", y_train)
    np.savetxt("pred", y_predicted)
    path_input_image = '../../Dataset/GehlerShi_input/'
    path_output = '../../Dataset/Prediction/'
    file_names = []
    for i in range(1, 569):
        file_names.append('00' + ut.zero_string(4-ut.nr_digits(i)) + str(i))
    for index,file_name in enumerate(file_names):
        image_blob = Image.open(os.path.join(path_input_image, file_name+".png"))
        gt_lumminance = y_train[index]
        pred_lumminance = y_predicted[index]
        White_bal_groundtruth = to_pil(white_balance(image_blob,gt_lumminance))
        White_bal_prediction = to_pil(white_balance(image_blob,pred_lumminance))
        White_bal_groundtruth.save(os.path.join(path_output, file_name+"_gt.png"))
        White_bal_prediction.save(os.path.join(path_output, file_name+"_pred.png"))

if __name__ == "__main__":
    main()
