"""
Author: bulletcross@gmail.com (Vishal Keshav)
"""
from keras.preprocessing import image
from keras import backend as K
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import h5py

hyper_param_hyp = {"filters": [128, 256],
                "kernel_size": [(8,8), (4,4)],
                "strides_conv": [(4,4), (2,2)],
                "nr_layers": 2,
                "type": 'hyp'}

hyper_param_sel = {"filters": [128, 256],
                "kernel_size": [(8,8), (4,4)],
                "strides_conv": [(4,4), (2,2)],
                "nr_layers": 2,
                "type": 'sel'}

def RGB_to_UV(image):
    # u = log(R/G), v = log(B/G)
    image = np.stack(np.ma.log(image[:,:,0]/image[:,:,2]),
                     np.ma.log(image[:,:,1]/image[:,:,2]))
    return image

def nr_digits(i):
    ret = 0
    while i !=0:
        ret = ret+1
        i = int(i/10)
    return ret

def zero_string(i):
    ret = ''
    for i in range(0,i):
        ret=ret+'0'
    return ret

#http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def normalize_data(uv_image):
    uv_image = uv_image - np.mean(uv_image, axis = 2) #Mean broadcasted
    uv_image = uv_image/np.std(uv_image, axis = 2) #Not described in paper
    return uv_image

def convert_to_uv(gt_rgb):
    min_rgb = min(min(gt_rgb[0],gt_rgb[1]),gt_rgb[2])
    R = min_rgb/gt_rgb[0]
    G = min_rgb/gt_rgb[1]
    B = min_rgb/gt_rgb[2]
    gt_uv = []
    gt_uv[0] = mp.ma.log(R/G)
    gt_uv[1] = np.ma.log(B/G)
    return gt_uv

def split_to_patches(X,Y):
    sess = K.InteractiveSession()
    with sess.as_default():
        patches =K.extract_image_patches(images=X, ksizes=[1,44,44, 1],
                                         strides=[1, 44, 44, 1], rates=[1, 1, 1, 1],
                                         padding="VALID").eval()
        num_train_X, num_patch_row, num_patch_col, depth = patches.shape
        patch_X = K.reshape(patches,
                             [num_train_X * num_patch_row * num_patch_col,
                             PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2]]).eval()
    patch_Y = np.repeat(Y, num_patch_row * num_patch_col, axis=0)
    return patch_X, patch_Y

def load_dataset():
    data_path = os.getcwd()
    if os.path.isfile("dataset.h5"):
        data_file = h5py.File('dataset.h5','r')
        group_data = data_file.get('dataset_group')
        x_train_sel = np.array(group_data.get('x_train_sel'))
        x_train_hyp = np.array(group_data.get('x_train_hyp'))
        y_train = np.array(group_data.get('y_train'))
        data_file.close()
    else:
        x_train_sel = np.zeros([568, 384, 256, 2], dtype = 'uint8')
        x_train_hyp = np.zeros([568, 384, 256, 2], dtype = 'uint8')
        y_train = np.zeros([568, 2], dtype = 'float32')
        path_input = '../../Dataset/GehlerShi_input/'
        path_output = '../../Dataset/GehlerShi_output/'
        #file_name = [000001, 000002, ... , 000568]
        file_names = []
        for i in range(1, 569):
            file_names.append('00' + zero_string(4-nr_digits(i)) + str(i))
        #print(file_names)
        for index,file_name in enumerate(file_names):
            image_blob = Image.open(os.path.join(path_input, file_name+".png"))
            image_array = np.array(image_blob.getdata())
            image_array = image_array.reshape(image_blob.size[0], image_blob.size[1],3)
            image_array = image_array.astype('float32')
            image_array = image_array/225
            image_array = RGB_to_UV(image_array)
            ground_truth = np.loadtxt(os.path.join(path_output, file_name+".txt"))
            ground_truth = convert_to_uv(ground_truth)
            x_train_sel[index] = image_array
            x_train_hyp[index] = normalize_data(image_array)
            y_train[index] = ground_truth
        #x_train = x_train.astype('float32')
        #x_train = x_train/225
        data_file = h5py.File('dataset.h5','w')
        group_data = data_file.create_group('dataset_group')
        group_data.create_dataset('x_train_sel', data=x_train_sel, compression='gzip')
        group_data.create_dataset('x_train_hyp', data=x_train_hyp, compression='gzip')
        group_data.create_dataset('y_train', data=y_train, compression='gzip')
        data_file.close()
    return (x_train, y_train)
