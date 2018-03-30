"""
Author: bulletcross@gmail.com (Vishal Keshav)
"""
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import h5py

hyper_param = {"filters": [64, 128, 256, 256, 512],
                "kernel_size": [(8,8), (4,4), (3,3), (3,3), (3,3)],
                "strides_conv": [(1,1), (2,2), (1,1), (1,1), (1,1)],
                "pool_size": [(4,6), (4,4), (2,2), (2,2), (2,2)],
                "strides_pool": [(4,6), (4,4), (2,2), (2,2), (2,2)]}
def load_dataset():
    data_path = os.getcwd()
    if os.path.isfile("dataset.h5"):
        data_file = h5py.File('dataset.h5','r')
        group_data = data_file.get('dataset_group')
        x_train = np.array(group_data.get('x_train'))
        y_train = np.array(group_data.get('y_train'))
        data_file.close()
    else:
        #Dont do anything as of now
    return (x_train, y_train)
