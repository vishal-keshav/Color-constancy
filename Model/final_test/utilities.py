
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import h5py

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

def load_dataset_GhelarShi():
    data_path = os.getcwd()
    if os.path.isfile("dataset.h5"):
        data_file = h5py.File('dataset.h5','r')
        group_data = data_file.get('dataset_group')
        x_train = np.array(group_data.get('x_train'))
        y_train = np.array(group_data.get('y_train'))
        data_file.close()
    else:
        x_train = np.zeros([568, 384, 256, 3], dtype = 'uint8')
        y_train = np.zeros([568, 3], dtype = 'float32')
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
            ground_truth = np.loadtxt(os.path.join(path_output, file_name+".txt"))
            x_train[index] = image_array
            y_train[index] = ground_truth
        x_train = x_train.astype('float32')
        #x_train = x_train/225
        data_file = h5py.File('dataset.h5','w')
        group_data = data_file.create_group('dataset_group')
        group_data.create_dataset('x_train', data=x_train, compression='gzip')
        group_data.create_dataset('y_train', data=y_train, compression='gzip')
        data_file.close()
    return (x_train, y_train)

def load_dataset_Cube():
    data_path = os.getcwd()
    if os.path.isfile("dataset.h5"):
        data_file = h5py.File('dataset.h5','r')
        group_data = data_file.get('dataset_group')
        x_train = np.array(group_data.get('x_train'))
        y_train = np.array(group_data.get('y_train'))
        data_file.close()
    else:
        x_train = np.zeros([1365, 384, 256, 3], dtype = 'uint8')
        y_train = np.zeros([1365, 3], dtype = 'float32')
        path_input = '../../Dataset/Cube_input/'
        path_output = '../../Dataset/Cude_output/'
        #file_name = [000001, 000002, ... , 000568]
        file_names = []
        for i in range(1, 1366):
            file_names.append(str(i))
        #print(file_names)
        for index,file_name in enumerate(file_names):
            image_blob = Image.open(os.path.join(path_input, file_name+".png"))
            image_array = np.array(image_blob.getdata())
            image_array = image_array.reshape(image_blob.size[0], image_blob.size[1],3)
            #ground_truth = np.loadtxt(os.path.join(path_output, file_name+".txt"))
            x_train[index] = image_array
            #y_train[index] = ground_truth
        ground_truth = np.loadtxt(os.path.join(path_output, "gt.txt"))
        y_train = ground_truth.tolist()
        x_train = x_train.astype('float32')
        #x_train = x_train/225
        data_file = h5py.File('dataset.h5','w')
        group_data = data_file.create_group('dataset_group')
        group_data.create_dataset('x_train', data=x_train, compression='gzip')
        group_data.create_dataset('y_train', data=y_train, compression='gzip')
        data_file.close()
    return (x_train, y_train)
