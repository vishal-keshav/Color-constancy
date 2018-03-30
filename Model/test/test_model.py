"""
Author: bulletcross@gmail.com (Vishal Keshav)
"""

from keras.layers import Input, Dense, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D, Concatenate, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
import pandas as pd
import numpy as np
from keras.models import Model, model_from_json
from keras.optimizers import Adam

import utilities

def build_inference_graph(input_shape, hyper_param, output_shape):
    hp_filters = hyper_param['filters']#[64, 128, 256, 256, 512]
    hp_kernel = hyper_param['kernel_size']#[(8,8), (4,4), (3,3), (3,3), (3,3)]
    hp_kernel_s = hyper_param['strides_conv']#[(1,1), (2,2), (1,1), (1,1), (1,1)]
    hp_pool = hyper_param['pool_size']#[(4,6), (4,4), (2,2), (2,2), (2,2)]
    hp_pool_s = hyper_param['strides_pool']#[(4,6), (4,4), (2,2), (2,2), (2,2)]
    model_input = Input(shape = input_shape)
    layer = model_input
    #Iteratively build the graph
    for i in range(0, hyper_param['nr_layers']):
        layer = Conv2D(filters = hp_filters[i], kernel_size = hp_kernel[i],
                    strides = hp_kernel_s[i], padding='same', activation='relu',
                    kernel_initializer='glorot_uniform')(layer)
        layer = maxPooling2D(pool_size = hp_pool[i], strides = hp_pool_s[i](layer)
    global_avgpool = Global_AveragePooling2D()(layer)
    fc_layer = Dense(units = 64, activation='relu',
                    kernel_initializer='glorot_uniform')(global_avgpool)
    fc_layer = Dense(units = output_shape, activation='relu',
                    kernel_initializer='glorot_uniform')(fc_layer)
    model = Model(inputs = model_input, outputs = fc_layer)
    return model

def save_model(model, model_name):
    model_json = model.to_json()
    with open(model_name+".json","w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name+".h5")
    print("Saved model in the name:" + model_name)

def load_model(model_name):
    json_file = open(model_name+".json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name+".h5")
return loaded_model

def main():
    (x_train, y_train) = utilities.load_dataset()
    if os.path.isfile('trained_model.json'):
        print("Model found, loading...")
        model_def = load_model('trained_model')
    else:
        model_def = build_inference_graph(input_shape, utilities.hyper_param, output_shape)
    model_def.summary()
    model_def.compile(optimizer = Adam(lr=0.001), loss = 'mse', metrics=['accuracy'])
    model_def.fit(x_train, y_train, batch_size = 64, epochs = 10)
    save_model(model_def, 'trained_model')

if __name__ == "__main__":
    main()
