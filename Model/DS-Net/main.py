"""
Author: bulletcross@gmail.com (Vishal Keshav)
"""

import utilities as ut
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D, Concatenate, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import os
from tqdm import tqdm
import h5py

def save_model(model, model_name):
    model_json = model.to_json()
    with open(model_name+".json","w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name+".h5")
    print("Saved model in the name:" + model_name)
    return

def load_model(model_name):
    json_file = open(model_name+".json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name+".h5")
    return loaded_model

def build_inference_graph(in_shape, param, out_shape):
    hp_filters = param['filters']
    hp_kernel = param['kernel_size']
    hp_kernel_s = param['strides_conv']
    model_input = Input(shape = in_shape)
    layer = model_input
    #Iteratively build the graph
    for i in range(0, param['nr_layers']):
        layer = Conv2D(filters = hp_filters[i], kernel_size = hp_kernel[i],
                    strides = hp_kernel_s[i], padding='valid', activation='relu',
                    kernel_initializer='glorot_uniform')(layer)

    # = GlobalAveragePooling2D()(layer)
    layer = Flatten()(layer)
    if param['type'] == 'hyp':
        layer = Dense(units = 256, activation='relu',
                        kernel_initializer='glorot_uniform')(layer)
        out_a = Dense(units = out_shape, activation='relu',
                        kernel_initializer='glorot_uniform')(layer)
        out_b = Dense(units = out_shape, activation='relu',
                        kernel_initializer='glorot_uniform')(layer)
        model = Model(inputs = model_input, outputs = (out_a, out_b))
    else:
        layer = Dense(units = 256, activation='relu',
                        kernel_initializer='glorot_uniform')(layer)
        out = Dense(units = out_shape, activation='softmax',
                        kernel_initializer='glorot_uniform')(layer)
        model = Model(inputs = model_input, outputs = out)
    return model

def hyp_loss(y_true, y_pred):
    #Winner-take-all loss
    out_a = y_pred[0]
    out_b = y_pred[1]
    error_a = tf.square(tf.subtract(out_a, y_true))
    error_b = tf.square(tf.subtract(out_b, y_true))
    min_error = tf.where(tf.less(error_a, error_b), error_a, error_b)
    return tf.reduce_mean(min_error)

def prepare_sel_data(out_a, out_b, label):
    error_a = tf.reduce_sum(tf.square(tf.subtract(out_a, label)), axis = 1)
    error_b = tf.reduce_sum(tf.square(tf.subtract(out_b, label)), axis = 1)
    zeros = tf.zeros_like(error_a)
    ones = tf.ones_like(error_a)
    option_a = tf.stack([ones,zeros], axis = 1)
    option_b = tf.stack([zeros,ones], axis = 1)
    return tf.where(tf.less(error_a, error_b), option_a, option_b)


def main():
    x_train_sel,x_train_hyp,y_train_hyp = ut.load_dataset()
    print(x_train_sel.shape)
    print(y_train_hyp.shape)
    #Training Hypothesis network
    if os.path.isfile('trained_model_hyp.json'):
        print("Model found, loading...")
        model_def_hyp = load_model('trained_model_hyp')
    else:
        input_shape = [44,44,2]
        output_shape = 2 #along two branches
        model_def_hyp = build_inference_graph(input_shape, ut.hyper_param_hyp, output_shape)
    model_def_hyp.summary()
    #Define loss for Hyp-Net

    model_def_hyp.compile(optimizer = Adam(lr=0.0001), loss = hyp_loss, metrics=['accuracy'])
    model_def_hyp.fit(x_train_hyp, y_train_hyp, batch_size = 1, epochs = 10)
    save_model(model_def_hyp, 'trained_model_hyp')
    #We need inference output of hypnet to train selnet
    y_train_sel_a, y_train_sel_b = model_def_hyp.predict(x_train_hyp)
    y_train_sel = prepare_sel_data(y_train_sel_a, y_train_sel_b, y_train_hyp)
    #Training Selection network
    if os.path.isfile('trained_model_sel.json'):
        print("Model found, loading...")
        model_def_hyp = load_model('trained_model_sel')
    else:
        input_shape = [44,44,2]
        output_shape = 2
        model_def_sel = build_inference_graph(input_shape, ut.hyper_param_sel, output_shape)
    model_def_sel.summary()
    model_def_sel.compile(optimizer = Adam(lr=0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
    model_def_sel.fit(x_train, y_train_sel, batch_size = 1, epochs = 10)
    save_model(model_def_sel, 'trained_model_sel')
    #Here we evaluate the DS-Net (HypNet+SelNet)

if __name__=="__main__":
    main()
