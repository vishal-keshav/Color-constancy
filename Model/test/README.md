# Test Model Architecture

```
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 384, 256, 3)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 384, 256, 64)      12352
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 64, 64, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 128)       131200
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 128)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 256)         295168
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 256)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 4, 256)         590080
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 2, 256)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 2, 512)         1180160
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
global_average_pooling2d_1 ( (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                32832
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 195
=================================================================
Total params: 2,241,987
Trainable params: 2,241,987
Non-trainable params: 0
```

# Benchmark Results

### Parameters used for training
Learning rate: 0.0001 with MSE loss, Adam Optimizer
Trained for 40 epochs with batch size of 16 images

Achieved training accuracy **99.82%**
```
Epoch 1/10
568/568 [==============================] - 3s 6ms/step - loss: 6.5175e-04 - acc: 0.9947
Epoch 2/10
568/568 [==============================] - 3s 5ms/step - loss: 5.9143e-04 - acc: 0.9930
Epoch 3/10
568/568 [==============================] - 3s 5ms/step - loss: 5.1535e-04 - acc: 0.9912
Epoch 4/10
568/568 [==============================] - 3s 5ms/step - loss: 4.8754e-04 - acc: 0.9965
Epoch 5/10
568/568 [==============================] - 3s 5ms/step - loss: 4.9264e-04 - acc: 0.9965
Epoch 6/10
568/568 [==============================] - 3s 5ms/step - loss: 4.2165e-04 - acc: 0.9947
Epoch 7/10
568/568 [==============================] - 3s 5ms/step - loss: 4.3836e-04 - acc: 0.9965
Epoch 8/10
568/568 [==============================] - 3s 5ms/step - loss: 4.0249e-04 - acc: 0.9965
Epoch 9/10
568/568 [==============================] - 3s 5ms/step - loss: 3.4436e-04 - acc: 0.9982
Epoch 10/10
568/568 [==============================] - 3s 5ms/step - loss: 3.3302e-04 - acc: 0.9982
Saved model in the name:trained_model
```

# Dataset and trained model
Dataset to run this project can be found at [LINK](https://1drv.ms/f/s!AmmAl9o7UugUinqb-liPoXqXBrEW)
For any query: [email](mailto: bulletcross@gmail.com)
