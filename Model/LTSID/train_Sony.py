#uniform content loss + adaptive threshold + per_class_input + recursive G
#improvement upon cqf37
from __future__ import division
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import pdb
import rawpy
import glob

import modified_network as new_net

# Input directory, has short exposure images
# Ground truth has long exposure
input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
# This will be created while training
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

print("Number of train_fns ", str(len(train_fns)))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

print("Number of test_fns ", str(len(test_fns)))

# This is the image feed for training
ps = 512 #patch size for training
save_freq = 500


# For debugging on example images
DEBUG = 0
if DEBUG == 1:
  save_freq = 1
  train_ids = train_ids[0:4]
  test_ids = test_ids[0:3]

def lrelu(x):
    return tf.maximum(x*0.2,x)

# Transposed on x1 is done to make it same shape as x2, then concatenation along 3rd axis
# then proactivily setting the first three shape as none
def upsample_and_concat(x1, x2, output_channels, in_channels, name):
    pool_size = 2
    with tf.name_scope(name) as scope:
        deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        # tf.shape(x2) is the output shape of deconv operation
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )
        # Number of output channels are now doubled
        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])
        return deconv_output


"""def network(input):
    # Nowhere we tell the spatial dimentions for other than weight tensors
    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME')
    print('conv1: ', conv1.get_shape().as_list())

    conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
    conv2=slim.conv2d(conv2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME')
    print('conv2: ', conv2.get_shape().as_list())

    conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
    conv3=slim.conv2d(conv3,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME')
    print('conv3: ', conv3.get_shape().as_list())

    conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
    conv4=slim.conv2d(conv4,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME')
    print('conv4: ', conv4.get_shape().as_list())

    conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
    conv5=slim.conv2d(conv5,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')
    print('conv5: ', conv5.get_shape().as_list())

    up6 =  upsample_and_concat( conv5, conv4, 256, 512 , 'up_conv1')
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')
    print('conv6: ', conv6.get_shape().as_list())

    up7 =  upsample_and_concat( conv6, conv3, 128, 256 , 'up_conv2')
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
    print('conv7: ', conv7.get_shape().as_list())

    up8 =  upsample_and_concat( conv7, conv2, 64, 128 , 'up_conv3')
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')
    print('conv8: ', conv8.get_shape().as_list())

    up9 =  upsample_and_concat( conv8, conv1, 32, 64 , 'up_conv4')
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')
    print('conv9: ', conv9.get_shape().as_list())

    conv10=slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    print('conv10: ', conv10.get_shape().as_list())
    out = tf.depth_to_space(conv10,2)
    print('out: ', out.get_shape().as_list())
    return out"""

# Take in raw image, and make four channels (RGBG)
#raw = rawpy.imread(raw_file), no postprocessing is done on this
def pack_raw(raw):
    #pack Bayer image to 4 channels
    # below is for view of Bayer-pattern RAW image, one channel without borders
    im = raw.raw_image_visible.astype(np.float32)
    # 2^14 = 16383, so the pixels in bayers pattern must be of 14 bits, normalizing below
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level
    # From shape = (h,w) to shape = (h,w,1), does same as np.newaxis
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html
    im = np.expand_dims(im,axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    # Returns normalized RGBG from bayers-pattern
    return out


# Created a deafault session
sess=tf.Session()
# This has not batch or channel dimentions
in_image=tf.placeholder(tf.float32,[None,None,None,4])
gt_image=tf.placeholder(tf.float32,[None,None,None,3])

# With that, we create the network
#out_image=network(in_image)
out_image = new_net.acc_network(in_image)

# Creating a loss function
G_loss=tf.reduce_mean(tf.abs(out_image - gt_image))

#t_vars=tf.trainable_variables()
# learning rate is a variable
lr=tf.placeholder(tf.float32)
# Defining optimizer
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

# Saver helps saves the graph and checkpoint
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

# this will automatically reads the latest checkpoints from directory
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    # If there is something, then restore in the default session
    saver.restore(sess,ckpt.model_checkpoint_path)

#Raw data takes long time to load. Keep them in memory after loaded.
# below creates [None, None, None, ... 6000 times]
gt_images=[None]*6000
# A ditionary for all types of exposures
input_images = {}
input_images['300'] = [None]*len(train_ids) # Similiarly, for each exposure
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)

# TO note down the losses, for graph
g_loss = np.zeros((5000,1))

# Just tracks that what was the last time it did the epoch
allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

# Start from the last epoch
learning_rate = 1e-4
for epoch in range(lastepoch,4001):
    # If there is a directory with that epoch, then continue with next epochs
    if os.path.isdir("result/%04d"%epoch):
        continue
    cnt=0
    # Reduce the learning rate after 200 epochs
    if epoch > 2000:
        learning_rate = 1e-5

    # Train_ids are the prefix numbers in train_fns, now iterated randomly
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        # Now, with this number, get all the files corresponding to this gt in input
        in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
        in_path = in_files[np.random.random_integers(0,len(in_files)-1)] # Gives one randomly generated number
        _, in_fn = os.path.split(in_path)
        # After randomizing, in_fn is a list of file names for a gt

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
        gt_path = gt_files[0] # Because we know, there is only one file
        _, gt_fn = os.path.split(gt_path) # Again, get the file name
        # In the filename itself, the exposure is there, recover that
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        # All that matters is the realtive exposure that the input has had w.r.t. gt
        ratio = min(gt_exposure/in_exposure,300)
        # These ratios can be 100, 250 or 300

        # To measure time of training on this gt
        st=time.time()
        cnt+=1

        # If our dictionary was empty, then fill it by reading raw data
        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path) # Just the raw from sensors
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio
            # Makes shape = (h,w,4) to shape = (1,h,w,4)
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0) # Normalized
            # This post=process the gt image and expand the dimention to reflect batch size 1


        #crop
        H = input_images[str(ratio)[0:3]][ind].shape[1] # Already dimentions are expanded to reflect batch of one
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch = gt_images[ind][:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:] # Corresponding to same patch, but double

        if np.random.randint(2,size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)# How come flipping along axis = 0?, it should be axis = 2
        if np.random.randint(2,size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))

        input_patch = np.minimum(input_patch,1.0) # Already normalized

        _,G_current,output=sess.run([G_opt,G_loss,out_image],feed_dict={in_image:input_patch,gt_image:gt_patch,lr:learning_rate})
        output = np.minimum(np.maximum(output,0),1) # Normalizing the network output
        g_loss[ind]=G_current # Note down the loss for that image

        print("%d %d Loss=%.3f Time=%.3f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),time.time()-st))
        # Make directory and save the result
        if epoch%save_freq==0:
          if not os.path.isdir(result_dir + '%04d'%epoch):
              os.makedirs(result_dir + '%04d'%epoch)
          # Save the two images side by side
          temp = np.concatenate((gt_patch[0,:,:,:],output[0,:,:,:]),axis=1)
          scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))
    saver.save(sess, checkpoint_dir + 'model.ckpt')
