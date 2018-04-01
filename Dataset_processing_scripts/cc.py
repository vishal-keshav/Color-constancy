# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import sys
import os
from tqdm import tqdm
import h5py

import pandas as pd

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os.path
from os import path

def from_pil(pimg):
    pimg = pimg.convert(mode='RGB')
    nimg = np.asarray(pimg)
    nimg.flags.writeable = True
    return nimg

def to_pil(nimg):
    return Image.fromarray(np.uint8(nimg))

def stretch_pre(nimg):
    """
    from 'Applicability Of White-Balancing Algorithms to Restoring Faded Colour Slides: An Empirical Evaluation'
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.maximum(nimg[0]-nimg[0].min(),0)
    nimg[1] = np.maximum(nimg[1]-nimg[1].min(),0)
    nimg[2] = np.maximum(nimg[2]-nimg[2].min(),0)
    return nimg.transpose(1, 2, 0)

def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)

def max_white(nimg):
    if nimg.dtype==np.uint8:
        #print("Input image type: unit8")
        brightest=float(2**8)
    elif nimg.dtype==np.uint16:
        #print("Input image type: unit16")
        brightest=float(2**16)
    elif nimg.dtype==np.uint32:
        #print("Input image type: unit32")
        brightest=float(2**32)
    else:
        brightest==float(2**8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
    nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
    nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def stretch(nimg):
    return max_white(stretch_pre(nimg))

def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0]*(mu_g/float(nimg[0].max())),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0]**2)
    max_r = nimg[0].max()
    max_r2 = max_r**2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2,sum_r],[max_r2,max_r]]),
                                  np.array([sum_g,max_g]))
    nimg[0] = np.minimum((nimg[0]**2)*coefficient[0] + nimg[0]*coefficient[1],255)
    sum_b = np.sum(nimg[1])
    sum_b2 = np.sum(nimg[1]**2)
    max_b = nimg[1].max()
    max_b2 = max_r**2
    coefficient = np.linalg.solve(np.array([[sum_b2,sum_b],[max_b2,max_b]]),
                                             np.array([sum_g,max_g]))
    nimg[1] = np.minimum((nimg[1]**2)*coefficient[0] + nimg[1]*coefficient[1],255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_with_adjust(nimg):
    return retinex_adjust(retinex(nimg))

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

def show_chroma_histogram(filename):
    img = mpimg.imread(filename)
    """pixels = img.shape[0]*img.shape[1]
    channels = 3
    data = np.reshape(img[:, :, :channels], (pixels, channels)).astype(np.float64)"""
    print(img.shape)
    pixels = img.shape[1]*img.shape[2]
    channels = 3
    data = np.reshape(img, (pixels, channels)).astype(np.float64)
    print(data.shape)
    x = np.ma.log(data[:, 1]) - np.ma.log(data[:, 0])
    y = np.ma.log(data[:, 1]) - np.ma.log(data[:, 2])

    df = pd.DataFrame({'U': x, 'V': y})
    df = df.dropna()

    fig = plt.figure()
    plt.hist2d(df['U'], df['V'], bins=256)
    #ax = fig.add_subplot(111, projection='2d')
    #ax.scatter(u,v)
    #ax.set_xlabel('U')
    #ax.set_ylabel('V')
    plt.title('UV log chrominance plot')
    plt.show()

if __name__=="__main__":
    img = Image.open(sys.argv[1])
    #img.show()
    Stretch_img = to_pil(stretch(from_pil(img)))
    #Stretch_img.show(title="strech")
    Stretch_img.save("strech.png")
    Grey_world =to_pil(grey_world(from_pil(img)))
    #Grey_world.show(title="grey_world")
    Grey_world.save("grey_world.png")
    Retinex = to_pil(retinex(from_pil(img)))
    #Retinex.show(title="retinex")
    Retinex.save("retinex.png")
    Max_white = to_pil(max_white(from_pil(img)))
    #Max_white.show(title="max_white.png")
    Retinex_adjust = to_pil(retinex_adjust(retinex(from_pil(img))))
    #Retinex_adjust.show(title="retinex_adjust")
    Retinex_adjust.save("retinex_adjust.png")

    if path.exists(sys.argv[1][:-3] + "txt"):
        gt = np.loadtxt(sys.argv[1][:-3]+"txt")
        White_bal = to_pil(white_balance(img,gt))
        White_bal.show()
        White_bal.save("ground_truth.png")
    show_chroma_histogram(sys.argv[1])
