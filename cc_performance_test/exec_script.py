import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os

import os.path as op
import adb
from adb import adb_commands
from adb import sign_m2crypto

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_list = ['cnn', 'dsnet', 'fc4-alex', 'fc4-squeeze', 'fpc', 'ours_conv', 'ours_pool']

# Run adb kill-server if pyadb gives some problems
def push_tflite(name):
    print("CONNECTING TO ADB....")
    signer = sign_m2crypto.M2CryptoSigner(op.expanduser('~/.android/adbkey'))
    device = adb_commands.AdbCommands()
    device.ConnectDevice(rsa_keys=[signer])
    # Check if tflite file is present on disk, then push it into the device
    file_name = 'model_' + name + '.tflite'
    destination_dir = '/data/local/tmp/' + file_name
    if op.exists(file_name):
        print(device.Push(file_name, destination_dir))
        print("FILE PUSHED")
        return True
    else:
        print("FILE NOT PRESENT")
        return False

def execute_tflite(name):
    print('EXECUTING')
    # Should connect again, unnecceary overhead
    signer = sign_m2crypto.M2CryptoSigner(op.expanduser('~/.android/adbkey'))
    device = adb_commands.AdbCommands()
    device.ConnectDevice(rsa_keys=[signer])
    # More checks are required, but for now, its okay!
    benchmark_file = "/data/local/tmp/label_image"
    image_file = "/data/local/tmp/grace_hopper.bmp"
    label_file = "/data/local/tmp/labels.txt"
    model_file = '/data/local/tmp/model_' + name + '.tflite'
    exec_command = "." + benchmark_file + " -c 100 -v 1 -i " +  \
                    image_file + " -l " + label_file + " -m " + \
                    model_file + " -t 1"
    print(exec_command)
    print(device.Shell(exec_command, timeout_ms=100000))
    return

def profile_mobile_exec(name):
    if push_tflite(name):
        execute_tflite(name)
    return


def main():
    print("............Testing execution................")
    for i in range(0, len(model_list)):
        profile_mobile_exec(model_list[i])
    return

if __name__ == "__main__":
    main()
