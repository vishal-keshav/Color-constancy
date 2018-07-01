import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os

import os.path as op
import adb
from adb import adb_commands
from adb import sign_m2crypto

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Run adb kill-server if pyadb gives some problems
def push_tflite():
    print("CONNECTING TO ADB....")
    signer = sign_m2crypto.M2CryptoSigner(op.expanduser('~/.android/adbkey'))
    device = adb_commands.AdbCommands()
    device.ConnectDevice(rsa_keys=[signer])
    # Check if tflite file is present on disk, then push it into the device
    destination_dir = '/data/local/tmp/model.tflite'
    file_name = 'model.tflite'
    if op.exists(file_name):
        print(device.Push(file_name, destination_dir))
        print("FILE PUSHED")
    else:
        print("FILE NOT PRESENT")

def execute_tflite():
    print('EXECUTING')
    # Should connect again, unnecceary overhead
    signer = sign_m2crypto.M2CryptoSigner(op.expanduser('~/.android/adbkey'))
    device = adb_commands.AdbCommands()
    device.ConnectDevice(rsa_keys=[signer])
    # More checks are required, but for now, its okay!
    benchmark_file = "/data/local/tmp/label_image"
    image_file = "/data/local/tmp/grace_hopper.bmp"
    label_file = "/data/local/tmp/labels.txt"
    model_file = "/data/local/tmp/model.tflite"
    exec_command = "." + benchmark_file + " -c 100 -v 1 -i " +  \
                    image_file + " -l " + label_file + " -m " + \
                    model_file + " -t 1"
    print(exec_command)
    print(device.Shell(exec_command, timeout_ms=100000))

def profile_mobile_exec():
    push_tflite()
    execute_tflite()

def main():
    print("............Testing execution................")
    profile_mobile_exec()
    return

if __name__ == "__main__":
    main()
