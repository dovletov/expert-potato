import tensorflow as tf
from tensorflow.python.client import device_lib

with tf.Session() as sess:
    print(device_lib.list_local_devices())
