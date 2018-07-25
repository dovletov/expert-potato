import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

with tf.Session() as sess:
    print("Only one GPU with id=0 is used")
