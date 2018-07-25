import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with tf.Session() as sess:
    print("Hello World!")
