import os
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
c = tf.constant(5, dtype=tf.int32, name="c")
result = a*tf.to_float(c)+b

with tf.Session(config=config) as sess:
    res = sess.run(result, {a: 2.0, b: 1.0})
    print(res)
