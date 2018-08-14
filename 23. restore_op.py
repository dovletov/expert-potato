import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
from tensorflow import initializers as tf_init

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

ckpt_dir = './checkpoints/'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Session(config=config) as sess:
    
    ckpt = ckpt_dir + '21/model.ckpt-1000'
    meta = ckpt + '.meta'
    imported_meta = tf.train.import_meta_graph(meta)
    imported_meta.restore(sess, ckpt)

    # output operations
    for tensor in tf.get_default_graph().get_operations():
        print(tensor.name)
    
    graph = tf.get_default_graph()

    # placeholders
    x = graph.get_tensor_by_name("input_x:0")
    y_ = graph.get_tensor_by_name("input_y_:0")
    keep_prob = graph.get_tensor_by_name("input_keep_prob:0")

    loss = graph.get_tensor_by_name("op_loss:0")
    accuracy = graph.get_tensor_by_name("op_accuracy:0")

    feed_test = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
    l,a = sess.run([loss, accuracy], feed_dict=feed_test)
    print('Loss %s, Accuracy %s' % (l, a))
