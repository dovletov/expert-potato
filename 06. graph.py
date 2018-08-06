import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

graph = tf.Graph()
with graph.as_default():
    variable = tf.Variable(42, name='foo')
    initialize = tf.global_variables_initializer()
    assign = variable.assign(13)

with tf.Session(graph=graph) as sess:
    sess.run(initialize)
    print(sess.run(variable))

    sess.run(assign)
    print(sess.run(variable))


with tf.Session(graph=graph) as sess2:
    print(sess2.run(variable))
    # tensorflow.python.framework.errors_impl.FailedPreconditionError:
    # Attempting to use uninitialized value foo