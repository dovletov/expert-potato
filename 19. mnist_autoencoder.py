import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
from tensorflow import initializers as tf_init

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])

# model
def cnn(x):
    images = tf.reshape(x, [-1, 28, 28, 1])
    print(images.shape)
    net = slim.conv2d(inputs = images,
                      num_outputs = 32,
                      kernel_size = 5,
                      padding = 'SAME',
                      activation_fn = tf.nn.relu,
                      weights_initializer = tf_init.truncated_normal(mean=0,
                        stddev=0.05),
                      biases_initializer = tf_init.zeros(),
                      scope = 'conv1'
                      )
    print(net.shape)
    net = slim.conv2d(inputs = net,
                      num_outputs = 16,
                      kernel_size = 5,
                      padding = 'SAME',
                      activation_fn = tf.nn.relu,
                      weights_initializer = tf_init.truncated_normal(mean=0,
                        stddev=0.05),
                      biases_initializer = tf_init.zeros(),
                      scope='conv2'
                      )
    print(net.shape)
    net = slim.conv2d(inputs = net,
                      num_outputs = 8,
                      kernel_size = 5,
                      padding = 'SAME',
                      activation_fn = tf.nn.relu,
                      weights_initializer = tf_init.truncated_normal(mean=0,
                        stddev=0.05),
                      biases_initializer = tf_init.zeros(),
                      scope='conv3'
                      )
    print(net.shape)
    
    net = slim.conv2d_transpose(inputs=net,
                                num_outputs=16,
                                kernel_size=5,
                                padding='SAME',
                                activation_fn = tf.nn.relu,
                                weights_initializer = tf_init.truncated_normal(mean=0,
                                stddev=0.05),
                                biases_initializer = tf_init.zeros(),
                                scope = 'tconv1')
    print(net.shape)
    net = slim.conv2d_transpose(inputs=net,
                                num_outputs=32,
                                kernel_size=5,
                                padding='SAME',
                                activation_fn = tf.nn.relu,
                                weights_initializer = tf_init.truncated_normal(mean=0,
                                stddev=0.05),
                                biases_initializer = tf_init.zeros(),
                                scope = 'tconv2')
    print(net.shape)
    net = slim.conv2d_transpose(inputs=net,
                                num_outputs=1,
                                kernel_size=5,
                                padding='SAME',
                                activation_fn = tf.nn.relu,
                                weights_initializer = tf_init.truncated_normal(mean=0,
                                stddev=0.05),
                                biases_initializer = tf_init.zeros(),
                                scope = 'tconv3')
    print(net.shape)
    return net

y = cnn(x)

# cross-entropy loss function
loss = tf.reduce_mean(tf.square(tf.reshape(x, [-1, 28, 28, 1]) - y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


# summaries
x_image = tf.summary.image('x_image', tf.reshape(x, [-1, 28, 28, 1]), 10)
y_image = tf.summary.image('y_image', tf.reshape(y, [-1, 28, 28, 1]), 10)
loss_scalar = tf.summary.scalar("loss", loss)
summaries = tf.summary.merge_all()

initializer = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(initializer)

    # op to write logs to Tensorboard
    writer_1 = tf.summary.FileWriter("./logs/19/tr", graph=tf.get_default_graph())
    writer_2 = tf.summary.FileWriter("./logs/19/vl")

    # training loop
    for step_id in range(200):
        
        batch = mnist.train.next_batch(1)
        feed_train = {x: batch[0]}
        train_step.run(feed_dict=feed_train)

        if step_id % 5 == 0:
            feed_test = {x: mnist.test.images[0:1]}
            vl_loss = sess.run(loss, feed_dict=feed_test)
            print('Step %d, validation loss %g' % (step_id, vl_loss))

            writer_1.add_summary(sess.run(summaries, feed_dict=feed_train), step_id)
            writer_2.add_summary(sess.run(summaries, feed_dict=feed_test), step_id)
