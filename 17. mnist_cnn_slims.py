import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
from tensorflow import initializers as tf_init

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)

x_image = tf.summary.image('image', tf.reshape(x, [-1, 28, 28, 1]), 10)

def cnn(x, keep_prob):
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
    net = slim.max_pool2d(inputs = net, kernel_size=[2,2])
    print(net.shape)
    net = slim.conv2d(inputs = net,
                      num_outputs = 64,
                      kernel_size = 5,
                      padding = 'SAME',
                      activation_fn = tf.nn.relu,
                      weights_initializer = tf_init.truncated_normal(mean=0,
                        stddev=0.05),
                      biases_initializer = tf_init.zeros(),
                      scope='conv2'
                      )
    print(net.shape)
    net = slim.max_pool2d(inputs = net, kernel_size=[2,2])
    print(net.shape)
    net = slim.flatten(net)
    print(net.shape)
    net = slim.fully_connected(inputs = net,
                               num_outputs = 1024,
                               scope = 'fc3',
                               weights_initializer = tf_init.truncated_normal(mean=0,
                                stddev=0.05),
                               biases_initializer = tf_init.zeros()
                               )
    
    net = slim.dropout(net, keep_prob)
    print(net.shape)
    net = slim.fully_connected(inputs = net,
                               num_outputs = 10,
                               scope = 'fc4',
                               weights_initializer = tf_init.truncated_normal(mean=0,
                                stddev=0.05),
                               biases_initializer = tf_init.zeros()
                               )
    return net

y_conv = cnn(x, 0.5)

# cross-entropy loss function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
loss_scalar = tf.summary.scalar("loss", loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_scalar = tf.summary.scalar("accuracy", accuracy)

initializer = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(initializer)

    # op to write logs to Tensorboard
    writer_1 = tf.summary.FileWriter("./logs/17/tr", graph=tf.get_default_graph())
    writer_2 = tf.summary.FileWriter("./logs/17/vl")

    # training loop
    for step_id in range(2000):
        
        batch = mnist.train.next_batch(100)
        feed_train = {x: batch[0], y_: batch[1], keep_prob: 0.5}
        train_step.run(feed_dict=feed_train)

        if step_id % 50 == 0:
            tr_loss, tr_acc = sess.run([loss, accuracy], feed_dict=feed_train)
            feed_test = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
            vl_loss, vl_acc = sess.run([loss, accuracy], feed_dict=feed_test)
            print('Step  %d, validation accuracy %g' % (step_id,vl_acc))

            writer_1.add_summary(sess.run(accuracy_scalar, feed_dict=feed_train), step_id)
            writer_1.add_summary(sess.run(loss_scalar, feed_dict=feed_train), step_id)
            writer_1.add_summary(sess.run(x_image, feed_dict=feed_train), step_id)
            
            writer_2.add_summary(sess.run(accuracy_scalar, feed_dict=feed_test), step_id)
            writer_2.add_summary(sess.run(loss_scalar, feed_dict=feed_test), step_id)
            writer_2.add_summary(sess.run(x_image, feed_dict=feed_test), step_id)
