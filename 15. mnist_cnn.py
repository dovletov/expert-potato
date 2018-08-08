import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# variables
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# cross-entropy loss function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('performance'):
    # Summaries need to be displayed. Whenever you need to record the loss/acc,
    # feed the values to these placeholders.
    tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    tf_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
    
    # Create a scalar summary objects for the loss/acc so it can be displayed
    tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)
    tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

    # Merge all summaries together
    performance_summaries = tf.summary.merge([tf_loss_summary,tf_accuracy_summary])

initializer = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(initializer)
  
    # op to write logs to Tensorboard
    writer_1 = tf.summary.FileWriter("./logs/15/tr", graph=tf.get_default_graph())
    writer_2 = tf.summary.FileWriter("./logs/15/vl", graph=tf.get_default_graph())

    # training loop
    for step_id in range(20000):
        
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
        if step_id % 50 == 0:
            tr_loss, tr_acc = sess.run([loss, accuracy], 
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            vl_loss, vl_acc = sess.run([loss, accuracy], 
                feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print('Step  %d, validation accuracy %g' % (step_id,vl_acc))
            
            # Execute the summaries defined above. Write the obtained summaries
            # to the file, so it can be displayed in the TensorBoard
            summ1 = sess.run(performance_summaries, 
                feed_dict={tf_loss_ph:tr_loss, tf_accuracy_ph:tr_acc})
            writer_1.add_summary(summ1, step_id)

            summ2 = sess.run(performance_summaries, 
                feed_dict={tf_loss_ph:vl_loss, tf_accuracy_ph:vl_acc})
            writer_2.add_summary(summ2, step_id)
