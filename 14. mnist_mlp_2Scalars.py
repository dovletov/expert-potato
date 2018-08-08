# based on https://github.com/tensorflow/tensorflow/issues/7089
# ===============================================================================

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# linear regression
y = tf.matmul(x,W) + b

# cross-entropy loss function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
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
    writer_1 = tf.summary.FileWriter("./logs/14/tr", graph=tf.get_default_graph())
    writer_2 = tf.summary.FileWriter("./logs/14/vl")

    # training loop
    for step_id in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        # evaluation
        if step_id%50 == 0:
            tr_loss, tr_acc = sess.run([loss, accuracy], 
                feed_dict={x: batch[0], y_: batch[1]})
            vl_loss, vl_acc = sess.run([loss, accuracy], 
                feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print('Step  %d, validation accuracy %g' % (step_id,vl_acc))
            
            # Execute the summaries defined above. Write the obtained summaries
            # to the file, so it can be displayed in the TensorBoard
            summ1 = sess.run(performance_summaries, 
                feed_dict={tf_loss_ph:tr_loss, tf_accuracy_ph:tr_acc})
            writer_1.add_summary(summ1, step_id)

            summ2 = sess.run(performance_summaries, 
                feed_dict={tf_loss_ph:vl_loss, tf_accuracy_ph:vl_acc})
            writer_2.add_summary(summ2, step_id)
