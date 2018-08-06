import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from tensorflow.examples.tutorials.mnist import input_data
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
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    # Create a scalar summary object for the loss so it can be displayed
    tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

    # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
    tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
    # Create a scalar summary object for the accuracy so it can be displayed
    tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

    # Merge all summaries together
    performance_summaries = tf.summary.merge([tf_loss_summary,tf_accuracy_summary])

initializer = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(initializer)

    # op to write logs to Tensorboard
    log_path = './logs/13'
    summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    # training loop
    for step_id in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        # evaluation
        if step_id%50 == 0:

            l, a = sess.run([loss, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print(l)
            
            # Execute the summaries defined above
            summ = sess.run(performance_summaries, feed_dict={tf_loss_ph:l, tf_accuracy_ph:a})

            # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
            summary_writer.add_summary(summ, step_id)
