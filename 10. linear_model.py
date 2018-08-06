import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    
    sess.run(init)

    # training loop
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

        # evaluate training accuracy
        if (i+1)%50==0:
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
            print("Step: %s  W: %s b: %s loss: %s"%(str(i).zfill(3), curr_W, curr_b, curr_loss))
