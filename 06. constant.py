import tensorflow as tf

a = tf.constant(5, name="a")


with tf.Session() as sess:
    print(a)
    print(sess.run(a))
