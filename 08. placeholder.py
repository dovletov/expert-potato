import tensorflow as tf
import numpy as np

a = tf.placeholder(dtype=tf.float32)


with tf.Session() as sess:
    print(a)
    try:
        print(sess.run(a))
    except:
        print("In order to print, first placeholder should get some value")

    random_int = np.random.randint(100)

    print(sess.run(a, feed_dict={a: random_int}))
