import os

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

import dirs
from util.mnist_tool import *

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss function
y_ = tf.placeholder(tf.float32, [None, 10])


saver = tf.train.Saver()

saver.restore(sess, dirs.save + "softmax.ckpt")
print("Model restored. dir:", dirs.save + "softmax.ckpt")


dir_name="test_num"
files = os.listdir(dir_name)
cnt=len(files)
for i in range(cnt):
    files[i] = dir_name + "/" + files[i]
    #print(files[i])
    test_images1, test_labels1 = GetImage([files[i]])

    mnist_test = DataSet(test_images1, test_labels1, dtype=tf.float32)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    res = sess.run(accuracy, feed_dict={x: mnist_test.images, y_: mnist_test.labels})

    # print(shape(mnist.test.images))
    # print (tf.argmax(y, 1))
    # print(y.eval())
    print("识别结果:", float(res))
    print("\n")