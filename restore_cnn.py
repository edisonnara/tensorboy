import os

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.examples.tutorials.mnist import input_data

import dirs
from util.mnist_tool import *

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()

# 占位符
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 第一层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

saver.restore(sess, dirs.save + "cnn.ckpt")
print("CNN模型被加载. dir:", dirs.save + "cnn.ckpt")

# print("CNN模型的正确率为： %g" %
#       accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

dir_name="img_custom"
files = os.listdir(dir_name)
cnt=len(files)
for i in range(cnt):
    files[i] = dir_name + "/" + files[i]
    #print(files[i])
    test_images1, test_labels1 = GetImage([files[i]])

    mnist_test = DataSet(test_images1, test_labels1, dtype=tf.float32)

    res = accuracy.eval(feed_dict={x: mnist_test.images, y_: mnist_test.labels, keep_prob: 1.0})

    # print(shape(mnist.test.images))
    # print (tf.argmax(y, 1))
    # print(y.eval())
    print("识别结果:", float(res))
    #print("\n")