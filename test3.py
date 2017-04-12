import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.examples.tutorials.mnist import input_data

from util.mnist_tool import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss function
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
#optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_:batch_ys})

#正确度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction_float = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction_float)

yc=tf.cast(tf.argmax(y, 1),tf.float32)
sj=tf.cast(tf.argmax(y_, 1),tf.float32)

#print(correct_prediction_float.eval({x: mnist.test.images, y_:mnist.test.labels}))
#print(sj.eval({x: mnist.test.images, y_:mnist.test.labels}))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

dir_name="test_num"
files = os.listdir(dir_name)
cnt=len(files)
for i in range(cnt):
    files[i] = dir_name + "/" + files[i]
    #print(files[i])
    test_images1, test_labels1 = GetImage([files[i]])

    mnist_test = DataSet(test_images1, test_labels1, dtype=tf.float32)
    res = accuracy.eval({x: mnist_test.images, y_: mnist_test.labels})

    # print(shape(mnist.test.images))
    # print (tf.argmax(y, 1))
    # print(y.eval())
    print("识别结果:", float(res))
    print("\n")
    # if(res==1):
    #   print("correct!\n")
    # else:
    #   print("wrong!\n")

    # print("input:",files[i].strip().split('/')[1][0])
