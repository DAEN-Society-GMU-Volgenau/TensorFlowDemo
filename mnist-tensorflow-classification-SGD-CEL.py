"""

Recognize hand-written digits using Stochastic Gradient Descent
Based on TensorFlow demo provided by Google, Inc.

Darron Fuller - November 2015 - Society of Data Analytics Engineers at Volgenau School of Engineering
Web Page: www.DAEN-Society.org
eMail:  contact@DAEN-Society.org
Twitter:  @da_engineers

"""
print(__doc__)

import tensorflow as tf
import input_data, time

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

print 'TRAINING data shape (cases,image dimension) : {:s}'.format(mnist.train.images.shape)
print 'TEST data shape (cases,image dimension) : {:s}'.format(mnist.test.images.shape)

# begin timing now (after loading data)
timeStart = time.clock()

x = tf.placeholder("float", [None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# implement the softmax model in one line of code!

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000) :
    batch_xs , batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x: batch_xs, y_ : batch_ys})

correct_prediction = tf.equal(tf.arg_max(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print 'Classificaton Accuracy: {:2.2f}'.format(sess.run(accuracy,feed_dict={x: mnist.test.images,y_ : mnist.test.labels}))

timeElapsed = (time.clock() - timeStart)
print 'Run Time: {:2.2f} seconds'.format(timeElapsed)