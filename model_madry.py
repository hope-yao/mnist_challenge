"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from fea_matching import identical_matching_net_conv1, identical_matching_net_conv2

class Model(object):
  def __init__(self, fea_dim, denoiser_flag=None):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32], name='W_conv1')
    b_conv1 = self._bias_variable([32], name='b_conv1')
    self.variable_conv1 = [W_conv1, b_conv1]
    self.h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
    if denoiser_flag=='resnet':
      with tf.variable_scope('matching_net_conv1') as scope:
        self.conv1_res = identical_matching_net_conv1(self.h_conv1)
        self.h_conv1 += self.conv1_res
        self.variable_conv1_denoiser = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_rob/matching_net_conv1')
    h_pool1 = self._max_pool_2x2(self.h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64], name='W_conv2')
    b_conv2 = self._bias_variable([64], name='b_conv2')
    self.variable_conv2 = [W_conv2, b_conv2]
    self.h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    if denoiser_flag=='resnet':
      with tf.variable_scope('matching_net_conv2') as scope:
        self.conv2_res = identical_matching_net_conv2(self.h_conv2)
        self.h_conv2 += self.conv2_res
        self.variable_conv2_denoiser = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_rob/matching_net_conv2')
    h_pool2 = self._max_pool_2x2(self.h_conv2)

    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 64, fea_dim], name='W_fc1')
    b_fc1 = self._bias_variable([fea_dim], name='b_fc1')
    self.variable_fc1 = [W_fc1, b_fc1]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    self.h_fc1 = h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # output layer
    W_fc2 = self._weight_variable([1024,10], name='W_fc2')
    b_fc2 = self._bias_variable([10], name='b_fc2')
    self.variable_fc2 = [W_fc2, b_fc2]
    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    self.all_variables = self.variable_conv1 + self.variable_conv2 + self.variable_fc1 + self.variable_fc2
    if denoiser_flag:
        self.all_variables += self.variable_conv1_denoiser
        self.all_variables += self.variable_conv2_denoiser

  def copy(self, sess, model):
    for i in range(2):
    # weights and bias
      sess.run(tf.assign(self.variable_conv1[i], model.variable_conv1[i]))
      sess.run(tf.assign(self.variable_conv2[i], model.variable_conv2[i]))
      sess.run(tf.assign(self.variable_fc1[i], model.variable_fc1[i]))
      sess.run(tf.assign(self.variable_fc2[i], model.variable_fc2[i]))

  @staticmethod
  def _weight_variable(shape, name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

  @staticmethod
  def _bias_variable(shape, name):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial, name=name)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
