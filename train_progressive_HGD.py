"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model_madry import Model
from pgd_attack import LinfPGDAttack
from fea_matching import FEA_MATCHING
FEA_MATCHING_FLAG = 1

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
fea_dim = config['fea_dim']
batch_size = config['training_batch_size']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(fea_dim)

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

def init_fea(model, layer_idx):
    tmp = tf.all_variables()
    if FEA_MATCHING_FLAG:
        if layer_idx=='conv1':
            # conv1
            output = model.h_conv1
            fea_variables = model.variable_conv1
        elif layer_idx=='conv2':
            # conv2
            output = model.h_conv2
            fea_variables = model.variable_conv2 # model.variable_conv1 + model.variable_conv2
        elif layer_idx=='fc1':
            # fc1
            output = model.fc1
            fea_variables = model.variable_fc1 # model.variable_conv1 + model.variable_conv2 + model.variable_fc1
        elif layer_idx=='fc2':
            # fc2
            output = model.pre_softmax
            fea_variables = model.variable_fc2 #model.variable_conv1 + model.variable_conv2 + model.variable_fc1 + model.variable_fc2
        else:
            return 0
    # build fea matching model
    fea_matching = FEA_MATCHING(model.x_input, output, fea_variables)
    fea_matching_optimizer_var = set(tf.all_variables()) - set(tmp)
    sess.run(tf.variables_initializer(fea_matching_optimizer_var))
    return fea_matching

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  # saver.restore(sess, '/home/hope-yao/Documents/mnist_challenge/models/a_very_robust_model_madry/checkpoint-99900')
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    if FEA_MATCHING_FLAG:
        if ii==20000:
            fea_matching = init_fea(model,layer_idx='conv1')
        if ii==40000:
            fea_matching = init_fea(model,layer_idx='conv2')
        if ii==60000:
            fea_matching = init_fea(model,layer_idx='fc1')
        if ii==80000:
            fea_matching = init_fea(model,layer_idx='fc2')
    x_batch, y_batch = mnist.train.next_batch(batch_size)

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    if FEA_MATCHING_FLAG:
        fea_matching.apply(sess, x_batch, x_batch_adv)
    end = timer()
    training_time += end - start
