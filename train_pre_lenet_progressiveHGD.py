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
from fea_matching import FEA_MATCHING, init_fea


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
with tf.name_scope('model_rob') as scope:
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
    os.system("cp *.py {}".format(model_dir))
    os.system("cp *.json {}".format(model_dir))
# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver()
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

hist_hinge_loss_value_conv1 = []
hist_hinge_loss_value_conv2 = []
hist_hinge_loss_value_fc1 = []
hist_hinge_loss_value_fc2 = []
hist_match_loss_value_conv1 = []
hist_match_loss_value_conv2 = []
hist_match_loss_value_fc1 = []
hist_match_loss_value_fc2 = []
with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  # saver.restore(sess, '/home/hope-yao/Documents/mnist_challenge/models/a_very_robust_model_madry/checkpoint-99900')
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}
    # PRETRAIN LENET
    start = timer()
    sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    training_time += end - start
    if ii == 10000:
        with tf.name_scope('model_fix') as scope:
            model_fix = Model(fea_dim)
            sess.run(tf.variables_initializer(model_fix.all_variables))
            model_fix.copy(sess, model)

        fea_matching = init_fea(sess, model, model_fix, distance_flag='L_inf')

    if ii >= 10000:
        # Compute Adversarial Perturbations
        start = timer()
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        end = timer()
        training_time += end - start
        adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch}
        # progressive feature matching
        fea_matching.apply(sess, x_batch, x_batch_adv)

    # Output to stdout
    if ii % num_output_steps == 0:
        # Compute Adversarial Perturbations
        start = timer()
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        end = timer()
        training_time += end - start
        adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch}
        nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
        adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
        print('Step {}:    ({})'.format(ii, datetime.now()))
        print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
        print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
        if ii>10000:
            hinge_loss_value, match_loss_value = fea_matching.get_loss_value(sess, x_batch, x_batch_adv)
            hist_hinge_loss_value_conv1 += [hinge_loss_value['conv1']]
            hist_hinge_loss_value_conv2 += [hinge_loss_value['conv2']]
            hist_hinge_loss_value_fc1 += [hinge_loss_value['fc1']]
            hist_hinge_loss_value_fc2 += [hinge_loss_value['fc2']]
            hist_match_loss_value_conv1 += [match_loss_value['conv1']]
            hist_match_loss_value_conv2 += [match_loss_value['conv2']]
            hist_match_loss_value_fc1 += [match_loss_value['fc1']]
            hist_match_loss_value_fc2 += [match_loss_value['fc2']]
            print('    training conv1 hinge loss {:.4}'.format(hinge_loss_value['conv1']))
            print('    training conv2 hinge loss {:.4}'.format(hinge_loss_value['conv2']))
            print('    training fc1 hinge loss {:.4}'.format(hinge_loss_value['fc1']))
            print('    training fc2 hinge loss {:.4}'.format(hinge_loss_value['fc2']))
            print('    training conv1 match loss {:.4}'.format(match_loss_value['conv1']))
            print('    training conv2 match loss {:.4}'.format(match_loss_value['conv2']))
            print('    training fc1 match loss {:.4}'.format(match_loss_value['fc1']))
            print('    training fc2 match loss {:.4}'.format(match_loss_value['fc2']))
            np.save('fea_loss',{"conv1_match_loss": hist_match_loss_value_conv1,
                                "conv2_match_loss": hist_match_loss_value_conv2,
                                "fc1_match_loss": hist_match_loss_value_fc1,
                                "fc2_match_loss": hist_match_loss_value_fc2,
                                "conv1_hinge_loss": hist_hinge_loss_value_conv1,
                                "conv2_hinge_loss": hist_hinge_loss_value_conv2,
                                "fc1_hinge_loss": hist_hinge_loss_value_fc1,
                                "fc2_hinge_loss": hist_hinge_loss_value_fc2,
                                })
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

