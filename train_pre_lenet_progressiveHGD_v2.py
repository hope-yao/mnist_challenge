"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from tqdm import tqdm
from timeit import default_timer as timer
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    sess.run(tf.global_variables_initializer())

    # pre-train LeNet
    start = timer()
    for ii in range(10000):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch}
        sess.run(train_step, feed_dict=nat_dict)
    end = timer()
    pretrain_time = end - start
    print('LeNet pretrain done... ')

    # fix the pre-trained model
    with tf.name_scope('model_fix') as scope:
        model_fix = Model(fea_dim)
        sess.run(tf.variables_initializer(model_fix.all_variables))
        model_fix.copy(sess, model)

    # Compute Adversarial Perturbations
    num_adv_batch = 100000
    if 0:
        x_pool_nat = np.zeros((num_adv_batch*batch_size, 784))
        x_pool_adv = np.zeros((num_adv_batch*batch_size, 784))
        y_pool = np.zeros((num_adv_batch*batch_size))
        start = timer()
        for ii in tqdm(range(num_adv_batch)):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            x_batch_adv = attack.perturb(x_batch, y_batch, sess)
            x_pool_nat[ii*batch_size:(ii+1)*batch_size] = x_batch
            x_pool_adv[ii*batch_size:(ii+1)*batch_size] = x_batch_adv
            y_pool[ii * batch_size:(ii + 1) * batch_size] = y_batch
        end = timer()
        PGD_time = end - start
        np.save('x_pool_adv', x_pool_adv)
        np.save('x_pool_nat', x_pool_nat)
        np.save('y_pool', y_pool)
    else:
        x_pool_adv = np.load('./mnist_adv_pool/x_pool_adv.npy')
        x_pool_nat = np.load('./mnist_adv_pool/x_pool_nat.npy')
        y_pool = np.load('./mnist_adv_pool/y_pool.npy')
    print('PGD adv gen done... ')


    # progressive feature matching
    fea_matching = init_fea(sess, model, model_fix, distance_flag='L_1')
    for i, tag_i in enumerate(fea_matching.tag_list):
        # layer by layer
        for adv_ep in range(5):
            for ii in range(num_adv_batch):
                # over all adversarial adata
                # if ii %100 == 0:
                #     matching_time = 0
                start = timer()

                # train feature matching
                x_batch_nat = x_pool_nat[ii*batch_size:(ii+1)*batch_size]
                x_batch_adv = x_pool_adv[ii*batch_size:(ii+1)*batch_size]
                fea_matching.apply(sess, x_batch_nat, x_batch_adv, tag_i)

                # monitor the accuracy
                if ii%100 == 0:
                    # x_batch_nat, y_batch = mnist.train.next_batch(batch_size)
                    # x_batch_adv = attack.perturb(x_batch, y_batch, sess)
                    nat_dict = {model.x_input: x_batch_nat,
                                model.y_input: y_batch}
                    nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                    adv_dict = {model.x_input: x_batch_adv,
                                model.y_input: y_batch}
                    adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
                    print('layer {} Step {}:    ({})'.format(tag_i, ii, datetime.now()))
                    print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                    print('    training adv accuracy {:.4}%'.format(adv_acc * 100))

                    # monitor the loss
                    hinge_loss_value, match_loss_value = fea_matching.get_loss_value(sess, x_batch_nat, x_batch_adv)
                    hist_hinge_loss_value_conv1 += [hinge_loss_value['conv1']]
                    # hist_hinge_loss_value_conv2 += [hinge_loss_value['conv2']]
                    # hist_hinge_loss_value_fc1 += [hinge_loss_value['fc1']]
                    # hist_hinge_loss_value_fc2 += [hinge_loss_value['fc2']]
                    hist_match_loss_value_conv1 += [match_loss_value['conv1']]
                    # hist_match_loss_value_conv2 += [match_loss_value['conv2']]
                    # hist_match_loss_value_fc1 += [match_loss_value['fc1']]
                    # hist_match_loss_value_fc2 += [match_loss_value['fc2']]
                    print('    training conv1 hinge loss {:.4}'.format(hinge_loss_value['conv1']))
                    # print('    training conv2 hinge loss {:.4}'.format(hinge_loss_value['conv2']))
                    # print('    training fc1 hinge loss {:.4}'.format(hinge_loss_value['fc1']))
                    # print('    training fc2 hinge loss {:.4}'.format(hinge_loss_value['fc2']))
                    print('    training conv1 match loss {:.4}'.format(match_loss_value['conv1']))
                    # print('    training conv2 match loss {:.4}'.format(match_loss_value['conv2']))
                    # print('    training fc1 match loss {:.4}'.format(match_loss_value['fc1']))
                    # print('    training fc2 match loss {:.4}'.format(match_loss_value['fc2']))
                    np.save('fea_loss',{"conv1_match_loss": hist_match_loss_value_conv1,
                                        # "conv2_match_loss": hist_match_loss_value_conv2,
                                        # "fc1_match_loss": hist_match_loss_value_fc1,
                                        # "fc2_match_loss": hist_match_loss_value_fc2,
                                        "conv1_hinge_loss": hist_hinge_loss_value_conv1,
                                        # "conv2_hinge_loss": hist_hinge_loss_value_conv2,
                                        # "fc1_hinge_loss": hist_hinge_loss_value_fc1,
                                        # "fc2_hinge_loss": hist_hinge_loss_value_fc2,
                                        })
                end = timer()


                # Tensorboard summaries
                if ii % num_summary_steps == 0:
                  summary = sess.run(merged_summaries, feed_dict=adv_dict)
                  summary_writer.add_summary(summary, global_step.eval(sess))
                # Write a checkpoint
                if ii % num_checkpoint_steps == 0:
                  saver.save(sess,
                             os.path.join(model_dir, 'checkpoint'),
                             global_step=global_step)

