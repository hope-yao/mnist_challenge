import numpy as np
import tensorflow as tf

class FEA_MATCHING():
    def __init__(self, x_input, fea, fea_variables):
        self.x_input = x_input
        self.fea_nat, self.fea_adv = tf.split(fea, 2)
        if 1:
            # using Euclidean distance
            # fea_distance = tf.norm(self.fea_nat - self.fea_adv, ord='euclidean', axis=1)
            fea_nat = tf.contrib.layers.flatten(self.fea_nat)
            fea_adv = tf.contrib.layers.flatten(self.fea_adv)
            fea_distance = tf.reduce_max(tf.abs(fea_nat - fea_adv))
        else:
            # using Cosine distance
            fea_nat = tf.contrib.layers.flatten(self.fea_nat)
            fea_adv = tf.contrib.layers.flatten(self.fea_adv)
            fea_distance = tf.reduce_sum( fea_nat * fea_adv, axis=(1,2,3) )\
                           / (tf.norm(fea_nat, ord='euclidean', axis=1) + 1e-8)\
                           / (tf.norm(fea_adv, ord='euclidean', axis=1) + 1e-8)
        alpha = 1.0
        self.loss = alpha*tf.reduce_mean(fea_distance)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss,var_list=fea_variables)

    def apply(self, sess, x_batch, x_batch_adv):
        fea_dict = {self.x_input: np.concatenate([x_batch, x_batch_adv], 0)}
        sess.run(self.train_step, fea_dict)

