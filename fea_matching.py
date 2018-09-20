import numpy as np
import tensorflow as tf

class FEA_MATCHING():
    def __init__(self, model):
        # using Euclidean distance
        self.fea_nat, self.fea_adv = tf.split(model.fea, 2)
        fea_distance = tf.norm(self.fea_nat - self.fea_adv, ord='euclidean', axis=1)
        self.loss = tf.reduce_mean(fea_distance)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss,var_list=model.fea_variables)

    def apply(self, sess, fea_dict):
        sess.run(self.train_step, fea_dict)

