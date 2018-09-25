import numpy as np
import tensorflow as tf

class FEA_MATCHING():
    def __init__(self, model):
        self.model = model
        self.fea_nat, self.fea_adv = tf.split(model.fea, 2)
        if 0:
            # using Euclidean distance
            fea_distance = tf.norm(self.fea_nat - self.fea_adv, ord='euclidean', axis=1)
        else:
            # using Cosine distance
            fea_distance = tf.reduce_sum( self.fea_nat * self.fea_adv )\
                           / (tf.norm(self.fea_nat, ord='euclidean', axis=1) + 1e-8)\
                           / (tf.norm(self.fea_adv, ord='euclidean', axis=1) + 1e-8)
        alpha = 1.0
        self.loss = alpha*tf.reduce_mean(fea_distance)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss,var_list=model.fea_variables)

    def apply(self, sess, x_batch, x_batch_adv):
        fea_dict = {self.model.x_input: np.concatenate([x_batch, x_batch_adv], 0)}
        sess.run(self.train_step, fea_dict)

