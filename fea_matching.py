import numpy as np
import tensorflow as tf

class FEA_MATCHING():
    def __init__(self, model, model_fix, distance_flag, alpha=1.0): #x_input, fea, fea_variables, distance_flag, alpha = 1.0):
        self.model_input = model.x_input
        self.model_fix_input = model_fix.x_input
        self.alpha = alpha
        self.distance_flag = distance_flag
        # self.x_input = model.x_input
        self.fea_nat_hinge = {}
        self.match_loss = {}
        self.hinge_loss = {}
        self.train_layer = {}

        self.tag_list = ['conv1', 'conv2', 'fc1', 'fc2']
        fix_fea_list = [model_fix.h_conv1, model_fix.h_conv2, model_fix.h_fc1, model_fix.pre_softmax]
        self.fix_fea_dict = dict(zip(self.tag_list, fix_fea_list))
        fea_list = [model.h_conv1, model.h_conv2, model.h_fc1, model.pre_softmax]
        self.fea_dict = dict(zip(self.tag_list, fea_list))
        fea_var_list = [model.variable_conv1, model.variable_conv2, model.variable_fc1, model.variable_fc2]
        self.fea_var_dict = dict(zip(self.tag_list, fea_var_list))

        for i, tag_i in enumerate(self.tag_list):
            fea_nat, fea_adv = tf.split(self.fea_dict[tag_i], 2)
            self.fea_nat_hinge[self.tag_list[i]] = fea_nat_hinge_i = tf.placeholder(tf.float32, fea_nat.get_shape().as_list(), name='fea_nat_hinge')
            fea_nat_vec = tf.contrib.layers.flatten(fea_nat)
            fea_adv_vec = tf.contrib.layers.flatten(fea_adv)
            fea_nat_hinge_vec = tf.contrib.layers.flatten(fea_nat_hinge_i)
            match_loss, hinge_loss = self.get_loss(fea_nat_vec, fea_adv_vec, fea_nat_hinge_vec, distance_flag)
            loss = self.alpha * tf.reduce_mean(match_loss) + tf.reduce_mean(hinge_loss)
            train_layer = tf.train.AdamOptimizer(1e-4).minimize(loss,var_list=self.fea_var_dict[tag_i])
            self.match_loss[self.tag_list[i]] = match_loss
            self.hinge_loss[self.tag_list[i]] = hinge_loss
            self.train_layer[self.tag_list[i]] = train_layer

    def get_loss(self, fea_nat, fea_adv, fea_nat_hinge, distance_flag):
        if distance_flag=='cosine':
            # using Cosine distance measurement
            match_loss = self.cos_dist(fea_nat, fea_adv)
            hinge_loss = self.cos_dist(fea_nat, fea_nat_hinge)
        else:
            if distance_flag == 'L_inf':
                # using L_inf distance measurement
                match_loss = tf.reduce_max(tf.abs(fea_adv - fea_nat_hinge), 1)
                hinge_loss = tf.reduce_max(tf.abs(fea_nat - fea_nat_hinge), 1)
                # fea_distance = tf.norm(self.fea_nat - self.fea_adv, ord='np.inf', axis=1)
            elif distance_flag == 'L_1':
                # using L_1 distance measurement
                match_loss = tf.norm(fea_adv - fea_nat_hinge, ord=1, axis=1)
                hinge_loss = tf.norm(fea_nat - fea_nat_hinge, ord=1, axis=1)
            elif distance_flag == 'L_2':
                # using L_2 distance measurement
                match_loss = tf.norm(fea_adv - fea_nat_hinge, ord=2, axis=1)
                hinge_loss = tf.norm(fea_nat - fea_nat_hinge, ord=2, axis=1)
            else:
                print("loss measure undefined")
                return 0
        return tf.reduce_mean(match_loss), tf.reduce_mean(hinge_loss)

    def cos_dist(self, x, y):
        # x and y are bs*fea_dim
        cos_dist = tf.reduce_sum(x* y, axis=(1, 2, 3)) \
                       / (tf.norm(x, ord='euclidean', axis=1) + 1e-8) \
                       / (tf.norm(y, ord='euclidean', axis=1) + 1e-8)
        return cos_dist

    def apply(self, sess, x_batch_nat, x_batch_adv, tag_i):

        nat_fea_fix = sess.run(self.fix_fea_dict[tag_i], {self.model_fix_input: x_batch_nat})
        fea_dict = {self.model_input: np.concatenate([x_batch_nat, x_batch_adv], 0),
                    self.fea_nat_hinge[tag_i]: nat_fea_fix}
        sess.run(self.train_layer[tag_i], fea_dict)

    def get_loss_value(self, sess, x_batch, x_batch_adv):
        hinge_loss_val = {}
        match_loss_val = {}
        for i, tag_i  in enumerate(self.tag_list):
            nat_fea_fix = sess.run(self.fix_fea_dict[tag_i], {self.model_fix_input: x_batch})
            fea_dict = {self.model_input: np.concatenate([x_batch, x_batch_adv], 0),
                        self.fea_nat_hinge[tag_i]: nat_fea_fix}
            hinge_loss_val[tag_i] = sess.run(self.hinge_loss[tag_i], fea_dict)
            match_loss_val[tag_i] = sess.run(self.match_loss[tag_i], fea_dict)
        return hinge_loss_val, match_loss_val

def init_fea(sess, model, model_fix, distance_flag):
    tmp = tf.all_variables()
    fea_matching = FEA_MATCHING(model, model_fix, distance_flag)
    fea_matching_optimizer_var = set(tf.all_variables()) - set(tmp)
    sess.run(tf.variables_initializer(fea_matching_optimizer_var))
    return fea_matching

