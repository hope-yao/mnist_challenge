import numpy as np
import tensorflow as tf

class FEA_MATCHING():
    def __init__(self, x_input, fea, fea_variables, distance_flag):

        self.distance_flag = distance_flag
        self.x_input = x_input
        self.fea_nat, self.fea_adv = tf.split(fea, 2)
        self.fea_nat_hinge = tf.placeholder(tf.float32, self.fea_nat.get_shape().as_list(), name='fea_nat_hinge')
        fea_nat = tf.contrib.layers.flatten(self.fea_nat)
        fea_adv = tf.contrib.layers.flatten(self.fea_adv)
        fea_nat_hinge = tf.contrib.layers.flatten(self.fea_nat_hinge)

        if distance_flag=='cosine':
            # using Cosine distance measurement
            self.match_loss = self.cos_dist(fea_nat, fea_adv)
            self.hinge_loss = self.cos_dist(fea_nat, fea_nat_hinge)
        else:
            if distance_flag == 'L_inf':
                # using L_inf distance measurement
                self.match_loss = tf.reduce_max(tf.abs(fea_adv - fea_nat_hinge), 1)
                self.hinge_loss = tf.reduce_max(tf.abs(fea_nat - fea_nat_hinge), 1)
                # fea_distance = tf.norm(self.fea_nat - self.fea_adv, ord='np.inf', axis=1)
            elif distance_flag == 'L_1':
                # using L_1 distance measurement
                self.match_loss = tf.norm(fea_adv - fea_nat_hinge, ord=1, axis=1)
                self.hinge_loss = tf.norm(fea_nat - fea_nat_hinge, ord=1, axis=1)
            elif distance_flag == 'L_2':
                # using L_2 distance measurement
                self.match_loss = tf.norm(fea_adv - fea_nat_hinge, ord=2, axis=1)
                self.hinge_loss = tf.norm(fea_nat - fea_nat_hinge, ord=2, axis=1)
            else:
                return 0
            alpha = 1.0
        self.loss = tf.reduce_mean(self.match_loss) + alpha * tf.reduce_mean(self.hinge_loss)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss,var_list=fea_variables)

    def cos_dist(self, x, y):
        # x and y are bs*fea_dim
        cos_dist = tf.reduce_sum(x* y, axis=(1, 2, 3)) \
                       / (tf.norm(x, ord='euclidean', axis=1) + 1e-8) \
                       / (tf.norm(y, ord='euclidean', axis=1) + 1e-8)
        return cos_dist

    def apply(self, sess, x_batch, x_batch_adv, nat_fea_fix):
        fea_dict = {self.x_input: np.concatenate([x_batch, x_batch_adv], 0),
                    self.fea_nat_hinge: nat_fea_fix}
        # fea_dict = {self.x_input: np.concatenate([x_batch, x_batch_adv], 0)}
        sess.run(self.train_step, fea_dict)

    def get_loss_value(self, sess, x_batch, x_batch_adv, nat_fea_fix):
        fea_dict = {self.x_input: np.concatenate([x_batch, x_batch_adv], 0),
                    self.fea_nat_hinge: nat_fea_fix}
        hinge_loss, match_loss = sess.run([self.hinge_loss, self.match_loss], fea_dict)
        return tf.reduce_mean(hinge_loss), tf.reduce_mean(match_loss)

def init_fea(sess, model, layer_idx, distance_flag):
    tmp = tf.all_variables()
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
    fea_matching = FEA_MATCHING(model.x_input, output, fea_variables, distance_flag)
    fea_matching_optimizer_var = set(tf.all_variables()) - set(tmp)
    sess.run(tf.variables_initializer(fea_matching_optimizer_var))
    return fea_matching

