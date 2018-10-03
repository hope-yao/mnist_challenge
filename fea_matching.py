import numpy as np
import tensorflow as tf


def matching_net_conv1(x):
    wider_times = 1
    # encoder
    h1 = tf.contrib.slim.conv2d(x, wider_times * 32, [3, 3])  # 28x28
    p1 = tf.contrib.slim.max_pool2d(h1, [2, 2])
    h2 = tf.contrib.slim.conv2d(p1, wider_times * 64, [3, 3])  # 14x14
    # p2 = tf.contrib.slim.max_pool2d(h2, [2, 2])
    # h3 = tf.contrib.slim.conv2d(p2, wider_times*128, [3, 3])  # 7x7
    # p3 = tf.contrib.slim.max_pool2d(h3, [2, 2])
    # f3 = tf.contrib.slim.flatten(p3)
    # h4 = tf.contrib.slim.fully_connected(f3, 128)
    # # decoder
    # h5 = tf.contrib.slim.fully_connected(h4, wider_times*128 * 3 * 3)
    # h6 = tf.reshape(h5, (100, 3, 3, wider_times*128))
    # h7 = tf.contrib.slim.conv2d(h6, wider_times*128, [3, 3])
    # p7 = tf.image.resize_images(h7, [7, 7])
    # h8 = tf.contrib.slim.conv2d(p7, wider_times*64, [3, 3])
    # p8 = tf.image.resize_images(h8, [14, 14])
    h9 = tf.contrib.slim.conv2d(h2, wider_times * 32, [3, 3])
    p9 = tf.image.resize_images(h9, [28, 28])
    h10 = tf.contrib.slim.conv2d(p9, 32, [3, 3])
    output = h10
    return output

def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def _weight_variable_zero(shape):
    initial = tf.zeros(shape)
    #initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def _bias_variable_negative(shape):
    initial = tf.constant(-1., shape=shape)
    #initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def identical_matching_net_conv1(x):
    W_conv1 = _weight_variable_zero([3, 3, 32, 32])
    b_conv1 = _bias_variable_negative([32])
    h_conv1 = tf.nn.relu(_conv2d(x, W_conv1) + b_conv1)
    # W_conv2 = self._weight_variable_zero([3, 3, 32, 64])
    # b_conv2 = self._bias_variable_negative([64])
    # h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2) + b_conv2)
    # W_conv3 = self._weight_variable_zero([3, 3, 64, 32])
    # b_conv3 = self._bias_variable_negative([32])
    # h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3) + b_conv3)
    W_conv4 = _weight_variable_zero([3, 3, 32, 32])
    b_conv4 = _bias_variable_negative([32])
    h_conv4 = tf.nn.relu(_conv2d(h_conv1, W_conv4) + b_conv4)
    return h_conv4


def identical_matching_net_conv2(x):
    W_conv1 = _weight_variable_zero([3, 3, 64, 64])
    b_conv1 = _bias_variable_negative([64])
    h_conv1 = tf.nn.relu(_conv2d(x, W_conv1) + b_conv1)
    # W_conv2 = self._weight_variable_zero([3, 3, 32, 64])
    # b_conv2 = self._bias_variable_negative([64])
    # h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2) + b_conv2)
    # W_conv3 = self._weight_variable_zero([3, 3, 64, 32])
    # b_conv3 = self._bias_variable_negative([32])
    # h_conv3 = tf.nn.relu(self._conv2d(h_conv2, W_conv3) + b_conv3)
    W_conv4 = _weight_variable_zero([3, 3, 64, 64])
    b_conv4 = _bias_variable_negative([64])
    h_conv4 = tf.nn.relu(_conv2d(h_conv1, W_conv4) + b_conv4)
    return h_conv4

class FEA_MATCHING():
    def __init__(self, model, model_fix, distance_flag, alpha=1.): #x_input, fea, fea_variables, distance_flag, alpha = 1.0):
        self.model_input = model.x_input
        self.model_fix_input = model_fix.x_input
        self.model_logits = model.pre_softmax
        self.model_fix_logits = model_fix.pre_softmax
        self.alpha = alpha
        self.distance_flag = distance_flag
        # self.x_input = model.x_input
        self.fea_nat_hinge = {}
        self.match_loss = {}
        self.hinge_loss = {}
        self.train_layer = {}

        # self.tag_list = ['conv1', 'conv2', 'fc1', 'fc2']
        # fix_fea_list = [model_fix.h_conv1, model_fix.h_conv2, model_fix.h_fc1, model_fix.pre_softmax]
        # fea_list = [model.h_conv1, model.h_conv2, model.h_fc1, model.pre_softmax]
        # fea_var_list = [model.variable_conv1, model.variable_conv2, model.variable_fc1, model.variable_fc2]
        self.tag_list = ['conv1', 'conv2']
        fix_fea_list = [model_fix.h_conv1, model_fix.h_conv2]
        fea_list = [model.h_conv1, model.h_conv2]
        fea_var_list = [model.variable_conv1, model.variable_conv2]

        self.fix_fea_dict = dict(zip(self.tag_list, fix_fea_list))
        self.fea_dict = dict(zip(self.tag_list, fea_list))
        self.fea_var_dict = dict(zip(self.tag_list, fea_var_list))
        self.fea_var_dict['conv1'] += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_fix/matching_net_conv1')
        self.fea_var_dict['conv2'] += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_fix/matching_net_conv2')

        self.label = tf.placeholder(tf.int64, shape=[50]) # batch size

        for i, tag_i in enumerate(self.tag_list):
            # clean and adversarial feature
            fea_nat, fea_adv = tf.split(self.fea_dict[tag_i], 2)
            # hinge feature
            self.fea_nat_hinge[self.tag_list[i]] = fea_nat_hinge_i = tf.placeholder(tf.float32, fea_nat.get_shape().as_list(), name='fea_nat_hinge')
            # matching network
#            with tf.variable_scope('matching_net_conv1') as scope:
#                self.fea_res = self.identical_matching_net_conv1(self.fea_dict[tag_i])
#                self.fea_var_dict[tag_i] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='matching_net_conv1')
#            fea_nat_res, fea_adv_res = tf.split(self.fea_res, 2)

            self.fea_nat_vec = fea_nat_vec = tf.contrib.layers.flatten(fea_nat)
            self.fea_adv_vec = fea_adv_vec = tf.contrib.layers.flatten(fea_adv)
            fea_nat_hinge_vec = tf.contrib.layers.flatten(fea_nat_hinge_i)
            match_loss, hinge_loss = self.get_loss(fea_nat_vec, fea_adv_vec, fea_nat_hinge_vec, distance_flag)
            self.match_loss[self.tag_list[i]] = match_loss
            self.hinge_loss[self.tag_list[i]] = hinge_loss

            logits_nat, logits_adv = tf.split(self.model_logits,2) # clean and adv logtis
            self.xent_nat, self.xent_adv = self.get_loss_xent(logits_adv, logits_nat, self.label)

            # loss = self.alpha * tf.reduce_mean(match_loss) + tf.reduce_mean(hinge_loss) #*1000
            loss = self.alpha * tf.reduce_mean(self.xent_adv) + tf.reduce_mean(self.xent_nat) #*1000
            train_layer = tf.train.AdamOptimizer(2e-5).minimize(loss,var_list=self.fea_var_dict[tag_i])
            self.train_layer[self.tag_list[i]] = train_layer

    def get_loss_xent(self, logits_adv, logits_nat, label):
        xent_adv = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=logits_adv)
        xent_nat = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=logits_nat)
        return tf.reduce_mean(xent_nat), tf.reduce_mean(xent_adv)

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

    def apply(self, sess, x_batch_nat, x_batch_adv, y_batch, tag_i):

        nat_fea_fix = sess.run(self.fix_fea_dict[tag_i], {self.model_fix_input: x_batch_nat})
        fea_dict = {self.model_input: np.concatenate([x_batch_nat, x_batch_adv], 0),
                    self.fea_nat_hinge[tag_i]: nat_fea_fix,
                    self.label: y_batch}
        for i in range(5):
            sess.run(self.train_layer[tag_i], fea_dict)

    def get_loss_value(self, sess, x_batch, x_batch_adv, y_batch):
        nat_fea_fix_l1 = {}
        hinge_loss_val = {}
        match_loss_val = {}
        for i, tag_i  in enumerate(self.tag_list):
            nat_fea_fix = sess.run(self.fix_fea_dict[tag_i], {self.model_fix_input: x_batch})
            nat_fea_fix_l1[tag_i] = np.mean(np.abs(nat_fea_fix))
            fea_dict = {self.model_input: np.concatenate([x_batch, x_batch_adv], 0),
                        self.fea_nat_hinge[tag_i]: nat_fea_fix,
                        self.label: y_batch}
            hinge_loss_val[tag_i] = sess.run(self.hinge_loss[tag_i], fea_dict)
            match_loss_val[tag_i] = sess.run(self.match_loss[tag_i], fea_dict)
        return hinge_loss_val, match_loss_val, nat_fea_fix_l1


def init_fea(sess, model, model_fix, distance_flag):
    tmp = tf.all_variables()
    fea_matching = FEA_MATCHING(model, model_fix, distance_flag)
    fea_matching_optimizer_var = set(tf.all_variables()) - set(tmp)
    sess.run(tf.variables_initializer(fea_matching_optimizer_var))
    return fea_matching

