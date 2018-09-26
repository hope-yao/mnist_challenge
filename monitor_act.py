fea_nat = sess.run(model.x_input, feed_dict=nat_dict)
fea_adv = sess.run(model.x_input, feed_dict=adv_dict)
err = []
for i in range(batch_size):
    err = np.max(fea_nat[i].flatten()-fea_adv[i].flatten())
diff_x_input = np.mean(err)

fea_nat = sess.run(model.h_conv1, feed_dict=nat_dict)
fea_adv = sess.run(model.h_conv1, feed_dict=adv_dict)
err = []
for i in range(batch_size):
    err = np.max(fea_nat[i].flatten()-fea_adv[i].flatten())
diff_h_conv1 = np.mean(err)

fea_nat = sess.run(model.h_conv2, feed_dict=nat_dict)
fea_adv = sess.run(model.h_conv2, feed_dict=adv_dict)
err = []
for i in range(batch_size):
    err = np.max(fea_nat[i].flatten()-fea_adv[i].flatten())
diff_h_conv2 = np.mean(err)

fea_nat = sess.run(model.fc1, feed_dict=nat_dict)
fea_adv = sess.run(model.fc1, feed_dict=adv_dict)
err = []
for i in range(batch_size):
    err = np.max(fea_nat[i].flatten()-fea_adv[i].flatten())
diff_fc1 = np.mean(err)

fea_nat = sess.run(model.pre_softmax, feed_dict=nat_dict)
fea_adv = sess.run(model.pre_softmax, feed_dict=adv_dict)
err = []
for i in range(batch_size):
    err = np.max(fea_nat[i].flatten()-fea_adv[i].flatten())
diff_pre_softmax = np.mean(err)

import seaborn as sns
import matplotlib.pyplot as plt
plt.plot([diff_x_input,diff_h_conv1,diff_h_conv2,diff_fc1,diff_pre_softmax], '.')
plt.axis([-0.1,4.1,0,3])