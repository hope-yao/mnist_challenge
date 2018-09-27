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





# plot out activation
import matplotlib.pyplot as plt
fea_hinge = sess.run(model_fix.h_conv1, {model_fix.x_input: x_batch})
aa = np.zeros((28*4,28*8))
for i in range(4):
    for j in range(8):
        aa[28*i:28*(i+1), 28*j:28*(j+1)] = fea_hinge[0,:,:,8*i+j]
plt.figure()
plt.imshow(aa)

# plot out filter
filter = sess.run(model.variable_conv1[0])
import matplotlib.pyplot as plt
aa = np.zeros((5*4,5*8))
for i in range(4):
    for j in range(8):
        aa[5*i:5*(i+1), 5*j:5*(j+1)] = filter[:,:,0,8*i+j]
plt.figure()
plt.imshow(aa)