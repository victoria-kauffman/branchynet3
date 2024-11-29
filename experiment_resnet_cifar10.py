from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda


# Define Network

# In[3]:

from networks import resnet_cifar10

print("Get network")

branchyNet = resnet_cifar10.get_network()
if cuda.available:
    print("to gpu")
    branchyNet.to_gpu()

print("Training")
branchyNet.training()


# Import Data

# In[4]:
print("Get data")

from datasets import pcifar10

x_train,y_train,x_test,y_test = pcifar10.get_data()


# Settings

# In[ ]:

TRAIN_BATCHSIZE = 64
TEST_BATCHSIZE = 1
TRAIN_NUM_EPOCHS = 100


# Train Main Network

# In[ ]:
print("Training main network")

main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)

print("Main loss: ", main_loss)
print("Main acc: ", main_acc)
print("Main time: ", main_time)
# Train BranchyNet

# In[ ]:

TRAIN_NUM_EPOCHS = 100
print("Training branchynet network")

branch_loss, branch_acc, branch_time = utils.train(branchyNet, x_train, y_train, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)
print("Branch loss: ", branch_loss)
print("Branch acc: ", branch_acc)
print("Branch time: ", branch_time)

#set network to inference mode
branchyNet.testing()

print("BranchyNet.testing()")

# Visualizing Network Training

# In[ ]:
print("Attempt to plot")
visualize.plot_layers(main_loss, xlabel='Epochs', ylabel='Training Loss')
visualize.plot_layers(main_acc, xlabel='Epochs', ylabel='Training Accuracy')


# In[ ]:

visualize.plot_layers(list(zip(*branch_loss)), xlabel='Epochs', ylabel='Training Loss')
visualize.plot_layers(list(zip(*branch_acc)), xlabel='Epochs', ylabel='Training Accuracy')


# Run test suite and visualize

# In[ ]:

print("test suite")

#set network to inference mode
branchyNet.testing()
branchyNet.verbose = False
if cuda.available:
    branchyNet.to_gpu()

print("utils test")
g_baseacc, g_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(y_test))) * 1000.

print("g_baseacc: ", g_baseacc)
print("g_basediff: ", g_basediff)

print("utils test 2")

# #branchyNet.to_cpu()
# c_baseacc, c_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
# c_basediff = (c_basediff / float(len(y_test))) * 1000.


# # In[ ]:

# # Specify thresholds
# thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 3., 5., 10.]


# In[ ]:
print("screen branchy")

#GPU
if cuda.available:
    branchyNet.to_gpu()
g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
                                                    batchsize=TEST_BATCHSIZE, verbose=True)
# g_ts, g_accs, g_diffs, g_exits = utils.screen_leaky(leakyNet, x_test, y_test, thresholds, inc_amt=-0.1,
#                                                     batchsize=TEST_BATCHSIZE, verbose=True)

print("g_ts: ", g_ts)
print("g_accs: ", g_accs)
print("g_diffs: ", g_diffs)
print("g_exits: ", g_exits)
#convert to ms
g_diffs *= 1000.


# Save model/data

# In[ ]:
print("Dill dump")
import dill
branchyNet.to_cpu()
with open("_models/resnet_cifar10.bn", "w") as f:
    dill.dump(branchyNet, f)
with open("_models/resnet_cifar10_gpu_results.pkl", "w") as f:
    dill.dump({'accs': g_accs, 'rt': g_diffs, 'exits': g_exits, 'ts': g_ts, 'baseacc': g_baseacc, 'basediff': g_basediff}, f)
with open("_models/resnet_cifar10_cpu_results.pkl", "w") as f:
    dill.dump({'accs': c_accs, 'rt': c_diffs, 'exits': c_exits, 'ts': c_ts, 'baseacc': c_baseacc, 'basediff': c_basediff}, f)



print("plot shit")


# In[ ]:

visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff, all_samples=False, inc_amt=-0.0001000,
                             our_label='BranchyResNet', orig_label='ResNet', xlabel='Runtime (ms)', 
                             title='ResNet GPU', output_path='_figs/resnet_gpu.pdf')


# In[ ]:

#CPU
branchyNet.to_cpu()
c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
                                                     batchsize=TEST_BATCHSIZE, verbose=True)
# c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, g_ts, inc_amt=0.01,
#                                                      batchsize=TEST_BATCHSIZE, prescreen=False, verbose=True)
#convert to ms
c_diffs *= 1000.


# In[ ]:

visualize.plot_line_tradeoff(c_accs, c_diffs, c_ts, c_exits, c_baseacc, c_basediff, all_samples=False, inc_amt=-0.0001000,
                             our_label='BranchyResNet', orig_label='ResNet', xlabel='Runtime (ms)',
                             title='ResNet CPU', output_path='_figs/resnet_cpu.pdf')


# In[ ]:

#Compute table results
utils.branchy_table_results(c_baseacc, c_basediff, g_basediff, c_accs, c_diffs, g_accs, g_diffs, inc_amt=0.000, 
                          network='ResNet')


