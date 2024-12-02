from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda
import sys
import random
import dill

# Define Network

# In[3]:

from networks import alex_cifar10

branch_locs = sys.argv[1:]
print("Get network")
branch_locs = [int(i) for i in branch_locs]
print("Branch locs: ", branch_locs)

branchyNet = alex_cifar10.get_network()
print("Got them -")
# if cuda.available:
#     print("to gpu")
#     branchyNet.to_gpu()
print("Training")

branchyNet.training()
rand_it = str(random.randint(1, 9999999))
print("ID ", rand_it)

# Import Data

# In[ ]:

from datasets import pcifar10

print("Import data")
x_train,y_train,x_test,y_test = pcifar10.get_data()

# Settings

# In[5]:

TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 1
TRAIN_NUM_EPOCHS = 2


# Train Main Network

# In[6]:
# print("Training main network")
# main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=TRAIN_BATCHSIZE,
#                                              num_epoch=TRAIN_NUM_EPOCHS)

# print("Main loss: ", main_loss)
# print("Main acc: ", main_acc)
# print("Main time: ", main_time)

# Train BranchyNet

# In[7]:
print("Training branchynet network")

TRAIN_NUM_EPOCHS = 15
TRAIN_NUM_EPOCHS = 2
branch_loss, branch_acc, branch_time = utils.train(branchyNet, x_train, y_train, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)
print("Branch loss: ", branch_loss)
print("Branch acc: ", branch_acc)
print("Branch time: ", branch_time)

# branchyNet.to_cpu()
with open("_models/main_alex_cifar" + rand_it + ".bn", "wb") as f:
    dill.dump(branchyNet, f)

#set network to inference mode
print("BranchyNet.testing()")

branchyNet.testing()


# Visualizing Network Training


# In[8]:
# print("Attempt to plot")

# visualize.plot_layers(main_loss, xlabel='Epochs', ylabel='Training Loss')
# visualize.plot_layers(main_acc, xlabel='Epochs', ylabel='Training Accuracy')


# # In[9]:

# visualize.plot_layers(list(zip(*branch_loss)), xlabel='Epochs', ylabel='Training Loss')
# visualize.plot_layers(list(zip(*branch_acc)), xlabel='Epochs', ylabel='Training Accuracy')


# Run test suite and visualize

# In[11]:

print("test suite")
#set network to inference mode
branchyNet.testing()
branchyNet.verbose = False
# if cuda.available:
#     branchyNet.to_gpu()

print("utils test")

g_baseacc, g_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(y_test))) * 1000.

print("g_baseacc: ", g_baseacc)
print("g_basediff: ", g_basediff)

with open("_models/alex_cifar" + rand_it + ".bn", "wb") as f:
    dill.dump(branchyNet, f)

print("utils test 2")

# #branchyNet.to_cpu()
# c_baseacc, c_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
# c_basediff = (c_basediff / float(len(y_test))) * 1000.

# print("c base acc: ", c_baseacc)
# print("c base diff: ", c_basediff)

# # In[30]:

# # Specify thresholds
# thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 3., 5., 10.]


# # In[20]:
# print("screen branchy")

# #GPU
# # if cuda.available:
# #     branchyNet.to_gpu()
# g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
#                                                     batchsize=TEST_BATCHSIZE, verbose=True)
# # g_ts, g_accs, g_diffs, g_exits = utils.screen_leaky(leakyNet, x_test, y_test, thresholds, inc_amt=-0.1,
# #                                                     batchsize=TEST_BATCHSIZE, verbose=True)

# print("g_ts: ", g_ts)
# print("g_accs: ", g_accs)
# print("g_diffs: ", g_diffs)
# print("g_exits: ", g_exits)
# #convert to ms
# g_diffs *= 1000.

# # print("plot shit")

# # # In[ ]:

# # # In[32]:
# # # print("screen branchy 2")

# # # #CPU
# # # branchyNet.to_cpu()
# # # c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
# # #                                                      batchsize=TEST_BATCHSIZE, verbose=True)
# # # # c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, g_ts, inc_amt=0.01,
# # # #                                                      batchsize=TEST_BATCHSIZE, prescreen=False, verbose=True)
# # # #convert to ms
# # # c_diffs *= 1000.


# # # # In[22]:
# # # print("plot again")

# # # print("c_accs: ", c_accs)
# # print("c_diffs: ", c_diffs)
# # print("c_ts: ", c_ts)
# # print("c_exits: ", c_exits)
# # print("c_baseacc: ", c_baseacc)
# # print("c_basediff: ", c_basediff)

# # Save model/data

# # # In[40]:
# # print("dill dump")

# # import dill
# # branchyNet.to_cpu()
# # with open("_models/lenet_mnist.bn", "wb") as f:
# #     dill.dump(branchyNet, f)
# # # with open("_models/lenet_mnist_gpu_results.pkl", "w") as f:
# #     dill.dump({'accs': g_accs, 'rt': g_diffs, 'exits': g_exits, 'ts': g_ts, 'baseacc': g_baseacc, 'basediff': g_basediff}, f)
# # with open("_models/lenet_mnist_cpu_results.pkl", "w") as f:
# #     dill.dump({'accs': c_accs, 'rt': c_diffs, 'exits': c_exits, 'ts': c_ts, 'baseacc': c_baseacc, 'basediff': c_basediff}, f)

# # visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff, all_samples=False, inc_amt=-0.0001000,
# #                              our_label='BranchyLeNet', orig_label='LeNet', xlabel='Runtime (ms)', 
# #                              title='LeNet GPU', output_path='_figs/lenet_gpu.pdf')



# # visualize.plot_line_tradeoff(c_accs, c_diffs, c_ts, c_exits, c_baseacc, c_basediff, all_samples=False, inc_amt=-0.0001000,
# #                              our_label='BranchyLeNet', orig_label='LeNet', xlabel='Runtime (ms)',
# #                              title='LeNet CPU', output_path='_figs/lenet_cpu.pdf')


# # # In[ ]:

# # print("branchy table results")

# # #Compute table results
# # utils.branchy_table_results(c_baseacc, c_basediff, g_basediff, c_accs, c_diffs, g_accs, g_diffs, inc_amt=0.000, 
# #                           network='LeNet')



