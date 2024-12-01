from branchynet.links.links import *
import chainer.links as L
import chainer.functions as F
import math


# output_height = lambda input_h, padding, ksize, stride:
#     math.floor((input_h + 2 * padding - ksize) // 2) + 1

PAD = 1
STRIDE = 1
KSIZE = 3
POOLING_STRIDE = 2
OUT_SIZE = 10

get_branch = lambda branch_in_channels, branch_out_channels, out_size: [FL(F.max_pooling_2d, KSIZE, POOLING_STRIDE),
    FL(F.relu),

    L.Convolution2D(branch_in_channels, branch_out_channels, KSIZE, pad=PAD, stride=STRIDE), 

    FL(F.max_pooling_2d, KSIZE, POOLING_STRIDE),
    FL(F.relu),
    L.Linear(OUT_SIZE)]

# Add a branch to a given network at location [loc]
# We assume that the network already ends with Branch([L.Linear(in, out)])
def add_branches(network, loc):
    layer_idx = 0

    # Based on the assumed network setup of ending with branch
    out_size = network[-1][0].out_size

    # Find the convolutional layer(s)
    for i in range(len(network)):
        layer = network[i]
        # print(layer)
        if isinstance(layer, L.Convolution2D):
            layer_idx += 1
            if layer_idx in loc:
                # Insert layer?
                branch_in = layer.out_channels
                branch_out = branch_in * 2
                branch = Branch(get_branch(branch_in, branch_out, out_size))
                network.insert(i + 1, branch)
    # for l in network:
    #     print(l)

# branch_in_channels, branch_out_channels, ksize
# ksize, pooling_stride 


# round_down((branch_out_channels - ksize) / pooling_stride) + 1 = 15

# (input_h - ksize + 2 * padding) / pooling_stride + 1
# (32 - 2) / 2 + 1 = 6

# Post conv2d 1:
# (input_h - ksize + 2 * padding) / pooling_stride + 1
# 