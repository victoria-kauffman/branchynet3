from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.functions as F
import chainer.links as L

def norm():
    return [FL(F.relu), FL(F.max_pooling_2d, 3, 2),
            FL(F.local_response_normalization,n=3, alpha=5e-05, beta=0.75)]

conv = lambda n: [L.Convolution2D(n, 32,  3, pad=1, stride=1), FL(F.relu)]
cap =  lambda n: [FL(F.max_pooling_2d, 3, 2), L.Linear(n, 10)]

PAD = 1
STRIDE = 1
KSIZE = 3
POOLING_STRIDE = 2
OUT_SIZE = 10


get_branch = lambda branch_in_channels, branch_out_channels, linear_in: [FL(F.max_pooling_2d, KSIZE, POOLING_STRIDE),
    FL(F.relu),

    L.Convolution2D(branch_in_channels, branch_out_channels, KSIZE, pad=PAD, stride=STRIDE), 

    FL(F.max_pooling_2d, KSIZE, POOLING_STRIDE),
    FL(F.relu),
    L.Linear(linear_in, 10)]

def gen_2b():
    network = [
        L.Convolution2D(3, 32, 5, pad=2, stride=1),
        FL(F.relu),
        FL(F.max_pooling_2d, 3, 2),
        FL(F.local_response_normalization,n=3, alpha=5e-05, beta=0.75),
        L.Convolution2D(32, 64,  5, pad=2, stride=1),

        Branch(get_branch(64, 96, 1536)), # Branch 1

        FL(F.relu),
        FL(F.max_pooling_2d, 3, 2),
        FL(F.local_response_normalization,n=3, alpha=5e-05, beta=0.75),
        L.Convolution2D(64, 96,  3, pad=1, stride=1),

        # Branch(get_branch(96, 96, 384)), # Branch 2

        FL(F.relu),
        L.Convolution2D(96, 96,  3, pad=1, stride=1),

        # Branch(get_branch(96, 64, 256)), # Branch 3

        FL(F.relu),
        L.Convolution2D(96, 64,  3, pad=1, stride=1),

        # Branch(get_branch(64, 64, 256)), # Branch 4

        FL(F.relu),
        FL(F.max_pooling_2d, 3, 2),
        L.Linear(1024, 256),
        FL(F.relu),
        SL(FL(F.dropout,0.5,train=True)),
        L.Linear(256, 128),
        FL(F.relu),
        SL(FL(F.dropout,0.5,train=True)),
        Branch([L.Linear(128, 10)])
    ]
    
    return network

def get_network(percentTrainKeeps=1):
    network = gen_2b()
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
