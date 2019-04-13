import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
import time
import math
import argparse
import random
import sys
import os
import matplotlib.pyplot as plt

from tensorlayer.prepro import *
from tensorlayer.layers import *
from termcolor import colored, cprint

from subspace_dense_layer import *
from transpose_dense_layer import *

from Kuhn_Munkres import KM

from time import gmtime, strftime

# Input format: inputs - [batch_size, N, C]
def kNNGConvLayer_naive_wDist_wMask(inputs, card, k, act, pos_range, channels, name = 'kNNGConvNaive'):
    
    with tf.variable_scope(name):
        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N-1, k)

        # Let's compute masks!
        # Very happy right?
        # No.

        inf_num = 10000.0
        d1_mask = tf.broadcast_to(tf.reshape(tf.cast(tf.range(N), tf.float32), [1, N]), [bs, N])
        card_bc = tf.broadcast_to(tf.reshape(card, [bs, 1]), [bs, N])
        d1_mask = tf.math.greater_equal(d1_mask, card_bc)
        out_mask = tf.cast(tf.reshape(tf.math.logical_not(d1_mask), [bs, N, 1]), tf.float32)
        dist_mask = tf.math.logical_or(tf.reshape(d1_mask, [bs, 1, N]), tf.reshape(d1_mask, [bs, N, 1])) # [bs, N, N]

        pos = inputs[:, :, :pos_range]
        drow = tf.reshape(pos, [bs, N, 1, pos_range]) # duplicate for row
        dcol = tf.reshape(pos, [bs, 1, N, pos_range]) # duplicate for column
        minusdist = -tf.sqrt(tf.reduce_sum(tf.square(drow - dcol), axis = 3))

        # Apply the mask
        minusdist = tf.where(dist_mask, tf.ones_like(minusdist) * (-inf_num), minusdist)

        _kNNEdg, _kNNIdx = tf.nn.top_k(minusdist, k + 1)
        kNNIdx = _kNNIdx[:, :, 1:] # No self-loops? (Separated branch for self-conv)
        # kNNIdx = _kNNIdx # Have self-loops?
        kNNEdg = - _kNNEdg[:, :, 1:] # Better methods?

        conv_mask = tf.cast(tf.reshape(tf.math.less(tf.cast(kNNIdx,tf.float32), tf.reshape(card, [bs, 1, 1])), [bs, N, k, 1]), tf.float32)

        # Build NxKxC Neighboor tensor
        # Create indices
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1, 1]), [bs, N, k])
        # Ns = tf.broadcast_to(tf.reshape(tf.range(N), [1, N, 1]), [bs, N, k])
        gather_indices = tf.stack([batches, kNNIdx], axis = -1)
        neighbors = tf.gather_nd(inputs, gather_indices)

        # Translate to local space
        origins = tf.reshape(inputs[:,:,:pos_range], [bs, N, 1, pos_range])
        neighbors = tf.concat(\
            [neighbors[:,:,:,:pos_range] - origins,\
             neighbors[:,:,:,pos_range:]], -1)

        # Reshape to conv
        rs_neighbors = tf.reshape(neighbors, [bs, N*k, C])
        rs_knnedg = tf.reshape(kNNEdg, [bs, N*k, 1])
        rs_neighbors = tf.concat([rs_neighbors, rs_knnedg], -1) # embed edge data in it

        # Do the convolution
        W_init = tf.truncated_normal_initializer(stddev=0.1)
        b_init = tf.constant_initializer(value=0.0)
        W_neighbor = tf.get_variable('W_neighbor', shape = [1, C + 1, channels], initializer = W_init, trainable=True)
        # b_neighbor = tf.get_variable('b_neighbor', shape = [channels], initializer = b_init)
        W_self = tf.get_variable('W_self', shape = [1, C, channels], initializer = W_init, trainable=True)
        b_kNNG = tf.get_variable('b', shape = [channels], initializer = b_init, trainable=True)

        resnbr = tf.nn.conv1d(rs_neighbors, W_neighbor, 1, padding = 'SAME')
        resnbr = tf.reshape(resnbr, [bs, N, k, channels])
        resnbr = tf.multiply(resnbr, conv_mask)
        resnbr = tf.reduce_sum(resnbr, axis = 2) # combine_method?

        resslf = tf.nn.conv1d(inputs, W_self, 1, padding = 'SAME')
        resslf = tf.multiply(resslf, out_mask)
        resnbr = resnbr + resslf

        resnbr = tf.nn.bias_add(resnbr, b_kNNG)
        resnbr = act(resnbr)
        resnbr = tf.multiply(resnbr, out_mask)

    return resnbr, [W_neighbor, W_self, b_kNNG] # [bs, N, channels]

def kNNGConvLayer_naive_wDist(inputs, card, k, act, pos_range, channels, name = 'kNNGConvNaive'):
    
    with tf.variable_scope(name):
        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N-1, k)

        pos = inputs[:, :, :pos_range]
        drow = tf.reshape(pos, [bs, N, 1, pos_range]) # duplicate for row
        dcol = tf.reshape(pos, [bs, 1, N, pos_range]) # duplicate for column
        minusdist = -tf.sqrt(tf.reduce_sum(tf.square(drow - dcol), axis = 3))
        _kNNEdg, _kNNIdx = tf.nn.top_k(minusdist, k + 1)
        kNNIdx = _kNNIdx[:, :, 1:] # No self-loops? (Separated branch for self-conv)
        # kNNIdx = _kNNIdx # Have self-loops?
        kNNEdg = - _kNNEdg[:, :, 1:] # Better methods?

        # Build NxKxC Neighboor tensor
        # Create indices
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1, 1]), [bs, N, k])
        # Ns = tf.broadcast_to(tf.reshape(tf.range(N), [1, N, 1]), [bs, N, k])
        gather_indices = tf.stack([batches, kNNIdx], axis = -1)
        neighbors = tf.gather_nd(inputs, gather_indices)

        # Translate to local space
        origins = tf.reshape(inputs[:,:,:pos_range], [bs, N, 1, pos_range])
        neighbors = tf.concat(\
            [neighbors[:,:,:,:pos_range] - origins,\
             neighbors[:,:,:,pos_range:]], -1)

        # Reshape to conv
        rs_neighbors = tf.reshape(neighbors, [bs, N*k, C])
        rs_knnedg = tf.reshape(kNNEdg, [bs, N*k, 1])
        rs_neighbors = tf.concat([rs_neighbors, rs_knnedg], -1) # embed edge data in it

        # Do the convolution
        W_init = tf.truncated_normal_initializer(stddev=0.1)
        b_init = tf.constant_initializer(value=0.0)
        W_neighbor = tf.get_variable('W_neighbor', shape = [1, C + 1, channels], initializer = W_init, trainable=True)
        # b_neighbor = tf.get_variable('b_neighbor', shape = [channels], initializer = b_init)
        W_self = tf.get_variable('W_self', shape = [1, C, channels], initializer = W_init, trainable=True)
        b_kNNG = tf.get_variable('b', shape = [channels], initializer = b_init, trainable=True)

        resnbr = tf.nn.conv1d(rs_neighbors, W_neighbor, 1, padding = 'SAME')
        resnbr = tf.reshape(resnbr, [bs, N, k, channels])
        resnbr = tf.reduce_sum(resnbr, axis = 2) # combine_method?

        resslf = tf.nn.conv1d(inputs, W_self, 1, padding = 'SAME')
        resnbr = resnbr + resslf

        resnbr = tf.nn.bias_add(resnbr, b_kNNG)
        resnbr = act(resnbr)

    return resnbr, [W_neighbor, W_self, b_kNNG] # [bs, N, channels]

# Inputs: [bs, N, C]
def kNNG_gen(inputs, k, pos_range, name = 'kNNG_gen'):

    with tf.variable_scope(name):
        
        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N-1, k)

        pos = inputs[:, :, :pos_range]
        drow = tf.reshape(pos, [bs, N, 1, pos_range]) # duplicate for row
        dcol = tf.reshape(pos, [bs, 1, N, pos_range]) # duplicate for column
        
        local_pos = drow - dcol #[bs, N, N, 3]
        minusdist = -tf.sqrt(tf.reduce_sum(tf.square(local_pos), axis = 3))

        _kNNEdg, _TopKIdx = tf.nn.top_k(minusdist, k + 1)
        TopKIdx = _TopKIdx[:, :, 1:] # No self-loops? (Separated branch for self-conv)
        # TopKIdx = _TopKIdx # Have self-loops?
        kNNEdg = -_kNNEdg[:, :, 1:] # Better methods?
        kNNEdg = tf.stop_gradient(kNNEdg) # Don't flow gradients here to avoid nans generated for unselected edges
        kNNEdg = tf.reshape(kNNEdg, [bs, N, k, 1])

        # Build NxKxC Neighboor tensor
        # Create indices
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1, 1]), [bs, N, k])
        kNNIdx = tf.stack([batches, TopKIdx], axis = -1)
        
        Ns = tf.broadcast_to(tf.reshape(tf.range(N), [1, N, 1]), [bs, N, k])
        gather_lpos_indices = tf.stack([batches, Ns, TopKIdx], axis = -1)

        # [x, y, z], 1st order moment
        neighbor_pos = tf.gather_nd(local_pos, gather_lpos_indices) # [bs, N, k, 3]

        # [xx, xy, xz, yx, yy, yz, zx, zy, zz], 2nd order moment
        neighbor_pos_rs = tf.reshape(neighbor_pos, [bs, N, k, 3, 1])
        neighbor_quadratic = tf.reshape(tf.multiply(neighbor_pos_rs, tf.transpose(neighbor_pos_rs, perm = [0, 1, 2, 4, 3])), [bs, N, k, 9])

        kNNEdg = tf.concat([kNNEdg, neighbor_pos, neighbor_quadratic], axis = -1) # [bs, N, k, eC]

        return pos, kNNIdx, kNNEdg

def Laplacian(bs, N, k, kNNIdx, name = 'kNNG_laplacian'):

    with tf.variable_scope(name):
        
        Ns = tf.broadcast_to(tf.reshape(tf.range(N), [1, N, 1]), [bs, N, k])
        _ustack = tf.unstack(kNNIdx, axis = -1) # list of [bs, N, k]
        kNNIdx_withN = tf.stack([_ustack[0], Ns, _ustack[1]], axis = -1) # [bs, N, k, 3], containing [#batch, #start, #end] with #start #end in [0, N)

        # Translate a directed graph to undirected graph by removing the direction of edges, in order to obtain a real symmtric matrix L.
        A = tf.scatter_nd(kNNIdx_withN, tf.constant(True, shape = [bs, N, k]), [bs, N, N], name = 'A')
        print(A.shape)
        A_T = tf.transpose(A, [0, 2, 1])
        A_undirected = tf.math.logical_or(A, A_T)
        A = tf.cast(A_undirected, tf.float32, name = 'A_undirected') # [bs, N, N]
        print(A.shape)

        D = tf.matrix_set_diag(tf.zeros([bs, N, N], tf.float32), tf.reduce_sum(A, axis = -1)) # [bs, N] -> [bs, N, N]
        print(D.shape)
        L = D - A
        print(L.shape)

        # Normalizations for the laplacian?

        return L, A, D

# Inputs: [bs, N, C]
# Builds edges X -> Y
def bip_kNNG_gen(Xs, Ys, k, pos_range, name = 'kNNG_gen'):

    with tf.variable_scope(name):
        
        bs = Xs.shape[0]
        Nx = Xs.shape[1]
        Ny = Ys.shape[1]
        Cx = Xs.shape[2]
        Cy = Ys.shape[2]
        k = min(Ny, k)

        posX = Xs[:, :, :pos_range]
        posY = Ys[:, :, :pos_range]
        drow = tf.reshape(posX, [bs, Nx, 1, pos_range]) # duplicate for row
        dcol = tf.reshape(posY, [bs, 1, Ny, pos_range]) # duplicate for column
        
        local_pos = drow - dcol #[bs, Nx, Ny, 3]
        minusdist = -tf.sqrt(tf.reduce_sum(tf.square(local_pos), axis = 3))

        _kNNEdg, _TopKIdx = tf.nn.top_k(minusdist, k)
        TopKIdx = _TopKIdx[:, :, :] # No self-loops? (Separated branch for self-conv)
        # TopKIdx = _TopKIdx # Have self-loops?
        kNNEdg = -_kNNEdg[:, :, :] # Better methods?
        kNNEdg = tf.stop_gradient(kNNEdg) # Don't flow gradients here to avoid nans generated for unselected edges
        kNNEdg = tf.reshape(kNNEdg, [bs, Nx, k, 1])

        # Build NxKxC Neighboor tensor
        # Create indices
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1, 1]), [bs, Nx, k])
        kNNIdx = tf.stack([batches, TopKIdx], axis = -1)
        
        Ns = tf.broadcast_to(tf.reshape(tf.range(Nx), [1, Nx, 1]), [bs, Nx, k])
        gather_lpos_indices = tf.stack([batches, Ns, TopKIdx], axis = -1)

        # [x, y, z], 1st order moment
        neighbor_pos = tf.gather_nd(local_pos, gather_lpos_indices) # [bs, Nx, k, 3]

        # [xx, xy, xz, yx, yy, yz, zx, zy, zz], 2nd order moment
        neighbor_pos_rs = tf.reshape(neighbor_pos, [bs, Nx, k, 3, 1])
        neighbor_quadratic = tf.reshape(tf.multiply(neighbor_pos_rs, tf.transpose(neighbor_pos_rs, perm = [0, 1, 2, 4, 3])), [bs, Nx, k, 9])

        kNNEdg = tf.concat([kNNEdg, neighbor_pos, neighbor_quadratic], axis = -1) # [bs, Nx, k, eC]

        return posX, posY, kNNIdx, kNNEdg

# Inputs: [bs, Nx, Cx] [bs, Ny, Cy]
# kNNIdx: [bs, Nx, k]
# kNNEdg: [bs, Nx, k, eC]
# Edges are X -> Y
def bip_kNNGConvLayer_edgeMask(Xs, Ys, kNNIdx, kNNEdg, act, channels, no_act_final = False, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGConvNaive'):
    
    with tf.variable_scope(name):

        bs = Xs.shape[0]
        Nx = Xs.shape[1]
        Ny = Ys.shape[1]
        Cx = Xs.shape[2]
        Cy = Ys.shape[2]
        k = kNNIdx.shape[2]
        eC = kNNEdg.shape[3]

        neighbors = tf.gather_nd(Ys, kNNIdx)
        # neighbors: Edge u-v = [u;v;edg]
        neighbors = tf.concat([neighbors, tf.broadcast_to(tf.reshape(Xs, [bs, Nx, 1, Cx]), [bs, Nx, k, Cx]), kNNEdg], axis = -1) # [bs, Nx, Cx+Cy+eC]

        # Reshape to conv
        rs_neighbors = tf.reshape(neighbors, [bs, Nx*k, Cx+Cy+eC])
        rs_knnedg = tf.reshape(kNNEdg, [bs, Nx*k, eC])
        # rs_neighbors = tf.concat([rs_neighbors, rs_knnedg], -1) # embed edge data in it

        ### Do the convolution ###
        # TODO: MLP instead of 1x fc?

        # Collect neightbors ("M" stage)
        W_neighbor = tf.get_variable('W_neighbor', shape = [1, Cx+Cy+eC, channels], initializer = W_init, trainable=True)
        b_neighbor = tf.get_variable('b_neighbor', shape = [channels], initializer = b_init, trainable=True)

        resnbr = tf.nn.conv1d(rs_neighbors, W_neighbor, 1, padding = 'SAME')
        resnbr = tf.nn.bias_add(resnbr, b_neighbor)
        resnbr = act(resnbr)
        # resnbr = tf.reshape(resnbr, [bs, Nx, k, channels])

        # Collect edge masks
        W_edges = tf.get_variable("W_edges", shape = [1, eC, channels], initializer = W_init, trainable=True)
        b_edges = tf.get_variable("b_edges", shape = [channels], initializer = b_init, trainable=True)

        resedg = tf.nn.conv1d(rs_knnedg, W_edges, 1, padding = 'SAME')
        resedg = tf.nn.bias_add(resedg, b_edges)
        resedg = act(resedg)
        # resedg = tf.nn.softmax(resedg, axis = -1)
        # resedg = tf.reshape(resedg, [bs, Nx, k, channels])

        # resnbr = tf.multiply(resnbr, resedg)
        resnbr = tf.concat([resnbr, resedg], axis = -1)
        W_nb2 = tf.get_variable('W_neighbor2', shape = [1, channels*2, channels], initializer = W_init, trainable=True)
        b_nb2 = tf.get_variable('b_neighbor2', shape = [channels], initializer = b_init, trainable=True)
        resnbr = tf.nn.conv1d(resnbr, W_nb2, 1, padding = 'SAME')
        resnbr = tf.nn.bias_add(resnbr, b_nb2)
        resnbr = act(resnbr)

        resnbr = tf.reshape(resnbr, [bs, Nx, k, channels])
        resnbr = tf.reduce_sum(resnbr, axis = 2) # combine_method?

        W_self = tf.get_variable('W_self', shape = [1, Cx + channels, channels], initializer = W_init, trainable=True)
        b_self = tf.get_variable('b', shape = [channels], initializer = b_init, trainable=True)
        res    = tf.nn.conv1d(tf.concat([Xs, resnbr], axis = -1), W_self, 1, padding = 'SAME')
        
        res = tf.nn.bias_add(res, b_self)

        if not no_act_final:
            res = act(res)

    return resnbr, [W_neighbor, b_neighbor, W_edges, b_edges, W_self, b_self] # [bs, Nx, channels]

# Inputs: [bs, N, C]
# kNNIdx: [bs, N, k]
# kNNEdg: [bs, N, k, eC]
def kNNGConvLayer_edgeMask(inputs, kNNIdx, kNNEdg, act, channels, no_act_final = False, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGConvNaive'):
    
    with tf.variable_scope(name):

        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = kNNIdx.shape[2]
        eC = kNNEdg.shape[3]

        neighbors = tf.gather_nd(inputs, kNNIdx)
        # neighbors: Edge u-v = [u;v;edg]
        neighbors = tf.concat([neighbors, tf.broadcast_to(tf.reshape(inputs, [bs, N, 1, C]), [bs, N, k, C]), kNNEdg], axis = -1)

        # Reshape to conv
        rs_neighbors = tf.reshape(neighbors, [bs, N*k, 2*C+eC])
        rs_knnedg = tf.reshape(kNNEdg, [bs, N*k, eC])
        # rs_neighbors = tf.concat([rs_neighbors, rs_knnedg], -1) # embed edge data in it

        ### Do the convolution ###
        # TODO: MLP instead of 1x fc?

        # Collect neightbors ("M" stage)
        W_neighbor = tf.get_variable('W_neighbor', shape = [1, 2*C+eC, channels], initializer = W_init, trainable=True)
        b_neighbor = tf.get_variable('b_neighbor', shape = [channels], initializer = b_init, trainable=True)

        resnbr = tf.nn.conv1d(rs_neighbors, W_neighbor, 1, padding = 'SAME')
        resnbr = tf.nn.bias_add(resnbr, b_neighbor)
        resnbr = act(resnbr)
        resnbr = tf.reshape(resnbr, [bs, N, k, channels])

        # Collect edge masks
        W_edges = tf.get_variable("W_edges", shape = [1, eC, channels], initializer = W_init, trainable=True)
        b_edges = tf.get_variable("b_edges", shape = [channels], initializer = b_init, trainable=True)

        resedg = tf.nn.conv1d(rs_knnedg, W_edges, 1, padding = 'SAME')
        resedg = tf.nn.bias_add(resedg, b_edges)
        resedg = tf.nn.softmax(resedg, axis = -1)
        resedg = tf.reshape(resedg, [bs, N, k, channels])

        resnbr = tf.multiply(resnbr, resedg)
        resnbr = tf.reduce_sum(resnbr, axis = 2) # combine_method?

        W_self = tf.get_variable('W_self', shape = [1, C + channels, channels], initializer = W_init, trainable=True)
        b_self = tf.get_variable('b', shape = [channels], initializer = b_init, trainable=True)
        res    = tf.nn.conv1d(tf.concat([inputs, resnbr], axis = -1), W_self, 1, padding = 'SAME')
        
        res = tf.nn.bias_add(res, b_self)

        if not no_act_final:
            res = act(res)

    return resnbr, [W_neighbor, b_neighbor, W_edges, b_edges, W_self, b_self] # [bs, N, channels]

# Inputs: [bs, N, C]
# kNNIdx: [bs, N, k]
# kNNEdg: [bs, N, k, eC]
def kNNGConvLayer_concat(inputs, kNNIdx, kNNEdg, act, channels, no_act_final = False, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGConvNaive'):
    
    with tf.variable_scope(name):

        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = kNNIdx.shape[2]
        eC = kNNEdg.shape[3]

        neighbors = tf.gather_nd(inputs, kNNIdx)
        neighbors = tf.concat([neighbors, tf.broadcast_to(tf.reshape(inputs, [bs, N, 1, C]), [bs, N, k, C]), kNNEdg], axis = -1)

        # Reshape to conv
        rs_neighbors = tf.reshape(neighbors, [bs, N*k, 2*C+eC])

        ### Do the convolution ###
        # TODO: MLP instead of 1x fc?

        # Collect neightbors ("M" stage)
        W_neighbor = tf.get_variable('W_neighbor', shape = [1, 2*C+eC, channels], initializer = W_init, trainable=True)
        b_neighbor = tf.get_variable('b_neighbor', shape = [channels], initializer = b_init, trainable=True)

        resnbr = tf.nn.conv1d(rs_neighbors, W_neighbor, 1, padding = 'SAME')
        resnbr = tf.nn.bias_add(resnbr, b_neighbor)
        resnbr = act(resnbr)
        resnbr = tf.reshape(resnbr, [bs, N, k, channels])

        resnbr = tf.reduce_sum(resnbr, axis = 2) # combine_method?

        W_self = tf.get_variable('W_self', shape = [1, C + channels, channels], initializer = W_init, trainable=True)
        b_self = tf.get_variable('b', shape = [channels], initializer = b_init, trainable=True)
        res    = tf.nn.conv1d(tf.concat([inputs, resnbr], axis = -1), W_self, 1, padding = 'SAME')
        
        res = tf.nn.bias_add(res, b_self)

        if not no_act_final:
            res = act(res)

    return resnbr, [W_neighbor, b_neighbor, W_self, b_self] # [bs, N, channels]

# Inputs: [bs, N, C]
#    Pos: [bs, N, 3]
def kNNGPooling_GUnet(inputs, pos, k, masking = True, channels = 1, W_init = tf.truncated_normal_initializer(stddev=0.1), name = 'kNNGPool'):

    with tf.variable_scope(name):

        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N, k)

        W = tf.get_variable('W', shape = [1, C, channels], initializer=W_init, trainable=True)
        norm = tf.sqrt(tf.reduce_sum(tf.square(W), axis = 1, keepdims = True)) # [1, 1, channels]
        
        y = tf.nn.conv1d(inputs, W, 1, padding = 'SAME') # [bs, C, channels]
        y = tf.multiply(y, 1.0 / norm)
        y = tf.reduce_mean(y, axis = -1) # [bs, C]
        val, idx = tf.nn.top_k(y, k) # [bs, k]

        # Pick them
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, k])
        gather_idx = tf.stack([batches, idx], axis = -1)
        pool_features = tf.gather_nd(inputs, gather_idx) # [bs, k, C]
        pool_position = tf.gather_nd(pos, gather_idx) # [bs, k, 3]

        if masking == True:
            pool_features = tf.multiply(pool_features, tf.reshape(tf.nn.tanh(val), [bs, k, 1]))
    
    return pool_position, pool_features, [W]

# Inputs: [bs, N, C]
#    Pos: [bs, N, 3]
def kNNGPooling_HighFreqLoss_GUnet(inputs, pos, k, laplacian, masking = True, channels = 1, W_init = tf.truncated_normal_initializer(stddev=0.1), name = 'kNNGPool'):

    with tf.variable_scope(name):

        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N, k)

        # inputs = tf.stop_gradient(inputs)

        W = tf.get_variable('W', shape = [1, C, channels], initializer=W_init, trainable=True)
        norm = tf.sqrt(tf.reduce_sum(tf.square(W), axis = 1, keepdims = True)) # [1, 1, channels]
        
        y = tf.nn.conv1d(inputs, W, 1, padding = 'SAME') # [bs, N, channels]
        y = tf.multiply(y, 1.0 / norm)
        y = tf.reduce_mean(y, axis = -1) # [bs, N]

        # Freq Loss
        print(laplacian.shape)
        norm_Ly = tf.sqrt(tf.reduce_sum(tf.square(tf.matmul(laplacian, tf.reshape(y, [bs, N, 1]), name = 'L_y')), axis = [1, 2]))
        norm_y = tf.sqrt(tf.reduce_sum(tf.square(y), axis = 1))
        freq_loss = norm_Ly / norm_y # Maximize this
        freq_loss = 0 - freq_loss # Minimize negate

        val, idx = tf.nn.top_k(y, k) # [bs, k]

        # Pick them
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, k])
        gather_idx = tf.stack([batches, idx], axis = -1)
        pool_features = tf.gather_nd(inputs, gather_idx) # [bs, k, C]
        pool_position = tf.gather_nd(pos, gather_idx) # [bs, k, 3]

        if masking == True:
            pool_features = tf.multiply(pool_features, tf.reshape(tf.nn.tanh(val), [bs, k, 1]))
    
    return pool_position, pool_features, [W], freq_loss

def Conv1dWrapper(inputs, filters, kernel_size, stride, padding, act, W_init, b_init, bias = True, name = 'conv'):

    with tf.variable_scope(name):

        N = inputs.shape[1]
        C = inputs.shape[2]
        variables = []

        W = tf.get_variable('W', shape = [kernel_size, C, filters], initializer=W_init, trainable=True)
        variables.append(W)

        y = tf.nn.conv1d(inputs, W, stride, padding = padding)

        if bias == True:
            b = tf.get_variable('b', shape = [filters], initializer=b_init, trainable=True)
            y = tf.nn.bias_add(y, b)
            variables.append(b)

        if act is not None:
            y = act(y)

        return y, variables

def kNNGPosition_refine(input_position, input_feature, act, hidden = 128, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGPosRefine'):

    with tf.variable_scope(name):

        bs = input_position.shape[0]
        N = input_position.shape[1]
        C = input_feature.shape[2]
        pC = input_position.shape[2]

        assert N == input_feature.shape[1] and bs == input_feature.shape[0]

        pos_feature, vars1 = Conv1dWrapper(tf.concat([input_position, input_feature], axis = -1), hidden, 1, 1, 'SAME', act, W_init, b_init, True, 'hidden')
        pos_res, vars2 = Conv1dWrapper(pos_feature, pC, 1, 1, 'SAME', None, W_init, b_init, True, 'refine')

        tf.summary.histogram('Position_Refine_%s' % name, pos_res)

        refined_pos = tf.add(input_position, pos_res)

        return refined_pos, [vars1, vars2]

def bip_kNNGConvBN_wrapper(Xs, Ys, kNNIdx, kNNEdg, batch_size, gridMaxSize, particle_hidden_dim, act, decay = 0.999, is_train = True, name = 'gconv', W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0)):
    
    with tf.variable_scope(name):
        # n, v = kNNGConvLayer_concat(inputs, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        n, v = bip_kNNGConvLayer_edgeMask(Xs, Ys, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        n = InputLayer(tf.reshape(n, [batch_size * gridMaxSize, particle_hidden_dim]), name = 'bn/input')
        n = BatchNormLayer(n, decay = 0.999, act = act, is_train = is_train, name = 'bn').outputs
        n = tf.reshape(n, [batch_size, gridMaxSize, particle_hidden_dim])

    return n, v

def kNNGConvBN_wrapper(inputs, kNNIdx, kNNEdg, batch_size, gridMaxSize, particle_hidden_dim, act, decay = 0.999, is_train = True, name = 'gconv', W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0)):
    
    with tf.variable_scope(name):
        # n, v = kNNGConvLayer_concat(inputs, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        # n, v = kNNGConvLayer_edgeMask(inputs, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        n, v = bip_kNNGConvLayer_edgeMask(inputs, inputs, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        n = InputLayer(tf.reshape(n, [batch_size * gridMaxSize, particle_hidden_dim]), name = 'bn/input')
        n = BatchNormLayer(n, decay = 0.999, act = act, is_train = is_train, name = 'bn').outputs
        n = tf.reshape(n, [batch_size, gridMaxSize, particle_hidden_dim])

    return n, v
        
# TODO: position re-fine layer

class model_particles:

    def __init__(self, gridMaxSize, latent_dim, batch_size, optimizer):
        
        # Size of each grid
        self.gridMaxSize = gridMaxSize
        self.particle_latent_dim = latent_dim
        self.particle_hidden_dim = 64
        self.cluster_feature_dim = 128
        self.cluster_count = 128
        # self.latent_dim = latent_dim
        self.combine_method = tf.reduce_sum
        self.loss_func = tf.abs
        self.resSize = 1
        self.batch_size = batch_size
        self.knn_k = 16

        # self.act = (lambda x: 0.8518565165255 * tf.exp(-2 * tf.pow(x, 2)) - 1) # normalization constant c = (sqrt(2)*pi^(3/2)) / 3, 0.8518565165255 = c * sqrt(5).
        self.act = tf.nn.elu
        self.convact = tf.nn.elu
        # self.act = tf.nn.relu

        self.encoder_arch = 'plain' # plain, plain_noNorm, plain_shallow, attractor, attractor_attention, attractor_affine
        self.decoder_arch = 'plain' # plain, advanced_score, distribution_weight, distribution_conditional, attractor

        self.wdev=0.1

        # self.initial_grid_size = 6.0 # TODO: make this larger? (Done in dataLoad)
        # self.total_world_size = 96.0
        self.loss_metric = 'chamfer' # or 'earthmover'

        self.ph_X = tf.placeholder('float32', [self.batch_size, self.gridMaxSize, 7]) # x y z vx vy vz 1
        self.ph_card = tf.placeholder('float32', [self.batch_size]) # card
        self.ph_max_length = tf.placeholder('int32', [2])

        self.optimizer = optimizer

    # 1 of a batch goes in this function at once.
    def particleEncoder(self, input_particle, output_dim, is_train = False, reuse = False, returnPool = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleEncoder", reuse = reuse) as vs:

            if self.encoder_arch == 'RANDOM':

                # input_particle : [batch_size, gridMaxSize, 7]

                # n = InputLayer(tf.reshape(input_particle, [self.batch_size * self.gridMaxSize, 7]), name = 'input')
                n = tf.random.normal([self.batch_size, output_dim])

                return n, []

            if self.encoder_arch == 'plain_ln':

                # input_particle : [batch_size, gridMaxSize, 7]

                # n = InputLayer(tf.reshape(input_particle, [self.batch_size * self.gridMaxSize, 7]), name = 'input')
                n = InputLayer(input_particle, name = 'input')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv1', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv1/ln')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv2', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv2/ln')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv3', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv3/ln')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv4', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv4/ln')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv5', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv5/ln')
                n = Conv1d(n, n_filter = output_dim, filter_size = 1, stride = 1, act = None, name = 'conv6', W_init = w_init)

                n = self.combine_method(n.outputs, axis = 1)

                return n, []

            if self.encoder_arch == 'plain_bn':

                # input_particle : [batch_size, gridMaxSize, 7]

                n = InputLayer(tf.reshape(input_particle, [self.batch_size * self.gridMaxSize, 7]), name = 'input')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc1', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc1/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc2', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc2/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc3', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc3/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc4', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc4/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc5', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc5/bn')
                n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc6', W_init = w_init)

                n = tf.reshape(n.outputs, [self.batch_size, self.gridMaxSize, output_dim])
                n = self.combine_method(n, axis = 1)

                return n, []

            if self.encoder_arch == 'plain_mask_bn':

                # input_particle : [batch_size, gridMaxSize, 7]

                n = InputLayer(tf.reshape(input_particle, [self.batch_size * self.gridMaxSize, 7]), name = 'input')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc1', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc1/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc2', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc2/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc3', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc3/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc4', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc4/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc5', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc5/bn')
                n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc6', W_init = w_init)

                # Compute mask
                mask = InputLayer(tf.reshape(input_particle[:, :, 0:3], [self.batch_size * self.gridMaxSize, 3]), name = 'posmask/input')
                mask = DenseLayer(mask, n_units = 256, act = self.act, name = 'mask/fc1', W_init = w_init)
                mask = DenseLayer(mask, n_units = output_dim, act = None, name = 'mask/fc2', W_init = w_init)
                mask = tf.nn.softmax(mask.outputs, axis = -1)
                n = tf.multiply(n.outputs, mask)

                n = tf.reshape(n, [self.batch_size, self.gridMaxSize, output_dim])
                n = self.combine_method(n, axis = 1)

                return n, []
            
            if self.encoder_arch == 'graph_naive_mixed_ln':

                g1_pos, g1_idx, g1_edg = kNNG_gen(input_particle, self.knn_k, 3, name = 'ggen1')

                n = InputLayer(input_particle, name = 'input')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv1', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv1/ln')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv2', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv2/ln')

                n, vars1 = kNNGConvLayer_edgeMask(n.outputs, g1_idx, g1_edg, self.act, no_act_final = True, channels = self.particle_hidden_dim, W_init = w_init, b_init = b_init, name = 'gconv1')
                n = InputLayer(n, name = 'gconv1/ln_input')
                n = LayerNormLayer(n, act = self.act, name = 'gconv1/ln')
                n, vars2 = kNNGConvLayer_edgeMask(n.outputs, g1_idx, g1_edg, self.act, no_act_final = True, channels = self.particle_hidden_dim, W_init = w_init, b_init = b_init, name = 'gconv2')
                n = InputLayer(n, name = 'gconv2/ln_input')
                n = LayerNormLayer(n, act = self.act, name = 'gconv2/ln')

                # n, w1 = kNNGConvLayer_naive_wDist(midstage, self.ph_card, self.knn_k, act = self.act, pos_range = 3, channels = self.particle_hidden_dim, name = 'gconv1')
                # n, w2 = kNNGConvLayer_naive(n, self.ph_card, self.knn_k, act = self.act, pos_range = 3, channels = self.particle_hidden_dim, name = 'gconv2')

                # midstage2 = tf.concat([input_particle[:, :, 5:6], n], 2)
                # n = InputLayer(midstage2, name = 'midstage_input')
                # n = InputLayer(n, name = 'midstage_input')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv3', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv3/ln')
                n = Conv1d(n, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'conv4', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv4/ln')
                n = Conv1d(n, n_filter = output_dim, filter_size = 1, stride = 1, act = None, name = 'conv5', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'conv5/ln')

                n = n.outputs
                n = self.combine_method(n, axis = 1)

                return n, [vars1, vars2]
                # return n, [w1, w2]
            
            if self.encoder_arch == 'graph_naive_mixed_bn':

                g1_pos, g1_idx, g1_edg = kNNG_gen(input_particle, self.knn_k, 3, name = 'ggen1')
                
                n = InputLayer(tf.reshape(input_particle, [self.batch_size * self.gridMaxSize, 7]), name = 'input')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc1', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc1/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc2', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc2/bn')

                n = tf.reshape(n.outputs, [self.batch_size, self.gridMaxSize, self.particle_hidden_dim])
                
                n, vars1 = kNNGConvLayer_edgeMask(n, g1_idx, g1_edg, self.act, channels = self.particle_hidden_dim, W_init = w_init, b_init = b_init, name = 'gconv1')
                n, vars2 = kNNGConvLayer_edgeMask(n, g1_idx, g1_edg, self.act, channels = self.particle_hidden_dim, W_init = w_init, b_init = b_init, name = 'gconv2')
                # n, w2 = kNNGConvLayer_naive(n, self.ph_card, self.knn_k, act = self.act, pos_range = 3, channels = self.particle_hidden_dim, name = 'gconv2')

                # midstage2 = tf.concat([input_particle[:, :, 5:6], n], 2)
                n = InputLayer(tf.reshape(n, [self.batch_size * self.gridMaxSize, self.particle_hidden_dim]), name = 'midstage_input')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc3', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc3/bn')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc4', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc4/bn')
                n = DenseLayer(n, n_units = output_dim, act = None, name = 'fc5', W_init = w_init)
                n = BatchNormLayer(n, decay = 0.999, act = self.act, is_train = is_train, name = 'fc5/bn')

                # Compute mask
                mask = InputLayer(tf.reshape(input_particle[:, :, 0:3], [self.batch_size * self.gridMaxSize, 3]), name = 'posmask/input')
                mask = DenseLayer(mask, n_units = 256, act = self.act, name = 'mask/fc1', W_init = w_init)
                mask = DenseLayer(mask, n_units = output_dim, act = None, name = 'mask/fc2', W_init = w_init)
                mask = tf.nn.softmax(mask.outputs, axis = -1)
                n = tf.multiply(n.outputs, mask)

                n = tf.reshape(n, [self.batch_size, self.gridMaxSize, output_dim])
                n = self.combine_method(n, axis = 1)

                return n, [vars1, vars2]
                # return n, [w1, w2]

            if self.encoder_arch == 'graph_pure_bn':

                g1_pos, g1_idx, g1_edg = kNNG_gen(input_particle, self.knn_k, 3, name = 'ggen1')
                particle_features = input_particle[:, :, 0:]

                n, vars1 = kNNGConvBN_wrapper(particle_features, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv1')
                n, vars2 = kNNGConvBN_wrapper(n, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv2')
                n, vars3 = kNNGConvBN_wrapper(n, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv3')
                n, vars4 = kNNGConvBN_wrapper(n, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv4')
                n, vars5 = kNNGConvBN_wrapper(n, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, output_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv5')

                n = tf.reshape(n, [self.batch_size * self.gridMaxSize, output_dim])

                # Compute mask
                mask = InputLayer(tf.reshape(input_particle[:, :, 0:3], [self.batch_size * self.gridMaxSize, 3]), name = 'posmask/input')
                mask = DenseLayer(mask, n_units = 256, act = self.act, name = 'mask/fc1', W_init = w_init)
                mask = DenseLayer(mask, n_units = output_dim, act = None, name = 'mask/fc2', W_init = w_init)
                mask = tf.nn.softmax(mask.outputs, axis = -1)
                n = tf.multiply(n, mask)

                n = tf.reshape(n, [self.batch_size, self.gridMaxSize, output_dim])
                n = self.combine_method(n, axis = 1)

                return n, [vars1, vars2, vars3, vars4, vars5]

            if self.encoder_arch == 'full_graph_pool_bn':

                g1_pos, g1_idx, g1_edg = kNNG_gen(input_particle, self.knn_k, 3, name = 'ggen1')
                particle_features = input_particle[:, :, 6:]

                n, vars1 = kNNGConvBN_wrapper(particle_features, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim // 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv1')
                n, vars2 = kNNGConvBN_wrapper(n, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim // 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv2')

                g2_pos, n, vars3 = kNNGPooling_GUnet(n, g1_pos, 1280, W_init = w_init, name = 'gpool1')
                _, g2_idx, g2_edg = kNNG_gen(g2_pos, self.knn_k, 3, name = 'ggen2')

                n, vars4 = kNNGConvBN_wrapper(n, g2_idx, g2_edg, self.batch_size, 1280, self.particle_hidden_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv3')
                n, vars5 = kNNGConvBN_wrapper(n, g2_idx, g2_edg, self.batch_size, 1280, self.particle_hidden_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv4')

                g3_pos, n, vars6 = kNNGPooling_GUnet(n, g2_pos, 512, W_init = w_init, name = 'gpool2')
                _, g3_idx, g3_edg = kNNG_gen(g3_pos, self.knn_k, 3, name = 'ggen3')

                n, vars7 = kNNGConvBN_wrapper(n, g3_idx, g3_edg, self.batch_size, 512, self.particle_hidden_dim * 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv5')
                n, vars8 = kNNGConvBN_wrapper(n, g3_idx, g3_edg, self.batch_size, 512, self.particle_hidden_dim * 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv6')
                n, vars9 = kNNGConvBN_wrapper(n, g3_idx, g3_edg, self.batch_size, 512, self.particle_hidden_dim * 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv7')
                n, varsA = kNNGConvBN_wrapper(n, g3_idx, g3_edg, self.batch_size, 512, self.particle_hidden_dim * 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv8')

                g4_pos, n, varsB = kNNGPooling_GUnet(n, g3_pos, 256, W_init = w_init, name = 'gpool3')
                _, g4_idx, g4_edg = kNNG_gen(g4_pos, self.knn_k, 3, name = 'ggen4')

                n, varsC = kNNGConvBN_wrapper(n, g4_idx, g4_edg, self.batch_size, 256, output_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv9')
                n, varsD = kNNGConvBN_wrapper(n, g4_idx, g4_edg, self.batch_size, 256, output_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv10')
                n, varsE = kNNGConvBN_wrapper(n, g4_idx, g4_edg, self.batch_size, 256, output_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv11')
                n, varsF = kNNGConvBN_wrapper(n, g4_idx, g4_edg, self.batch_size, 256, output_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv12')

                # Compute mask
                mask = InputLayer(tf.reshape(g4_pos, [self.batch_size * 256, 3]), name = 'posmask/input')
                mask = DenseLayer(mask, n_units = 256, act = self.act, name = 'mask/fc1', W_init = w_init)
                mask = DenseLayer(mask, n_units = output_dim, act = None, name = 'mask/fc2', W_init = w_init)
                mask = tf.nn.softmax(mask.outputs, axis = -1)
                
                n = tf.reshape(n, [self.batch_size * 256, output_dim])
                n = tf.multiply(n, mask)

                n = tf.reshape(n, [self.batch_size, 256, output_dim])
                n = self.combine_method(n, axis = 1)

                return n, [\
                    vars1,\
                    vars2,\
                    vars3,\
                    vars4,\
                    vars5,\
                    vars6,\
                    vars7,\
                    vars8,\
                    vars9,\
                    varsA,\
                    varsB,\
                    varsC,\
                    varsD,\
                    varsE,\
                    varsF] # OMG...

            if self.encoder_arch == 'full_graph_pool_prefine_longfeature_bn':

                g1_pos, g1_idx, g1_edg = kNNG_gen(input_particle, self.knn_k, 3, name = 'ggen1')
                particle_features = input_particle[:, :, 6:]

                n, vars1 = kNNGConvBN_wrapper(particle_features, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim // 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv1')
                n, vars2 = kNNGConvBN_wrapper(n, g1_idx, g1_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim // 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv2')

                g1_n = n
                g2_cnt = min(1280, self.gridMaxSize)
                g2_pos, n, vars3 = kNNGPooling_GUnet(n, g1_pos, g2_cnt, W_init = w_init, name = 'gpool1')
                # g2_pos, vars_prg2 = kNNGPosition_refine(g2_pos, n, self.act, W_init = w_init, b_init = b_init, name = 'gpref2')
                _, g2_idx, g2_edg = kNNG_gen(g2_pos, self.knn_k, 3, name = 'ggen2')

                n, vars4 = kNNGConvBN_wrapper(n, g2_idx, g2_edg, self.batch_size, g2_cnt, self.particle_hidden_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv3')
                n, vars5 = kNNGConvBN_wrapper(n, g2_idx, g2_edg, self.batch_size, g2_cnt, self.particle_hidden_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv4')

                g3_cnt = min(512, self.gridMaxSize)
                g3_pos, n, vars6 = kNNGPooling_GUnet(n, g2_pos, g3_cnt, W_init = w_init, name = 'gpool2')
                # g3_pos, vars_prg3 = kNNGPosition_refine(g3_pos, n, self.act, W_init = w_init, b_init = b_init, name = 'gpref3')
                _, g3_idx, g3_edg = kNNG_gen(g3_pos, self.knn_k, 3, name = 'ggen3')

                n, vars7 = kNNGConvBN_wrapper(n, g3_idx, g3_edg, self.batch_size, g3_cnt, self.particle_hidden_dim * 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv5')
                n, vars8 = kNNGConvBN_wrapper(n, g3_idx, g3_edg, self.batch_size, g3_cnt, self.particle_hidden_dim * 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv6')
                n, vars9 = kNNGConvBN_wrapper(n, g3_idx, g3_edg, self.batch_size, g3_cnt, self.particle_hidden_dim * 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv7')
                n, varsA = kNNGConvBN_wrapper(n, g3_idx, g3_edg, self.batch_size, g3_cnt, self.particle_hidden_dim * 2, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv8')

                g4_pos, n, varsB = kNNGPooling_GUnet(n, g3_pos, self.cluster_count, W_init = w_init, name = 'gpool3')
                # g4_pos, vars_prg4 = kNNGPosition_refine(g4_pos, n, self.act, W_init = w_init, b_init = b_init, name = 'gpref4')
                _, g4_idx, g4_edg = kNNG_gen(g4_pos, self.knn_k, 3, name = 'ggen4')

                tf.summary.histogram('Pooled clusters pos', g4_pos)

                n, varsC = kNNGConvBN_wrapper(n, g4_idx, g4_edg, self.batch_size, self.cluster_count, self.cluster_feature_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv9')
                n, varsD = kNNGConvBN_wrapper(n, g4_idx, g4_edg, self.batch_size, self.cluster_count, self.cluster_feature_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'gconv10')
                
                varRes = []
                tmp = n
                resblocks = 1
                for resi in range(resblocks):
                    nn, v = kNNGConvBN_wrapper(n, g4_idx, g4_edg, self.batch_size, self.cluster_count, self.cluster_feature_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'res%d/gconv1' % resi)
                    varRes.append(v)
                    nn, v = kNNGConvBN_wrapper(nn, g4_idx, g4_edg, self.batch_size, self.cluster_count, self.cluster_feature_dim, self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'res%d/gconv2' % resi)
                    varRes.append(v)
                    n = n + nn

                # n = n + tmp

                clu_pos = tf.reshape(g4_pos, [-1, self.cluster_count * 3])
                clu_lat = tf.reshape(n, [-1, self.cluster_count * self.cluster_feature_dim])

                # Compute mask
                # mask = InputLayer(tf.reshape(g4_pos, [self.batch_size * 256, 3]), name = 'posmask/input')
                # mask = DenseLayer(mask, n_units = 256, act = self.act, name = 'mask/fc1', W_init = w_init)
                # mask = DenseLayer(mask, n_units = output_dim, act = None, name = 'mask/fc2', W_init = w_init)
                # mask = tf.nn.softmax(mask.outputs, axis = -1)
                
                # n = tf.reshape(n, [self.batch_size * 256, output_dim])
                # n = tf.multiply(n, mask)

                # n = tf.reshape(n, [self.batch_size, 256, output_dim])
                n, varsConvOut = Conv1dWrapper(n, output_dim, 1, 1, 'SAME', None, w_init, b_init, True, 'convFinal') # final conv
                n = self.combine_method(n, axis = 1)

                n = tf.concat([n, clu_pos, clu_lat], axis = -1)

                vars_array = [\
                    vars1,\
                    vars2,\
                    vars3,\
                    vars4,\
                    vars5,\
                    vars6,\
                    vars7,\
                    vars8,\
                    vars9,\
                    varsA,\
                    varsB,\
                    varsC,\
                    varsD,\
                   # vars_prg2, vars_prg3, vars_prg4,\
                    varsConvOut] + varRes # OMG...

                if returnPool == True:
                    return n, vars_array, [g2_pos, g3_pos, g4_pos]

                return n, vars_array

            # We are going to use a way deeper model than before. Please refer to model_particlesTest_backup.py for original codes.
            if self.encoder_arch == 'full_graph_pool_prefine_longfeature_poolFreqLoss_bn':

                # I hate *** code.
                # blocks = 5
                # particles_count = [2560, 1280, 640, 320, self.cluster_count]
                # conv_count = [2, 2, 4, 1, 1]
                # res_count = [0, 0, 0, 2, 4]
                # kernel_size = [int(self.knn_k / 1.5), int(self.knn_k / 1.2), self.knn_k, self.knn_k, self.knn_k]
                # hd = self.particle_hidden_dim
                # channels = [int(hd / 3.2), hd // 2, hd, int(hd * 1.5), hd * 2]
                
                blocks = 4
                particles_count = [2560, 1280, 512, self.cluster_count]
                conv_count = [3, 2, 3, 2]
                res_count = [0, 0, 0, 1]
                kernel_size = [self.knn_k, self.knn_k, self.knn_k, self.knn_k]
                hd = self.particle_hidden_dim
                channels = [hd // 3, hd // 2, hd, hd * 2]

                gPos = input_particle[:, :, :3]
                n = input_particle[:, :, 6:]
                var_list = []
                pool_pos = []
                freq_loss = 0

                for i in range(blocks):
                    
                    if i > 0:
                        
                        # Pooling
                        prev_n = n
                        prev_pos = gPos
                        gPos, n, v, fl = kNNGPooling_HighFreqLoss_GUnet(n, gPos, particles_count[i], MatL, W_init = w_init, name = 'gpool%d' % i)
                        var_list.append(v)

                        pool_pos.append(gPos)
                        freq_loss = freq_loss + fl

                        # Collect features after pool
                        _, _, bpIdx, bpEdg = bip_kNNG_gen(gPos, prev_pos, kernel_size[i], 3, name = 'gpool%d/ggen' % i)
                        n, _ = bip_kNNGConvBN_wrapper(n, prev_n, bpIdx, bpEdg, self.batch_size, particles_count[i], channels[i], self.act, is_train = True, W_init = w_init, b_init = b_init, name = 'gpool%d/gconv' % i)

                    gPos, gIdx, gEdg = kNNG_gen(gPos, kernel_size[i], 3, name = 'ggen%d' % i)
                    MatL, MatA, MatD = Laplacian(self.batch_size, particles_count[i], kernel_size[i], gIdx, name = 'gLaplacian%d' % i)

                    for c in range(conv_count[i]):

                        n, v = kNNGConvBN_wrapper(n, gIdx, gEdg, self.batch_size, particles_count[i], channels[i], self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'g%d/gconv%d' % (i, c))
                        var_list.append(v)

                    tmp = n
                    for r in range(res_count[i]):
                    
                        nn, v = kNNGConvBN_wrapper(n, gIdx, gEdg, self.batch_size, particles_count[i], channels[i], self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'g%d/res%d/conv1' % (i, r))
                        var_list.append(v)
                        
                        nn, v = kNNGConvBN_wrapper(nn, gIdx, gEdg, self.batch_size, particles_count[i], channels[i], self.act, 0.999, is_train = True, W_init = w_init, b_init = b_init, name = 'g%d/res%d/conv2' % (i, r))
                        var_list.append(v)
                        n = n + nn
                    
                    if res_count[i] > 1:
                        n = n + tmp
                
                tf.summary.histogram('Pooled_clusters_pos', gPos)
                n, v = Conv1dWrapper(n, self.cluster_feature_dim, 1, 1, 'SAME', None, w_init, b_init, True, 'convOut')
                var_list.append(v)

                clu_pos = tf.reshape(gPos, [-1, self.cluster_count * 3])
                clu_lat = tf.reshape(n, [-1, self.cluster_count * self.cluster_feature_dim])

                # Compute mask
                # mask = InputLayer(tf.reshape(g4_pos, [self.batch_size * 256, 3]), name = 'posmask/input')
                # mask = DenseLayer(mask, n_units = 256, act = self.act, name = 'mask/fc1', W_init = w_init)
                # mask = DenseLayer(mask, n_units = output_dim, act = None, name = 'mask/fc2', W_init = w_init)
                # mask = tf.nn.softmax(mask.outputs, axis = -1)
                
                # n = tf.reshape(n, [self.batch_size * 256, output_dim])
                # n = tf.multiply(n, mask)

                # n = tf.reshape(n, [self.batch_size, 256, output_dim])
                n, v = Conv1dWrapper(n, output_dim, 1, 1, 'SAME', None, w_init, b_init, True, 'convFinal') # final conv
                var_list.append(v)
                n = self.combine_method(n, axis = 1)

                n = tf.concat([n, clu_pos, clu_lat], axis = -1)

                if returnPool == True:
                    return n, var_list, pool_pos, freq_loss

                return n, var_list, freq_loss

            # TODO: finish it (annoying!)            
            # if self.encoder_arch == 'attractor':

            #     n = InputLayer(tf.reshape(input_particle, [self.batch_size * self.gridMaxSize, 7]), name = 'input')

            #     l = self.initial_grid_size / 4
            #     attractors = tf.constant(\
            #         [\
            #             [ l,  l,  l],\
            #             [ l,  l, -l],\
            #             [ l, -l,  l],\
            #             [ l, -l, -l],\
            #             [-l,  l,  l],\
            #             [-l,  l, -l],\
            #             [-l, -l,  l],\
            #             [-l, -l, -l],\
            #             [ 0,  0,  0],\
            #         ], dtype = 'float'\
            #     )

            #     score = tf.square(attractors - )

            #     n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc1', W_init = w_init)
            #     n = LayerNormLayer(n, act = self.act, name = 'fc1/ln')
            #     n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc2', W_init = w_init)
            #     n = LayerNormLayer(n, act = self.act, name = 'fc2/ln')
            #     n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc3', W_init = w_init)
            #     n = LayerNormLayer(n, act = self.act, name = 'fc3/ln')
            #     n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'fc4', W_init = w_init)
            #     n = LayerNormLayer(n, act = self.act, name = 'fc4/ln')
            #     n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc5', W_init = w_init)

            #     n = tf.reshape(n.outputs, [self.batch_size, self.gridMaxSize, output_dim])
            #     n = self.combine_method(n, axis = 1)
    
    def particleDecoder(self, input_latent, groundTruth_card, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev=0.03*self.wdev)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleDecoder", reuse = reuse) as vs:

            if self.decoder_arch == 'plain_ln':

                # input_latent : [batch_size, channels]

                net_input = InputLayer(input_latent, name = 'input')
                
                card = DenseLayer(net_input, n_units = 1, act = None, name = 'card/out', W_init = w_init)
                card = tf.reshape(card.outputs, [self.batch_size])

                alter_particles = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'alter/fc1', W_init = w_init)
                alter_particles = LayerNormLayer(alter_particles, act = self.act, name = 'alter/fc1/ln')
                alter_particles = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'alter/fc2', W_init = w_init)
                alter_particles = LayerNormLayer(alter_particles, act = self.act, name = 'alter/fc2/ln')
                alter_particles = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'alter/fc3', W_init = w_init)
                alter_particles = LayerNormLayer(alter_particles, act = self.act, name = 'alter/fc3/ln')
                alter_particles = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'alter/fc4', W_init = w_init)
                alter_particles = LayerNormLayer(alter_particles, act = self.act, name = 'alter/fc4/ln')
                alter_particles = DenseLayer(net_input, n_units = output_dim * self.gridMaxSize, act = None, name = 'alter/out', W_init = w_init)

                alter_particles = tf.reshape(alter_particles.outputs, [self.batch_size, self.gridMaxSize, 6])

                if is_train == True:
                    card_used = groundTruth_card
                else:
                    card_used = tf.round(card)
                    card_used = tf.minimum(card_used, self.gridMaxSize)
                    card_used = tf.maximum(card_used, 0)

                card_mask, card_match = tf.py_func(self.generate_match_canonical, [card_used], [tf.float32, tf.int32], name='card_mask')

                masked_output = tf.multiply(alter_particles, card_mask, name = 'masked_output')
                final_outputs = tf.gather_nd(masked_output, card_match, name = 'final_outputs')

                return card, final_outputs, card_used

            if self.decoder_arch == 'plain_bn':

                # input_latent : [batch_size, channels]

                net_input = InputLayer(input_latent, name = 'input')
                
                card = DenseLayer(net_input, n_units = 1, act = None, name = 'card/out', W_init = w_init)
                card = tf.reshape(card.outputs, [self.batch_size])

                alter_particles = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'alter/fc1', W_init = w_init)
                alter_particles = BatchNormLayer(alter_particles, decay = 0.999, act = self.act, is_train = is_train, name = 'alter/fc1/bn')
                alter_particles = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'alter/fc2', W_init = w_init)
                alter_particles = BatchNormLayer(alter_particles, decay = 0.999, act = self.act, is_train = is_train, name = 'alter/fc2/bn')
                alter_particles = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'alter/fc3', W_init = w_init)
                alter_particles = BatchNormLayer(alter_particles, decay = 0.999, act = self.act, is_train = is_train, name = 'alter/fc3/bn')
                alter_particles = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'alter/fc4', W_init = w_init)
                alter_particles = BatchNormLayer(alter_particles, decay = 0.999, act = self.act, is_train = is_train, name = 'alter/fc4/bn')
                alter_particles = DenseLayer(net_input, n_units = output_dim * self.gridMaxSize, act = None, name = 'alter/out', W_init = w_init)

                alter_particles = tf.reshape(alter_particles.outputs, [self.batch_size, self.gridMaxSize, 6])

                if is_train == True:
                    card_used = groundTruth_card
                else:
                    card_used = tf.round(card)
                    card_used = tf.minimum(card_used, self.gridMaxSize)
                    card_used = tf.maximum(card_used, 0)

                card_mask, card_match = tf.py_func(self.generate_match_canonical, [card_used], [tf.float32, tf.int32], name='card_mask')

                masked_output = tf.multiply(alter_particles, card_mask, name = 'masked_output')
                final_outputs = tf.gather_nd(masked_output, card_match, name = 'final_outputs')

                return card, final_outputs, card_used
            
            if self.decoder_arch == 'distribution_weight':

                # input_latent : [batch_size, channels]

                net_input = InputLayer(input_latent, name = 'input')
                
                # predict card (amount of particles inside voxel) first

                card = DenseLayer(net_input, n_units = 1, act = None, name = 'card/out', W_init = w_init)
                card = tf.reshape(card.outputs, [self.batch_size])

                # generate random noise

                z = tf.random.normal([self.batch_size, self.gridMaxSize, output_dim])

                # predict network weights (omg so big...)

                n = DenseLayer(net_input, n_units = self.particle_hidden_dim, act = None, name = 'weight/fc1', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'weight/fc1/ln')
                n = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'weight/fc2', W_init = w_init)
                n = LayerNormLayer(n, act = self.act, name = 'weight/fc2/ln')

                w1 = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'weight/w1/fc1', W_init = w_init)
                w1 = LayerNormLayer(w1, act = self.act, name = 'weight/w1/fc1/ln')
                w1 = DenseLayer(w1, n_units = output_dim * self.particle_hidden_dim, act = None, name = 'weight/w1/out', W_init = w_init).outputs
                w1 = tf.reshape(w1, [self.batch_size, output_dim, self.particle_hidden_dim])

                w2 = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'weight/w2/fc1', W_init = w_init)
                w2 = LayerNormLayer(w2, act = self.act, name = 'weight/w2/fc1/ln')
                w2 = DenseLayer(w2, n_units = output_dim * self.particle_hidden_dim, act = None, name = 'weight/w2/out', W_init = w_init).outputs
                w2 = tf.reshape(w2, [self.batch_size, self.particle_hidden_dim, output_dim])

                b1 = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'weight/b1/fc1', W_init = w_init)
                b1 = LayerNormLayer(b1, act = self.act, name = 'weight/b1/fc1/ln')
                b1 = DenseLayer(b1, n_units = self.particle_hidden_dim, act = None, name = 'weight/b1/out', W_init = w_init).outputs
                b1 = tf.reshape(b1, [self.batch_size, 1, self.particle_hidden_dim])

                b2 = DenseLayer(n, n_units = self.particle_hidden_dim, act = None, name = 'weight/b2/fc1', W_init = w_init)
                b2 = LayerNormLayer(b2, act = self.act, name = 'weight/b2/fc1/ln')
                b2 = DenseLayer(b2, n_units = output_dim, act = None, name = 'weight/b2/out', W_init = w_init).outputs
                b2 = tf.reshape(b2, [self.batch_size, 1, output_dim])

                # calculate refined position from predicted weights

                hidden = self.act(tf.matmul(z, w1) + b1)
                alter_particles = tf.matmul(hidden, w2) + b2

                # Get the masked output

                if is_train == True:
                    card_used = groundTruth_card
                else:
                    card_used = tf.round(card)
                    card_used = tf.minimum(card_used, self.gridMaxSize)
                    card_used = tf.maximum(card_used, 0)

                card_mask, card_match = tf.py_func(self.generate_match_canonical, [card_used], [tf.float32, tf.int32], name='card_mask')

                masked_output = tf.multiply(alter_particles, card_mask, name = 'masked_output')
                final_outputs = tf.gather_nd(masked_output, card_match, name = 'final_outputs')

                return card, final_outputs, card_used

            if self.decoder_arch == 'fold_ln': # Folding-Net decoder structure (3D variation)

                # input_latent : [batch_size, channels]

                net_input = InputLayer(input_latent, name = 'input')
                
                # predict card (amount of particles inside voxel) first

                card = DenseLayer(net_input, n_units = 1, act = None, name = 'card/out', W_init = w_init)
                card = tf.reshape(card.outputs, [self.batch_size])

                # generate random noise

                z = tf.random.normal([self.batch_size, self.gridMaxSize, output_dim])

                # conditional generative network

                perparticle_latent = \
                    tf.broadcast_to\
                    (\
                        tf.reshape(input_latent, [self.batch_size, 1, self.particle_latent_dim]),\
                        [self.batch_size, self.gridMaxSize, self.particle_latent_dim]\
                    )

                conditional_input = tf.concat([z, perparticle_latent], axis = -1)

                c = InputLayer(conditional_input, name = 'fold/input')

                c = Conv1d(c, n_filter = 512, filter_size = 1, stride = 1, act = None, name = 'fold/convin', W_init = w_init)
                c = LayerNormLayer(c, act = self.act, name = 'fold/convin/ln')

                c = Conv1d(c, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'fold/conv1', W_init = w_init)
                c = LayerNormLayer(c, act = self.act, name = 'fold/conv1/ln')
                c = Conv1d(c, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'fold/conv2', W_init = w_init)
                c = LayerNormLayer(c, act = self.act, name = 'fold/conv2/ln')
                c = Conv1d(c, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'fold/conv3', W_init = w_init)
                c = LayerNormLayer(c, act = self.act, name = 'fold/conv3/ln')
                c = Conv1d(c, n_filter = output_dim, filter_size = 1, stride = 1, act = None, name = 'fold/midout', W_init = w_init)

                conditional_input_2nd = tf.concat([c.outputs, perparticle_latent], axis = -1) # Inspired by Folding-Net
                c = InputLayer(conditional_input_2nd, name = 'fold/2ndinput')

                c = Conv1d(c, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'fold/conv4', W_init = w_init)
                c = LayerNormLayer(c, act = self.act, name = 'fold/conv4/ln')
                c = Conv1d(c, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'fold/conv5', W_init = w_init)
                c = LayerNormLayer(c, act = self.act, name = 'fold/conv5/ln')
                c = Conv1d(c, n_filter = self.particle_hidden_dim, filter_size = 1, stride = 1, act = None, name = 'fold/conv6', W_init = w_init)
                c = LayerNormLayer(c, act = self.act, name = 'fold/conv6/ln')
                alter_particles = Conv1d(c, n_filter = output_dim, filter_size = 1, stride = 1, act = None, name = 'fold/out', W_init = w_init)

                alter_particles = alter_particles.outputs

                # Get the masked output

                if is_train == True:
                    card_used = groundTruth_card
                else:
                    card_used = tf.round(card)
                    card_used = tf.minimum(card_used, self.gridMaxSize)
                    card_used = tf.maximum(card_used, 0)

                card_mask, card_match = tf.py_func(self.generate_match_canonical, [card_used], [tf.float32, tf.int32], name='card_mask')

                masked_output = tf.multiply(alter_particles, card_mask, name = 'masked_output')
                final_outputs = tf.gather_nd(masked_output, card_match, name = 'final_outputs')

                return card, final_outputs, card_used
            
            if self.decoder_arch == 'fold_bn': # Folding-Net decoder structure (3D variation)

                # input_latent : [batch_size, channels]

                net_input = InputLayer(input_latent, name = 'input')
                
                # predict card (amount of particles inside voxel) first

                card = DenseLayer(net_input, n_units = 1, act = None, name = 'card/out', W_init = w_init)
                card = tf.reshape(card.outputs, [self.batch_size])

                # generate random noise

                z = tf.random.normal([self.batch_size * self.gridMaxSize, output_dim])

                # conditional generative network

                flatten_latent = \
                tf.reshape\
                (\
                    tf.broadcast_to\
                    (\
                        tf.reshape(input_latent, [self.batch_size, 1, self.particle_latent_dim]),\
                        [self.batch_size, self.gridMaxSize, self.particle_latent_dim]\
                    ),\
                    [self.batch_size * self.gridMaxSize, self.particle_latent_dim]\
                )

                conditional_input = tf.concat([z, flatten_latent], axis = -1)

                c = InputLayer(conditional_input, name = 'cond/input')

                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc1', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc1/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc2', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc2/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc3', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc3/bn')
                c = DenseLayer(c, n_units = output_dim, act = None, name = 'cond/midout', W_init = w_init)

                conditional_input_2nd = tf.concat([c.outputs, flatten_latent], axis = -1) # Inspired by Folding-Net
                c = InputLayer(conditional_input_2nd, name = 'cond/midinput')

                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc4', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc4/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc5', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc5/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc6', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc6/bn')
                alter_particles = DenseLayer(c, n_units = output_dim, act = None, name = 'cond/out', W_init = w_init)

                alter_particles = tf.reshape(alter_particles.outputs, [self.batch_size, self.gridMaxSize, output_dim])

                # Get the masked output

                # if is_train == True:
                #     card_used = groundTruth_card
                # else:
                #     card_used = tf.round(card)
                #     card_used = tf.minimum(card_used, self.gridMaxSize)
                #     card_used = tf.maximum(card_used, 0)

                # card_mask, card_match = tf.py_func(self.generate_match_canonical, [card_used], [tf.float32, tf.int32], name='card_mask')

                # masked_output = tf.multiply(alter_particles, card_mask, name = 'masked_output')
                # final_outputs = tf.gather_nd(masked_output, card_match, name = 'final_outputs')

                # return card, final_outputs, card_used
                return 0, alter_particles, 0
            
            if self.decoder_arch == 'deep_fold_bn': # More generator-liked structure

                # input_latent : [batch_size, channels]

                net_input = InputLayer(input_latent, name = 'input')
                
                # FIXME: no card in this model

                # generate random noise
                z = tf.random.normal([self.batch_size * self.gridMaxSize, output_dim])

                # conditional generative network
                latents = \
                tf.reshape\
                (\
                    tf.broadcast_to\
                    (\
                        tf.reshape(input_latent, [self.batch_size, 1, self.particle_latent_dim]),\
                        [self.batch_size, self.gridMaxSize, self.particle_latent_dim]\
                    ),\
                    [self.batch_size * self.gridMaxSize, self.particle_latent_dim]\
                )
                pos = z

                conditional_input = tf.concat([pos, latents], axis = -1)

                c = InputLayer(conditional_input, name = 'cond/input')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc1', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc1/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc2', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc2/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc3', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc3/bn')

                tmp = c

                for i in range(6):

                    conditional_input = tf.concat([c.outputs, latents], axis = -1)

                    cc = InputLayer(conditional_input, name = 'res%d/cond/input' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc1' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc1/bn' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc2' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc2/bn' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc3' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc3/bn' % i)

                    c = ElementwiseLayer([c, cc], tf.add, name = 'res%d/add' % i)

                c = ElementwiseLayer([c, tmp], tf.add, name = 'resout/add')
                c = DenseLayer(c, n_units = output_dim, act = None, name = 'res%d/cond/resout' % i, W_init = w_init)
                pos = c.outputs

                alter_particles = tf.reshape(pos, [self.batch_size, self.gridMaxSize, output_dim])
                
                return 0, alter_particles, 0
            
            if self.decoder_arch == 'fold_graph_bn': # More generator-liked structure

                # input_latent : [batch_size, channels]
                
                # input_latent - [Global latents(512); Positions(256*3); Local latents(256*512)]
                
                global_latent = input_latent[:, :self.particle_latent_dim]
                cluster_pos = input_latent[:, self.particle_latent_dim:(self.particle_latent_dim+self.cluster_count*3)]
                local_feature = input_latent[:, (self.particle_latent_dim+self.cluster_count*3):]
                cluster_pos = tf.reshape(cluster_pos, [-1, self.cluster_count, 3])
                local_feature = tf.reshape(local_feature, [-1, self.cluster_count, self.cluster_feature_dim])

                # Folding stage
                fold_particles_count = self.gridMaxSize - self.cluster_count
                # net_input = InputLayer(input_latent, name = 'input')
                
                # FIXME: no card in this model

                # generate random noise
                z = tf.random.normal([self.batch_size * fold_particles_count, output_dim])

                # conditional generative network
                latents = \
                tf.reshape\
                (\
                    tf.broadcast_to\
                    (\
                        tf.reshape(global_latent, [self.batch_size, 1, self.particle_latent_dim]),\
                        [self.batch_size, fold_particles_count, self.particle_latent_dim]\
                    ),\
                    [self.batch_size * fold_particles_count, self.particle_latent_dim]\
                )
                pos = z

                conditional_input = tf.concat([pos, latents], axis = -1)

                c = InputLayer(conditional_input, name = 'cond/input')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc1', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc1/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc2', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc2/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc3', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc3/bn')

                tmp = c
                resCount = 2

                for i in range(resCount):

                    conditional_input = tf.concat([c.outputs, latents], axis = -1)

                    cc = InputLayer(conditional_input, name = 'res%d/cond/input' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc1' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc1/bn' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc2' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc2/bn' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc3' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc3/bn' % i)

                    # if i < (resCount - 1):
                    c = ElementwiseLayer([c, cc], tf.add, name = 'res%d/add' % i)

                c = ElementwiseLayer([c, tmp], tf.add, name = 'resout/add')
                c = DenseLayer(c, n_units = output_dim, act = None, name = 'resFinal/cond/resout', W_init = w_init)
                pos = c.outputs

                alter_particles = tf.reshape(pos, [self.batch_size, fold_particles_count, output_dim])
                fold_before_prefine = tf.concat([alter_particles, cluster_pos], axis = 1)

                tf.summary.histogram('Particles_AfterFolding', alter_particles)

                # Graph pos-refinement stage
                # Obtain features for alter particles

                # Create the graph
                posAlter, posRefer, gp_idx, gp_edg = bip_kNNG_gen(alter_particles, cluster_pos, self.knn_k // 2, 3, name = 'bi_ggen_pre')

                # Create a empty feature (0.0)
                n = tf.reduce_mean(tf.zeros_like(alter_particles), axis = -1, keepdims = True)

                # Do the convolution
                convSteps = 3
                varsGConv = []
                for i in range(convSteps):
                    n, v = bip_kNNGConvBN_wrapper(n, local_feature, gp_idx, gp_edg, self.batch_size, fold_particles_count, self.particle_hidden_dim // 2, self.act, is_train = True, name = 'gconv%d_pre' % i, W_init = w_init)
                    varsGConv.append(v)
           
                fold_particle_features = n

                # Reduce clusters' features
                # clusters: [bs, N_clusters, cluster_feature_dim]
                n, vars3 = Conv1dWrapper(local_feature, self.particle_hidden_dim // 2, 1, 1, 'SAME', self.act, w_init, b_init, True, 'conv1')
                ref_particle_features = n

                # Combine them to a single graph
                pos = tf.concat([posRefer, posAlter], axis = 1) # [bs, N, 3]
                n = tf.concat([ref_particle_features, fold_particle_features], axis = 1) # [bs, N, phd]

                # Position Refinement
                refine_loops = 3
                vars_loop = []

                _, gr_idx, gr_edg = kNNG_gen(pos, self.knn_k // 2, 3, name = 'grefine/ggen')
                tmp = n

                for i in range(refine_loops):
                    
                    # Pos-refinement
                    # pos, v = kNNGPosition_refine(pos, n, self.act, W_init = w_init_pref, b_init = b_init, name = 'gloop%d/pos_refine' % i)
                    # vars_loop.append(v)

                    # Graph generation
                    # _, gl_idx, gl_edg = kNNG_gen(pos, self.knn_k, 3, name = 'gloop%d/ggen' % i)

                    # Convolution
                    nn, v = kNNGConvBN_wrapper(n, gr_idx, gr_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim // 2, self.act, is_train = is_train, name = 'gloop%d/gconv1' % i, W_init = w_init, b_init = b_init)
                    vars_loop.append(v)
                    
                    nn, v = kNNGConvBN_wrapper(nn, gr_idx, gr_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim // 2, self.act, is_train = is_train, name = 'gloop%d/gconv2' % i, W_init = w_init, b_init = b_init)
                    vars_loop.append(v)

                    n = n + nn

                n = n + tmp
                pos, v = kNNGPosition_refine(pos, n, self.act, W_init = w_init_pref, b_init = b_init, name = 'grefine/refine')

                return 0, [pos, fold_before_prefine], 0
            
            if self.decoder_arch == 'graph_bn': # More generator-liked structure

                # input_latent : [batch_size, channels]
                
                # input_latent - [Global latents(512); Positions(256*3); Local latents(256*512)]
                
                global_latent = input_latent[:, :self.particle_latent_dim]
                cluster_pos = input_latent[:, self.particle_latent_dim:(self.particle_latent_dim+self.cluster_count*3)]
                local_feature = input_latent[:, (self.particle_latent_dim+self.cluster_count*3):]
                cluster_pos = tf.reshape(cluster_pos, [-1, self.cluster_count, 3])
                local_feature = tf.reshape(local_feature, [-1, self.cluster_count, self.cluster_feature_dim])

                # Folding stage
                fold_particles_count = self.gridMaxSize - self.cluster_count
                # net_input = InputLayer(input_latent, name = 'input')
                
                # FIXME: no card in this model

                # generate random noise
                z = tf.random.normal([self.batch_size * fold_particles_count, output_dim])

                # conditional generative network
                latents = \
                tf.reshape\
                (\
                    tf.broadcast_to\
                    (\
                        tf.reshape(global_latent, [self.batch_size, 1, self.particle_latent_dim]),\
                        [self.batch_size, fold_particles_count, self.particle_latent_dim]\
                    ),\
                    [self.batch_size * fold_particles_count, self.particle_latent_dim]\
                )
                pos = z

                conditional_input = tf.concat([pos, latents], axis = -1)

                c = InputLayer(conditional_input, name = 'cond/input')
                # c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc1', W_init = w_init)
                # c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc1/bn')
                # c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc2', W_init = w_init)
                # c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc2/bn')
                # c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc3', W_init = w_init)
                # c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc3/bn')

                # tmp = c
                resCount = 0

                for i in range(resCount):

                    conditional_input = tf.concat([c.outputs, latents], axis = -1)

                    cc = InputLayer(conditional_input, name = 'res%d/cond/input' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc1' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc1/bn' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc2' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc2/bn' % i)
                    cc = DenseLayer(cc, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc3' % i, W_init = w_init)
                    cc = BatchNormLayer(cc, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc3/bn' % i)

                    # if i < (resCount - 1):
                    c = ElementwiseLayer([c, cc], tf.add, name = 'res%d/add' % i)

                # c = ElementwiseLayer([c, tmp], tf.add, name = 'resout/add')
                c = DenseLayer(c, n_units = output_dim, act = None, name = 'resFinal/cond/resout', W_init = w_init)
                pos = c.outputs

                alter_particles = tf.reshape(pos, [self.batch_size, fold_particles_count, output_dim])
                fold_before_prefine = tf.concat([alter_particles, cluster_pos], axis = 1)

                tf.summary.histogram('Particles_AfterFolding', alter_particles)

                # Graph pos-refinement stage
                # Obtain features for alter particles

                # Create the graph
                posAlter, posRefer, gp_idx, gp_edg = bip_kNNG_gen(alter_particles, cluster_pos, self.knn_k - 6, 3, name = 'bi_ggen_pre')

                # Create a empty feature (0.0)
                n = tf.reduce_mean(tf.zeros_like(alter_particles), axis = -1, keepdims = True)

                # Do the convolution
                convSteps = 3
                varsGConv = []
                for i in range(convSteps):
                    n, v = bip_kNNGConvBN_wrapper(n, local_feature, gp_idx, gp_edg, self.batch_size, fold_particles_count, self.particle_hidden_dim // 2, self.act, is_train = True, name = 'gconv%d_pre' % i, W_init = w_init)
                    varsGConv.append(v)
           
                fold_particle_features = n

                # Reduce clusters' features
                # clusters: [bs, N_clusters, cluster_feature_dim]
                n, vars3 = Conv1dWrapper(local_feature, self.particle_hidden_dim // 2, 1, 1, 'SAME', self.act, w_init, b_init, True, 'conv1')
                ref_particle_features = n

                # Combine them to a single graph
                pos = tf.concat([posRefer, posAlter], axis = 1) # [bs, N, 3]
                n = tf.concat([ref_particle_features, fold_particle_features], axis = 1) # [bs, N, phd]

                # Position Refinement
                refine_loops = 2
                refine_res_blocks = 2
                vars_loop = []

                for r in range(refine_loops):
                
                    _, gr_idx, gr_edg = kNNG_gen(pos, self.knn_k, 3, name = 'grefine%d/ggen' % r)
                    tmp = n

                    for i in range(refine_res_blocks):
                        
                        # Pos-refinement
                        # pos, v = kNNGPosition_refine(pos, n, self.act, W_init = w_init_pref, b_init = b_init, name = 'gloop%d/pos_refine' % i)
                        # vars_loop.append(v)

                        # Graph generation
                        # _, gl_idx, gl_edg = kNNG_gen(pos, self.knn_k, 3, name = 'gloop%d/ggen' % i)

                        # Convolution
                        nn, v = kNNGConvBN_wrapper(n, gr_idx, gr_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim // 2, self.act, is_train = is_train, name = 'gr%d/gloop%d/gconv1' % (r, i), W_init = w_init, b_init = b_init)
                        vars_loop.append(v)
                        
                        nn, v = kNNGConvBN_wrapper(nn, gr_idx, gr_edg, self.batch_size, self.gridMaxSize, self.particle_hidden_dim // 2, self.act, is_train = is_train, name = 'gr%d/gloop%d/gconv2' % (r, i), W_init = w_init, b_init = b_init)
                        vars_loop.append(v)

                        n = n + nn

                    n = n + tmp
                    pos, v = kNNGPosition_refine(pos, n, self.act, W_init = w_init_pref, b_init = b_init, name = 'gr%d/grefine/refine' % r)
                    vars_loop.append(v)

                return 0, [pos, fold_before_prefine], 0


            if self.decoder_arch == 'deep_mask_fold_bn': # Folding-Net decoder structure (3D variation)

                # input_latent : [batch_size, channels]

                net_input = InputLayer(input_latent, name = 'input')
                
                # FIXME: no card in this model

                # generate random noise
                z = tf.random.normal([self.batch_size * self.gridMaxSize, output_dim])

                # conditional generative network
                latents = \
                tf.reshape\
                (\
                    tf.broadcast_to\
                    (\
                        tf.reshape(input_latent, [self.batch_size, 1, self.particle_latent_dim]),\
                        [self.batch_size, self.gridMaxSize, self.particle_latent_dim]\
                    ),\
                    [self.batch_size * self.gridMaxSize, self.particle_latent_dim]\
                )
                pos = z

                m = InputLayer(pos, name = 'cond/mask/input')
                m = DenseLayer(m, n_units = self.particle_latent_dim, act = self.act, name = 'cond/mask/fc1', W_init = w_init)
                m = DenseLayer(m, n_units = self.particle_latent_dim, act = None, name = 'cond/mask/fc2', W_init = w_init)
                m = tf.nn.softmax(m.outputs, axis = -1)
                masked_latents = tf.multiply(m, latents)

                conditional_input = tf.concat([pos, masked_latents], axis = -1)

                c = InputLayer(conditional_input, name = 'cond/input')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc1', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc1/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc2', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc2/bn')
                c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'cond/fc3', W_init = w_init)
                c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'cond/fc3/bn')
                c = DenseLayer(c, n_units = output_dim, act = None, name = 'cond/midout', W_init = w_init)

                pos = c.outputs

                for i in range(3):
                    
                    m = InputLayer(pos, name = 'res%d/cond/mask/input' % i)
                    m = DenseLayer(m, n_units = self.particle_latent_dim, act = self.act, name = 'res%d/cond/mask/fc1' % i, W_init = w_init)
                    m = DenseLayer(m, n_units = self.particle_latent_dim, act = None, name = 'res%d/cond/mask/fc2' % i, W_init = w_init)
                    m = tf.nn.softmax(m.outputs, axis = -1)
                    masked_latents = tf.multiply(m, latents)

                    conditional_input = tf.concat([pos, masked_latents], axis = -1)

                    c = InputLayer(conditional_input, name = 'res%d/cond/input' % i)
                    c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc1' % i, W_init = w_init)
                    c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc1/bn' % i)
                    c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc2' % i, W_init = w_init)
                    c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc2/bn' % i)
                    c = DenseLayer(c, n_units = self.particle_hidden_dim, act = None, name = 'res%d/cond/fc3' % i, W_init = w_init)
                    c = BatchNormLayer(c, decay = 0.999, act = self.act, is_train = is_train, name = 'res%d/cond/fc3/bn' % i)
                    c = DenseLayer(c, n_units = output_dim, act = None, name = 'res%d/cond/resout' % i, W_init = w_init)

                    pos = pos + c.outputs

                    # Modify latent?

                alter_particles = tf.reshape(pos, [self.batch_size, self.gridMaxSize, output_dim])
                return 0, alter_particles, 0

    def generate_match_canonical(self, card):

        # card: [bs]
        batch_size = card.shape[0]

        mask = np.zeros((batch_size, self.gridMaxSize, 3), dtype = 'f')

        for b in range(batch_size):
            for i in range(int(card[b])):
                mask[b, i, :] = 1
            # Uncomment for randomly choosing
            # np.random.shuffle(mask[b, :])

        match = np.zeros((batch_size, self.gridMaxSize, 2), dtype = np.int32)
        # TODO: implement match

        for b in range(batch_size):
            cnt = 0
            for i in range(self.gridMaxSize):
                if mask[b, i, 0] > 0.2: # randomly chosen 0.2 (any 0~1)
                    match[b, cnt, :] = [b, i]
                    cnt += 1
            
            # fill lefts
            for i in range(self.gridMaxSize):
                if mask[b, i, 0] <= 0.2:
                    match[b, cnt, :] = [b, i]
                    cnt += 1

        return mask, match
    
    def generate_canonical_mask(self, card):

        # card: [bs]
        batch_size = card.shape[0]

        mask = np.zeros((batch_size, self.gridMaxSize, 3), dtype = 'f')

        for b in range(batch_size):
            for i in range(int(card[b])):
                mask[b, i, :] = 1
            # Uncomment for randomly choosing
            # np.random.shuffle(mask[b, :])

        return mask
    
    def generate_score_label(self, src, cards):

        result = np.zeros((self.batch_size, self.gridMaxSize), dtype = 'f')

        for b in range(self.batch_size):
            for p in range(self.gridMaxSize):
                if src[b, p] < cards[b]:
                    result[b, p] = 1 / cards[b]
        
        return result
    
    def generate_KM_match(self, src):

        result = np.zeros((self.batch_size, self.gridMaxSize, 2), dtype = np.int32)

        for b in range(self.batch_size):
            for p in range(self.gridMaxSize):
                result[b, src[b, p]] = np.asarray([b, p]) # KM match order reversed (ph_Y -> output => output -> ph_Y)
        
        return result

    def no_return_assign(self, ref, value):

        tf.assign(ref, value)
        return 0

    def chamfer_metric(self, particles, groundtruth, pos_range, loss_func):
            
        # test - shuffle the groundtruth and calculate the loss
        # rec_particles = tf.stack(list(map(lambda x: tf.random.shuffle(x), tf.unstack(self.ph_X[:, :, 0:6]))))
        # rec_particles = tf.random.uniform([self.batch_size, self.gridMaxSize, 3], minval = -1.0, maxval = 1.0)

        bs = groundtruth.shape[0]
        N  = groundtruth.shape[1]

        # NOTE: current using position (0:3) only here for searching nearest point.
        row_predicted = tf.reshape(  particles[:, :, 0:pos_range], [bs, N, 1, pos_range])
        col_groundtru = tf.reshape(groundtruth[:, :, 0:pos_range], [bs, 1, N, pos_range])
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(row_predicted, col_groundtru)), axis = -1))
        
        rearrange_predicted_N = tf.argmin(distance, axis = 1, output_type = tf.int32)
        rearrange_groundtru_N = tf.argmin(distance, axis = 2, output_type = tf.int32)
        
        batch_subscript = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, N])
        rearrange_predicted = tf.stack([batch_subscript, rearrange_predicted_N], axis = 2)
        rearrange_groundtru = tf.stack([batch_subscript, rearrange_groundtru_N], axis = 2)

        nearest_predicted = tf.gather_nd(  particles[:, :, 0:3], rearrange_predicted)
        nearest_groundtru = tf.gather_nd(groundtruth[:, :, 0:3], rearrange_groundtru)

        chamfer_loss =\
            tf.reduce_mean(loss_func(        particles - nearest_groundtru     )) +\
            tf.reduce_mean(loss_func(nearest_predicted - groundtruth[:, :, 0:3]))
        
        return chamfer_loss

    def build_network(self, is_train, reuse):

        # Go through the particle AE

        normalized_X = self.ph_X / 48.0
        # tf.summary.histogram('GroundTruth', normalized_X[:, :, 0:3])
        if self.encoder_arch == 'full_graph_pool_prefine_longfeature_poolFreqLoss_bn':
            latent, additional_vars, floss = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse)
        else:
            latent, additional_vars = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse)
            floss = 0.0
        rec_card, rec_particles, _ = self.particleDecoder(latent, self.ph_card, 3, is_train = is_train, reuse = reuse)
        if self.decoder_arch == 'fold_graph_bn' or self.decoder_arch == 'graph_bn':
            rec_final_particles, rec_fold_particles = rec_particles[0], rec_particles[1]
            rec_particles = rec_final_particles
            rec_fold_particles = rec_fold_particles * 48.0
        
        rec_particles = rec_particles * 48.0

        # EMD
        if self.loss_metric == 'earthmover':
            KM_matches = tf.py_func(KM, [rec_particles, self.ph_X, self.ph_card, self.ph_max_length], tf.int32, name = 'KM_matches')
            gather_KM_matches = tf.py_func(self.generate_KM_match, [KM_matches], tf.int32, name = 'gather_KM_matches')
            final_particles = tf.gather_nd(rec_particles, gather_KM_matches, 'final_outputs_after_KM')

            particle_network_loss =\
                tf.reduce_mean(self.loss_func(final_particles - self.ph_X[:, :, 0:6]))
        
        if self.loss_metric == 'chamfer':
            particle_network_loss = self.chamfer_metric(rec_particles, self.ph_X, 3, self.loss_func)
            if self.decoder_arch == 'fold_graph_bn':
                particle_raw_loss = particle_network_loss
                tf.summary.scalar('RealParticleLoss', particle_raw_loss)
                particle_fold_loss = self.chamfer_metric(rec_fold_particles, self.ph_X, 3, self.loss_func)
                tf.summary.scalar('FoldParticleLoss', particle_fold_loss)
                particle_network_loss += particle_fold_loss
            if self.decoder_arch == 'graph_bn':
                tf.summary.scalar('RealParticleLoss', particle_network_loss)

        # particle_card_loss =\
        #     tf.reduce_mean(self.loss_func(rec_card - self.ph_card)) +\
        #     tf.reduce_mean(floss)
        
        particle_card_loss = 0.004 * tf.reduce_mean(floss)

        particle_net_vars =\
            tl.layers.get_variables_with_name('ParticleEncoder', True, True) +\
            tl.layers.get_variables_with_name('ParticleDecoder', True, True) +\
            additional_vars

        return particle_network_loss, particle_card_loss, particle_net_vars, rec_particles, latent

    def build_predict(self, reuse = False):

        is_train = False

        includePool = False
        if self.encoder_arch == 'full_graph_pool_prefine_longfeature_bn' or self.encoder_arch == 'full_graph_pool_prefine_longfeature_poolFreqLoss_bn':
            includePool = True

        normalized_X = self.ph_X / 48.0

        if includePool == True:
            if self.encoder_arch == 'full_graph_pool_prefine_longfeature_poolFreqLoss_bn':
                latent, additional_vars, pool_pos, _ = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse, returnPool = True)
            else:
                latent, additional_vars, pool_pos = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse, returnPool = True)
        else:
            latent, additional_vars = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse, returnPool = False)
            pool_pos = []

        rec_card, rec_particles, _ = self.particleDecoder(latent, self.ph_card, 3, is_train = is_train, reuse = reuse)
        if self.decoder_arch == 'fold_graph_bn' or self.decoder_arch == 'graph_bn':
            rec_final_particles, rec_fold_particles = rec_particles[0], rec_particles[1]
            rec_particles = rec_final_particles
            rec_fold_particles = rec_fold_particles * 48.0
        else:
            rec_fold_particles = tf.zeros_like(rec_particles)

        rec_particles = rec_particles * 48.0
        
        val_loss = self.chamfer_metric(rec_particles, self.ph_X, 3, self.loss_func)
        
        for i in range(len(pool_pos)):
            pool_pos[i] = pool_pos[i] * 48.0

        return val_loss, rec_particles, rec_fold_particles, pool_pos

    def build_model(self):

        # Train & Validation
        self.train_particleRawLoss, self.train_particleCardLoss, self.particle_vars,\
        self.train_particles, self.train_latents =\
            self.build_network(True, False)

        # self.val_particleRawLoss, self.val_particleCardLoss, _, _, _ =\
        #     self.build_network(False, True)

        # self.train_particleLoss = self.train_particleCardLoss
        # self.val_particleLoss = self.val_particleCardLoss

        self.train_particleLoss = self.train_particleRawLoss + self.train_particleCardLoss
        # self.train_particleLoss = self.train_particleCardLoss + 100 * self.train_particleRawLoss
        # self.val_particleLoss = self.val_particleRawLoss
        # self.val_particleLoss = self.val_particleCardLoss + 100 * self.val_particleRawLoss

        self.train_op = self.optimizer.minimize(self.train_particleLoss)
        # self.train_op = self.optimizer.minimize(self.train_particleLoss, var_list = self.particle_vars)
        # self.train_op = tf.constant(0, shape=[10, 10])
