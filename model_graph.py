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

# from Kuhn_Munkres import KM
# from BN16 import BatchNormalizationF16

from time import gmtime, strftime

from external.structural_losses.tf_approxmatch import approx_match, match_cost
from external.sampling.tf_sampling import farthest_point_sample, prob_sample

default_dtype = tf.float32
SN = False

def batch_norm(inputs, decay, is_train, name):

    decay = 0.965

    # Disable norm
    return inputs

    # if default_dtype == tf.float32:
    #     return tf.keras.layers.BatchNormalization(momentum = decay)(inputs, training = is_train)
    # if default_dtype == tf.float16:
    #     return BatchNormalizationF16(momentum = decay)(inputs, training = is_train)
   
    # return tf.keras.layers.BatchNormalization(momentum = decay)(inputs, training = is_train)
    # return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, fused = True)
    
    # Batch norm
    # return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, scope = name, fused = True)
    
    # Layer norm
    # return tf.contrib.layers.layer_norm(inputs, scope = name)
    
    # Instance norm 
    if False:
        if default_dtype == tf.float32:
            return tf.contrib.layers.instance_norm(inputs, scope = name)
        else:
            return tf.contrib.layers.instance_norm(inputs, epsilon = 1e-3, scope = name)
    # return tf.contrib.layers.group_norm(inputs, 

# TODO: use Spec norm
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def fc_as_conv_SN(inputs, outDim, act = None, bias = True, name = 'fc'):
    
    input_shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, input_shape[-1]])

    with tf.variable_scope(name):
        
        w = tf.get_variable('W', shape = [input_shape[-1], outDim], dtype = default_dtype)

        if SN:
            x = tf.matmul(inputs, spectral_norm(w))
        else:
            x = tf.matmul(inputs, w)

        if bias == True:
            b = tf.get_variable('b', shape = [outDim], dtype = default_dtype)
            x = tf.nn.bias_add(x, b)
        
        x_shape = input_shape
        x_shape[-1] = outDim
        x = tf.reshape(x, x_shape)

        if act is not None:
            x = act(x)
        return x

# Laplacian is always FP32
def Laplacian(bs, N, k, kNNIdx, name = 'kNNG_laplacian'):

    with tf.variable_scope(name):
        
        Ns = tf.broadcast_to(tf.reshape(tf.range(N), [1, N, 1]), [bs, N, k])
        _ustack = tf.unstack(kNNIdx, axis = -1) # list of [bs, N, k]
        kNNIdx_withN = tf.stack([_ustack[0], Ns, _ustack[1]], axis = -1) # [bs, N, k, 3], containing [#batch, #start, #end] with #start #end in [0, N)

        # Translate a directed graph to undirected graph by removing the direction of edges, in order to obtain a real symmtric matrix L.
        A = tf.scatter_nd(kNNIdx_withN, tf.constant(True, shape = [bs, N, k]), [bs, N, N], name = 'A')
        # print(A.shape)
        A_T = tf.transpose(A, [0, 2, 1])
        A_undirected = tf.math.logical_or(A, A_T)
        A = tf.cast(A_undirected, tf.float16, name = 'A_undirected') # [bs, N, N]
        # print(A.shape)

        # D = tf.matrix_set_diag(tf.zeros([bs, N, N], tf.float32), tf.reduce_sum(A, axis = -1)) # [bs, N] -> [bs, N, N]
        # print(D.shape)
        L = tf.matrix_set_diag(-A, tf.reduce_sum(A, axis = -1) - 1) # We have self-loops
        # print(L.shape)

        # Normalizations for the laplacian?

        return tf.cast(L, default_dtype), 0, 0

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

        print("bip-kNNG-gen: %4d -> %4d, kernel = %3d" % (Nx, Ny, k))

        posX = Xs[:, :, :pos_range]
        posY = Ys[:, :, :pos_range]
        drow = tf.cast(tf.reshape(posX, [bs, Nx, 1, pos_range]), tf.float16) # duplicate for row
        dcol = tf.cast(tf.reshape(posY, [bs, 1, Ny, pos_range]), tf.float16) # duplicate for column
        
        local_pos = drow - dcol #[bs, Nx, Ny, 3]
        # minusdist = -tf.sqrt(tf.reduce_sum(tf.square(local_pos), axis = 3))
        # minusdist = -tf.sqrt(tf.add_n(tf.unstack(tf.square(local_pos), axis = 3))) # Will this be faster?
        minusdist = -tf.norm(local_pos, ord = 'euclidean', axis = -1)

        _kNNEdg, _TopKIdx = tf.nn.top_k(minusdist, k)
        TopKIdx = _TopKIdx[:, :, :] # No self-loops? (Separated branch for self-conv)
        # TopKIdx = _TopKIdx # Have self-loops?
        kNNEdg = -_kNNEdg[:, :, :] # Better methods?
        kNNEdg = tf.stop_gradient(kNNEdg) # Don't flow gradients here to avoid nans generated for unselected edges
        kNNEdg = tf.cast(tf.reshape(kNNEdg, [bs, Nx, k, 1]), default_dtype)

        # Build NxKxC Neighboor tensor
        # Create indices
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1, 1]), [bs, Nx, k])
        kNNIdx = tf.stack([batches, TopKIdx], axis = -1)
        
        Ns = tf.broadcast_to(tf.reshape(tf.range(Nx), [1, Nx, 1]), [bs, Nx, k])
        gather_lpos_indices = tf.stack([batches, Ns, TopKIdx], axis = -1)

        # [x, y, z], 1st order moment
        neighbor_pos = tf.cast(tf.gather_nd(local_pos, gather_lpos_indices), default_dtype) # [bs, Nx, k, 3]

        # [xx, xy, xz, yx, yy, yz, zx, zy, zz], 2nd order moment
        # neighbor_pos_rs = tf.reshape(neighbor_pos, [bs, Nx, k, 3, 1])
        # neighbor_quadratic = tf.reshape(tf.multiply(neighbor_pos_rs, tf.transpose(neighbor_pos_rs, perm = [0, 1, 2, 4, 3])), [bs, Nx, k, 9])

        kNNEdg = tf.concat([neighbor_pos], axis = -1) # [bs, Nx, k, eC]
        # kNNEdg = tf.concat([kNNEdg, neighbor_pos, neighbor_quadratic], axis = -1) # [bs, Nx, k, eC]

        return posX, posY, kNNIdx, kNNEdg

def kNNG_gen(inputs, k, pos_range, name = 'kNNG_gen'):

    p, _, idx, edg = bip_kNNG_gen(inputs, inputs, k, pos_range, name)
    return p, idx, edg

# Inputs: [bs, Nx, Cx] [bs, Ny, Cy]
# kNNIdx: [bs, Nx, k]
# kNNEdg: [bs, Nx, k, eC]
# Edges are X -> Y
def bip_kNNGConvLayer_concatMLP(Xs, Ys, kNNIdx, kNNEdg, act, channels, no_act_final = False, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGConvNaive'):
    
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
        W_neighbor = tf.get_variable('W_neighbor', dtype = default_dtype, shape = [1, Cx+Cy+eC, channels], initializer = W_init, trainable=True)
        b_neighbor = tf.get_variable('b_neighbor', dtype = default_dtype, shape = [channels], initializer = b_init, trainable=True)

        resnbr = tf.nn.conv1d(rs_neighbors, W_neighbor, 1, padding = 'SAME')
        resnbr = tf.nn.bias_add(resnbr, b_neighbor)
        resnbr = act(resnbr)
        # resnbr = tf.reshape(resnbr, [bs, Nx, k, channels])

        # Collect edge masks
        W_edges = tf.get_variable("W_edges", dtype = default_dtype, shape = [1, eC, channels], initializer = W_init, trainable=True)
        b_edges = tf.get_variable("b_edges", dtype = default_dtype, shape = [channels], initializer = b_init, trainable=True)

        resedg = tf.nn.conv1d(rs_knnedg, W_edges, 1, padding = 'SAME')
        resedg = tf.nn.bias_add(resedg, b_edges)
        resedg = act(resedg)
        # resedg = tf.nn.softmax(resedg, axis = -1)
        # resedg = tf.reshape(resedg, [bs, Nx, k, channels])

        # resnbr = tf.multiply(resnbr, resedg)
        resnbr = tf.concat([resnbr, resedg], axis = -1)
        W_nb2 = tf.get_variable('W_neighbor2', dtype = default_dtype, shape = [1, channels*2, channels], initializer = W_init, trainable=True)
        b_nb2 = tf.get_variable('b_neighbor2', dtype = default_dtype, shape = [channels], initializer = b_init, trainable=True)
        resnbr = tf.nn.conv1d(resnbr, W_nb2, 1, padding = 'SAME')
        resnbr = tf.nn.bias_add(resnbr, b_nb2)
        resnbr = act(resnbr)

        resnbr = tf.reshape(resnbr, [bs, Nx, k, channels])
        resnbr = tf.reduce_sum(resnbr, axis = 2) # combine_method?

        W_self = tf.get_variable('W_self', dtype = default_dtype, shape = [1, Cx + channels, channels], initializer = W_init, trainable=True)
        b_self = tf.get_variable('b', dtype = default_dtype, shape = [channels], initializer = b_init, trainable=True)
        res    = tf.nn.conv1d(tf.concat([Xs, resnbr], axis = -1), W_self, 1, padding = 'SAME')
        
        res = tf.nn.bias_add(res, b_self)

        if not no_act_final:
            res = act(res)

    return res, [W_neighbor, b_neighbor, W_edges, b_edges, W_self, b_self] # [bs, Nx, channels]

def bip_kNNGConvLayer_concat(Xs, Ys, kNNIdx, kNNEdg, act, channels, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGConvNaive'):
    
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
        neighbors = tf.concat([neighbors, tf.broadcast_to(tf.reshape(Xs, [bs, Nx, 1, Cx]), [bs, Nx, k, Cx]), kNNEdg], axis = -1) # [bs, Nx, k, Cx+Cy+eC]

        ### Do the convolution ###

        # Collect neightbors ("M" stage)
        W_neighbor = tf.get_variable('W_neighbor', dtype = default_dtype, shape = [1, 1, Cx+Cy+eC, channels], initializer = W_init, trainable=True)
        b_neighbor = tf.get_variable('b_neighbor', dtype = default_dtype, shape = [channels], initializer = b_init, trainable=True)

        res = tf.nn.conv2d(neighbors, W_neighbor, [1, 1, 1, 1], padding = 'SAME')
        # res = tf.reduce_max(res, axis = 2) # combine_method?
        res = tf.reduce_sum(res, axis = 2) # combine_method?
        # res = tf.add_n(tf.unstack(res, axis = 2)) # combine_method? # nearly the same performance
        res = tf.nn.bias_add(res, b_neighbor)

        if act:
            res = act(res)

    return res, [W_neighbor, b_neighbor] # [bs, Nx, channels]

def bip_kNNGConvLayer_feature(Xs, Ys, kNNIdx, kNNEdg, act, channels, is_train, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGConvNaive'):
    
    global global_is_train

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
        # neighbors = tf.concat([neighbors, kNNEdg], axis = -1) # [bs, Nx, k, Cx+Cy+eC]

        fCh = 6
        if channels > 32:
            fCh = 12

        ### Do the convolution ###
        mlp = [channels]
        n = kNNEdg
        for i in range(len(mlp)):
            if SN:
                n = fc_as_conv_SN(n, mlp[i], tf.nn.elu, name = 'kernel/mlp%d' % i)
            else:
                n = tf.contrib.layers.conv2d(n, mlp[i], [1, 1], padding = 'SAME', activation_fn = tf.nn.elu, scope = 'kernel/mlp%d' % i, weights_initializer = W_init)
                n = batch_norm(n, 0.999, is_train, 'kernel/bn')
        
        if SN:
            n = fc_as_conv_SN(n, channels * fCh, tf.nn.tanh, name = 'kernel/mlp_out')
        else:
            n = tf.contrib.layers.conv2d(n, channels * fCh, [1, 1], padding = 'SAME', activation_fn = tf.nn.tanh, scope = 'kernel/mlp_out', weights_initializer = W_init)
        
        cW = tf.reshape(n, [bs, Nx, k, channels, fCh])
        
        # Batch matmul won't work for more than 65535 matrices ???
        # n = tf.matmul(n, tf.reshape(neighbors, [bs, Nx, k, Cy, 1]))
        # Fallback solution
        if SN:
            n = fc_as_conv_SN(neighbors, channels * fCh, None, name = 'feature/feature_combine')
        else:
            n = tf.contrib.layers.conv2d(neighbors, channels * fCh, [1, 1], padding = 'SAME', activation_fn = None, scope = 'feature/feature_combine', weights_initializer = W_init)
            n = batch_norm(n, 0.999, is_train, 'feature/bn')

        # MatMul
        n = tf.reshape(n, [bs, Nx, k, channels, fCh])
        n = tf.reduce_sum(tf.multiply(cW, n), axis = -1)

        print(n.shape)
        print("Graph cConv: [%3d x %2d] = %4d" % (channels, fCh, channels * fCh))
        # n = tf.reshape(n, [bs, Nx, k, channels])

        b = tf.get_variable('b_out', dtype = default_dtype, shape = [channels], initializer = b_init, trainable = True)
        n = tf.reduce_mean(n, axis = 2)
        n = tf.nn.bias_add(n, b)
        
        if act is not None:
            n = act(n)

    return n, [b] # [bs, Nx, channels]

def bip_kNNGConvLayer_edgeMask(Xs, Ys, kNNIdx, kNNEdg, act, channels, no_act_final = False, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGConvNaive'):

    with tf.variable_scope(name):

        bs = Xs.shape[0]
        Nx = Xs.shape[1]
        Ny = Ys.shape[1]
        Cx = Xs.shape[2]
        Cy = Ys.shape[2]
        k = kNNIdx.shape[2]
        eC = kNNEdg.shape[3]

        W_edge = tf.get_variable('W_edge', dtype = default_dtype, shape = [1, 1, eC, channels], initializer = W_init, trainable = True)
        b_edge = tf.get_variable('b_edge', dtype = default_dtype, shape = [channels], initializer = b_init, trainable = True)

        mask = tf.nn.conv2d(kNNEdg, W_edge, [1, 1, 1, 1], padding = 'SAME')
        mask = tf.nn.bias_add(mask, b_edge)
        mask = tf.nn.sigmoid(mask)

        neighbors = tf.gather_nd(Ys, kNNIdx)
        # neighbors: Edge u-v = [u;v;edg]
        neighbors = tf.concat([neighbors, tf.broadcast_to(tf.reshape(Xs, [bs, Nx, 1, Cx]), [bs, Nx, k, Cx]), kNNEdg], axis = -1) # [bs, Nx, k, Cx+Cy+eC]

        ### Do the convolution ###

        # Collect neightbors ("M" stage)
        W_neighbor = tf.get_variable('W_neighbor', dtype = default_dtype, shape = [1, 1, Cx+Cy+eC, channels], initializer = W_init, trainable=True)
        b_neighbor = tf.get_variable('b_neighbor', dtype = default_dtype, shape = [channels], initializer = b_init, trainable=True)

        res = tf.nn.conv2d(neighbors, W_neighbor, [1, 1, 1, 1], padding = 'SAME')
        res = tf.multiply(res, mask)
        res = tf.reduce_sum(res, axis = 2) # combine_method?
        # res = tf.add_n(tf.unstack(res, axis = 2)) # combine_method? # nearly the same performance
        res = tf.nn.bias_add(res, b_neighbor)

        if act:
            res = act(res)
    return res, [W_edge, b_edge, W_neighbor, b_neighbor]

# Inputs: [bs, N, C]
#    Pos: [bs, N, 3]
def kNNGPooling_farthest(inputs, pos, k):

    # with tf.variable_scope(name):

    bs = pos.shape[0]
    N = pos.shape[1]
    k = min(N, k)

    idx = farthest_point_sample(k, pos) # [bs, k]

    # Pick them
    batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, k])
    gather_idx = tf.stack([batches, idx], axis = -1)
    pool_features = tf.gather_nd(inputs, gather_idx) # [bs, k, C]
    pool_position = tf.gather_nd(pos, gather_idx) # [bs, k, 3]
    
    return pool_position, pool_features, 0.0, [], 0.0

# Inputs: [bs, N, C]
#    Pos: [bs, N, 3]
def kNNGPooling_rand(inputs, pos, bs, N, k, laplacian, masking = True, channels = 1, W_init = tf.truncated_normal_initializer(stddev=0.1), name = 'kNNGPool'):

    with tf.variable_scope(name):

        k = min(N, k)

        y = tf.random.uniform([bs, N]) # [bs, N]
        val, idx = tf.nn.top_k(y, k) # [bs, k]

        # Pick them
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, k])
        gather_idx = tf.stack([batches, idx], axis = -1)
        pool_features = tf.gather_nd(inputs, gather_idx) # [bs, k, C]
        pool_position = tf.gather_nd(pos, gather_idx) # [bs, k, 3]

        if False:
            pool_features = tf.multiply(pool_features, tf.reshape(tf.nn.tanh(val), [bs, k, 1]))
    
    return pool_position, pool_features, y, [], 0.0

# Inputs: [bs, N, C]
#    Pos: [bs, N, 3]
def kNNGPooling_GUnet(inputs, pos, k, masking = True, channels = 1, W_init = tf.truncated_normal_initializer(stddev=0.1), name = 'kNNGPool'):

    with tf.variable_scope(name):

        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N, k)

        W = tf.get_variable('W', dtype = default_dtype, shape = [1, C, channels], initializer=W_init, trainable=True)
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
def kNNGPooling_CAHQ(inputs, pos, k, kNNIdx, kNNEdg, laplacian, is_train, masking = True, channels = 1, W_init = tf.truncated_normal_initializer(stddev=0.1), name = 'kNNGPool', stopGradient = False, act = tf.nn.relu, b_init = None):

    with tf.variable_scope(name):

        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N, k)

        if stopGradient == True:
            inputs = tf.stop_gradient(inputs)

        kNNEdg_dist = tf.norm(kNNEdg, axis = -1, keepdims = True)
        imp = tf.contrib.layers.conv1d(inputs, 1, 1, padding = 'SAME', activation_fn = tf.nn.tanh, scope = 'importance')

        layers = 4
        f = imp
        for i in range(layers):
            tmp = f
            f = tf.gather_nd(f, kNNIdx)
            f = tf.concat([f, kNNEdg_dist], axis = -1) # [bs, N, k, 2]
            f = tf.contrib.layers.conv2d(f, 16, 1, padding = 'SAME', scope = 'mlp%d/h1' % i)
            f = tf.contrib.layers.conv2d(f, 1,  1, padding = 'SAME', scope = 'mlp%d/h2' % i) # [bs, N, k, 1]
            f = tf.reduce_mean(f, axis = 2) # [bs, N, 1]
            f = tf.contrib.layers.conv1d(f, 16, 1, padding = 'SAME', scope = 'mlp%d/ro/h1' % i)
            f = tf.contrib.layers.conv1d(f, 1,  1, padding = 'SAME', scope = 'mlp%d/ro/h2' % i)
            f = batch_norm(f, 0.999, is_train, 'mlp%d/bn' % i)
            f = f + tmp

        y = tf.reshape(f, [bs, N])

        # Freq Loss
        print(laplacian.shape)
        norm_Ly = tf.sqrt(tf.reduce_sum(tf.cast(tf.square(tf.matmul(laplacian, tf.reshape(y, [bs, N, 1]), name = 'L_y')), tf.float32), axis = [1, 2]) + 1e-3)
        norm_y = tf.sqrt(tf.reduce_sum(tf.cast(tf.square(y), tf.float32), axis = 1) + 1e-3)
        freq_loss = norm_Ly / (norm_y + 1e-3) # Maximize this
        freq_loss = 0 - freq_loss # Minimize negate
        # freq_loss = 0

        val, idx = tf.nn.top_k(y, k) # [bs, k]

        # Pick them
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, k])
        gather_idx = tf.stack([batches, idx], axis = -1)
        pool_features = tf.gather_nd(inputs, gather_idx) # [bs, k, C]
        pool_position = tf.gather_nd(pos, gather_idx) # [bs, k, 3]

        if masking == True:
            # pool_features = tf.multiply(pool_features, tf.reshape(tf.nn.tanh(val), [bs, k, 1]))
            pool_features = tf.multiply(pool_features, tf.reshape(val, [bs, k, 1]))
    
    return pool_position, pool_features, y, [], tf.cast(freq_loss, default_dtype)

# Inputs: [bs, N, C]
#    Pos: [bs, N, 3]
def kNNGPooling_HighFreqLoss_GUnet(inputs, pos, k, laplacian, masking = True, channels = 1, W_init = tf.truncated_normal_initializer(stddev=0.1), name = 'kNNGPool', stopGradient = False, act = tf.nn.relu, b_init = None):

    with tf.variable_scope(name):

        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N, k)

        if stopGradient == True:
            inputs = tf.stop_gradient(inputs)

        # Fuse freq features
        fuse_freq = tf.get_variable('freq', dtype = default_dtype, shape = [2], trainable = True, initializer = tf.ones_initializer)
        fuse_1 = tf.math.sin(pos * fuse_freq[0])
        fuse_2 = tf.math.sin(pos * fuse_freq[1])
        tf.summary.scalar('Fuse_freq', fuse_freq[0])
        inputs = tf.concat([inputs, fuse_1, fuse_2], axis = -1)

        W = tf.get_variable('W', dtype = default_dtype, shape = [1, C, channels], initializer=W_init, trainable=True)
        norm = tf.sqrt(tf.reduce_sum(tf.square(W), axis = 1, keepdims = True)) # [1, 1, channels]
        
        # y = tf.nn.conv1d(inputs, W, 1, padding = 'SAME') # [bs, N, channels]
        # y = tf.multiply(y, 1.0 / (norm + 1e-3))
        # y = tf.reduce_mean(y, axis = -1) # [bs, N]

        mlp = [C*2]
        y = inputs
        for l in range(len(mlp)):
            y, _ = Conv1dWrapper(y, mlp[l], 1, 1, 'SAME', act, W_init = W_init, b_init = b_init, name = 'fc%d' % l)
        y, _ = Conv1dWrapper(y, 1, 1, 1, 'SAME', tf.nn.tanh, W_init = W_init, b_init = b_init, name = 'fcOut')
        y = tf.reshape(y, [bs, N])

        # Freq Loss
        print(laplacian.shape)
        norm_Ly = tf.sqrt(tf.reduce_sum(tf.cast(tf.square(tf.matmul(laplacian, tf.reshape(y, [bs, N, 1]), name = 'L_y')), tf.float32), axis = [1, 2]) + 1e-3)
        norm_y = tf.sqrt(tf.reduce_sum(tf.cast(tf.square(y), tf.float32), axis = 1) + 1e-3)
        freq_loss = norm_Ly / (norm_y + 1e-3) # Maximize this
        freq_loss = 0 - freq_loss # Minimize negate
        # freq_loss = 0

        val, idx = tf.nn.top_k(y, k) # [bs, k]

        # Pick them
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, k])
        gather_idx = tf.stack([batches, idx], axis = -1)
        pool_features = tf.gather_nd(inputs, gather_idx) # [bs, k, C]
        pool_position = tf.gather_nd(pos, gather_idx) # [bs, k, 3]

        if masking == True:
            pool_features = tf.multiply(pool_features, tf.reshape(tf.nn.tanh(val), [bs, k, 1]))
            # pool_features = tf.multiply(pool_features, tf.reshape(val, [bs, k, 1]))
    
    return pool_position, pool_features, y, [W], tf.cast(freq_loss, default_dtype)

def Conv1dWrapper(inputs, filters, kernel_size, stride, padding, act, W_init, b_init, bias = True, name = 'conv'):

    if SN:
        return fc_as_conv_SN(inputs, filters, act, name = name), []
    else:
        with tf.variable_scope(name):

            N = inputs.shape[1]
            C = inputs.shape[2]
            variables = []

            W = tf.get_variable('W', dtype = default_dtype, shape = [kernel_size, C, filters], initializer=W_init, trainable=True)
            variables.append(W)

            y = tf.nn.conv1d(inputs, W, stride, padding = padding)

            if bias == True:
                b = tf.get_variable('b', dtype = default_dtype, shape = [filters], initializer=b_init, trainable=True)
                y = tf.nn.bias_add(y, b)
                variables.append(b)

            if act is not None:
                y = act(y)

            return y, variables

def kNNGPosition_refine(input_position, input_feature, refine_maxLength, act, hidden = 128, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGPosRefine'):

    with tf.variable_scope(name):

        bs = input_position.shape[0]
        N = input_position.shape[1]
        C = input_feature.shape[2]
        pC = input_position.shape[2]

        assert N == input_feature.shape[1] and bs == input_feature.shape[0]

        # pos_feature, vars1 = Conv1dWrapper(tf.concat([input_position, input_feature], axis = -1), hidden, 1, 1, 'SAME', act, W_init, b_init, True, 'hidden')
        pos_res, v = Conv1dWrapper(input_feature, pC, 1, 1, 'SAME', None, W_init, b_init, True, 'refine') # [bs, N, pC]
        pos_norm = tf.norm(pos_res, axis = -1, keepdims = True) # [bs, N, 1]
        pos_norm_tun = tf.nn.tanh(pos_norm) * refine_maxLength
        pos_res = pos_res / (pos_norm + 1) * pos_norm_tun
        # pos_res *= refine_maxLength

        # tf.summary.histogram('Position_Refine_%s' % name, pos_res)

        refined_pos = tf.add(input_position, pos_res)

        return refined_pos, [v]

def bip_kNNGConvBN_wrapper(Xs, Ys, kNNIdx, kNNEdg, batch_size, gridMaxSize, particle_hidden_dim, act, decay = 0.999, is_train = True, name = 'gconv', W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0)):

    with tf.variable_scope(name):
        n, v = bip_kNNGConvLayer_feature(Xs, Ys, kNNIdx, kNNEdg, act = None, channels = particle_hidden_dim, is_train = is_train, W_init = W_init, b_init = b_init, name = 'gc')
        # n, v = bip_kNNGConvLayer_edgeMask(Xs, Ys, kNNIdx, kNNEdg, act = None, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        # n, v = bip_kNNGConvLayer_concat(Xs, Ys, kNNIdx, kNNEdg, act = None, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        # n, v = bip_kNNGConvLayer_concatMLP(Xs, Ys, kNNIdx, kNNEdg, act = act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        
        if False:
        # if True:
            ch = particle_hidden_dim
            mlp = [ch * 2, ch * 2, ch]
            vs = []
            n, v = bip_kNNGConvLayer_concat(Xs, Ys, kNNIdx, kNNEdg, act = None, channels = mlp[0], W_init = W_init, b_init = b_init, name = 'gconv')
            vs.append(v)
            for i in range(1, len(mlp)):
                n = batch_norm(n, decay, is_train, name = 'bn%d' % (i-1))
                if act:
                    n = act(n)
                n, v = Conv1dWrapper(n, mlp[i], 1, 1, 'SAME', None, W_init, b_init, True, 'fc%d' % i)
        
        n = batch_norm(n, decay, is_train, name = 'bn')
        if act:
            n = act(n)

    return n, v

def kNNGConvBN_wrapper(inputs, kNNIdx, kNNEdg, batch_size, gridMaxSize, particle_hidden_dim, act, decay = 0.999, is_train = True, name = 'gconv', W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0)):
    return bip_kNNGConvBN_wrapper(inputs, inputs, kNNIdx, kNNEdg, batch_size, gridMaxSize, particle_hidden_dim, act, decay, is_train, name, W_init, b_init)
        
# TODO: position re-fine layer

class model_particles:

    def __init__(self, gridMaxSize, latent_dim, batch_size, optimizer, outDim):
        
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

        self.doSim = True
        self.doLoop = True
        self.loops = 30
        self.normalize = 1.0

        self.outDim = outDim

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

        self.ph_X           = tf.placeholder(default_dtype, [self.batch_size, self.gridMaxSize, outDim + 1]) # x y z vx vy vz 1
        self.ph_Y           = tf.placeholder(default_dtype, [self.batch_size, self.gridMaxSize, outDim + 1])
        self.ph_L           = tf.placeholder(default_dtype, [self.batch_size, self.gridMaxSize, outDim + 1]) # Loop simulation (under latent space) ground truth

        self.ph_card        = tf.placeholder(default_dtype, [self.batch_size]) # card
        self.ph_max_length  = tf.placeholder('int32', [2])

        self.optimizer = optimizer

    # 1 of a batch goes in this function at once.
    def particleEncoder(self, input_particle, output_dim, is_train = False, reuse = False, returnPool = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        w_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleEncoder", reuse = reuse) as vs:

            # We are going to use a way deeper model than before. Please refer to model_particlesTest_backup.py for original codes.

            # I hate *** code.
            # blocks = 5
            # particles_count = [2560, 1280, 640, 320, self.cluster_count]
            # conv_count = [2, 2, 4, 1, 1]
            # res_count = [0, 0, 0, 2, 4]
            # kernel_size = [int(self.knn_k / 1.5), int(self.knn_k / 1.2), self.knn_k, self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [int(hd / 3.2), hd // 2, hd, int(hd * 1.5), hd * 2]
            
            # LJP 2560
            # blocks = 4
            # particles_count = [self.gridMaxSize, 1280, 512, self.cluster_count]
            # conv_count = [3, 2, 2, 2]
            # res_count = [0, 0, 1, 1]
            # kernel_size = [self.knn_k, self.knn_k, self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [hd // 3, hd // 2, hd, hd * 2]
            
            # LJP shallow
            # blocks = 3
            particles_count = [self.gridMaxSize, 768, self.cluster_count]
            # conv_count = [2, 3, 2]
            # res_count = [0, 0, 1]
            kernel_size = [self.knn_k, self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [hd // 2, hd, hd * 2]

            # Test
            # blocks = 2
            # particles_count = [self.gridMaxSize, self.cluster_count]
            # conv_count = [3, 2]
            # res_count = [0, 0]
            # kernel_size = [self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [hd, hd]

            # LJP Deep
            # blocks = 5
            # particles_count = [self.gridMaxSize, 1024, 512, 256, self.cluster_count]
            # conv_count = [4, 2, 2, 0, 0]
            # res_count = [0, 0, 1, 1, 2]
            # kernel_size = [6, 8, 12, self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [16, 32, hd, hd*2, hd*4]

            # ShapeNet_NEWconvConcat and Fluid_NEWconvConcat
            # blocks = 5
            # particles_count = [self.gridMaxSize, 1920, 768, 256, self.cluster_count]
            # conv_count = [2, 3, 2, 2, 2]
            # res_count = [0, 0, 1, 2, 2]
            # kernel_size = [6, 8, 12, self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [16, 32, hd, hd*2, hd*4]

            # ShapeNet_shallow_uniform_NEWconvConcat
            # blocks = 3
            # particles_count = [self.gridMaxSize, 1920, self.cluster_count]
            # conv_count = [2, 3, 2]
            # res_count = [0, 0, 1]
            # kernel_size = [self.knn_k, self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [hd // 2, hd, hd*2]
            
            # ShapeNet_shallow_feature
            # blocks = 3
            # particles_count = [self.gridMaxSize, 1920, self.cluster_count]
            # conv_count = [1, 2, 0]
            # res_count = [0, 0, 2]
            # kernel_size = [self.knn_k, self.knn_k, self.knn_k]
            # bik = [0, 32, 64]
            # hd = self.particle_hidden_dim
            # channels = [hd // 2, hd, hd * 2]
            
            # ShapeNet_regular_featureSqz
            blocks = 5
            particles_count = [self.gridMaxSize, 1920, 768, max(256, self.cluster_count * 2), self.cluster_count]
            conv_count = [1, 2, 2, 0, 0]
            res_count = [0, 0, 0, 1, 2]
            kernel_size = [self.knn_k, self.knn_k, self.knn_k, self.knn_k, min(self.knn_k, particles_count[4])]
            bik = [0, 32, 32, 48, 64]
            hd = self.particle_hidden_dim
            channels = [hd // 2, 2 * hd // 3, hd, 3 * hd // 2, max(self.particle_latent_dim, hd * 2)]
            
            res_count[4] = 6

            # ShapeNet_SingleVector
            # blocks = 5
            # particles_count = [self.gridMaxSize, 1920, 768, max(256, self.cluster_count * 2), self.cluster_count]
            # conv_count = [1, 2, 2, 0, 1]
            # res_count = [0, 0, 0, 1, 0]
            # kernel_size = [self.knn_k, self.knn_k, self.knn_k, self.knn_k, min(self.knn_k, particles_count[4])]
            # bik = [0, 32, 32, 48, 256]
            # hd = self.particle_hidden_dim
            # channels = [hd // 2, 2 * hd // 3, hd, 3 * hd // 2, max(self.particle_latent_dim, hd * 2)]
            
            # bik = [0, 4, 32]
            # channels = [hd // 8, hd // 6, hd // 4]

            # ShapeNet_shallow_uniform_convConcatSimpleMLP
            # blocks = 3
            # particles_count = [self.gridMaxSize, 1920, self.cluster_count]
            # conv_count = [1, 1, 1]
            # res_count = [0, 0, 1]
            # kernel_size = [self.knn_k, self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [hd // 2, hd, hd*2]
            
            # ShapeNet_deepshallow_uniform_convConcatSimpleMLP
            # blocks = 7
            # particles_count = [self.gridMaxSize, 2560, 1280, 512, 256, 128, self.cluster_count]
            # conv_count = [2, 1, 1, 1, 0, 0, 0]
            # res_count = [0, 0, 0, 0, 1, 1, 2]
            # kernel_size = [self.knn_k for i in range(7)]
            # hd = self.particle_hidden_dim
            # channels = [hd // 2, hd // 2, hd, hd, hd, hd * 2, hd * 2]

            # ShapeNet_deep_uniform_edgeMask
            # blocks = 5
            # particles_count = [self.gridMaxSize, 1920, 768, 256, self.cluster_count]
            # conv_count = [4, 2, 0, 0, 0]
            # res_count = [0, 0, 1, 2, 2]
            # kernel_size = [self.knn_k // 2, self.knn_k, self.knn_k, self.knn_k, self.knn_k]
            # hd = self.particle_hidden_dim
            # channels = [hd // 2, int(hd / 1.4), hd, int(hd * 1.5), hd * 2]

            self.pool_count = blocks - 1
            self.pCount = particles_count

            try:
                bik
            except NameError:
                bik = kernel_size
            
            gPos = input_particle[:, :, :3]
            n = input_particle[:, :, self.outDim:] # Ignore velocity
            var_list = []
            pool_pos = []
            pool_eval_func = []
            freq_loss = 0

            for i in range(blocks):
                
                if i > 0:
                    
                    # Pooling
                    prev_n = n
                    prev_pos = gPos
                    # gPos, n, eval_func, v, fl = kNNGPooling_HighFreqLoss_GUnet(n, gPos, particles_count[i], MatL, W_init = w_init, name = 'gpool%d' % i, stopGradient = True)
                    # gPos, n, eval_func, v, fl = kNNGPooling_CAHQ(n, gPos, particles_count[i], gIdx, gEdg, MatL, masking = True, W_init = w_init, name = 'gpool%d' % i, stopGradient = True)
                    #gPos, n, eval_func, v, fl = kNNGPooling_rand(n, gPos, self.batch_size, particles_count[i-1], particles_count[i], MatL, masking = True, W_init = w_init, name = 'gpool%d' % i)
                    gPos, n, eval_func, v, fl = kNNGPooling_farthest(n, gPos, particles_count[i])
                    
                    # Single point
                    # if i == 4:
                    #     gPos = tf.zeros_like(gPos)

                    var_list.append(v)
                    # pool_eval_func.append(tf.concat([prev_pos, tf.reshape(eval_func, [self.batch_size, particles_count[i-1], 1])], axis = -1))

                    pool_pos.append(gPos)
                    freq_loss = freq_loss + fl

                    # Collect features after pool
                    _, _, bpIdx, bpEdg = bip_kNNG_gen(gPos, prev_pos, bik[i], 3, name = 'gpool%d/ggen' % i)
                    n, _ = bip_kNNGConvBN_wrapper(n, prev_n, bpIdx, bpEdg, self.batch_size, particles_count[i], channels[i], self.act, is_train = is_train, W_init = w_init, b_init = b_init, name = 'gpool%d/gconv' % i)

                gPos, gIdx, gEdg = kNNG_gen(gPos, kernel_size[i], 3, name = 'ggen%d' % i)
                # MatL, MatA, MatD = Laplacian(self.batch_size, particles_count[i], kernel_size[i], gIdx, name = 'gLaplacian%d' % i)

                for c in range(conv_count[i]):

                    n, v = kNNGConvBN_wrapper(n, gIdx, gEdg, self.batch_size, particles_count[i], channels[i], self.act, 0.999, is_train = is_train, W_init = w_init, b_init = b_init, name = 'g%d/gconv%d' % (i, c))
                    var_list.append(v)

                tmp = n
                for r in range(res_count[i]):
                
                    nn, v = kNNGConvBN_wrapper(n, gIdx, gEdg, self.batch_size, particles_count[i], channels[i], self.act, 0.999, is_train = is_train, W_init = w_init, b_init = b_init, name = 'g%d/res%d/conv1' % (i, r))
                    var_list.append(v)
                    
                    nn, v = kNNGConvBN_wrapper(nn, gIdx, gEdg, self.batch_size, particles_count[i], channels[i], self.act, 0.999, is_train = is_train, W_init = w_init, b_init = b_init, name = 'g%d/res%d/conv2' % (i, r))
                    var_list.append(v)
                    n = n + nn
                
                if res_count[i] > 1:
                    n = n + tmp
            
            # tf.summary.histogram('Pooled_clusters_pos', gPos)
            n, v = Conv1dWrapper(n, self.cluster_feature_dim, 1, 1, 'SAME', None, w_init, b_init, True, 'convOut')
            var_list.append(v)
            
            if returnPool == True:
                return gPos, n, var_list, pool_pos, freq_loss, pool_eval_func

            return gPos, n, var_list, freq_loss, pool_eval_func
    
    def particleDecoder(self, cluster_pos, local_feature, groundTruth_card, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        w_init_fold = tf.random_normal_initializer(stddev= 1.0*self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev=0.03*self.wdev)
        
        w_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype)
        w_init_fold = w_init
        w_init_pref = w_init
        
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleDecoder", reuse = reuse) as vs:
 
            CC = local_feature.shape[2]

            if False: # Original approach

                global_latent, v = Conv1dWrapper(local_feature, self.particle_latent_dim, 1, 1, 'SAME', None, w_init, b_init, True, 'convGlobal')
                global_latent = self.combine_method(global_latent, axis = 1)

                # Folding stage
                fold_particles_count = self.gridMaxSize - self.cluster_count
                # net_input = InputLayer(input_latent, name = 'input')
                
                # FIXME: no card in this model

                # generate random noise
                pos_range = 3
                # use gaussian for fluid
                # z = tf.random.normal([self.batch_size, fold_particles_count, self.particle_latent_dim * 2], dtype = default_dtype)
                # but uniform should be way better
                # z = tf.random.uniform([self.batch_size, fold_particles_count, self.particle_latent_dim * 2], minval = -1., maxval = 1., dtype = default_dtype)
                z = tf.random.uniform([self.batch_size, fold_particles_count, 3], minval = -1., maxval = 1., dtype = default_dtype)

                # conditional generative network (FOLD Stage)
                latents = \
                    tf.broadcast_to\
                    (\
                        tf.reshape(global_latent, [self.batch_size, 1, self.particle_latent_dim]),\
                        [self.batch_size, fold_particles_count, self.particle_latent_dim]\
                    )

                pos = z

                c = tf.concat([pos, latents], axis = -1)

                global_fold = 3
                for i in range(global_fold):

                    c, v = Conv1dWrapper(c, self.particle_hidden_dim, 1, 1, 'SAME', None, w_init_fold, b_init, True, 'fold/fc%d' % i)
                    c = batch_norm(c, 0.999, is_train, name = 'fold/fc%d/bn' % i)
                    c = self.act(c)

                alter_particles, v = Conv1dWrapper(c, pos_range, 1, 1, 'SAME', None, w_init, b_init, True, 'fold/fc_out')

                fold_before_prefine = tf.concat([alter_particles, cluster_pos], axis = 1)

                # tf.summary.histogram('Particles_AfterFolding', alter_particles)

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
                    n, v = bip_kNNGConvBN_wrapper(n, local_feature, gp_idx, gp_edg, self.batch_size, fold_particles_count, self.particle_hidden_dim // 2, self.act, is_train = is_train, name = 'gconv%d_pre' % i, W_init = w_init)
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
                # refine_loops = 0
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

                final_particles = pos
                
                if output_dim > pos_range:
                    n, _ = Conv1dWrapper(n, output_dim - pos_range, 1, 1, 'SAME', None, w_init, b_init, True, 'finalConv')
                    final_particles = tf.concat([pos, n], -1)
                return 0, [final_particles, fold_before_prefine], 0
            
            else: # New approach, local generation, fold-refine blocks
                
                hd = self.particle_hidden_dim
                ld = self.particle_latent_dim
                _k = self.knn_k

                # Single decoding stage
                coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
                blocks = 1
                pcnt = [self.gridMaxSize] # particle count
                generator = [6] # Generator depth
                refine = [0] # refine steps (each refine step = 1x res block (2x gconv))
                refine_res = [1]
                refine_maxLength = [0.6]
                hdim = [self.particle_hidden_dim // 3]
                fdim = [self.particle_latent_dim] # dim of features used for folding
                gen_hdim = [self.particle_latent_dim]
                knnk = [self.knn_k // 2]

                # Multiple stacks
                # coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
                # blocks = 2
                # pcnt = [768, self.gridMaxSize] # particle count
                # generator = [4, 3] # Generator depth
                # refine = [2, 1] # refine steps (each refine step = 1x res block (2x gconv))
                # hdim = [self.particle_hidden_dim, self.particle_hidden_dim // 3]
                # fdim = [self.particle_latent_dim, self.particle_latent_dim] # dim of features used for folding
                # gen_hdim = [self.particle_latent_dim, self.particle_latent_dim]
                # knnk = [self.knn_k, self.knn_k // 2]
                
                # [fullFC_regular, fullGen_regular] Setup for full generator - fully-connected
                # coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
                # blocks = 3
                # pcnt = [256, 1280, self.gridMaxSize] # particle count
                # generator = [4, 4, 4] # Generator depth
                # refine = [2, 1, 1] # Regular setup
                # refine_maxLength = [2.0, 1.0, 0.5]
                # refine = [1, 0, 0] # variant / refine steps (each refine step = 1x res block (2x gconv))
                # refine_res = [1, 1, 1]
                # hdim = [hd * 2, hd, hd // 3]
                # fdim = [ld, ld, ld // 2] # dim of features used for folding
                # gen_hdim = [ld, ld, ld]
                # knnk = [_k, _k, _k // 2]
                
                # [fullGen_shallow]
                coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
                blocks = 2
                pcnt = [1280, self.gridMaxSize] # particle count
                generator = [6, 3] # Generator depth
                refine = [0, 0] # refine steps (each refine step = 1x res block (2x gconv))
                refine_res = [1, 1]
                hdim = [self.particle_hidden_dim, self.particle_hidden_dim // 3]
                fdim = [self.particle_latent_dim, self.particle_latent_dim] # dim of features used for folding
                gen_hdim = [self.particle_latent_dim, self.particle_latent_dim]
                knnk = [self.knn_k, self.knn_k // 2]

                pos_range = 3

                gen_only = []

                regularizer = 0.0

                for bi in range(blocks):

                    with tf.variable_scope('gr%d' % bi):
 
                        if True: # Fully-connected generator (Non-distribution-based) & Full generators (pcnt[bi] instead of pcnt[bi] - coarse_cnt
                        
                            # Check for good setups
                            assert pcnt[bi] % coarse_cnt == 0

                            n_per_cluster = pcnt[bi] // coarse_cnt

                            if False: # fc
                                n = coarse_fea
                                for gi in range(generator[bi]):
                                    with tf.variable_scope('gen%d' % gi):
                                        n, v = Conv1dWrapper(n, fdim[bi], 1, 1, 'SAME', None, w_init, b_init, True, 'fc')
                                        n = batch_norm(n, 0.999, is_train, name = 'norm')
                                        n = self.act(n)
                                n, v = Conv1dWrapper(n, pos_range * n_per_cluster, 1, 1, 'SAME', None, w_init, b_init, True, 'gen_out')
                                n = tf.reshape(n, [self.batch_size, coarse_cnt, n_per_cluster, pos_range])

                                # Back to world space
                                n = n + tf.reshape(coarse_pos, [self.batch_size, coarse_cnt, 1, pos_range])
                                
                                ap = tf.reshape(n, [self.batch_size, pcnt[bi], pos_range])

                            else: # generator
                                z = tf.random.uniform([self.batch_size, coarse_cnt, n_per_cluster, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)
                                fuse_fea, v = Conv1dWrapper(coarse_fea, fdim[bi], 1, 1, 'SAME', None, w_init_fold, b_init, True, 'feaFuse')
                                z = tf.concat([z, tf.broadcast_to(tf.reshape(fuse_fea, [self.batch_size, coarse_cnt, 1, fdim[bi]]), [self.batch_size, coarse_cnt, n_per_cluster, fdim[bi]])], axis = -1)
                                
                                n = tf.reshape(z, [self.batch_size, pcnt[bi], fdim[bi] * 2])
                                
                                for gi in range(generator[bi]):
                                    with tf.variable_scope('gen%d' % gi):
                                        n, v = Conv1dWrapper(n, fdim[bi], 1, 1, 'SAME', None, w_init, b_init, True, 'fc')
                                        n = batch_norm(n, 0.999, is_train, name = 'norm')
                                        n = self.act(n)
                                n, v = Conv1dWrapper(n, pos_range, 1, 1, 'SAME', None, w_init, b_init, True, 'gen_out')
                                n = tf.reshape(n, [self.batch_size, coarse_cnt, n_per_cluster, pos_range])

                                # Back to world space
                                n = n + tf.reshape(coarse_pos, [self.batch_size, coarse_cnt, 1, pos_range])
                                
                                ap = tf.reshape(n, [self.batch_size, pcnt[bi], pos_range])

                            # General operations for full generators
                            gen_only.append(ap)

                            # Empty feature
                            # n = tf.zeros([self.batch_size, pcnt[bi], 1], dtype = default_dtype)

                            # Outputs of this stage
                            pos = ap
                            # n = n

                        else:
                            
                            # Folding stage
                            fold_particles_count = pcnt[bi] - coarse_cnt
                            
                            # Mixture
                            mix = tf.random.uniform([self.batch_size, fold_particles_count, 1], maxval = coarse_cnt, dtype = tf.int32)

                            # Coarse graph: [bs, coarse_cnt, coarse_hdim]
                            bs_idx = tf.broadcast_to(tf.reshape(tf.range(self.batch_size), [self.batch_size, 1, 1]), [self.batch_size, fold_particles_count, 1])
                            gather_idx = tf.concat([bs_idx, mix], axis = -1)
                            origin_pos = tf.gather_nd(coarse_pos, gather_idx)
                            origin_fea = tf.gather_nd(coarse_fea, gather_idx)

                            z = tf.random.uniform([self.batch_size, fold_particles_count, fdim[bi] * 2], minval = -1., maxval = 1., dtype = default_dtype)

                            if False: # Fuse feature to every layer, maybe stupid...?
                                for gi in range(generator[bi]):
                                    
                                    with tf.variable_scope('gen%d' % gi):
                                        fuse_fea, v = Conv1dWrapper(origin_fea, fdim[bi], 1, 1, 'SAME', self.act, w_init_fold, b_init, True, 'feaFuse')
                                        
                                        z = tf.concat([z, fuse_fea], axis = -1)
                                        z, v = Conv1dWrapper(z, fdim[bi], 1, 1, 'SAME', None, w_init_fold, b_init, True, 'fc')
                                        z = batch_norm(z, 0.999, is_train, name = 'bn')
                                        z = self.act(z)

                            elif False: # Regular small generator
                                fuse_fea, v = Conv1dWrapper(origin_fea, fdim[bi], 1, 1, 'SAME', self.act, w_init_fold, b_init, True, 'feaFuse')
                                z = tf.concat([z, fuse_fea], axis = -1)

                                for gi in range(generator[bi]):
                                    with tf.variable_scope('gen%d' % gi):
                                        z, v = Conv1dWrapper(z, 2 * fdim[bi], 1, 1, 'SAME', None, w_init_fold, b_init, True, 'fc')
                                        z = batch_norm(z, 0.999, is_train, name = 'bn')
                                        z = self.act(z)
                                
                                z, v = Conv1dWrapper(z, pos_range, 1, 1, 'SAME', None, w_init, b_init, True, 'gen/fc_out') 
                            
                            else: # Advanced conditioned small generator
                                
                                with tf.variable_scope('weight_gen'):
                                    l, v = Conv1dWrapper(origin_fea, gen_hdim[bi], 1, 1, 'SAME', self.act, w_init_fold, b_init, True, 'mlp1')
                                    l = batch_norm(l, 0.999, is_train, name = 'mlp1/bn')
                                    l, v = Conv1dWrapper(l, gen_hdim[bi], 1, 1, 'SAME', self.act, w_init_fold, b_init, True, 'mlp2')
                                    l = batch_norm(l, 0.999, is_train, name = 'mlp2/bn')
                                    w, v = Conv1dWrapper(l, pos_range * fdim[bi], 1, 1, 'SAME', None, w_init_fold, b_init, True, 'mlp_weights_out')
                                    b, v = Conv1dWrapper(l, pos_range, 1, 1, 'SAME', None, w_init_fold, b_init, True, 'mlp_bias_out')
                                    t, v = Conv1dWrapper(l, pos_range * pos_range, 1, 1, 'SAME', None, w_init_fold, b_init, True, 'mlp_transform_out')
                                    w = tf.reshape(w, [self.batch_size, fold_particles_count, fdim[bi], pos_range])
                                    w = tf.nn.softmax(w, axis = 2)
                                    t = tf.reshape(t, [self.batch_size, fold_particles_count, pos_range, pos_range])

                                    # Entropy loss
                                    entropy = tf.reduce_mean(-tf.reduce_sum(w * tf.log(w + 1e-4), axis = 2)) # We want minimize entropy of W
                                    tf.summary.scalar('entropy', entropy)
                                    regularizer += entropy * 0.1
                                    # Ortho of t?

                                z = tf.random.uniform([self.batch_size, fold_particles_count, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)

                                for gi in range(generator[bi]):
                                    with tf.variable_scope('gen%d' % gi):
                                        z, v = Conv1dWrapper(z, fdim[bi], 1, 1, 'SAME', None, w_init_fold, b_init, True, 'fc')
                                        z = batch_norm(z, 0.999, is_train, name = 'bn')
                                        z = self.act(z)
                                z, v = Conv1dWrapper(z, fdim[bi], 1, 1, 'SAME', None, w_init_fold, b_init, True, 'fc_final')
                                
                                # Collect features
                                z = tf.multiply(w, tf.reshape(z, [self.batch_size, fold_particles_count, fdim[bi], 1]))
                                z = tf.reduce_sum(z, axis = 2)
                                z = z + b

                                # Linear transformation
                                z = tf.multiply(t, tf.reshape(z, [self.batch_size, fold_particles_count, pos_range, 1]))
                                z = tf.reduce_sum(z, axis = 2)

                            # ap, v = Conv1dWrapper(z, pos_range, 1, 1, 'SAME', None, w_init, b_init, True, 'gen/fc_out')
                            ap = z
                            ap = ap + origin_pos # ap is alter_particles

                            gen_only.append(tf.concat([ap, coarse_pos], axis = 1))

                            # Position refinement stage
                            # Bipartite graph
                            posAlter, posRefer, gp_idx, gp_edg = bip_kNNG_gen(ap, coarse_pos, knnk[bi], pos_range, name = 'bi_ggen_pre')
                            # Empty feature
                            n = tf.zeros([self.batch_size, fold_particles_count, 1], dtype = default_dtype)
                            n, v = bip_kNNGConvBN_wrapper(n, coarse_fea, gp_idx, gp_edg, self.batch_size, fold_particles_count, hdim[bi], self.act, is_train = is_train, name = 'bip/conv', W_init = w_init)
                            gen_features = n

                            # Existing features
                            n, v = Conv1dWrapper(coarse_fea, hdim[bi], 1, 1, 'SAME', self.act, w_init, b_init, True, 'pre/conv')
                            ref_features = n

                            # Combine to get graph
                            pos = tf.concat([posRefer, posAlter], axis = 1)
                            n = tf.concat([ref_features, gen_features], axis = 1)

                        ### General part for full and partial generators

                        # Position Refinement

                        # get feature
                        # Bipartite graph
                        posAlter, posRefer, gp_idx, gp_edg = bip_kNNG_gen(pos, coarse_pos, knnk[bi], pos_range, name = 'bi_ggen_gRefine')
                        # Empty feature
                        n = tf.zeros([self.batch_size, pcnt[bi], 1], dtype = default_dtype)
                        n, v = bip_kNNGConvBN_wrapper(n, coarse_fea, gp_idx, gp_edg, self.batch_size, pcnt[bi], hdim[bi], self.act, is_train = is_train, name = 'gRefine/bip/conv', W_init = w_init)

                        # refine_loops = 0
                        refine_res_blocks = refine_res[bi]
                        vars_loop = []

                        for r in range(refine[bi]):
                        
                            _, gr_idx, gr_edg = kNNG_gen(pos, knnk[bi], 3, name = 'grefine%d/ggen' % r)
                            tmp = n

                            for i in range(refine_res_blocks):
                                
                                # Convolution
                                nn, v = kNNGConvBN_wrapper(n, gr_idx, gr_edg, self.batch_size, pcnt[bi], hdim[bi], self.act, is_train = is_train, name = 'gr%d/gloop%d/gconv1' % (r, i), W_init = w_init, b_init = b_init)
                                vars_loop.append(v)
                                
                                nn, v = kNNGConvBN_wrapper(nn, gr_idx, gr_edg, self.batch_size, pcnt[bi], hdim[bi], self.act, is_train = is_train, name = 'gr%d/gloop%d/gconv2' % (r, i), W_init = w_init, b_init = b_init)
                                vars_loop.append(v)

                                n = n + nn

                            n = n + tmp
                            pos, v = kNNGPosition_refine(pos, n, refine_maxLength[bi], self.act, W_init = w_init_pref, b_init = b_init, name = 'gr%d/grefine/refine' % r)
                            vars_loop.append(v)

                        # get feature
                        # Bipartite graph
                        posAlter, posRefer, gp_idx, gp_edg = bip_kNNG_gen(pos, coarse_pos, knnk[bi], pos_range, name = 'bi_ggen_featureEx')
                        # Empty feature
                        n = tf.zeros([self.batch_size, pcnt[bi], 1], dtype = default_dtype)
                        n, v = bip_kNNGConvBN_wrapper(n, coarse_fea, gp_idx, gp_edg, self.batch_size, pcnt[bi], hdim[bi], self.act, is_train = is_train, name = 'featureEx/bip/conv', W_init = w_init)

                        _, gidx, gedg = kNNG_gen(pos, knnk[bi], 3, name = 'featureEx/ggen')
                        n, v = kNNGConvBN_wrapper(n, gidx, gedg, self.batch_size, pcnt[bi], hdim[bi], self.act, is_train = is_train, name = 'featureEx/gconv1', W_init = w_init, b_init = b_init)
                        n, v = kNNGConvBN_wrapper(n, gidx, gedg, self.batch_size, pcnt[bi], hdim[bi], self.act, is_train = is_train, name = 'featureEx/gconv2', W_init = w_init, b_init = b_init)

                        coarse_pos = pos
                        coarse_fea = n
                        coarse_cnt = pcnt[bi]

                final_particles = coarse_pos
                n = coarse_fea

                if output_dim > pos_range:
                    n, _ = Conv1dWrapper(n, output_dim - pos_range, 1, 1, 'SAME', None, w_init, b_init, True, 'finalConv')
                    final_particles = tf.concat([pos, n], -1)

                regularizer = regularizer / blocks

                return 0, [final_particles, gen_only[0]], 0, regularizer

    def simulator(self, pos, particles, name = 'Simluator', is_train = True, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev=0.03*self.wdev)
        b_init = tf.constant_initializer(value=0.0)

        with tf.variable_scope(name, reuse = reuse) as vs:
            
            _, gIdx, gEdg = kNNG_gen(pos, self.knn_k, 3, name = 'simulator/ggen')
            layers = 1
            n = particles
            Np = particles.shape[1]
            C = particles.shape[2]
            var_list = []

            nn = n

            for i in range(layers):
                nn, v = kNNGConvBN_wrapper(nn, gIdx, gEdg, self.batch_size, Np, C, self.act, is_train = is_train, name = 'simulator/gconv%d' % i, W_init = w_init, b_init = b_init)
                var_list.append(v)

            n = n + nn
            pos, v = kNNGPosition_refine(pos, n, self.act, W_init = w_init_pref, b_init = b_init, name = 'simulator/grefine')
            var_list.append(v)

        return pos, particles, var_list

    def simulator_old(self, pos, particles, name = 'Simulator', is_train = True, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev=0.03*self.wdev)
        b_init = tf.constant_initializer(value=0.0)

        with tf.variable_scope(name, reuse = reuse) as vs:
            
            _, gIdx, gEdg = kNNG_gen(pos, self.knn_k, 3, name = 'simulator/ggen')
            layers = 1
            n = particles
            Np = particles.shape[1]
            C = particles.shape[2]
            var_list = []

            nn = n

            for i in range(layers):
                nn, v = kNNGConvBN_wrapper(nn, gIdx, gEdg, self.batch_size, Np, C, self.act, is_train = is_train, name = 'simulator/gconv%d' % i, W_init = w_init, b_init = b_init, bnact = None)
                # n, v = kNNGConvBN_wrapper(n, gIdx, gEdg, self.batch_size, Np, C, self.act, is_train = is_train, name = 'simulator/gconv%d' % i, W_init = w_init, b_init = b_init)
                var_list.append(v)

            n = n + nn
            pos, v = kNNGPosition_refine(pos, n, self.act, W_init = w_init_pref, b_init = b_init, name = 'simulator/grefine')
            var_list.append(v)

        return pos, particles, var_list

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

    def chamfer_metric(self, particles, groundtruth, pos_range, loss_func, EMD = False):
        
        if EMD == True:
            
            bs = groundtruth.shape[0]
            Np = particles.shape[1]
            Ng = groundtruth.shape[1]
            
            match = approx_match(groundtruth, particles) # [bs, Np, Ng]
            row_predicted = tf.reshape(  particles[:, :, 0:pos_range], [bs, Np, 1, pos_range])
            col_groundtru = tf.reshape(groundtruth[:, :, 0:pos_range], [bs, 1, Ng, pos_range])
            distance = tf.sqrt(tf.add_n(tf.unstack(tf.square(row_predicted - col_groundtru), axis = -1)))
            distance = distance * match
            distance_loss = tf.reduce_mean(tf.reduce_sum(distance, axis = -1))
        
        else:
            
            # test - shuffle the groundtruth and calculate the loss
            # rec_particles = tf.stack(list(map(lambda x: tf.random.shuffle(x), tf.unstack(self.ph_X[:, :, 0:6]))))
            # rec_particles = tf.random.uniform([self.batch_size, self.gridMaxSize, 3], minval = -1.0, maxval = 1.0)

            bs = groundtruth.shape[0]
            Np = particles.shape[1]
            Ng = groundtruth.shape[1]

            assert groundtruth.shape[2] == particles.shape[2]

            # NOTE: current using position (0:3) only here for searching nearest point.
            row_predicted = tf.reshape(  particles[:, :, 0:pos_range], [bs, Np, 1, pos_range])
            col_groundtru = tf.reshape(groundtruth[:, :, 0:pos_range], [bs, 1, Ng, pos_range])
            # distance = tf.norm(row_predicted - col_groundtru, ord = 'euclidean', axis = -1)
            distance = tf.sqrt(tf.add_n(tf.unstack(tf.square(row_predicted - col_groundtru), axis = -1)))
            
            rearrange_predicted_N = tf.argmin(distance, axis = 1, output_type = tf.int32)
            rearrange_groundtru_N = tf.argmin(distance, axis = 2, output_type = tf.int32)
            
            batch_subscriptG = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, Ng])
            batch_subscriptP = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, Np])
            rearrange_predicted = tf.stack([batch_subscriptG, rearrange_predicted_N], axis = 2)
            rearrange_groundtru = tf.stack([batch_subscriptP, rearrange_groundtru_N], axis = 2)

            nearest_predicted = tf.gather_nd(  particles[:, :, :], rearrange_predicted)
            nearest_groundtru = tf.gather_nd(groundtruth[:, :, :], rearrange_groundtru)

            if loss_func == tf.abs:
                distance_loss =\
                    tf.reduce_mean(loss_func(tf.cast(        particles, tf.float32) - tf.cast(nearest_groundtru, tf.float32))) +\
                    tf.reduce_mean(loss_func(tf.cast(nearest_predicted, tf.float32) - tf.cast(groundtruth      , tf.float32)))
            else:
                distance_loss =\
                    tf.reduce_mean(tf.sqrt(tf.reduce_sum(loss_func(tf.cast(        particles, tf.float32) - tf.cast(nearest_groundtru, tf.float32)), axis = -1))) +\
                    tf.reduce_mean(tf.sqrt(tf.reduce_sum(loss_func(tf.cast(nearest_predicted, tf.float32) - tf.cast(      groundtruth, tf.float32)), axis = -1)))

        return tf.cast(distance_loss, default_dtype)

    # pos [bs, N, pRange]
    # imp [bs, N]
    # FIXME: this is not good ...
    def pool_coverage(self, pos, importance, evaluation, k, ctrlRange = 0.01, falloff = 2.0, smin = 64.0, eplison = 1e-5):

        bs = pos.shape[0]
        N  = pos.shape[1]
        pr = pos.shape[2]

        rx = 1.0 / (ctrlRange / 1.0)
        row = tf.reshape(pos, [bs, N, 1, pr])
        col = tf.reshape(pos, [bs, 1, N, pr])
        dist = tf.sqrt(tf.add_n(tf.unstack(tf.square(row - col), axis = -1)))
        subCover = tf.pow(tf.nn.tanh(1.0 / (rx * dist + eplison)), falloff)
        
        # normalize importance, fake k-max
        importance = k * importance / tf.reduce_sum(importance, axis = -1, keepdims = True)
        importance = tf.pow(importance, 4.0)

    def custom_dtype_getter(self, getter, name, shape=None, dtype=default_dtype, *args, **kwargs):
        
        if dtype is tf.float16:
            
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        
        else:
            
            return getter(name, shape, dtype, *args, **kwargs)

    def build_network(self, is_train, reuse, loopSim = True, includeSim = True):

        normalized_X = self.ph_X / self.normalize
        normalized_Y = self.ph_Y / self.normalize
        normalized_L = self.ph_L / self.normalize
       
        # tf.summary.histogram('groundTruth_pos', self.ph_X[:, :, 0:3])
        # tf.summary.histogram('groundTruth_vel', self.ph_X[:, :, 3:6])

        # Mixed FP16 & FP32
        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
        # with tf.variable_scope('net'):

            # Go through the particle AE

            # tf.summary.histogram('GroundTruth', normalized_X[:, :, 0:3])
            
            var_list = []
            floss = 0
            regularizer = 0

            # Enc(X)
            posX, feaX, _v, _floss, eX = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse)
            var_list.append(_v)
            floss += _floss
            encX = tf.concat([posX, feaX], -1)

            if includeSim == True:
                
                # Enc(Y)
                posY, feaY, _v, _floss, eY = self.particleEncoder(normalized_Y, self.particle_latent_dim, is_train = is_train, reuse = True)
                var_list.append(_v)
                floss += _floss
                encY = tf.concat([posY, feaY], -1)
                # tf.summary.histogram('Encoded Y', encY)
                
                if loopSim == True:

                    # Enc(L)
                    posL, feaL, _v, _floss, _ = self.particleEncoder(normalized_L, self.particle_latent_dim, is_train = is_train, reuse = True)
                    var_list.append(_v)
                    floss += _floss
                    encL = tf.concat([posL, feaL], -1)


            # tf.summary.histogram('Clusters X', posX)

            outDim = self.outDim
            
            if includeSim == True:

                # Sim: X -> Y
                sim_posY, sim_feaY, _v = self.simulator(posX, feaX, 'Simulator', is_train, reuse)
                var_list.append(_v)
                
                # SimInv: Y -> X
                sim_posYX, sim_feaYX, _v = self.simulator(posY, feaY, 'Simulator_Inv', is_train, reuse)
                var_list.append(_v)

                simY  = tf.concat([ sim_posY,  sim_feaY], -1)
                simYX = tf.concat([sim_posYX, sim_feaYX], -1)
                
                # tf.summary.histogram('simulated Y', simY)

                # Decoders
                # _, [rec_YX, _], _ = self.particleDecoder(sim_posYX, sim_feaYX, self.ph_card, 6, True, reuse)
                # _, [ rec_Y, _], _ = self.particleDecoder( sim_posY,  sim_feaY, self.ph_card, 6, True,  True)
                _, [rec_YX, fold_X], _, r = self.particleDecoder( posX, feaX, self.ph_card, outDim, True, reuse)
                regularizer += r * 0.5
                _, [ rec_Y, _], _, r = self.particleDecoder( posY, feaY, self.ph_card, outDim, True,  True)
                regularizer += r * 0.5

                # tf.summary.histogram('Reconstructed X (from SInv(Y))', rec_YX[:, :, 0:3])
                
                if loopSim == True:

                    # SimInv: L -> X
                    sim_posLX, sim_feaLX = posL, feaL
                    for i in range(self.loops):
                        sim_posLX, sim_feaLX, _v = self.simulator(sim_posLX, sim_feaLX, 'Simulator_Inv', is_train, True)
                    
                    # Sim: X -> L
                    sim_posL, sim_feaL = posX, feaX
                    for i in range(self.loops):
                        sim_posL, sim_feaL, _v = self.simulator(sim_posL, sim_feaL, 'Simulator', is_train, True)
                    
                    simL  = tf.concat([ sim_posL,  sim_feaL], -1)
                    simLX = tf.concat([sim_posLX, sim_feaLX], -1)
                    
                    _, [rec_LX, _], _, r = self.particleDecoder(sim_posLX, sim_feaLX, self.ph_card, outDim, True,  True)
                    regularizer += r * 0.5
                    # _, [ rec_L, _], _ = self.particleDecoder( sim_posL,  sim_feaL, self.ph_card, outDim, True,  True)
                    _, [ rec_L, _], _, r = self.particleDecoder( posL,  feaL, self.ph_card, outDim, True,  True)
                    regularizer += r * 0.5
                    
                    regularizer *= 0.5
            
            else:

                _, [rec_X, fold_X], _, r = self.particleDecoder(posX, feaX, self.ph_card, outDim, True, reuse)
                regularizer += r
                # tf.summary.histogram('Reconstructed X (from X)', rec_X[:, :, 0:3])

        reconstruct_loss = 0.0
        simulation_loss  = 0.0
         
        # EMD
        if self.loss_metric == 'earthmover':
            raise NotImplementedError

        if self.loss_metric == 'chamfer':

            # Do some stop_gradient?

            # reconstruct_loss += self.chamfer_metric(rec_YX, normalized_X[:, :, 0:6], 3, self.loss_func)
            # reconstruct_loss += self.chamfer_metric(rec_LX, normalized_X[:, :, 0:6], 3, self.loss_func)
            # reconstruct_loss += self.chamfer_metric(rec_Y , normalized_Y[:, :, 0:6], 3, self.loss_func)
            # reconstruct_loss += self.chamfer_metric(rec_L , normalized_L[:, :, 0:6], 3, self.loss_func)
            
            pool_align_loss = 0

            if includeSim == True:

                if loopSim == True:
            
                    reconstruct_loss += self.chamfer_metric(rec_YX, normalized_X[:, :, 0:outDim], 3, self.loss_func)
                    reconstruct_loss += self.chamfer_metric(rec_LX, normalized_X[:, :, 0:outDim], 3, self.loss_func)
                    reconstruct_loss += self.chamfer_metric(rec_Y , normalized_Y[:, :, 0:outDim], 3, self.loss_func)
                    reconstruct_loss += self.chamfer_metric(rec_L , normalized_L[:, :, 0:outDim], 3, self.loss_func)

                    simulation_loss  += self.chamfer_metric(simY , encY, 3, self.loss_func)
                    simulation_loss  += self.chamfer_metric(simL , encL, 3, self.loss_func)
                    simulation_loss  += self.chamfer_metric(simYX, encX, 3, self.loss_func)
                    simulation_loss  += self.chamfer_metric(simLX, encX, 3, self.loss_func)

                    reconstruct_loss *= 10.0
                    simulation_loss *= 10.0

                else:

                    reconstruct_loss += self.chamfer_metric(rec_YX, normalized_X[:, :, 0:outDim], 3, self.loss_func)
                    reconstruct_loss += self.chamfer_metric(rec_Y , normalized_Y[:, :, 0:outDim], 3, self.loss_func)
                    
                    simulation_loss  += self.chamfer_metric(simY , encY, 3, self.loss_func)
                    simulation_loss  += self.chamfer_metric(simYX, encX, 3, self.loss_func)
            
                    reconstruct_loss *= 20.0
                    simulation_loss *= 20.0

                for ei in range(len(eX)):
                    pool_align_loss += self.chamfer_metric(eX[ei], eY[ei], 3, self.loss_func)
                pool_align_loss *= 10.0 / len(eX)

            else:

                reconstruct_loss += self.chamfer_metric(rec_X, normalized_X[:, :, 0:outDim], 3, self.loss_func, EMD = True)
                # reconstruct_loss += self.chamfer_metric(rec_X[:, self.cluster_count:, 0:outDim], normalized_X[:, :, 0:outDim], 3, self.loss_func, EMD = True)
                reconstruct_loss *= 40.0
                raw_error = 0.0
                # raw_error = self.chamfer_metric(rec_X, normalized_X[:, :, 0:outDim], 3, tf.abs) * 40.0
        
        # reconstruct_loss += self.chamfer_metric(fold_X[:, :, 0:3], normalized_X[:, :, 0:3], 3, self.loss_func) * 40.0
        # reconstruct_loss *= 0.5

        hqpool_loss = 0.01 * tf.reduce_mean(floss) + regularizer
        # hqpool_loss = 0.0
        
        if includeSim == True:
            hqpool_loss *= 0.5
        
        if includeSim == True and loopSim == True:
            hqpool_loss *= 0.5

        particle_net_vars =\
            tl.layers.get_variables_with_name('ParticleEncoder', True, True) +\
            tl.layers.get_variables_with_name('ParticleDecoder', True, True) 
        
        # rec_L  = rec_L  * self.normalize
        # rec_LX = rec_LX * self.normalize
        # rec_X = rec_X * self.normalize
        # rec_YX = rec_YX * self.normalize
        # rec_Y  = rec_Y  * self.normalize

        return reconstruct_loss, simulation_loss, hqpool_loss, pool_align_loss, particle_net_vars, raw_error

    # Only encodes X
    def build_predict_Enc(self, normalized_X, is_train = False, reuse = False):

        # Mixed FP16 & FP32
        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
        # with tf.variable_scope('net'):

            # Go through the particle AE

            # tf.summary.histogram('GroundTruth', normalized_X[:, :, 0:3])
            
            var_list = []
            floss = 0

            # Enc(X)
            posX, feaX, _v, pPos, _floss, evals = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse, returnPool = True)
            var_list.append(_v)
            floss += _floss

        return posX, feaX, pPos, evals
    
    # Only simulates posX & feaX for a single step
    def build_predict_Sim(self, pos, fea, is_train = False, reuse = False):

        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
            
            sim_posY, sim_feaY, _v = self.simulator(pos, fea, 'Simulator', is_train, reuse)
        
        return sim_posY, sim_feaY

    # Decodes Y
    def build_predict_Dec(self, pos, fea, gt, is_train = False, reuse = False, outDim = 6):

        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
            
            _, [rec, rec_f], _, _ = self.particleDecoder(pos, fea, self.ph_card, outDim, is_train, reuse)

        rec = rec
        reconstruct_loss = self.chamfer_metric(rec, gt, 3, self.loss_func, True) * 40.0

        return rec * self.normalize, rec_f * self.normalize, reconstruct_loss

    def build_model(self):

        # Train & Validation
        self.train_particleRecLoss, self.train_particleSimLoss, self.train_HQPLoss, self.train_PALoss,\
        self.particle_vars, self.train_error =\
            self.build_network(True, False, self.doLoop, self.doSim)

        self.val_particleRecLoss, self.val_particleSimLoss, _, _, _, self.val_error =\
            self.build_network(False, True, self.doLoop, self.doSim)

        # self.train_particleLoss = self.train_particleCardLoss
        # self.val_particleLoss = self.val_particleCardLoss

        self.train_particleLoss = self.train_particleRecLoss + self.train_particleSimLoss + self.train_HQPLoss + self.train_PALoss
        self.val_particleLoss = self.val_particleRecLoss
        # self.train_particleLoss = self.train_particleCardLoss + 100 * self.train_particleRawLoss
        # self.val_particleLoss = self.val_particleRawLoss
        # self.val_particleLoss = self.val_particleCardLoss + 100 * self.val_particleRawLoss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = self.optimizer.compute_gradients(self.train_particleLoss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.) if grad is not None else None, var) for grad, var in gvs]
            # self.train_op = self.optimizer.minimize(self.train_particleLoss)
            self.train_op = self.optimizer.apply_gradients(capped_gvs)
        
        # self.train_op = self.optimizer.minimize(self.train_particleLoss, var_list = self.particle_vars)
        # self.train_op = tf.constant(0, shape=[10, 10])
