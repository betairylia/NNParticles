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

default_dtype = tf.float32

def batch_norm(inputs, decay, is_train):

    # if default_dtype == tf.float32:
    #     return tf.keras.layers.BatchNormalization(momentum = decay)(inputs, training = is_train)
    # if default_dtype == tf.float16:
    #     return BatchNormalizationF16(momentum = decay)(inputs, training = is_train)
   
    # return tf.keras.layers.BatchNormalization(momentum = decay)(inputs, training = is_train)
    return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, fused = True)

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
        A = tf.cast(A_undirected, tf.float32, name = 'A_undirected') # [bs, N, N]
        # print(A.shape)

        D = tf.matrix_set_diag(tf.zeros([bs, N, N], tf.float32), tf.reduce_sum(A, axis = -1)) # [bs, N] -> [bs, N, N]
        # print(D.shape)
        L = D - A
        # print(L.shape)

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

def kNNG_gen(inputs, k, pos_range, name = 'kNNG_gen'):

    p, _, idx, edg = bip_kNNG_gen(inputs, inputs, k, pos_range, name)
    return p, idx, edg

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

    return resnbr, [W_neighbor, b_neighbor, W_edges, b_edges, W_self, b_self] # [bs, Nx, channels]

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
def kNNGPooling_HighFreqLoss_GUnet(inputs, pos, k, laplacian, masking = True, channels = 1, W_init = tf.truncated_normal_initializer(stddev=0.1), name = 'kNNGPool', stopGradient = False):

    with tf.variable_scope(name):

        bs = inputs.shape[0]
        N = inputs.shape[1]
        C = inputs.shape[2]
        k = min(N, k)

        if stopGradient == True:
            inputs = tf.stop_gradient(inputs)

        W = tf.get_variable('W', dtype = default_dtype, shape = [1, C, channels], initializer=W_init, trainable=True)
        norm = tf.sqrt(tf.reduce_sum(tf.square(W), axis = 1, keepdims = True)) # [1, 1, channels]
        
        y = tf.nn.conv1d(inputs, W, 1, padding = 'SAME') # [bs, N, channels]
        y = tf.multiply(y, 1.0 / (norm + 1e-3))
        y = tf.reduce_mean(y, axis = -1) # [bs, N]

        # Freq Loss
        print(laplacian.shape)
        _y = tf.cast(y, tf.float32)
        norm_Ly = tf.sqrt(tf.reduce_sum(tf.square(tf.matmul(laplacian, tf.reshape(_y, [bs, N, 1]), name = 'L_y')), axis = [1, 2]) + 1e-3)
        norm_y = tf.sqrt(tf.reduce_sum(tf.square(_y), axis = 1) + 1e-3)
        freq_loss = norm_Ly / (norm_y + 1e-3) # Maximize this
        freq_loss = 0 - freq_loss # Minimize negate

        val, idx = tf.nn.top_k(y, k) # [bs, k]

        # Pick them
        batches = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, k])
        gather_idx = tf.stack([batches, idx], axis = -1)
        pool_features = tf.gather_nd(inputs, gather_idx) # [bs, k, C]
        pool_position = tf.gather_nd(pos, gather_idx) # [bs, k, 3]

        if masking == True:
            pool_features = tf.multiply(pool_features, tf.reshape(tf.nn.tanh(val), [bs, k, 1]))
    
    return pool_position, pool_features, [W], tf.cast(freq_loss, default_dtype)

def Conv1dWrapper(inputs, filters, kernel_size, stride, padding, act, W_init, b_init, bias = True, name = 'conv'):

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

def kNNGPosition_refine(input_position, input_feature, act, hidden = 128, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGPosRefine'):

    with tf.variable_scope(name):

        bs = input_position.shape[0]
        N = input_position.shape[1]
        C = input_feature.shape[2]
        pC = input_position.shape[2]

        assert N == input_feature.shape[1] and bs == input_feature.shape[0]

        pos_feature, vars1 = Conv1dWrapper(tf.concat([input_position, input_feature], axis = -1), hidden, 1, 1, 'SAME', act, W_init, b_init, True, 'hidden')
        pos_res, vars2 = Conv1dWrapper(pos_feature, pC, 1, 1, 'SAME', None, W_init, b_init, True, 'refine')

        # tf.summary.histogram('Position_Refine_%s' % name, pos_res)

        refined_pos = tf.add(input_position, pos_res)

        return refined_pos, [vars1, vars2]

def bip_kNNGConvBN_wrapper(Xs, Ys, kNNIdx, kNNEdg, batch_size, gridMaxSize, particle_hidden_dim, act, decay = 0.999, is_train = True, name = 'gconv', W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0)):
    
    with tf.variable_scope(name):
        # n, v = kNNGConvLayer_concat(inputs, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        n, v = bip_kNNGConvLayer_edgeMask(Xs, Ys, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        n = batch_norm(n, decay, is_train)
        n = act(n)

    return n, v

def kNNGConvBN_wrapper(inputs, kNNIdx, kNNEdg, batch_size, gridMaxSize, particle_hidden_dim, act, decay = 0.999, is_train = True, name = 'gconv', W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0)):
    
    with tf.variable_scope(name):
        # n, v = kNNGConvLayer_concat(inputs, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        # n, v = kNNGConvLayer_edgeMask(inputs, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        n, v = bip_kNNGConvLayer_edgeMask(inputs, inputs, kNNIdx, kNNEdg, act, no_act_final = True, channels = particle_hidden_dim, W_init = W_init, b_init = b_init, name = 'gc')
        n = batch_norm(n, decay, is_train)
        n = act(n)

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

        self.doSim = True
        self.doLoop = True

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

        self.ph_X           = tf.placeholder(default_dtype, [self.batch_size, self.gridMaxSize, 7]) # x y z vx vy vz 1
        self.ph_Y           = tf.placeholder(default_dtype, [self.batch_size, self.gridMaxSize, 7])
        self.ph_L           = tf.placeholder(default_dtype, [self.batch_size, self.gridMaxSize, 7]) # Loop simulation (under latent space) ground truth

        self.ph_card        = tf.placeholder(default_dtype, [self.batch_size]) # card
        self.ph_max_length  = tf.placeholder('int32', [2])

        self.optimizer = optimizer

    # 1 of a batch goes in this function at once.
    def particleEncoder(self, input_particle, output_dim, is_train = False, reuse = False, returnPool = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
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
            
            blocks = 4
            particles_count = [2560, 1280, 512, self.cluster_count]
            conv_count = [3, 2, 3, 2]
            res_count = [0, 0, 0, 1]
            kernel_size = [self.knn_k, self.knn_k, self.knn_k, self.knn_k]
            hd = self.particle_hidden_dim
            channels = [hd // 3, hd // 2, hd, hd * 2]

            gPos = input_particle[:, :, :3]
            n = input_particle[:, :, 3:]
            var_list = []
            pool_pos = []
            freq_loss = 0

            for i in range(blocks):
                
                if i > 0:
                    
                    # Pooling
                    prev_n = n
                    prev_pos = gPos
                    gPos, n, v, fl = kNNGPooling_HighFreqLoss_GUnet(n, gPos, particles_count[i], MatL, W_init = w_init, name = 'gpool%d' % i, stopGradient = True)
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
            
            # tf.summary.histogram('Pooled_clusters_pos', gPos)
            n, v = Conv1dWrapper(n, self.cluster_feature_dim, 1, 1, 'SAME', None, w_init, b_init, True, 'convOut')
            var_list.append(v)
            
            if returnPool == True:
                return gPos, n, var_list, pool_pos, freq_loss

            return gPos, n, var_list, freq_loss
    
    def particleDecoder(self, cluster_pos, local_feature, groundTruth_card, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev=0.03*self.wdev)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleDecoder", reuse = reuse) as vs:
 
            CC = local_feature.shape[2]
            global_latent, v = Conv1dWrapper(local_feature, CC * 4, 1, 1, 'SAME', None, w_init, b_init, True, 'convGlobal')
            global_latent = self.combine_method(global_latent, axis = 1)

            # Folding stage
            fold_particles_count = self.gridMaxSize - self.cluster_count
            # net_input = InputLayer(input_latent, name = 'input')
            
            # FIXME: no card in this model

            # generate random noise
            pos_range = 3
            z = tf.random.normal([self.batch_size, fold_particles_count, pos_range], dtype = default_dtype)

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

                c, v = Conv1dWrapper(c, self.particle_hidden_dim, 1, 1, 'SAME', None, w_init, b_init, True, 'fold/fc%d' % i)
                c = batch_norm(c, 0.999, is_train)
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

            n, _ = Conv1dWrapper(n, output_dim - pos_range, 1, 1, 'SAME', None, w_init, b_init, True, 'finalConv')
            final_particles = tf.concat([pos, n], -1)
            return 0, [final_particles, fold_before_prefine], 0

    def simulator(self, pos, particles, name = 'Simulator', is_train = True, reuse = False):

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

            for i in range(layers):
                n, v = kNNGConvBN_wrapper(n, gIdx, gEdg, self.batch_size, Np, C, self.act, is_train = is_train, name = 'simulator/gconv%d' % i, W_init = w_init, b_init = b_init)
                var_list.append(v)

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

    def chamfer_metric(self, particles, groundtruth, pos_range, loss_func):
            
        # test - shuffle the groundtruth and calculate the loss
        # rec_particles = tf.stack(list(map(lambda x: tf.random.shuffle(x), tf.unstack(self.ph_X[:, :, 0:6]))))
        # rec_particles = tf.random.uniform([self.batch_size, self.gridMaxSize, 3], minval = -1.0, maxval = 1.0)

        bs = groundtruth.shape[0]
        N  = groundtruth.shape[1]

        assert groundtruth.shape[2] == particles.shape[2]

        # NOTE: current using position (0:3) only here for searching nearest point.
        row_predicted = tf.reshape(  particles[:, :, 0:pos_range], [bs, N, 1, pos_range])
        col_groundtru = tf.reshape(groundtruth[:, :, 0:pos_range], [bs, 1, N, pos_range])
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(row_predicted, col_groundtru)), axis = -1))
        
        rearrange_predicted_N = tf.argmin(distance, axis = 1, output_type = tf.int32)
        rearrange_groundtru_N = tf.argmin(distance, axis = 2, output_type = tf.int32)
        
        batch_subscript = tf.broadcast_to(tf.reshape(tf.range(bs), [bs, 1]), [bs, N])
        rearrange_predicted = tf.stack([batch_subscript, rearrange_predicted_N], axis = 2)
        rearrange_groundtru = tf.stack([batch_subscript, rearrange_groundtru_N], axis = 2)

        nearest_predicted = tf.gather_nd(  particles[:, :, :], rearrange_predicted)
        nearest_groundtru = tf.gather_nd(groundtruth[:, :, :], rearrange_groundtru)

        chamfer_loss =\
            tf.reduce_mean(loss_func(tf.cast(        particles, tf.float32) - tf.cast(nearest_groundtru, tf.float32))) +\
            tf.reduce_mean(loss_func(tf.cast(nearest_predicted, tf.float32) - tf.cast(groundtruth      , tf.float32)))
        
        return tf.cast(chamfer_loss, default_dtype)

    def custom_dtype_getter(self, getter, name, shape=None, dtype=default_dtype, *args, **kwargs):
        
        if dtype is tf.float16:
            
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        
        else:
            
            return getter(name, shape, dtype, *args, **kwargs)

    def build_network(self, is_train, reuse, loopSim = True, includeSim = True):

        normalized_X = self.ph_X / 48.0
        normalized_Y = self.ph_Y / 48.0
        normalized_L = self.ph_L / 48.0
        
        # Mixed FP16 & FP32
        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
        # with tf.variable_scope('net'):

            # Go through the particle AE

            # tf.summary.histogram('GroundTruth', normalized_X[:, :, 0:3])
            
            var_list = []
            floss = 0

            # Enc(X)
            posX, feaX, _v, _floss = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse)
            var_list.append(_v)
            floss += _floss
            encX = tf.concat([posX, feaX], -1)

            if includeSim == True:
                
                # Enc(Y)
                posY, feaY, _v, _floss = self.particleEncoder(normalized_Y, self.particle_latent_dim, is_train = is_train, reuse = True)
                var_list.append(_v)
                floss += _floss
                encY = tf.concat([posY, feaY], -1)
                tf.summary.histogram('Encoded Y', encY)
                
                if loopSim == True:

                    # Enc(L)
                    posL, feaL, _v, _floss = self.particleEncoder(normalized_L, self.particle_latent_dim, is_train = is_train, reuse = True)
                    var_list.append(_v)
                    floss += _floss
                    encL = tf.concat([posL, feaL], -1)


            tf.summary.histogram('Clusters X', posX)
            
            if includeSim == True:

                # Sim: X -> Y
                sim_posY, sim_feaY, _v = self.simulator(posX, feaX, 'Simulator', is_train, reuse)
                var_list.append(_v)
                
                # SimInv: Y -> X
                sim_posYX, sim_feaYX, _v = self.simulator(posY, feaY, 'Simulator_Inv', is_train, reuse)
                var_list.append(_v)

                simY  = tf.concat([ sim_posY,  sim_feaY], -1)
                simYX = tf.concat([sim_posYX, sim_feaYX], -1)
                
                tf.summary.histogram('simulated Y', simY)

                # Decoders
                _, [rec_YX, _], _ = self.particleDecoder(sim_posYX, sim_feaYX, self.ph_card, 6, True, reuse)
                _, [ rec_Y, _], _ = self.particleDecoder( sim_posY,  sim_feaY, self.ph_card, 6, True,  True)

                rec_YX = rec_YX * 48.0
                rec_Y  = rec_Y  * 48.0

                tf.summary.histogram('Reconstructed X (from SInv(Y))', rec_YX[:, :, 0:3])
                
                if loopSim == True:

                    # SimInv: L -> X
                    sim_posLX, sim_feaLX = posL, feaL
                    for i in range(5):
                        sim_posLX, sim_feaLX, _v = self.simulator(sim_posLX, sim_feaLX, 'Simulator_Inv', is_train, True)
                    
                    # Sim: X -> L
                    sim_posL, sim_feaL = posX, feaX
                    for i in range(5):
                        sim_posL, sim_feaL, _v = self.simulator(sim_posL, sim_feaL, 'Simulator', is_train, True)
                    
                    simL  = tf.concat([ sim_posL,  sim_feaL], -1)
                    simLX = tf.concat([sim_posLX, sim_feaLX], -1)
                    
                    _, [rec_LX, _], _ = self.particleDecoder(sim_posLX, sim_feaLX, self.ph_card, 6, True,  True)
                    _, [ rec_L, _], _ = self.particleDecoder( sim_posL,  sim_feaL, self.ph_card, 6, True,  True)
                    
                    rec_L  = rec_L  * 48.0
                    rec_LX = rec_LX * 48.0
            
            else:

                _, [rec_X, _], _ = self.particleDecoder(posX, feaX, self.ph_card, 6, True, reuse)
                rec_X = rec_X * 48.0
                tf.summary.histogram('Reconstructed X (from X)', rec_X[:, :, 0:3])

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
            
            if includeSim == True:

                if loopSim == True:
            
                    reconstruct_loss += self.chamfer_metric(rec_YX, self.ph_X[:, :, 0:6], 3, self.loss_func)
                    reconstruct_loss += self.chamfer_metric(rec_LX, self.ph_X[:, :, 0:6], 3, self.loss_func)
                    reconstruct_loss += self.chamfer_metric(rec_Y , self.ph_Y[:, :, 0:6], 3, self.loss_func)
                    reconstruct_loss += self.chamfer_metric(rec_L , self.ph_L[:, :, 0:6], 3, self.loss_func)

                    simulation_loss  += self.chamfer_metric(simY , encY, 3, self.loss_func)
                    simulation_loss  += self.chamfer_metric(simL , encL, 3, self.loss_func)
                    simulation_loss  += self.chamfer_metric(simYX, encX, 3, self.loss_func)
                    simulation_loss  += self.chamfer_metric(simLX, encX, 3, self.loss_func)

                    reconstruct_loss *= 0.25
                    simulation_loss *= 0.25

                else:

                    reconstruct_loss += self.chamfer_metric(rec_YX, self.ph_X[:, :, 0:6], 3, self.loss_func)
                    reconstruct_loss += self.chamfer_metric(rec_Y , self.ph_Y[:, :, 0:6], 3, self.loss_func)
                    
                    simulation_loss  += self.chamfer_metric(simY , encY, 3, self.loss_func)
                    simulation_loss  += self.chamfer_metric(simYX, encX, 3, self.loss_func)
            
                    reconstruct_loss *= 0.5
                    simulation_loss *= 0.5

            else:

                reconstruct_loss += self.chamfer_metric(rec_X, self.ph_X[:, :, 0:6], 3, self.loss_func)

        hqpool_loss = 0.004 * tf.reduce_mean(floss)
        
        if includeSim == True:
            hqpool_loss *= 0.5
        
        if includeSim == True and loopSim == True:
            hqpool_loss *= 0.5

        particle_net_vars =\
            tl.layers.get_variables_with_name('ParticleEncoder', True, True) +\
            tl.layers.get_variables_with_name('ParticleDecoder', True, True) 

        return reconstruct_loss, simulation_loss, hqpool_loss, particle_net_vars

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
            posX, feaX, _v, _floss = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse)
            var_list.append(_v)
            floss += _floss

        return posX, feaX
    
    # Only simulates posX & feaX for a single step
    def build_predict_Sim(self, pos, fea, is_train = False, reuse = False):

        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
            
            sim_posY, sim_feaY, _v = self.simulator(pos, fea, 'Simulator', is_train, reuse)
        
        return sim_posY, sim_feaY

    # Decodes Y
    def build_predict_Dec(self, pos, fea, gt, is_train = False, reuse = False):

        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
            
            _, [rec, _], _ = self.particleDecoder(pos, fea, self.ph_card, 6, is_train, reuse)

        rec = rec * 48.0
        reconstruct_loss = self.chamfer_metric(rec, gt, 3, self.loss_func)

        return rec, reconstruct_loss

    def build_model(self):

        # Train & Validation
        self.train_particleRecLoss, self.train_particleSimLoss, self.train_HQPLoss,\
        self.particle_vars =\
            self.build_network(True, False, self.doLoop, self.doSim)

        # self.val_particleRawLoss, self.val_particleCardLoss, _, _, _ =\
        #     self.build_network(False, True)

        # self.train_particleLoss = self.train_particleCardLoss
        # self.val_particleLoss = self.val_particleCardLoss

        self.train_particleLoss = self.train_particleRecLoss + self.train_particleSimLoss + self.train_HQPLoss
        # self.train_particleLoss = self.train_particleCardLoss + 100 * self.train_particleRawLoss
        # self.val_particleLoss = self.val_particleRawLoss
        # self.val_particleLoss = self.val_particleCardLoss + 100 * self.val_particleRawLoss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.train_particleLoss)
        
        # self.train_op = self.optimizer.minimize(self.train_particleLoss, var_list = self.particle_vars)
        # self.train_op = tf.constant(0, shape=[10, 10])
