import tensorflow as tf
import numpy as np
import scipy
import time
import math
import argparse
import random
import sys
import os
import matplotlib.pyplot as plt

from termcolor import colored, cprint

# from Kuhn_Munkres import KM
# from BN16 import BatchNormalizationF16

from time import gmtime, strftime

from external.structural_losses.tf_approxmatch import approx_match, match_cost
from external.sampling.tf_sampling import farthest_point_sample, prob_sample

import ot
import ot.gpu
from Sinkhorn_wrapper import Sinkhorn_dist

default_dtype = tf.float32
summary_scope = None
SN = False

nearestNorm = False
PDFNorm = False

PDFNorm = PDFNorm and not nearestNorm

normalization_method = 'None'
max_pool_conv = False
convd_ch = 2

def norm(inputs, decay, is_train, name):

    decay = 0.99

    if normalization_method == 'None':
        return inputs
    elif normalization_method == 'BrN':
        return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, scope = name, fused = True, renorm = True)
    elif normalization_method == 'LN':
        return tf.contrib.layers.layer_norm(inputs, scope = name)
    elif normalization_method == 'IN':
        if default_dtype == tf.float32:
            return tf.contrib.layers.instance_norm(inputs, scope = name)
        else:
            return tf.contrib.layers.instance_norm(inputs, epsilon = 1e-3, scope = name)

def brn(inputs, decay, is_train, name):
    decay = 0.99
    return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, scope = name, fused = True, renorm = True)

def AdaIN(inputs, mean, std, axes = [2], name = 'AdaIN', epsilon = 1e-5):

    with tf.variable_scope(name):

        c_mean, c_var = tf.nn.moments(inputs, axes = axes, keep_dims = True)
        c_std = tf.sqrt(c_var + epsilon)

        return std * (inputs - c_mean) / c_std + mean

# Inputs: [bs, N, C]
# Builds edges X -> Y
def bip_kNNG_gen(Xs, Ys, k, pos_range, name = 'kNNG_gen', xysame = False, recompute = True):

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
        
        nearest_norm = None
        if nearestNorm == True:
            if recompute == True:
                if xysame == True:
                    # nearest_norm = tf.reduce_max(kNNEdg, axis = 2) # distance to kth nearest point
                    # nearest_norm = tf.ones_like(nearest_norm)
                    sigma = tf.get_variable('rbf_sigma', [1], dtype = tf.float16, initializer = tf.constant_initializer(-2.5))
                    nearest_norm = RBF_dist(kNNEdg, tf.exp(sigma))
                    nearest_norm = tf.reduce_mean(nearest_norm, axis = 2)
                    nearest_norm = 1.0 / k / (tf.cast(nearest_norm, default_dtype) + 1e-5)
                else:
                    print("!")
                    dist = tf.norm(drow - tf.transpose(drow, perm = [0, 2, 1, 3]), ord = 'euclidean', axis = -1)
                    dist = tf.linalg.set_diag(dist, tf.constant(100.0, shape = [bs, Nx], dtype = tf.float16))
                    nearest_norm = tf.reduce_min(dist, axis = -1)

                # nnmean = tf.reduce_mean(nearest_norm, keepdims = True)
                # nearest_norm = tf.minimum(nearest_norm, nnmean) / nnmean
                # nearest_norm = nearest_norm / nnmean

                # nearest_norm = tf.pow(tf.cast(nearest_norm, default_dtype), 3)
        
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

        kNNEdg = tf.stop_gradient(kNNEdg)

        return posX, posY, kNNIdx, kNNEdg, nearest_norm

def kNNG_gen(inputs, k, pos_range, name = 'kNNG_gen', recompute = True):

    p, _, idx, edg, nnnorm = bip_kNNG_gen(inputs, inputs, k, pos_range, name, xysame = True, recompute = recompute)
    return p, idx, edg, nnnorm

def RBF_dist(d, sigma):
    
    d = d / (2.0 * sigma)
    d = tf.exp(-d)

    return d

def RBF(x, y, sigma):

    distance_norm = tf.norm(x - y, ord = 'euclidean', axis = -1)
    rbf = distance_norm / (2.0 * sigma)
    rbf = tf.exp(-rbf)

    return rbf

def bip_kNNGConvLayer_IN(inputs, kNNIdx, kNNEdg, act, channels, fCh, mlp, is_train, W_init, b_init, name):
    
    with tf.variable_scope(name):

        bs = inputs.shape[0]
        Ni = inputs.shape[1]
        Ci = inputs.shape[2]
        N  = kNNIdx.shape[1]
        k  = kNNIdx.shape[2]
        eC = kNNEdg.shape[3]

        neighbors = tf.gather_nd(inputs, kNNIdx)
        origins = tf.broadcast_to(tf.reshape(inputs, [bs, Ni, 1, Ci]), [bs, Ni, k, Ci])

        # Encode edge
        e_enc_mlp = [64, 64]
        e_encoded = kNNEdg
        for i in range(len(e_enc_mlp)):
            e_encoded = autofc(e_encoded, e_enc_mlp[i], None, name = 'eEnc/mlp%d' % i)
            e_encoded = norm(e_encoded, 0.999, is_train, 'eEnc/mlp%d/norm' % i)
            e_encoded = tf.nn.elu(e_encoded)
        e_encoded = autofc(e_encoded, 64, None, name = 'eEnc/mlp_out')

        # Relation (edge) stage
        e_mlp = [256, 256]
        e_in = tf.concat([e_encoded, neighbors, origins], axis = -1)

        e = e_in
        for i in range(len(e_mlp)):
            e = autofc(e, e_mlp[i], tf.nn.elu, name = 'eStage/mlp%d' % i)
        e = autofc(e, 64, None, name = 'eStage/mlp_out')

        # Sum-up
        e_sum = tf.reduce_mean(e, axis = 2)

        # Node stage
        n_mlp = [256, 256]
        n_in = tf.concat([e_sum, inputs], axis = -1)

        n = n_in
        for i in range(len(n_mlp)):
            n = autofc(n, n_mlp[i], tf.nn.elu, name = 'nStage/mlp%d' % i)
        n = autofc(n, channels, act, name = 'nStage/mlp_out')

    return n # [bs, Nx, channels]

def bip_kNNGConvLayer_kernel(inputs, kNNIdx, kNNEdg, act, channels, filters, fCh, is_train, f_init, b_init, k_init, name):

    with tf.variable_scope(name):

        bs = inputs.shape[0]
        Ni = inputs.shape[1]
        Ci = inputs.shape[2]
        Ny = kNNIdx.shape[1]
        k  = kNNIdx.shape[2]
        eC = kNNEdg.shape[3]

        n = tf.gather_nd(inputs, kNNIdx)

        # Get layer variables
        W       = tf.get_variable('W', shape = [filters, Ci, channels], trainable = True, initializer = f_init, dtype = default_dtype)
        F       = tf.get_variable('F', shape = [filters, 3], trainable = True, initializer = f_init, dtype = default_dtype)
        b       = tf.get_variable('b', shape = [channels], trainable = True, initializer = b_init, dtype = default_dtype)
        sigma   = tf.get_variable('K', shape = [filters], dtype = default_dtype)

        sigma = tf.exp(sigma)

        # old approach
        if False:
            # reduce channels for inputs
            n = autofc(n, fCh * channels, None, name = 'feature/feature_combine')
            n = tf.reshape(n, [bs, Ny, k, 1, fCh, channels, 1])

            # combine feature and position
            print(kNNEdg.shape)
            n = tf.concat([n, tf.tile(tf.reshape(kNNEdg, [bs, Ny, k, 1, 1, 1, 3]), [1, 1, 1, 1, fCh, channels, 1])], axis = -1)
            
            # "Convolution"
            n = RBF(n, tf.reshape(F, [1, 1, 1, filters, fCh, channels, 4]), tf.reshape(sigma, [1, 1, 1, 1, fCh, channels]))
            n = tf.reduce_mean(n, axis = [2, 3, 4]) # => [bs, Ny, channels]
        # No way. Give up.
        else:
            _lpos = tf.reshape(kNNEdg, [bs, Ny, k,       1, 3])
            _fpos = tf.reshape(     F, [ 1,  1, 1, filters, 3])

        # bias and act
        n = tf.nn.bias_add(n, b)
        if act is not None:
            n = act(n)
        
        return n

def bip_kNNGConvLayer_feature_getKernel(inputs, channels, fCh, mlp, name):
    
    with tf.variable_scope(name):

        # bs = inputs.shape[0]
        # Ni = inputs.shape[1]
        # Ci = inputs.shape[2]

        is_train = False

        ### Do the convolution ###
        mlp = mlp
        n = inputs
        for i in range(len(mlp)):
            n = autofc(n, mlp[i], None, name = 'kernel/mlp%d' % i)
            n = norm(n, 0.999, is_train, 'kernel/mlp%d/norm' % i)
            n = tf.nn.leaky_relu(n)
            # n = tf.nn.elu(n)

        n = autofc(n, channels * fCh, None, name = 'kernel/mlp_out')

    return n # [bs, channels * fCh]

def bip_kNNGConvLayer_feature(inputs, kNNIdx, kNNEdg, act, channels, fCh, mlp, is_train, W_init, b_init, name, nnnorm):
    
    with tf.variable_scope(name):

        bs = inputs.shape[0]
        Ni = inputs.shape[1]
        Ci = inputs.shape[2]
        N  = kNNIdx.shape[1]
        k  = kNNIdx.shape[2]
        eC = kNNEdg.shape[3]

        neighbors = tf.gather_nd(inputs, kNNIdx)

        # fCh = 6
        # if channels > 32:
            # fCh = 12

        ### Do the convolution ###
        mlp = mlp
        n = kNNEdg
        for i in range(len(mlp)):
            n = autofc(n, mlp[i], None, name = 'kernel/mlp%d' % i)
            n = brn(n, 0.999, is_train, 'kernel/mlp%d/norm' % i)
            n = tf.nn.leaky_relu(n)
            # n = tf.nn.elu(n)

        n = autofc(n, channels * fCh, None, name = 'kernel/mlp_out')
    
        # n = autofc(n, channels * fCh, tf.nn.tanh, name = 'kernel/mlp_out')

        # print(summary_scope)
        if summary_scope is not None:
            with tf.variable_scope(summary_scope):
                tf.summary.histogram('kernel weights', n)
        
        cW = tf.reshape(n, [bs, N, k, channels, fCh])
        
        # normalize it to be a true "PDF"
        if PDFNorm == True:
            cW = tf.nn.relu(cW)
            normalize_factor = tf.reduce_mean(cW, axis = [0, 1, 2], keepdims = True)
            cW = cW / (normalize_factor + 1e-5)
            cW = tf.log(cW + 1e-5)
            
            scaling = tf.get_variable('cW_scale', shape = [1, 1, 1, channels, fCh], dtype = default_dtype)
            cW = scaling * cW

        # Batch matmul won't work for more than 65535 matrices ???
        # n = tf.matmul(n, tf.reshape(neighbors, [bs, Nx, k, Cy, 1]))
        # Fallback solution
        n = autofc(neighbors, channels * fCh, None, name = 'feature/feature_combine')
        # n = norm(n, 0.999, is_train, 'feature/norm')

        # MatMul
        n = tf.reshape(n, [bs, N, k, channels, fCh])
        n = tf.reduce_sum(tf.multiply(cW, n), axis = -1)

        if nearestNorm == True:
            # n     => [bs, N, k, channels]
            # nnnorm=> [bs, N]
            nnnorm_collected = tf.gather_nd(nnnorm, kNNIdx)
            n = tf.multiply(n, tf.broadcast_to(tf.reshape(nnnorm_collected, [bs, N, k, 1]), [bs, N, k, channels]))

        print(n.shape)
        print("Graph cConv: [%3d x %2d] = %4d" % (channels, fCh, channels * fCh))
        # n = tf.reshape(n, [bs, Nx, k, channels])

        b = tf.get_variable('b_out', dtype = default_dtype, shape = [channels], initializer = b_init, trainable = True)
        
        if max_pool_conv == False:
            n = tf.reduce_mean(n, axis = 2)
            n = tf.nn.bias_add(n, b)
            n = autofc(n, channels, None, name = 'kernel/feature_combine')
        else:
            n = tf.reduce_max(n, axis = 2)
            n = autofc(n, channels, None, name = 'kernel/evidance_combine')
        
        if act is not None:
            n = act(n)

    return n # [bs, Nx, channels]

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
    
    return pool_position, pool_features

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
    
    return pool_position, pool_features

def norm_tun(inputs, maxLength):

    _norm = tf.norm(inputs, axis = -1, keepdims = True) # [bs, N, 1]
    _norm_tun = tf.nn.tanh(_norm) * maxLength
    _res = inputs / (_norm + 1e-4) * _norm_tun
    return _res

def kNNGPosition_refine(input_position, input_feature, refine_maxLength, act, hidden = 128, W_init = tf.truncated_normal_initializer(stddev=0.1), b_init = tf.constant_initializer(value=0.0), name = 'kNNGPosRefine'):

    with tf.variable_scope(name):

        bs = input_position.shape[0]
        N = input_position.shape[1]
        C = input_feature.shape[2]
        pC = input_position.shape[2]

        assert N == input_feature.shape[1] and bs == input_feature.shape[0]

        pos_res = autofc(input_feature, pC, None, True, SN, 'refine') # [bs, N, pC]
        pos_res = norm_tun(pos_res, refine_maxLength)

        refined_pos = tf.add(input_position, pos_res)

        return refined_pos

def gconv(inputs, gidx, gedg, filters, act, use_norm = True, is_train = True, name = 'gconv', W_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype), b_init = tf.constant_initializer(value=0.0), mlp = None, nnnorm = None, kernel_filters = 16, k_init = tf.constant_initializer(value=1.0)):
    
    with tf.variable_scope(name):
        
        fCh = convd_ch
        if filters >= 256: 
            fCh = convd_ch

        # feature
        if True:
            if mlp == None:
                mlp = [filters * 2, filters * 2]
                # mlp = [filters * 3 // 2]
            
            n = bip_kNNGConvLayer_feature(inputs, gidx, gedg, None, filters, fCh, mlp, is_train, W_init, b_init, 'gconv', nnnorm = nnnorm)
            if use_norm:
                n = norm(n, 0.999, is_train, name = 'norm')
                pass
            if act:
                n = act(n)
        
        # kernel
        else:
            n = bip_kNNGConvLayer_kernel(inputs, gidx, gedg, None, filters, kernel_filters, fCh, is_train, W_init, b_init, k_init, name)
            if use_norm:
                n = norm(n, 0.999, is_train, name = 'norm')
                pass
            if act:
                n = act(n)

    return n

def convRes(inputs, gidx, gedg, num_conv, num_res, filters, act, use_norm = True, is_train = True, name = 'block', W_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype), b_init = tf.constant_initializer(value=0.0), mlp = None, nnnorm = None, kernel_filters = 16, k_init = tf.constant_initializer(value=1.0)):

    with tf.variable_scope(name):
        
        n = inputs
        tmp = n
        for r in range(num_res):
            nn = n
            with tf.variable_scope('res%d' % r):
                for c in range(num_conv):
                    nn = gconv(nn, gidx, gedg, filters, act, use_norm, is_train, 'conv%d' % c, W_init, b_init, mlp, nnnorm = nnnorm, kernel_filters = kernel_filters, k_init = k_init)
            
            n = n + nn
        
        if num_res > 1:
            n = n + tmp
        
        return n

def autofc(inputs, outDim, act = None, bias = True, name = 'fc', W_init = None):
    
    input_shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, input_shape[-1]])

    with tf.variable_scope(name):
        
        w = tf.get_variable('W', shape = [input_shape[-1], outDim], dtype = default_dtype, initializer = W_init)

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

def lr_mult(alpha):
    @tf.custom_gradient
    def _lr_mult(x):
        def grad(dy):
            return dy * alpha * tf.ones_like(x)
        return x, grad
    return _lr_mult
        
# TODO: position re-fine layer

class model_particles:

    def __init__(self, gridMaxSize, latent_dim, batch_size, optimizer, outDim, config):
        
        self.config = config

        # Size of each grid
        self.gridMaxSize = gridMaxSize
        self.particle_latent_dim = latent_dim
        self.particle_hidden_dim = 64
        self.cluster_feature_dim = config['cdim']
        self.cluster_count = config['ccnt']
        self.combine_method = tf.reduce_sum
        self.loss_func = tf.abs
        self.resSize = 1
        self.batch_size = batch_size
        self.knn_k = 16
        self.useVector = config['useVector']

        self.outDim = outDim

        self.act = tf.nn.leaky_relu
        self.convact = tf.nn.leaky_relu

        self.wdev=0.1

        self.stages = config['stages'] # 2 stages

        self.loss_metric = config['loss']

        self.sim_steps = config['steps']

        self.ph_X           = tf.placeholder(default_dtype, [self.batch_size, self.sim_steps, self.cluster_count, self.cluster_feature_dim + 3])
        self.ph_stepweights = tf.placeholder(default_dtype, [self.batch_size, self.sim_steps])

        self.optimizer = optimizer

        global normalization_method
        global max_pool_conv
        global convd_ch

        normalization_method = config['normalization']
        max_pool_conv = config['maxpoolconv']
        convd_ch = config['convd']

    # 1 of a batch goes in this function at once.
    def particleEncoder(self, input_particle, output_dim, early_stop = 0, is_train = False, reuse = False, returnPool = False):

        config = self.config

        # w_init = tf.random_normal_initializer(stddev=self.wdev)
        w_init = tf.random_normal_initializer(stddev = 0.01 * self.wdev)
        # w_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleEncoder", reuse = reuse) as vs:

            # We are going to use a way deeper model than before. Please refer to model_particlesTest_backup.py for original codes.
            
            # ShapeNet_regular_featureSqz
            blocks = config['encoder']['blocks']
            particles_count = config['encoder']['particles_count']
            conv_count = config['encoder']['conv_count']
            res_count = config['encoder']['res_count']
            kernel_size = config['encoder']['kernel_size']
            bik = config['encoder']['bik']
            channels = config['encoder']['channels']

            kfilters = kernel_size
            if 'kfilters' in config['encoder']:
                kfilters = config['encoder']['kfilters']

            bikfilters = kernel_size
            if 'bikfilters' in config['encoder']:
                bikfilters = config['encoder']['bikfilters']

            self.pool_count = blocks - 1
            self.pCount = particles_count
            self.encBlocks = blocks

            try:
                bik
            except NameError:
                bik = kernel_size
            
            gPos = input_particle[:, :, :3]
            if 3 >= self.outDim:
                n = tf.ones_like(input_particle[:, :, :3])
            else:
                n = input_particle[:, :, 3:self.outDim]

            var_list = []
            pool_pos = []
            pool_eval_func = []
            freq_loss = 0

            target_dim = self.cluster_feature_dim
            target_block = early_stop
            if early_stop == 0:
                target_block = blocks

            edg_sample = None

            nnnorm = None
            
            for i in range(target_block):
                
                with tf.variable_scope('enc%d' % i):

                    if i > 0:
                        
                        # Pooling
                        prev_n = n
                        prev_pos = gPos
                        print("Pooling...")
                        gPos, n = kNNGPooling_farthest(n, gPos, particles_count[i])
                        print("pooling finished")
                        # Single point
                        # if i == 4:
                        #     gPos = tf.zeros_like(gPos)

                        pool_pos.append(gPos)

                        # Collect features after pool
                        _, _, bpIdx, bpEdg, _ = bip_kNNG_gen(gPos, prev_pos, bik[i], 3, name = 'gpool/ggen', recompute = False)
                        n = gconv(prev_n, bpIdx, bpEdg, channels[i], self.act, True, is_train, 'gpool/gconv', w_init, b_init, nnnorm = nnnorm, kernel_filters = bikfilters[i])

                    if i == 1:
                        edg_sample = bpEdg[..., 0:3] # + tf.random.uniform([self.batch_size, particles_count[i], 1, 3], minval = -24., maxval = 24.)
                        edg_sample = tf.reshape(edg_sample, [-1, 3])
                        edg_sample = [edg_sample, [self.batch_size * particles_count[i]]]

                    gPos, gIdx, gEdg, nnnorm = kNNG_gen(gPos, kernel_size[i], 3, name = 'ggen')
                    
                    if i == 0:
                        gt_density = nnnorm

                    if i == 0:
                        n = gconv(n, gIdx, gEdg, channels[i], self.act, True, is_train, 'conv_first', w_init, b_init, nnnorm = nnnorm, kernel_filters = kfilters[i])

                    n = convRes(n, gIdx, gEdg, conv_count[i], 1, channels[i], self.act, True, is_train, 'conv', w_init, b_init, nnnorm = nnnorm, kernel_filters = kfilters[i])
                    n = convRes(n, gIdx, gEdg, 2,  res_count[i], channels[i], self.act, True, is_train, 'res', w_init, b_init, nnnorm = nnnorm, kernel_filters = kfilters[i])

            if self.useVector == False and early_stop == 0:
                n = autofc(n, target_dim, name = 'enc%d/convOut' % (blocks - 1))
            
            if self.useVector == True and early_stop == 0:
                with tf.variable_scope('enc%d' % blocks):
                    zeroPos = tf.zeros([self.batch_size, 1, 3])
                    _, _, bpIdx, bpEdg, _ = bip_kNNG_gen(zeroPos, gPos, particles_count[blocks - 1], 3, name = 'globalPool/bipgen', recompute = False)
                    n = gconv(n, bpIdx, bpEdg, 512, self.act, True, is_train, 'globalPool/gconv', w_init, b_init, mlp = [512, 512], nnnorm = nnnorm, kernel_filters = 64)
                    n = autofc(n, 512, self.act, name = 'globalPool/fc')
                    # n = norm(n, 0.999, is_train, name = 'globalPool/norm')
                    n = autofc(n, 512, name = 'globalPool/fc2')
                    
                    # n = tf.concat([gPos, n], axis = -1) # [bs, ccnt, cdim + prange]
                    # n = autofc(n, 512, name = 'fc1')
                    # n = norm(n, 0.999, is_train, name = 'norm')
                    # n = autofc(n, 512, name = 'fc2')
                    # n = tf.reduce_max(n, axis = 1, keepdims = True) # [bs, 1, 512]

                    # n = tf.reduce_max(n, axis = 1, keepdims = True)
                    # n = autofc(n, 512, name = 'fc1')
                    # n = autofc(n, 512, name = 'fc2')
                    # n = autofc(n, 512, name = 'fc3')

                    gPos = zeroPos

            if returnPool == True:
                return gPos, n, var_list, pool_pos, freq_loss, pool_eval_func, gt_density

            return gPos, n, var_list, freq_loss, pool_eval_func, edg_sample, gt_density
    
    def getKernelEmbeddings(self, input_particle, channels, fCh, mlp, layer_name):

        is_train = False

        # default values
        if mlp == None:
            mlp = [channels * 2, channels * 2]
        if fCh == None:
            fCh = 2

        with tf.variable_scope(layer_name.split('/')[0], reuse = True) as vs:
            return bip_kNNGConvLayer_feature_getKernel(input_particle, channels, fCh, mlp, '/'.join(layer_name.split('/')[1:]))

    def particleDecoder(self, cluster_pos, local_feature, groundTruth_card, output_dim, begin_block = 0, is_train = False, reuse = False, bs_override = None):

        config = self.config

        bs = self.batch_size
        if bs_override is not None:
            bs = bs_override

        w_init = tf.random_normal_initializer(     stddev = 0.01 * self.wdev)
        w_init_fold = tf.random_normal_initializer(stddev = 0.01 * self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev = 0.01 * self.wdev)
        
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleDecoder", reuse = reuse) as vs:
 
            CC = local_feature.shape[2]
                
            hd = self.particle_hidden_dim
            ld = self.particle_latent_dim
            _k = self.knn_k

            # Single decoding stage
            coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count

            if self.useVector == True:
                coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, 1

            blocks = config['decoder']['blocks']
            pcnt = config['decoder']['pcnt']
            generator = config['decoder']['generator']
            maxLen = config['decoder']['maxLen']
            nConv = config['decoder']['nConv']
            nRes = config['decoder']['nRes']
            hdim = config['decoder']['hdim']
            fdim = config['decoder']['fdim']
            gen_hdim = config['decoder']['gen_hdim']
            knnk = config['decoder']['knnk']
            generator_struct = config['decoder']['genStruct']
            genFeatures = False
            monotonic = False
            if 'mono' in config['decoder']:
                monotonic = config['decoder']['mono']
            if 'genFeatures' in config['decoder']:
                genFeatures = config['decoder']['genFeatures']

            pos_range = 3

            gen_only = []
            coarse_pos_ref = coarse_pos

            regularizer = 0.0
            nnnorm = None

            # Meta-data
            meta = []
            # int_meta = tf.reshape(tf.range(coarse_cnt), [bs, coarse_cnt, 1])

            self.decBlocks = blocks
            if begin_block > 0:
                coarse_cnt = pcnt[begin_block - 1]

            for _bi in range(blocks - begin_block):
                bi = begin_block + _bi

                with tf.variable_scope('dec%d' % bi):

                    # Fully-connected generator (Non-distribution-based) & Full generators (pcnt[bi] instead of pcnt[bi] - coarse_cnt
                
                    # Check for good setups
                    assert pcnt[bi] % coarse_cnt == 0
                    meta.append(coarse_cnt)

                    n_per_cluster = pcnt[bi] // coarse_cnt

                    if generator_struct == 'concat':
                        # generator
                        # int_meta = tf.broadcast_to(tf.reshape(int_meta, [bs, coarse_cnt, 1, -1]), [bs, coarse_cnt, n_per_cluster, 1])
                        z = tf.random.uniform([bs, coarse_cnt, n_per_cluster, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)
                        # z = tf.random.uniform([bs, coarse_cnt, n_per_cluster, 3], minval = -0.5, maxval = 0.5, dtype = default_dtype)
                        uniform_dist = z
                        fuse_fea = autofc(coarse_fea, fdim[bi], name = 'feaFuse')
                        z = tf.concat([z, tf.broadcast_to(tf.reshape(fuse_fea, [bs, coarse_cnt, 1, fdim[bi]]), [bs, coarse_cnt, n_per_cluster, fdim[bi]])], axis = -1)
                        
                        # n = tf.reshape(z, [bs, pcnt[bi], fdim[bi]])
                        n = tf.reshape(z, [bs, pcnt[bi], fdim[bi] * 2])
                        # n = tf.reshape(z, [bs, pcnt[bi], fdim[bi] + 3])
                        
                        for gi in range(generator[bi]):
                            with tf.variable_scope('gen%d' % gi):
                                if monotonic:
                                    mono_cnt = -1
                                    if gi == 0:
                                        mono_cnt = fdim[bi]
                                        pass
                                    n = autofc_mono(n, gen_hdim[bi], mono = mono_cnt, name = 'mono_fc')
                                else:
                                    n = autofc(n, gen_hdim[bi], name = 'fc')
                                # n = norm(n, 0.999, is_train, name = 'norm')
                                n = self.act(n)
                        
                        if genFeatures:
                            nf = autofc(n, hdim[bi], name = 'gen_feature_out')
                        
                        if monotonic:
                            n = autofc_mono(n, pos_range, name = 'mono_gen_out')
                        else:
                            n = autofc(n, pos_range, name = 'gen_out')

                        n = tf.reshape(n, [bs, coarse_cnt, n_per_cluster, pos_range])

                    elif generator_struct == 'AdaIN':
                        
                        # generator
                        z = tf.random.uniform([bs, coarse_cnt, n_per_cluster, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)
                        uniform_dist = z
                        
                        fuse_fea = autofc(coarse_fea, fdim[bi], name = 'feaFuse')
                       
                        n = z

                        for gi in range(generator[bi]):
                            with tf.variable_scope('gen%d' % gi):
                                if monotonic:
                                    mono_cnt = -1
                                    if gi == 0:
                                        mono_cnt = fdim[bi]
                                        pass
                                    n = autofc_mono(n, gen_hdim[bi], mono = mono_cnt, name = 'mono_fc')
                                else:
                                    n = autofc(n, gen_hdim[bi], name = 'fc')
                                
                                # s_mean = autofc(coarse_fea, gen_hdim[bi], name = 'feaFuse_mean')
                                # s_std  = autofc(coarse_fea, gen_hdim[bi], name = 'feaFuse_std')
                                s_mean = autofc(fuse_fea, gen_hdim[bi], name = 'feaFuse_mean')
                                s_std  = autofc(fuse_fea, gen_hdim[bi], name = 'feaFuse_std')

                                s_mean = tf.reshape(s_mean, [bs, coarse_cnt, 1, gen_hdim[bi]])
                                s_std  = tf.reshape(s_std,  [bs, coarse_cnt, 1, gen_hdim[bi]])

                                n = AdaIN(n, s_mean, s_std)
                                # n = AdaIN(n, s_mean, s_std, axes = [0, 1, 2])

                                n = self.act(n)
                        
                        if genFeatures:
                            nf = autofc(n, hdim[bi], name = 'gen_feature_out')
                        
                        if monotonic:
                            n = autofc_mono(n, pos_range, name = 'mono_gen_out')
                        else:
                            n = autofc(n, pos_range, name = 'gen_out')

                        n = tf.reshape(n, [bs, coarse_cnt, n_per_cluster, pos_range])
                        print(n)
                    
                    elif generator_struct == 'final_selection':
                        
                        # weight, bias, transformation generator
                        with tf.variable_scope('weight_gen'):
                            l = autofc(coarse_fea, gen_hdim[bi], name = 'mlp1')
                            # l = norm(l, 0.999, is_train, name = 'mlp1/norm')
                            l = self.act(l)
                            l = autofc(coarse_fea, gen_hdim[bi], name = 'mlp2')
                            # l = norm(l, 0.999, is_train, name = 'mlp2/norm')
                            l = self.act(l)

                            w = autofc(l, pos_range * fdim[bi] , name = 'mlp/w')
                            b = autofc(l, pos_range            , name = 'mlp/b')

                            w = tf.reshape(w, [bs, coarse_cnt, 1,  fdim[bi], pos_range])
                            # w = tf.nn.softmax(w, axis = 3)
                            if monotonic:
                                w = tf.exp(w) # monotonic
                            # t = tf.reshape(t, [bs, coarse_cnt, 1, pos_range, pos_range])
                            b = tf.reshape(b, [bs, coarse_cnt, 1, pos_range])

                        z = tf.random.uniform([bs, coarse_cnt, n_per_cluster, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)
                        uniform_dist = z

                        # Regular generator
                        for gi in range(generator[bi]):
                            with tf.variable_scope('gen%d' % gi):
                                if monotonic:
                                    z = autofc_mono(z, gen_hdim[bi], name = 'mono_fc')
                                else:
                                    z = autofc(z, gen_hdim[bi], name = 'fc')

                                if gi < (generator[bi] - 1):
                                    z = self.act(z)
                        # Collect features
                        z = tf.multiply(w, tf.reshape(z, [bs, coarse_cnt, n_per_cluster, fdim[bi], 1]))
                        z = tf.reduce_sum(z, axis = 3) # z <- [bs, coarse_cnt, n_per_cluster, pos_range]
                        z = z + b

                        # Linear transformation
                        # z = tf.multiply(t, tf.reshape(z, [bs, coarse_cnt, n_per_cluster, pos_range, 1]))
                        # z = tf.reduce_sum(z, axis = 3)

                        n = z

                    n_ref = n

                    # regularizer to keep in local space
                    reg_curr = maxLen[bi]
                    if reg_curr == None:
                        reg_curr = 0.01
                    regularizer += reg_curr * tf.reduce_mean(tf.norm(n, axis = -1))

                    # Back to world space
                    n = n + tf.reshape(coarse_pos, [bs, coarse_cnt, 1, pos_range])
                    n_ref = n_ref + tf.reshape(coarse_pos_ref, [bs, coarse_cnt, 1, pos_range])

                    ap = tf.reshape(n, [bs, pcnt[bi], pos_range])
                    ap_ref = tf.reshape(n_ref, [bs, pcnt[bi], pos_range])
                    nf = tf.reshape(nf, [bs, pcnt[bi], -1])

                    # General operations for full generators
                    gen_only.append(ap)

                    # Outputs of this stage
                    pos = ap
                    coarse_pos_ref = ap_ref

                    ## "Transposed convolution" 's
                    
                    # get feature
                    # Bipartite graph
                    if genFeatures:
                        n = nf
                    else:
                        _, _, gp_idx, gp_edg, nnnorm = bip_kNNG_gen(pos, coarse_pos, knnk[bi], pos_range, name = 'bipggen', xysame = False, recompute = True)
                        n = gconv(coarse_fea, gp_idx, gp_edg, hdim[bi], self.act, True, is_train, 'convt', w_init, b_init, nnnorm)

                    if nConv[bi] > 0:
                        _, gidx, gedg, nnnorm = kNNG_gen(pos, knnk[bi], 3, name = 'ggen')
                        n = convRes(n, gidx, gedg, nConv[bi], nRes[bi], hdim[bi], self.act, True, is_train, 'resblock', w_init, b_init, nnnorm = nnnorm)

                    coarse_pos = pos
                    coarse_fea = n
                    coarse_cnt = pcnt[bi]

                    print("Stage %d: " % bi + str(coarse_pos) + str(coarse_cnt))

            final_particles = coarse_pos
            final_particles_ref = coarse_pos_ref
            if is_train == False:
                final_particles_ref = coarse_pos
            n = coarse_fea
            
            print(final_particles)

            if output_dim > pos_range:
                n = autofc(n, output_dim - pos_range, name = 'dec%d/finalLinear' % (blocks - 1))
                final_particles = tf.concat([pos, n], -1)

            print(final_particles)

            return 0, [final_particles, final_particles_ref, gen_only[0]], 0, regularizer, meta

    def simulator(self, pos, particles, name = 'Simluator', is_train = True, reuse = False):

        config = self.config

        w_init = tf.random_normal_initializer(stddev = 0.005 * self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev = 0.001 * self.wdev)
        b_init = tf.constant_initializer(value=0.0)

        with tf.variable_scope(name, reuse = reuse) as vs:
            
            _, gIdx, gEdg, nnnorm = kNNG_gen(pos, config['simulator']['knnk'], 3, name = 'simulator/ggen')
            layers = config['simulator']['layers']
            n = particles
            Np = particles.shape[1]
            C = particles.shape[2]

            if config['simulator']['GRU'] == True:

                hidden_dim_GRU = config['simulator']['GRU_hd']

                n = autofc(n, hidden_dim_GRU - 3, None, False, 'GRU_inFC')
                n = tf.concat([n, pos], axis = -1)

                # GRU
                if config['simulator']['IN'] == False:

                    upd_gate_input  = gconv(n, gIdx, gEdg, hidden_dim_GRU, None, True, is_train = is_train, name = 'up_inp', W_init = w_init, b_init = b_init)
                    upd_gate = tf.nn.sigmoid(upd_gate_input)

                    res_gate_input  = gconv(n, gIdx, gEdg, hidden_dim_GRU, None, True, is_train = is_train, name = 'rs_inp', W_init = w_init, b_init = b_init)
                    res_gate = tf.nn.sigmoid(res_gate_input)

                    nxt_hid_input   = gconv(res_gate * n, gIdx, gEdg, hidden_dim_GRU, None, True, is_train = is_train, name = 'nh_inp', W_init = w_init, b_init = b_init)
                    next_hidden = (1 - upd_gate) * n + upd_gate * (tf.nn.tanh( nxt_hid_input))
                
                else:

                    upd_gate_input  = bip_kNNGConvLayer_IN(n, gIdx, gEdg, None, hidden_dim_GRU, 2, [hidden_dim_GRU], is_train, w_init, b_init, 'up_inp')
                    upd_gate = tf.nn.sigmoid(upd_gate_input)

                    res_gate_input  = bip_kNNGConvLayer_IN(n, gIdx, gEdg, None, hidden_dim_GRU, 2, [hidden_dim_GRU], is_train, w_init, b_init, 'rs_inp')
                    res_gate = tf.nn.sigmoid(res_gate_input)

                    nxt_hid_input   = bip_kNNGConvLayer_IN(res_gate * n, gIdx, gEdg, None, hidden_dim_GRU, 2, [hidden_dim_GRU], is_train, w_init, b_init, 'nh_inp')
                    next_hidden = (1 - upd_gate) * n + upd_gate * (tf.nn.tanh( nxt_hid_input))

                n = autofc(next_hidden, hidden_dim_GRU, None, False, 'GRU_outFC')

            else:

                n = tf.concat([n, pos], axis = -1)

                if config['simulator']['IN'] == False:

                    nn = n

                    for i in range(len(layers)):
                        nn = gconv(nn, gIdx, gEdg, layers[i], self.act, True, is_train = is_train, name = 'gconv%d' % i, W_init = w_init, b_init = b_init)
                        # nn = lnorm(nn, 0.999, is_train, 'gconv%d/norm' % i)

                    nn = gconv(nn, gIdx, gEdg, C, self.act, True, is_train = is_train, name = 'gconvLast', W_init = w_init, b_init = b_init)
                    
                    # nn = lnorm(nn, 0.999, is_train, 'gconv%d/norm' % i)

                else:

                    nn = bip_kNNGConvLayer_IN(n, gIdx, gEdg, None, C, 6, [64], is_train, w_init, b_init, 'gconv0')
                    # nn = lnorm(nn, 0.999, is_train, 'gconv0/norm')
                
                n = n[:, :, :-3] + nn
                n = tf.nn.tanh(n)
                # nn = tf.concat([n, pos], axis = -1)

            nn = n

            pmlp = [128]
            for i in range(len(pmlp)):
                nn = autofc(nn, pmlp[i], self.act, name = 'pRefine/mlp%d' % i, W_init = w_init_pref)
            dPos = autofc(nn, 3, None, name = 'pRefine/mlp_out', W_init = w_init_pref)
            # dPos = norm(dPos, 0.999, is_train, 'pRefine/norm')

            pos += dPos

        return pos, n

    def chamfer_metric(self, particles, particles_ref, groundtruth, pos_range, loss_func, EMD = False, Sinkhorn = False, is_train = True):
        
        config = self.config

        if config['loss'] == 'sinkhorn' and is_train:

            bs = groundtruth.shape[0]
            Np = particles.shape[1]
            Ng = groundtruth.shape[1]

            # NOTE: current using position (0:3) only here for searching nearest point.
            row_predicted = tf.reshape(particles_ref[:, :, 0:pos_range], [bs, Np, 1, pos_range])
            col_groundtru = tf.reshape(  groundtruth[:, :, 0:pos_range], [bs, 1, Ng, pos_range])
            # distance = tf.norm(row_predicted - col_groundtru, ord = 'euclidean', axis = -1)
            distance = tf.sqrt(tf.add_n(tf.unstack(tf.square(row_predicted - col_groundtru), axis = -1)))

            _a = tf.ones([bs, Np], tf.float32)
            _b = tf.ones([bs, Ng], tf.float32)

            _a /= config['decoder']['pcnt'][len(config['decoder']['pcnt']) - 1]
            _b /= config['decoder']['pcnt'][len(config['decoder']['pcnt']) - 1]

            transport_mat = tf.py_func(Sinkhorn_dist, [particles_ref, groundtruth, _a, _b, distance], tf.float32)
            distance_loss = distance * transport_mat
            distance_loss = tf.reduce_sum(distance_loss, axis = [1, 2])

        elif config['loss'] == 'EMDUB' or not is_train:
        # elif False:

            bs = groundtruth.shape[0]
            Np = particles.shape[1]
            Ng = groundtruth.shape[1]
            
            match = approx_match(groundtruth[:, :, 0:3], particles_ref[:, :, 0:3]) # [bs, Np, Ng]
            row_predicted = tf.reshape(  particles[:, :, :], [bs, Np, 1, -1])
            col_groundtru = tf.reshape(groundtruth[:, :, :], [bs, 1, Ng, -1])
            distance = tf.sqrt(tf.add_n(tf.unstack(tf.square(row_predicted - col_groundtru), axis = -1)))
            distance = distance * match
            distance_loss = tf.reduce_mean(tf.reduce_sum(distance, axis = -1), axis = -1)
        
        elif config['loss'] == 'chamfer' and is_train:
        # elif True:

            raise NotImplementedError
            
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

        else:

            print("Unknown loss type: %s, expected - [chamfer, EMDUB, sinkhorn]" % config['loss'])
            raise NotImplementedError

        return tf.cast(distance_loss, default_dtype)

    def custom_dtype_getter(self, getter, name, shape=None, dtype=default_dtype, *args, **kwargs):
        
        if dtype is tf.float16:
            
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        
        else:
            
            return getter(name, shape, dtype, *args, **kwargs)

    def build_network(self, is_train, reuse):

        normalized_X = (self.ph_X - tf.broadcast_to(self.normalize['mean'], self.ph_X.shape)) / tf.broadcast_to(self.normalize['std'], self.ph_X.shape)

        # pX, fX = normalized_X[:, :, :, :3], normalized_X[:, :, :, 3:]
        pX_cur, fX_cur = normalized_X[:, 0, :, :3], normalized_X[:, 0, :, 3:]
        loss_total = 0.0
        
        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):

            for ls in range(self.sim_steps - 1):
                pX_cur, fX_cur = self.simulator(pX_cur, fX_cur, name = 'Simulator', is_train = is_train, reuse = False if ls == 0 and reuse == False else True)
                X_cur = tf.concat([pX_cur, fX_cur], axis = -1)
                loss = self.chamfer_metric(X_cur, X_cur, normalized_X[:, ls+1, :, :], 3, tf.square, EMD = True)
                loss_total += tf.reduce_mean(loss * self.ph_stepweights[:, ls+1]) / (self.sim_steps - 1)
        
        return loss_total

    # Only simulates posX & feaX for a single step
    def build_predict_Sim(self, pos, fea, is_train = False, reuse = False):

        _X = tf.concat([pos, fea], axis = -1)
        normalized_X = (_X - tf.broadcast_to(self.normalize['mean'], _X.shape)) / tf.broadcast_to(self.normalize['std'], _X.shape)

        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
            
            sim_posY, sim_feaY = self.simulator(normalized_X[:, :, :3], normalized_X[:, :, 3:], 'Simulator', is_train, reuse)

        sRec = tf.concat([sim_posY, sim_feaY], axis = -1)
        sRec = sRec * tf.broadcast_to(self.normalize['std'], sRec.shape) + tf.broadcast_to(self.normalize['mean'], sRec.shape)
        
        return sRec[:, :, :3], sRec[:, :, 3:]

    # Decodes Y
    def build_predict_Dec(self, pos, fea, gt, dec_normalize, is_train = False, reuse = False, outDim = 6, bs_override = None):

        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
            
            # 0, [final_particles, final_particles_ref, gen_only[0]], 0, regularizer, meta
            _, [rec, _, rec_f], _, _, meta = self.particleDecoder(pos, fea, None, outDim, is_train = is_train, reuse = reuse, bs_override = bs_override)

        rec = rec

        rec = rec * tf.broadcast_to(dec_normalize['std'], rec.shape) + tf.broadcast_to(dec_normalize['mean'], rec.shape)

        reconstruct_loss = tf.reduce_mean(self.chamfer_metric(rec, rec, gt[:, :, 0:outDim], 3, tf.square, EMD = True))

        return rec, reconstruct_loss

    def build_test_output(self, gt_shape, gt_norm, reuse = True, decReuse = False):

        # gt - np array with [bs, 2048, 6]
        bs = gt_shape[0]
        Np = gt_shape[1]
        od = gt_shape[2]

        self.ph_tst_p = tf.placeholder(default_dtype, [bs, self.cluster_count, 3])
        self.ph_tst_f = tf.placeholder(default_dtype, [bs, self.cluster_count, self.cluster_feature_dim])
        self.ph_rec_g = tf.placeholder(default_dtype, [bs, Np, od])

        self.spos, self.sfea = self.build_predict_Sim(self.ph_tst_p, self.ph_tst_f, False, reuse)
        self.rec, self.rLoss = self.build_predict_Dec(self.ph_tst_p, self.ph_tst_f, self.ph_rec_g, gt_norm, False, decReuse, od, bs_override = bs)

    def get_test_output(self, inp, gt, sess, verbose = False):

        bs = gt.shape[0]
        ss = gt.shape[1]
        Np = gt.shape[2]
        od = gt.shape[3]

        # inp - [bs, ccnt, cdim + 3]
        _spos = inp[:, :, :3]
        _sfea = inp[:, :, 3:]
        
        loss = np.zeros((ss))
        rec = np.zeros((bs, ss, Np, od))

        _rec, _rloss = sess.run([self.rec, self.rLoss], feed_dict = {self.ph_tst_p: _spos, self.ph_tst_f: _sfea, self.ph_rec_g: gt[:, 0, :, :]})
        rec[:, 0, :, :] = _rec
        if verbose:
                print(colored("Test It 0000", 'red') + ' - %7.4f' % np.mean(_rloss))

        for s in range(ss - 1):

            cur_step = s + 1
            _spos, _sfea = sess.run([self.spos, self.sfea], feed_dict = {self.ph_tst_p: _spos, self.ph_tst_f: _sfea})
            _rec, _rloss = sess.run([self.rec, self.rLoss], feed_dict = {self.ph_tst_p: _spos, self.ph_tst_f: _sfea, self.ph_rec_g: gt[:, cur_step, :, :]})
            
            loss[cur_step] = np.mean(_rloss)
            rec[:, cur_step, :, :] = _rec

            if verbose:
                print(colored("Test It %04d" % cur_step, 'red') + ' - %7.4f' % np.mean(_rloss))
        
        return rec, loss

    def build_model(self):

        # Train & Validation
        self.train_loss = self.build_network(True, False)
        self.val_loss   = self.build_network(False, True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = self.optimizer.compute_gradients(self.train_loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.) if grad is not None else None, var) for grad, var in gvs]
            # self.train_op = self.optimizer.minimize(self.train_particleLoss)
            self.train_op = self.optimizer.apply_gradients(capped_gvs)
