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

def norm(inputs, decay, is_train, name):

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

def bip_kNNGConvLayer_feature(inputs, kNNIdx, kNNEdg, act, channels, fCh, is_train, W_init, b_init, name):
    
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
        mlp = [channels]
        n = kNNEdg
        for i in range(len(mlp)):
            n = autofc(n, mlp[i], tf.nn.elu, name = 'kernel/mlp%d' % i)
            n = norm(n, 0.999, is_train, 'kernel/norm')
        
        n = autofc(n, channels * fCh, tf.nn.tanh, name = 'kernel/mlp_out')
        
        cW = tf.reshape(n, [bs, N, k, channels, fCh])
        
        # Batch matmul won't work for more than 65535 matrices ???
        # n = tf.matmul(n, tf.reshape(neighbors, [bs, Nx, k, Cy, 1]))
        # Fallback solution
        n = autofc(neighbors, channels * fCh, None, name = 'feature/feature_combine')
        # n = norm(n, 0.999, is_train, 'feature/norm')

        # MatMul
        n = tf.reshape(n, [bs, N, k, channels, fCh])
        n = tf.reduce_sum(tf.multiply(cW, n), axis = -1)

        print(n.shape)
        print("Graph cConv: [%3d x %2d] = %4d" % (channels, fCh, channels * fCh))
        # n = tf.reshape(n, [bs, Nx, k, channels])

        b = tf.get_variable('b_out', dtype = default_dtype, shape = [channels], initializer = b_init, trainable = True)
        n = tf.reduce_mean(n, axis = 2)
        n = tf.nn.bias_add(n, b)
        
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
    _norm_tun = tf.nn.tanh(_norm) * refine_maxLength
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

def gconv(inputs, gidx, gedg, filters, act, norm = True, is_train = True, name = 'gconv', W_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype), b_init = tf.constant_initializer(value=0.0)):
    
    with tf.variable_scope(name):
        
        fCh = 6
        if filters >= 256: 
            fCh = 4
        
        n = bip_kNNGConvLayer_feature(inputs, gidx, gedg, None, filters, fCh, is_train, W_init, b_init, 'gconv')
        n = norm(n, decay, is_train, name = 'norm')
        if act:
            n = act(n)
    
    return n

def convRes(inputs, gidx, gedg, num_conv, num_res, filters, act, norm = True, is_train = True, name = 'block', W_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype), b_init = tf.constant_initializer(value=0.0)):

    with tf.variable_scope(name):
        
        n = inputs
        tmp = n
        for r in range(num_res):
            nn = n
            with tf.variable_scope('res%d' % r):
                for c in range(num_conv):
                    nn = gconv(nn, gidx, gedg, filters, act, norm, is_train, 'conv%d' % c, W_init, b_init)
            n = n + nn
        
        if num_res > 1:
            n = n + tmp
        
        return n

def autofc(inputs, outDim, act = None, bias = True, name = 'fc'):
    
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
        self.useVector = True

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
                    gPos, n = kNNGPooling_farthest(n, gPos, particles_count[i])
                    
                    # Single point
                    # if i == 4:
                    #     gPos = tf.zeros_like(gPos)

                    pool_pos.append(gPos)

                    # Collect features after pool
                    _, _, bpIdx, bpEdg = bip_kNNG_gen(gPos, prev_pos, bik[i], 3, name = 'gpool%d/ggen' % i)
                    n = gconv(prev_n, bpIdx, bpEdg, channels[i], self.act, True, is_train, 'gpool%d/gconv' % i, w_init, b_init)

                gPos, gIdx, gEdg = kNNG_gen(gPos, kernel_size[i], 3, name = 'ggen%d' % i)

                n = convRes(n, gIdx, gEdg, conv_count[i], 1, channels[i], self.act, True, is_train, 'g%d/conv' % i, w_init, b_init)
                n = convRes(n, gIdx, gEdg, 2,  res_count[i], channels[i], self.act, True, is_train, 'g%d/res' % i, w_init, b_init)

            n = autofc(n, self.cluster_feature_dim, 'convOut')
            
            if self.useVector == True:
                zeroPos = tf.zeros([self.batch_size, 1, 3])
                _, _, bpIdx, bpEdg = bip_kNNG_gen(zeroPos, gPos, particles_count[blocks - 1], 3, name = 'globalPool/bipgen')
                n = gconv(n, bpIdx, bpEdg, 512, self.act, True, is_train, 'globalPool/gconv', w_init, b_init)
                n = autofc(n, 512, name = 'globalPool/fc')
                gPos = zeroPos

            if returnPool == True:
                return gPos, n, var_list, pool_pos, freq_loss, pool_eval_func

            return gPos, n, var_list, freq_loss, pool_eval_func
    
    def particleDecoder(self, cluster_pos, local_feature, groundTruth_card, output_dim, is_train = False, reuse = False):

        # w_init = tf.random_normal_initializer(stddev=self.wdev)
        # w_init_fold = tf.random_normal_initializer(stddev= 1.0*self.wdev)
        # w_init_pref = tf.random_normal_initializer(stddev=0.03*self.wdev)
        
        w_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype)
        w_init_fold = w_init
        w_init_pref = w_init
        
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleDecoder", reuse = reuse) as vs:
 
            CC = local_feature.shape[2]
                
            hd = self.particle_hidden_dim
            ld = self.particle_latent_dim
            _k = self.knn_k

            # Single decoding stage
            coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
            blocks = 1
            pcnt = [self.gridMaxSize] # particle count
            generator = [6] # Generator depth
            maxLen = [1.0]
            nConv = [0]
            nRes = [0]
            hdim = [self.particle_hidden_dim // 3]
            fdim = [self.particle_latent_dim] # dim of features used for folding
            gen_hdim = [self.particle_latent_dim]
            knnk = [self.knn_k // 2]
            
            # [fullFC_regular, fullGen_regular] Setup for full generator - fully-connected
            # coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
            # blocks = 3
            # pcnt = [256, 1280, self.gridMaxSize] # particle count
            # generator = [4, 4, 4] # Generator depth
            # hdim = [hd * 2, hd, hd // 3]
            # fdim = [ld, ld, ld // 2] # dim of features used for folding
            # gen_hdim = [ld, ld, ld]
            # knnk = [_k, _k, _k // 2]
            
            # [fullGen_shallow]
            # coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
            # blocks = 2
            # pcnt = [1280, self.gridMaxSize] # particle count
            # generator = [6, 3] # Generator depth
            # maxLen = [1.0, 0.2]
            # nConv = [2, 0]
            # nRes = [2, 0]
            # hdim = [self.particle_hidden_dim, self.particle_hidden_dim // 3]
            # fdim = [self.particle_latent_dim, self.particle_latent_dim] # dim of features used for folding
            # gen_hdim = [self.particle_latent_dim, self.particle_latent_dim]
            # knnk = [self.knn_k, self.knn_k // 2]

            if self.useVector == True:

                coarse_pos, coarse_fea, coarse_cnt = cluster_pos, local_feature, self.cluster_count
                blocks = 2
                pcnt = [coarse_cnt, self.gridMaxSize] # particle count
                generator = [6, 6] # Generator depth
                maxLen = [None, 1.0]
                nConv = [2, 0]
                nRes = [4, 0]
                hdim = [self.particle_hidden_dim, self.particle_hidden_dim // 3]
                fdim = [512, self.particle_latent_dim] # dim of features used for folding
                gen_hdim = [512, self.particle_latent_dim]
                knnk = [self.knn_k, self.knn_k // 2]

            pos_range = 3

            gen_only = []

            regularizer = 0.0

            for bi in range(blocks):

                with tf.variable_scope('gr%d' % bi):

                    # Fully-connected generator (Non-distribution-based) & Full generators (pcnt[bi] instead of pcnt[bi] - coarse_cnt
                
                    # Check for good setups
                    assert pcnt[bi] % coarse_cnt == 0

                    n_per_cluster = pcnt[bi] // coarse_cnt

                    if False: # fc
                        n = coarse_fea
                        for gi in range(generator[bi]):
                            with tf.variable_scope('gen%d' % gi):
                                n = autofc(n, fdim[bi], 'fc')
                                n = norm(n, 0.999, is_train, name = 'norm')
                                n = self.act(n)
                        n = autofc(n, pos_range * n_per_cluster, 'gen_out')
                        n = tf.reshape(n, [self.batch_size, coarse_cnt, n_per_cluster, pos_range])

                        if maxLen[bi]:
                            n = norm_tun(n, maxLen[bi])

                        # Back to world space
                        n = n + tf.reshape(coarse_pos, [self.batch_size, coarse_cnt, 1, pos_range])
                        
                        ap = tf.reshape(n, [self.batch_size, pcnt[bi], pos_range])

                    else: # generator
                        z = tf.random.uniform([self.batch_size, coarse_cnt, n_per_cluster, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)
                        fuse_fea = autofc(coarse_fea, fdim[bi], name = 'feaFuse')
                        z = tf.concat([z, tf.broadcast_to(tf.reshape(fuse_fea, [self.batch_size, coarse_cnt, 1, fdim[bi]]), [self.batch_size, coarse_cnt, n_per_cluster, fdim[bi]])], axis = -1)
                        
                        n = tf.reshape(z, [self.batch_size, pcnt[bi], fdim[bi] * 2])
                        
                        for gi in range(generator[bi]):
                            with tf.variable_scope('gen%d' % gi):
                                n = autofc(n, fdim[bi], 'fc')
                                n = norm(n, 0.999, is_train, name = 'norm')
                                n = self.act(n)
                        n = autofc(n, pos_range, 'gen_out')
                        n = tf.reshape(n, [self.batch_size, coarse_cnt, n_per_cluster, pos_range])

                        if maxLen[bi]:
                            n = norm_tun(n, maxLen[bi])

                        # Back to world space
                        n = n + tf.reshape(coarse_pos, [self.batch_size, coarse_cnt, 1, pos_range])
                        
                        ap = tf.reshape(n, [self.batch_size, pcnt[bi], pos_range])

                    # General operations for full generators
                    gen_only.append(ap)

                    # Outputs of this stage
                    pos = ap

                    ## "Transposed convolution" 's
                    
                    # get feature
                    # Bipartite graph
                    _, _, gp_idx, gp_edg = bip_kNNG_gen(pos, coarse_pos, knnk[bi], pos_range, name = 'bipggen')
                    n = gconv(coarse_fea, gp_idx, gp_edg, hdim[bi], self.act, True, is_train, 'convt', w_init, b_init)

                    _, gidx, gedg = kNNG_gen(pos, knnk[bi], 3, name = 'ggen' % r)
                    n = convRes(n, gidx, gedg, nConv[bi], nRes[bi], hdim[bi], True, is_train, 'resblock', w_init, b_init)

                    coarse_pos = pos
                    coarse_fea = n
                    coarse_cnt = pcnt[bi]

                final_particles = coarse_pos
                n = coarse_fea

                if output_dim > pos_range:
                    n = autofc(n, output_dim - pos_range, 'finalLinear')
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
