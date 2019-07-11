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
from config_graph_sim import config

from time import gmtime, strftime

from external.structural_losses.tf_approxmatch import approx_match, match_cost
from external.sampling.tf_sampling import farthest_point_sample, prob_sample

default_dtype = tf.float32
summary_scope = None
SN = False

def lnorm(inputs, decay, is_train, name):

    decay = 0.99

    # Disable norm
    # return inputs

    # if default_dtype == tf.float32:
    #     return tf.keras.layers.BatchNormalization(momentum = decay)(inputs, training = is_train)
    # if default_dtype == tf.float16:
    #     return BatchNormalizationF16(momentum = decay)(inputs, training = is_train)
   
    # return tf.keras.layers.BatchNormalization(momentum = decay)(inputs, training = is_train)
    # return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, fused = True)
    
    # Batch re-norm
    # return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, scope = name, fused = True, renorm = True)
    # return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, scope = name, fused = True)
    
    # Layer norm
    return tf.contrib.layers.layer_norm(inputs, scope = name)
    
    # Instance norm 
    if False:
        if default_dtype == tf.float32:
            return tf.contrib.layers.instance_norm(inputs, scope = name)
        else:
            return tf.contrib.layers.instance_norm(inputs, epsilon = 1e-3, scope = name)
    # return tf.contrib.layers.group_norm(inputs, 

def AdaIN(inputs, mean, std, axes = [2], name = 'AdaIN', epsilon = 1e-5):

    with tf.variable_scope(name):

        c_mean, c_var = tf.nn.moments(inputs, axes = axes, keep_dims = True)
        c_std = tf.sqrt(c_var + epsilon)

        return std * (inputs - c_mean) / c_std + mean

def norm(inputs, decay, is_train, name):

    decay = 0.99

    # Disable norm
    # return inputs

    # if default_dtype == tf.float32:
    #     return tf.keras.layers.BatchNormalization(momentum = decay)(inputs, training = is_train)
    # if default_dtype == tf.float16:
    #     return BatchNormalizationF16(momentum = decay)(inputs, training = is_train)
   
    # return tf.keras.layers.BatchNormalization(momentum = decay)(inputs, training = is_train)
    # return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, fused = True)
    
    # Batch re-norm
    return tf.contrib.layers.batch_norm(inputs, decay = decay, is_training = is_train, scope = name, fused = True, renorm = True)
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

        kNNEdg = tf.stop_gradient(kNNEdg)

        return posX, posY, kNNIdx, kNNEdg

def kNNG_gen(inputs, k, pos_range, name = 'kNNG_gen'):

    p, _, idx, edg = bip_kNNG_gen(inputs, inputs, k, pos_range, name)
    return p, idx, edg

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

def bip_kNNGConvLayer_feature(inputs, kNNIdx, kNNEdg, act, channels, fCh, mlp, is_train, W_init, b_init, name):
    
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
            n = norm(n, 0.999, is_train, 'kernel/mlp%d/norm' % i)
            n = tf.nn.elu(n)

        # n = autofc(n, channels * fCh, None, name = 'kernel/mlp_out')
        n = autofc(n, channels * fCh, tf.nn.tanh, name = 'kernel/mlp_out')

        # print(summary_scope)
        if summary_scope is not None:
            with tf.variable_scope(summary_scope):
                tf.summary.histogram('kernel weights', n)
        
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

def gconv(inputs, gidx, gedg, filters, act, use_norm = True, is_train = True, name = 'gconv', W_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype), b_init = tf.constant_initializer(value=0.0), mlp = None):
    
    with tf.variable_scope(name):
        
        fCh = 2
        if filters >= 256: 
            fCh = 2

        if mlp == None:
            # mlp = [filters * 2, filters * 2]
            mlp = [filters * 3 // 2]
        
        n = bip_kNNGConvLayer_feature(inputs, gidx, gedg, None, filters, fCh, mlp, is_train, W_init, b_init, 'gconv')
        if use_norm:
            # n = norm(n, 0.999, is_train, name = 'norm')
            pass
        if act:
            n = act(n)
    
    return n

def convRes(inputs, gidx, gedg, num_conv, num_res, filters, act, use_norm = True, is_train = True, name = 'block', W_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype), b_init = tf.constant_initializer(value=0.0), mlp = None):

    with tf.variable_scope(name):
        
        n = inputs
        tmp = n
        for r in range(num_res):
            with tf.variable_scope('res%d' % r):
                n = gconv(n, gidx, gedg, filters, act, use_norm, is_train, 'conv0', W_init, b_init, mlp)
                nn = n
                for c in range(num_conv - 1):
                    nn = gconv(nn, gidx, gedg, filters, act, use_norm, is_train, 'conv%d' % (c+1), W_init, b_init, mlp)
            
            if num_conv >= 1:
                n = n + nn
        
        if num_res > 1:
            n = n + tmp
        
        return n

def autofc(inputs, outDim, act = None, bias = True, name = 'fc', forceSN = False, W_init = None):
    
    input_shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, input_shape[-1]])

    with tf.variable_scope(name):
        
        w = tf.get_variable('W', shape = [input_shape[-1], outDim], dtype = default_dtype, initializer = W_init)

        if SN or forceSN:
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
        self.useVector = config['useVector']

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

        self.stages = config['stages'] # 2 stages

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
    def particleEncoder(self, input_particle, output_dim, early_stop = 0, is_train = False, reuse = False, returnPool = False):

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

            self.pool_count = blocks - 1
            self.pCount = particles_count
            self.encBlocks = blocks

            try:
                bik
            except NameError:
                bik = kernel_size
            
            gPos = input_particle[:, :, :3]
            n = input_particle[:, :, 3:] # DO NOT Ignore velocity
            var_list = []
            pool_pos = []
            pool_eval_func = []
            freq_loss = 0

            target_dim = self.cluster_feature_dim
            target_block = early_stop
            if early_stop == 0:
                target_block = blocks

            edg_sample = None
            
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
                        _, _, bpIdx, bpEdg = bip_kNNG_gen(gPos, prev_pos, bik[i], 3, name = 'gpool/ggen')
                        n = gconv(prev_n, bpIdx, bpEdg, channels[i], self.act, True, is_train, 'gpool/gconv', w_init, b_init)

                    if i == 1:
                        edg_sample = bpEdg[..., 0:3] + tf.random.uniform([self.batch_size, particles_count[i], 1, 3], minval = -24., maxval = 24.)
                        edg_sample = tf.reshape(edg_sample, [-1, 3])
                        edg_sample = [edg_sample, [self.batch_size * particles_count[i]]]

                    gPos, gIdx, gEdg = kNNG_gen(gPos, kernel_size[i], 3, name = 'ggen')

                    n = convRes(n, gIdx, gEdg, conv_count[i], 1, channels[i], self.act, True, is_train, 'conv', w_init, b_init)
                    n = convRes(n, gIdx, gEdg, 2,  res_count[i], channels[i], self.act, True, is_train, 'res', w_init, b_init)

            if self.useVector == False and early_stop == 0:
                n = autofc(n, target_dim, name = 'enc%d/convOut' % (blocks - 1))
            
            if self.useVector == True and early_stop == 0:
                with tf.variable_scope('enc%d' % blocks):
                    zeroPos = tf.zeros([self.batch_size, 1, 3])
                    # _, _, bpIdx, bpEdg = bip_kNNG_gen(zeroPos, gPos, particles_count[blocks - 1], 3, name = 'globalPool/bipgen')
                    # n = gconv(n, bpIdx, bpEdg, 512, self.act, True, is_train, 'globalPool/gconv', w_init, b_init, mlp = [512, 512])
                    # n = autofc(n, 512, self.act, name = 'globalPool/fc')
                    # n = norm(n, 0.999, is_train, name = 'globalPool/norm')
                    # n = autofc(n, 512, name = 'globalPool/fc2')
                    
                    n = tf.concat([gPos, n], axis = -1) # [bs, ccnt, cdim + prange]
                    n = autofc(n, 512, name = 'fc1')
                    # n = norm(n, 0.999, is_train, name = 'norm')
                    n = autofc(n, 512, name = 'fc2')
                    n = tf.reduce_max(n, axis = 1, keepdims = True) # [bs, 1, 512]

                    # n = tf.reduce_max(n, axis = 1, keepdims = True)
                    # n = autofc(n, 512, name = 'fc1')
                    # n = autofc(n, 512, name = 'fc2')
                    # n = autofc(n, 512, name = 'fc3')

                    gPos = zeroPos

            if returnPool == True:
                return gPos, n, var_list, pool_pos, freq_loss, pool_eval_func

            return gPos, n, var_list, freq_loss, pool_eval_func, edg_sample
    
    def particleDecoder(self, cluster_pos, local_feature, groundTruth_card, output_dim, begin_block = 0, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(     stddev = 0.01 * self.wdev)
        w_init_fold = tf.random_normal_initializer(stddev = 0.01 * self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev = 0.01 * self.wdev)
        
        # w_init = tf.contrib.layers.xavier_initializer(dtype = default_dtype)
        # w_init_fold = w_init
        # w_init_pref = w_init
        
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
            # int_meta = tf.reshape(tf.range(coarse_cnt), [self.batch_size, coarse_cnt, 1])

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
                        # int_meta = tf.broadcast_to(tf.reshape(int_meta, [self.batch_size, coarse_cnt, 1, -1]), [self.batch_size, coarse_cnt, n_per_cluster, 1])
                        z = tf.random.uniform([self.batch_size, coarse_cnt, n_per_cluster, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)
                        # z = tf.random.uniform([self.batch_size, coarse_cnt, n_per_cluster, 3], minval = -0.5, maxval = 0.5, dtype = default_dtype)
                        uniform_dist = z
                        fuse_fea = autofc(coarse_fea, fdim[bi], name = 'feaFuse')
                        z = tf.concat([z, tf.broadcast_to(tf.reshape(fuse_fea, [self.batch_size, coarse_cnt, 1, fdim[bi]]), [self.batch_size, coarse_cnt, n_per_cluster, fdim[bi]])], axis = -1)
                        
                        # n = tf.reshape(z, [self.batch_size, pcnt[bi], fdim[bi]])
                        n = tf.reshape(z, [self.batch_size, pcnt[bi], fdim[bi] * 2])
                        # n = tf.reshape(z, [self.batch_size, pcnt[bi], fdim[bi] + 3])
                        
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

                        n = tf.reshape(n, [self.batch_size, coarse_cnt, n_per_cluster, pos_range])

                    elif generator_struct == 'AdaIN':
                        
                        # generator
                        z = tf.random.uniform([self.batch_size, coarse_cnt, n_per_cluster, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)
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

                                s_mean = tf.reshape(s_mean, [self.batch_size, coarse_cnt, 1, gen_hdim[bi]])
                                s_std  = tf.reshape(s_std,  [self.batch_size, coarse_cnt, 1, gen_hdim[bi]])

                                n = AdaIN(n, s_mean, s_std)
                                # n = AdaIN(n, s_mean, s_std, axes = [0, 1, 2])

                                n = self.act(n)
                        
                        if genFeatures:
                            nf = autofc(n, hdim[bi], name = 'gen_feature_out')
                        
                        if monotonic:
                            n = autofc_mono(n, pos_range, name = 'mono_gen_out')
                        else:
                            n = autofc(n, pos_range, name = 'gen_out')

                        n = tf.reshape(n, [self.batch_size, coarse_cnt, n_per_cluster, pos_range])
                    
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

                            w = tf.reshape(w, [self.batch_size, coarse_cnt, 1,  fdim[bi], pos_range])
                            # w = tf.nn.softmax(w, axis = 3)
                            if monotonic:
                                w = tf.exp(w) # monotonic
                            # t = tf.reshape(t, [self.batch_size, coarse_cnt, 1, pos_range, pos_range])
                            b = tf.reshape(b, [self.batch_size, coarse_cnt, 1, pos_range])

                        z = tf.random.uniform([self.batch_size, coarse_cnt, n_per_cluster, fdim[bi]], minval = -0.5, maxval = 0.5, dtype = default_dtype)
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
                        z = tf.multiply(w, tf.reshape(z, [self.batch_size, coarse_cnt, n_per_cluster, fdim[bi], 1]))
                        z = tf.reduce_sum(z, axis = 3) # z <- [bs, coarse_cnt, n_per_cluster, pos_range]
                        z = z + b

                        # Linear transformation
                        # z = tf.multiply(t, tf.reshape(z, [self.batch_size, coarse_cnt, n_per_cluster, pos_range, 1]))
                        # z = tf.reduce_sum(z, axis = 3)

                        n = z

                    n_ref = n
                    # if maxLen[bi] is not None:
                        # n = norm_tun(n, maxLen[bi])
                        # n = maxLen[bi] * n
                        # n_ref = norm_tun(n_ref, maxLen[bi])

                    # regularizer to keep in local space
                    reg_curr = maxLen[bi]
                    if reg_curr == None:
                        reg_curr = 0.01
                    regularizer += reg_curr * tf.reduce_mean(tf.norm(n, axis = -1))

                    # Back to world space
                    n = n + tf.reshape(coarse_pos, [self.batch_size, coarse_cnt, 1, pos_range])
                    n_ref = n_ref + tf.reshape(coarse_pos_ref, [self.batch_size, coarse_cnt, 1, pos_range])

                    ap = tf.reshape(n, [self.batch_size, pcnt[bi], pos_range])
                    ap_ref = tf.reshape(n_ref, [self.batch_size, pcnt[bi], pos_range])
                    nf = tf.reshape(nf, [self.batch_size, pcnt[bi], -1])

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

            # regularizer = regularizer / blocks
            # if monotonic:
            #     jacobian = tf.stack(tf.gradients(pos, uniform_dist, name = 'monoConstrint'))
            #     mono_reg = 10.0 * tf.reduce_mean(tf.square(tf.nn.relu(-jacobian)))
            #     regularizer = mono_reg

            return 0, [final_particles, final_particles_ref, gen_only[0]], 0, regularizer, meta
    
    def simulator(self, pos, particles, name = 'Simluator', is_train = True, reuse = False):

        w_init = tf.random_normal_initializer(stddev = 0.005 * self.wdev)
        w_init_pref = tf.random_normal_initializer(stddev = 0.001 * self.wdev)
        b_init = tf.constant_initializer(value=0.0)

        with tf.variable_scope(name, reuse = reuse) as vs:
            
            _, gIdx, gEdg = kNNG_gen(pos, config['simulator']['knnk'], 3, name = 'simulator/ggen')
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

    def chamfer_metric(self, particles, particles_ref, groundtruth, pos_range, loss_func, EMD = False):
        
        if EMD == True:
            
            bs = groundtruth.shape[0]
            Np = particles.shape[1]
            Ng = groundtruth.shape[1]
            
            match = approx_match(groundtruth[:, :, 0:pos_range], particles_ref[:, :, 0:pos_range]) # [bs, Np, Ng]
            row_predicted = tf.reshape(  particles[:, :, 0:self.outDim], [bs, Np, 1, -1])
            col_groundtru = tf.reshape(groundtruth[:, :, 0:self.outDim], [bs, 1, Ng, -1])
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

    def custom_dtype_getter(self, getter, name, shape=None, dtype=default_dtype, *args, **kwargs):
        
        if dtype is tf.float16:
            
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        
        else:
            
            return getter(name, shape, dtype, *args, **kwargs)

    def build_network(self, is_train, reuse, loopSim = True, includeSim = True):

        normalized_X = (self.ph_X[:, :, 0:self.outDim] - tf.broadcast_to(self.normalize['mean'], [self.batch_size, self.gridMaxSize, self.outDim])) / tf.broadcast_to(self.normalize['std'], [self.batch_size, self.gridMaxSize, self.outDim])
        # normalized_X = self.ph_X / self.normalize

        pos = []
        fea = []
        rec = []
        vls = []
        loss = []
        simLoss = 0
        
        # Mixed FP16 & FP32
        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):

            # Go through the particle AE
            posX, feaX, _v, _floss, eX, esamp = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse)
            outDim = self.outDim
            _, [rec_X, rec_X_ref, fold_X], _, r, meta = self.particleDecoder(posX, feaX, self.ph_card, outDim, is_train = is_train, reuse = reuse)
            
            AE_REC_loss = self.chamfer_metric(rec_X, rec_X, normalized_X[:, :, 0:outDim], 3, self.loss_func, EMD = True)

            loss = 0

            if includeSim == True:
                
                normalized_Y = (self.ph_Y[:, :, 0:self.outDim] - tf.broadcast_to(self.normalize['mean'], [self.batch_size, self.gridMaxSize, self.outDim])) / tf.broadcast_to(self.normalize['std'], [self.batch_size, self.gridMaxSize, self.outDim])
                # normalized_Y = self.ph_Y / self.normalize
                posY, feaY, _v, _floss, eY, esamp = self.particleEncoder(normalized_Y, self.particle_latent_dim, is_train = is_train, reuse = True)

                # X => Y
                psimY, fsimY = self.simulator(posX, feaX, name = 'Simulator', is_train = is_train, reuse = reuse)

                # Y => X
                psimX, fsimX = self.simulator(posY, feaY, name = 'SimulatorInv', is_train = is_train, reuse = reuse)

                # Decoders
                _, [rec_sim_X, _, _], _, r, meta = self.particleDecoder(psimX, fsimX, self.ph_card, outDim, is_train = is_train, reuse = True)
                _, [rec_sim_Y, _, _], _, r, meta = self.particleDecoder(psimY, fsimY, self.ph_card, outDim, is_train = is_train, reuse = True)

                forward_AE_loss = self.chamfer_metric(rec_sim_X, rec_sim_X, normalized_X[:, :, 0:outDim], 3, self.loss_func, EMD = True)
                backwrd_AE_loss = self.chamfer_metric(rec_sim_Y, rec_sim_Y, normalized_Y[:, :, 0:outDim], 3, self.loss_func, EMD = True)
                backwrd_loss = self.chamfer_metric(tf.concat([psimX, fsimX], axis = -1), tf.concat([psimX, fsimX], axis = -1), tf.concat([posX, feaX], axis = -1), 3, self.loss_func, EMD = True)
                forward_loss = self.chamfer_metric(tf.concat([psimY, fsimY], axis = -1), tf.concat([psimY, fsimY], axis = -1), tf.concat([posY, feaY], axis = -1), 3, self.loss_func, EMD = True)

                AE_REC_loss += forward_AE_loss + backwrd_AE_loss

                if loopSim == True:
                    # backwrd_loss = 0
                    # forward_loss = 0
                    pass

                loss += forward_loss + backwrd_loss
                simLoss = forward_loss + backwrd_loss

                if loopSim == True:
                    normalized_L = (self.ph_L[:, :, 0:self.outDim] - tf.broadcast_to(self.normalize['mean'], [self.batch_size, self.gridMaxSize, self.outDim])) / tf.broadcast_to(self.normalize['std'], [self.batch_size, self.gridMaxSize, self.outDim])
                    # normalized_L = self.ph_Y / self.normalize
                    posL, feaL, _v, _floss, eY, esamp = self.particleEncoder(normalized_L, self.particle_latent_dim, is_train = is_train, reuse = True)

                    # X => L
                    plsimL = posX
                    flsimL = feaX
                    for li in range(self.loops):
                        plsimL, flsimL = self.simulator(plsimL, flsimL, name = 'Simulator', is_train = is_train, reuse = True)

                    # L => X
                    plsimX = posL
                    flsimX = feaX
                    for li in range(self.loops):
                        plsimX, flsimX = self.simulator(plsimX, flsimX, name = 'SimulatorInv', is_train = is_train, reuse = True)

                    # Decoders
                    _, [rec_lsim_X, _, _], _, r, meta = self.particleDecoder(plsimX, flsimX, self.ph_card, outDim, is_train = is_train, reuse = True)
                    _, [rec_lsim_L, _, _], _, r, meta = self.particleDecoder(plsimL, flsimL, self.ph_card, outDim, is_train = is_train, reuse = True)

                    backwrd_l_loss = self.chamfer_metric(tf.concat([plsimX, flsimX], axis = -1), tf.concat([plsimX, flsimX], axis = -1), tf.concat([posX, feaX], axis = -1), 3, self.loss_func, EMD = True)
                    forward_l_loss = self.chamfer_metric(tf.concat([plsimL, flsimL], axis = -1), tf.concat([plsimL, flsimL], axis = -1), tf.concat([posL, feaL], axis = -1), 3, self.loss_func, EMD = True)
                    forward_AE_l_loss = self.chamfer_metric(rec_lsim_X, rec_lsim_X, normalized_X[:, :, 0:outDim], 3, self.loss_func, EMD = True)
                    backwrd_AE_l_loss = self.chamfer_metric(rec_lsim_L, rec_lsim_L, normalized_L[:, :, 0:outDim], 3, self.loss_func, EMD = True)
            
                    AE_REC_loss += forward_AE_l_loss + backwrd_AE_l_loss
                    loss += forward_l_loss + backwrd_l_loss
                    lsimLoss = forward_l_loss + backwrd_l_loss

            loss += AE_REC_loss
            
            if is_train == True:
                rec = rec_X
                vls = []
                pos = posX
                fea = feaX
            else:
                rec = rec_X
                loss = self.chamfer_metric(rec_X, rec_X, normalized_X[:, :, 0:outDim], 3, tf.square, EMD = True) # Keep use L2 for validation loss.
                vls = []
                pos = posX
                fea = feaX

        return rec, normalized_X[:, :, 0:outDim], loss, vls, meta, esamp, simLoss, lsimLoss

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
            posX, feaX, _v, pPos, _floss, evals = self.particleEncoder(normalized_X, self.particle_latent_dim, is_train = is_train, reuse = reuse)
            var_list.append(_v)
            # floss += _floss

        return posX, feaX, pPos, evals
    
    # Only simulates posX & feaX for a single step
    def build_predict_Sim(self, pos, fea, is_train = False, reuse = False):

        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
            
            sim_posY, sim_feaY = self.simulator(pos, fea, 'Simulator', is_train, reuse)
        
        return sim_posY, sim_feaY

    # Decodes Y
    def build_predict_Dec(self, pos, fea, gt, is_train = False, reuse = False, outDim = 6):

        with tf.variable_scope('net', custom_getter = self.custom_dtype_getter):
            
            _, [rec, _, rec_f], _, _, _ = self.particleDecoder(pos, fea, self.ph_card, outDim, is_train = is_train, reuse = reuse)

        rec = rec
        reconstruct_loss = self.chamfer_metric(rec, rec, gt, 3, tf.square, EMD = True)

        return rec, rec_f, reconstruct_loss

    def build_model(self):

        # Train & Validation
        _, _,\
        self.train_particleLoss, self.particle_vars, _, _, self.train_simLoss, self.train_lsimLoss =\
            self.build_network(True, False, self.doLoop, self.doSim)

        self.val_rec, self.val_gt,\
        self.val_particleLoss, _, self.particle_meta, self.edge_sample, self.val_simLoss , self.val_lsimLoss=\
            self.build_network(False, True, self.doLoop, self.doSim)

        self.train_ops = []
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            gvs = self.optimizer.compute_gradients(self.train_particleLoss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.) if grad is not None else None, var) for grad, var in gvs]
            self.train_op = self.optimizer.apply_gradients(capped_gvs)
