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
from periodic_conv import *

from Kuhn_Munkres_modified import KM

from time import gmtime, strftime

class model_dual:

    def __init__(self, size, gridMaxSize, latent_dim, batch_size, optimizer):
        
        # Size of each grid
        self.size = size
        self.gridMaxSize = gridMaxSize
        self.particle_latent_dim = 16
        self.latent_dim = latent_dim
        self.combine_method = tf.reduce_sum
        self.loss_func = tf.abs
        self.resSize = 1
        self.batch_size = batch_size
        self.bn_decay = 0.9995

        # self.act = (lambda x: 0.8518565165255 * tf.exp(-2 * tf.pow(x, 2)) - 1) # normalization constant c = (sqrt(2)*pi^(3/2)) / 3, 0.8518565165255 = c * sqrt(5).
        self.act = tf.nn.elu
        self.convact = tf.nn.elu
        # self.act = tf.nn.relu

        self.norm_method = 'layerNorm'

        self.wdev=0.1
        self.onorm_lambda = 0.0

        self.initial_grid_size = 6.0
        self.total_world_size = 96.0

        self.latent_simulate_steps = 1

        self.train_particle_net_on_main_phase = False

        self.ph_X = tf.placeholder('float32', [batch_size, size, 7]) # x y z vx vy vz
        self.ph_Y = tf.placeholder('float32', [batch_size, size, 7]) # x y z vx vy vz
        self.ph_Y_progress = tf.placeholder('float32', [batch_size]) # -1.0 ~ 1.0
        # self.ph_Ycard = tf.placeholder('float32', [batch_size])

        self.ph_voxels = tf.placeholder('float32', [batch_size * 8, gridMaxSize, 7])
        self.ph_voxels_card = tf.placeholder('float32', [batch_size * 8])
        self.ph_voxels_pos = tf.placeholder('int32', [batch_size * 8, 4])
        self.ph_max_length = tf.placeholder('int32', [2])

        self.optimizer = optimizer

    # 1 of a batch goes in this function at once.
    def particleNetwork(self, input_particle, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleNet", reuse = reuse) as vs:

            gridCount = int(self.total_world_size // self.initial_grid_size)

            # Assume particle array ranked 2 and entries 0, 1, 2 contains x, y, z coordinates.
            particle_grid =\
                (tf.floordiv(input_particle[:, 0], self.initial_grid_size) + gridCount // 2) * (gridCount ** 2) +\
                (tf.floordiv(input_particle[:, 1], self.initial_grid_size) + gridCount // 2) *  gridCount +\
                (tf.floordiv(input_particle[:, 2], self.initial_grid_size) + gridCount // 2)
            particle_grid = tf.dtypes.cast(particle_grid, tf.int32)

            normalized_particle = tf.concat([tf.mod(input_particle[:, 0:3], self.initial_grid_size) - (self.initial_grid_size / 2), input_particle[:, 3:]], axis = 1)
            
            n = InputLayer(normalized_particle, name = 'input')

            n = DenseLayer(n, n_units = 128, act = self.act, name = 'fc1', W_init = w_init, b_init = b_init)
            n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc2', W_init = w_init, b_init = b_init)

            # X-Y-Z-C
            return  tf.reshape(tf.unsorted_segment_sum(n.outputs, particle_grid, gridCount ** 3, name = 'segSum'), [gridCount, gridCount, gridCount, output_dim], name = 'reshape')
                    # tf.reshape(tf.unsorted_segment_sum(tf.ones([self.batch_size, self.size]), particle_grid, gridCount ** 3, name = 'gtCard'), [gridCount, gridCount, gridCount], name = 'gtCard_reshape')
    
    def particleDecoder_naive(self, batch_size, input_latent, groundTruth_card, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleDecoder", reuse = reuse) as vs:

            n = InputLayer(input_latent, name = 'input')

            n = DenseLayer(n, n_units = 128, act = self.act, name = 'fc1', W_init = w_init, b_init = b_init)
            alter_particles = DenseLayer(n, n_units = output_dim * self.gridMaxSize, name = 'alter/fc', W_init = w_init, b_init = b_init)
            alter_particles = tf.reshape(alter_particles.outputs, [batch_size, self.gridMaxSize, output_dim])

            card = DenseLayer(n, n_units = 1, name = 'card/out', W_init = w_init, b_init = b_init)
            card = tf.reshape(card.outputs, [batch_size])

            if is_train == True:
                card_used = groundTruth_card
            else:
                card_used = tf.floor(card)
                card_used = tf.minimum(card_used, self.gridMaxSize)
                card_used = tf.maximum(card_used, 0)

            card_mask, card_match = tf.py_func(self.generate_match, [card_used], [tf.float32, tf.int32], name='card_mask')
            
            # final_outputs = tf.multiply(alter_particles, card_mask, name = 'masked_output')
            masked_output = tf.multiply(alter_particles, card_mask, name = 'masked_output')
            final_outputs = tf.gather_nd(masked_output, card_match, name = 'final_outputs')
            # final_outputs = tf.reshape(final_outputs, [batch_size, self.gridMaxSize, output_dim], name = 'reshaped_final_outputs')

            return card, final_outputs, card_used

    def particleDecoder_semi_advanced(self, batch_size, input_latent, groundTruth_card, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleDecoder", reuse = reuse) as vs:

            n = InputLayer(input_latent, name = 'input')

            # n = DenseLayer(n, n_units = 128, act = self.act, name = 'fc1', W_init = w_init, b_init = b_init)
            alter_particles = DenseLayer(n, act = self.act, n_units = 256, name = 'particles/fc1', W_init = w_init, b_init = b_init)
            alter_particles = DenseLayer(alter_particles, n_units = output_dim * self.gridMaxSize, name = 'particles/out', W_init = w_init, b_init = b_init)
            alter_particles = tf.reshape(alter_particles.outputs, [batch_size, self.gridMaxSize, output_dim])

            score_logit = DenseLayer(n, act = self.act, n_units = 512, name = 'score/fc', W_init = w_init, b_init = b_init)
            score_logit = DenseLayer(score_logit, n_units = self.gridMaxSize, name = 'score/out', W_init = w_init, b_init = b_init)

            card = DenseLayer(n, act = self.act, n_units = 128, name = 'card/fc', W_init = w_init, b_init = b_init)
            card = DenseLayer(card, n_units = 1, name = 'card/out', W_init = w_init, b_init = b_init)
            card = tf.reshape(card.outputs, [batch_size])

            return alter_particles, score_logit.outputs, card

    # TODO: complete it?
    def particleDecoder_advanced(self, batch_size, input_latent, groundTruth_card, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("ParticleDecoder", reuse = reuse) as vs:

            n = InputLayer(input_latent, name = 'input')

            # n = DenseLayer(n, n_units = 128, act = self.act, name = 'fc1', W_init = w_init, b_init = b_init)
            alter_particles = DenseLayer(n, n_units = output_dim * self.gridMaxSize, name = 'fcAlter', W_init = w_init, b_init = b_init)

            alter_particles = tf.reshape(alter_particles.outputs, [batch_size, self.gridMaxSize, output_dim])
            broadcast_input = tf.broadcast_to(input_latent, [batch_size, self.gridMaxSize, self.particle_latent_dim])
            score_input = tf.concat([alter_particles, broadcast_input], axis = 2)

            score = InputLayer(score_input, name = 'score_input')

            score = Conv1dLayer(score, act = None, shape = (1, output_dim + self.particle_latent_dim, 64), W_init = w_init, b_init = b_init, name = 'score/conv1')
            score = Conv1dLayer(score, act = None, shape = (1, 64, 1), W_init = w_init, b_init = b_init, name = 'score/conv2')

            card = DenseLayer(n, n_units = 1, name = 'cardOut', W_init = w_init, b_init = b_init)

            #TODO: softmax, select top N entries, mask-out & align to left.

    def Encoder(self, input_data, input_channels, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("Encoder", reuse = reuse) as vs:

            if self.norm_method == 'batchNorm':

                n = InputLayer(input_data, name = 'input')

                n = Conv3dLayer(n, shape = (3, 3, 3, input_channels, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'resin/conv1', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'resin/conv1/bn')

                temp = n

                for i in range(self.resSize):
                    nn = Conv3dLayer(n, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv1' % i, W_init = w_init, b_init = b_init)
                    nn = BatchNormLayer(nn, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'res%d/conv1/bn' % i)
                    nn = Conv3dLayer(n, shape = (1, 1, 1, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv2' % i, W_init = w_init, b_init = b_init)
                    nn = BatchNormLayer(nn, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'res%d/conv2/bn' % i)
                    nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                    n = nn
                
                n = Conv3dLayer(n, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'resout/conv1', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'resout/conv1/bn')
                n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

                n = Conv3dLayer(n, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 2, 2, 2, 1), name = 'conv1', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'conv1/bn')
                # reduced 1/2 (1/2)

                n = Conv3dLayer(n, shape = (3, 3, 3, output_dim // 2, output_dim), strides = (1, 1, 1, 1, 1), name = 'conv2', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'conv2/bn')

                n = Conv3dLayer(n, shape = (1, 1, 1, output_dim, output_dim), strides = (1, 1, 1, 1, 1), name = 'conv3', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'conv3/bn')

                # reduced 1/2 (1/2)
                return n.outputs

            if self.norm_method == 'layerNorm':

                n = InputLayer(input_data, name = 'input')

                n = PeriodicConv3dLayer(n, shape = (3, 3, 3, input_channels, output_dim // 2), strides = (1, 1, 1, 1, 1), padding = 1, name = 'resin/conv1', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'resin/conv1/ln')

                temp = n

                for i in range(self.resSize):
                    nn = PeriodicConv3dLayer(n, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), padding = 1, name = 'res%d/conv1' % i, W_init = w_init, b_init = b_init)
                    nn = LayerNormLayer(nn, act = self.convact, name = 'res%d/conv1/ln' % i)
                    nn = Conv3dLayer(n, shape = (1, 1, 1, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv2' % i, W_init = w_init, b_init = b_init)
                    nn = LayerNormLayer(nn, act = self.convact, name = 'res%d/conv2/ln' % i)
                    nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                    n = nn
                
                n = PeriodicConv3dLayer(n, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), padding = 1, name = 'resout/conv1', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'resout/conv1/ln')
                n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

                n = Conv3dLayer(n, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 2, 2, 2, 1), name = 'conv1', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'conv1/ln')
                # reduced 1/2 (1/2)

                n = PeriodicConv3dLayer(n, shape = (3, 3, 3, output_dim // 2, output_dim), strides = (1, 1, 1, 1, 1), padding = 1, name = 'conv2', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'conv2/ln')

                n = Conv3dLayer(n, shape = (1, 1, 1, output_dim, output_dim), strides = (1, 1, 1, 1, 1), name = 'conv3', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'conv3/ln')

                # reduced 1/2 (1/2)
                return n.outputs
            
            if self.norm_method == 'None':

                n = InputLayer(input_data, name = 'input')

                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, input_channels, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'resin/conv1', W_init = w_init, b_init = b_init)

                temp = n

                for i in range(self.resSize):
                    nn = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv1' % i, W_init = w_init, b_init = b_init)
                    nn = Conv3dLayer(n, act = self.convact, shape = (1, 1, 1, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv2' % i, W_init = w_init, b_init = b_init)
                    nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                    n = nn
                
                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 1, 1, 1, 1), name = 'resout/conv1', W_init = w_init, b_init = b_init)
                n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, output_dim // 2, output_dim // 2), strides = (1, 2, 2, 2, 1), name = 'conv1', W_init = w_init, b_init = b_init)
                # reduced 1/2 (1/2)

                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, output_dim // 2, output_dim), strides = (1, 1, 1, 1, 1), name = 'conv2', W_init = w_init, b_init = b_init)

                n = Conv3dLayer(n, act = self.convact, shape = (1, 1, 1, output_dim, output_dim), strides = (1, 1, 1, 1, 1), name = 'conv3', W_init = w_init, b_init = b_init)

                # reduced 1/2 (1/2)
                return n.outputs
    
    def Decoder(self, input_data, input_channels, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("Decoder", reuse = reuse) as vs:

            if self.norm_method == 'batchNorm':

                n = InputLayer(input_data, name = 'input')

                n = Conv3dLayer(n, shape = (1, 1, 1, input_channels, input_channels), strides = (1, 1, 1, 1, 1), name = 'conv1', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'conv1/bn')

                n = Conv3dLayer(n, shape = (3, 3, 3, input_channels, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'conv2', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'conv2/bn')

                n = DeConv3d(n, n_filter = input_channels // 2, filter_size = (3, 3, 3), strides = (2, 2, 2), name = 'convt3', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'convt3/bn')
                # 2x

                n = Conv3dLayer(n, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'resin/conv1', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'resin/conv1/bn')

                temp = n

                for i in range(self.resSize):
                    nn = Conv3dLayer(n, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv1' % i, W_init = w_init, b_init = b_init)
                    nn = BatchNormLayer(nn, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'res%d/conv1/bn' % i)
                    nn = Conv3dLayer(n, shape = (1, 1, 1, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv2' % i, W_init = w_init, b_init = b_init)
                    nn = BatchNormLayer(nn, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'res%d/conv2/bn' % i)
                    nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                    n = nn
                
                n = Conv3dLayer(n, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'resout/conv1', W_init = w_init, b_init = b_init)
                n = BatchNormLayer(n, act = self.convact, decay = self.bn_decay, is_train = is_train, gamma_init = g_init, name = 'resout/conv1/bn')
                n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, input_channels // 2, output_dim), strides = (1, 1, 1, 1, 1), name = 'outrefine', W_init = w_init, b_init = b_init)

                return n.outputs
            
            if self.norm_method == 'layerNorm':

                n = InputLayer(input_data, name = 'input')

                n = Conv3dLayer(n, shape = (1, 1, 1, input_channels, input_channels), strides = (1, 1, 1, 1, 1), name = 'conv1', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'conv1/ln')

                n = PeriodicConv3dLayer(n, shape = (3, 3, 3, input_channels, input_channels // 2), strides = (1, 1, 1, 1, 1), padding = 1, name = 'conv2', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'conv2/ln')

                n = DeConv3d(n, n_filter = input_channels // 2, filter_size = (3, 3, 3), strides = (2, 2, 2), name = 'convt3', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'convt3/ln')
                # 2x

                n = PeriodicConv3dLayer(n, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), padding = 1, name = 'resin/conv1', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'resin/conv1/ln')

                temp = n

                for i in range(self.resSize):
                    nn = PeriodicConv3dLayer(n, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), padding = 1, name = 'res%d/conv1' % i, W_init = w_init, b_init = b_init)
                    nn = LayerNormLayer(nn, act = self.convact, name = 'res%d/conv1/ln' % i)
                    nn = Conv3dLayer(nn,  shape = (1, 1, 1, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv2' % i, W_init = w_init, b_init = b_init)
                    nn = LayerNormLayer(nn, act = self.convact, name = 'res%d/conv2/ln' % i)
                    nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                    n = nn
                
                n = PeriodicConv3dLayer(n, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), padding = 1, name = 'resout/conv1', W_init = w_init, b_init = b_init)
                n = LayerNormLayer(n, act = self.convact, name = 'resout/conv1/ln')
                n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

                n = PeriodicConv3dLayer(n, shape = (3, 3, 3, input_channels // 2, output_dim), strides = (1, 1, 1, 1, 1), padding = 1, name = 'outrefine', W_init = w_init, b_init = b_init)

                return n.outputs
            
            if self.norm_method == 'None':

                n = InputLayer(input_data, name = 'input')

                n = Conv3dLayer(n, act = self.convact, shape = (1, 1, 1, input_channels, input_channels), strides = (1, 1, 1, 1, 1), name = 'conv1', W_init = w_init, b_init = b_init)

                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, input_channels, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'conv2', W_init = w_init, b_init = b_init)

                n = DeConv3d(n, act = self.convact, n_filter = input_channels // 2, filter_size = (3, 3, 3), strides = (2, 2, 2), name = 'convt3', W_init = w_init, b_init = b_init)
                # 2x

                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'resin/conv1', W_init = w_init, b_init = b_init)

                temp = n

                for i in range(self.resSize):
                    nn = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv1' % i, W_init = w_init, b_init = b_init)
                    nn = Conv3dLayer(n, act = self.convact, shape = (1, 1, 1, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'res%d/conv2' % i, W_init = w_init, b_init = b_init)
                    nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                    n = nn
                
                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, input_channels // 2, input_channels // 2), strides = (1, 1, 1, 1, 1), name = 'resout/conv1', W_init = w_init, b_init = b_init)
                n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

                n = Conv3dLayer(n, act = self.convact, shape = (3, 3, 3, input_channels // 2, output_dim), strides = (1, 1, 1, 1, 1), name = 'outrefine', W_init = w_init, b_init = b_init)

                return n.outputs
    
    def Simulator(self, input_map, output_dim, is_train = False, reuse = False):
        
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)

        with tf.variable_scope("Simulator", reuse = reuse) as vs:

            n = InputLayer(input_map, name = 'input')

            # Fast parallel fc layers
            n = Conv3dLayer(n, shape = (1, 1, 1, output_dim, output_dim), act = self.act, strides = (1, 1, 1, 1, 1), name = 'conv1', W_init = w_init, b_init = b_init)

            return n.outputs
    
    def SimulatorInverse(self, input_map, output_dim, is_train = False, reuse = False):
        
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)

        with tf.variable_scope("SimulatorInv", reuse = reuse) as vs:

            n = InputLayer(input_map, name = 'input')

            # Fast parallel fc layers
            n = Conv3dLayer(n, shape = (1, 1, 1, output_dim, output_dim), act = self.act, strides = (1, 1, 1, 1, 1), name = 'conv1', W_init = w_init, b_init = b_init)

            return n.outputs

    # def cardinalityNetwork(self, input_grids, is_train = False, reuse = False):
    #     # TODO

    # def outputNetwork(self, input_latent, output_dim, is_train = False, reuse = False):
    #     # TODO

    def generate_match(self, card):

        # card: [bs]
        batch_size = card.shape[0]

        mask = np.zeros((batch_size, self.gridMaxSize, 7), dtype = 'f')

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

        mask = np.zeros((batch_size, self.gridMaxSize, 7), dtype = 'f')

        for b in range(batch_size):
            for i in range(int(card[b])):
                mask[b, i, :] = 1
            # Uncomment for randomly choosing
            # np.random.shuffle(mask[b, :])

        return mask
    
    def generate_score_label(self, src, cards):

        result = np.zeros((8 * self.batch_size, self.gridMaxSize), dtype = 'f')

        for b in range(8 * self.batch_size):
            for p in range(self.gridMaxSize):
                if src[b, p] < cards[b]:
                    result[b, p] = 1 / cards[b]
        
        return result
    
    def generate_KM_match(self, src):

        result = np.zeros((8 * self.batch_size, self.gridMaxSize, 2), dtype = np.int32)

        for b in range(8 * self.batch_size):
            for p in range(self.gridMaxSize):
                result[b, src[b, p]] = np.asarray([b, p]) # KM match order reversed (ph_Y -> output => output -> ph_Y)
        
        return result

    def no_return_assign(self, ref, value):
        tf.assign(ref, value)
        return 0

    def build_network(self, is_train, reuse):

        # Collect 3D feature maps and encode to latent space
        feature_maps_X = []
        feature_maps_Y = []
        
        for b in range(self.batch_size):
            
            _x = self.particleNetwork(self.ph_X[b], self.particle_latent_dim, is_train = is_train, reuse = tf.AUTO_REUSE)
            _y = self.particleNetwork(self.ph_Y[b], self.particle_latent_dim, is_train = is_train, reuse = tf.AUTO_REUSE)

            feature_maps_X.append(_x)
            feature_maps_Y.append(_y)

        batch_feature_map_X = tf.stack(feature_maps_X)
        batch_feature_map_Y = tf.stack(feature_maps_Y)

        # Calculate mask
        _absX = tf.reduce_sum(tf.abs(batch_feature_map_X), axis = -1)
        _absY = tf.reduce_sum(tf.abs(batch_feature_map_Y), axis = -1)
        _nonZeroX = tf.not_equal(_absX, 0)
        _nonZeroY = tf.not_equal(_absY, 0)
        _preLossMask = tf.logical_or(_nonZeroX, _nonZeroY)
        _randomNum = tf.random.uniform([self.batch_size, 16, 16, 16])
        _randomMask = tf.greater(_randomNum, 0.9)
        _lossMask = tf.logical_or(_preLossMask, _randomMask)
        reconstruct_loss_mask = tf.reshape(tf.to_float(_lossMask), [self.batch_size, 16, 16, 16, 1])

        voxel_latents_X = self.Encoder(batch_feature_map_X, self.particle_latent_dim, self.latent_dim, is_train = is_train, reuse = reuse)
        voxel_latents_Y = self.Encoder(batch_feature_map_Y, self.particle_latent_dim, self.latent_dim, is_train = is_train, reuse = True)

        # Cycle latent reconstruction
        forward_prediction = self.Simulator(voxel_latents_X, self.latent_dim, is_train = is_train, reuse = reuse)
        backward_prediction = self.SimulatorInverse(voxel_latents_Y, self.latent_dim, is_train = is_train, reuse = reuse)

        # Decoding
        # decoded_feature_map_Y = self.Decoder( forward_prediction, self.latent_dim, self.particle_latent_dim, is_train = is_train, reuse = reuse)
        # decoded_feature_map_X = self.Decoder(backward_prediction, self.latent_dim, self.particle_latent_dim, is_train = is_train, reuse = True)
        decoded_feature_map_Y = self.Decoder(voxel_latents_Y, self.latent_dim, self.particle_latent_dim, is_train = is_train, reuse = reuse)
        decoded_feature_map_X = self.Decoder(voxel_latents_X, self.latent_dim, self.particle_latent_dim, is_train = is_train, reuse = True)

        # Particle network training stage
        particle_decoder_inputs = tf.gather_nd(batch_feature_map_X, self.ph_voxels_pos, name = 'pDecoderInput')

        #### Naive particle network ####

        pd_card, pd_particles, _ = self.particleDecoder_naive(8 * self.batch_size, particle_decoder_inputs, self.ph_voxels_card, 7, is_train = is_train, reuse = reuse)
        KM_matches = tf.py_func(KM, [pd_particles, self.ph_voxels, self.ph_voxels_card, self.ph_max_length], tf.int32, name = 'KM_matches')
        gather_KM_matches = tf.py_func(self.generate_KM_match, [KM_matches], tf.int32, name = 'gather_KM_matches')
        final_particles = tf.gather_nd(pd_particles, gather_KM_matches, 'final_outputs_after_KM')
        _score_loss = 0

        #### End of Naive particle network ####
        
        #### Advanced particle network ####

        # pd_particles, pd_score, pd_card = self.particleDecoder_semi_advanced(8 * self.batch_size, particle_decoder_inputs, self.ph_voxels_card, 7, is_train = is_train, reuse = reuse)

        # Gather final outputs (particles)
        # KM_matches = tf.py_func(KM, [pd_particles, self.ph_voxels, self.ph_voxels_card, self.ph_max_length], tf.int32, name = 'KM_matches')
        # gather_KM_matches = tf.py_func(self.generate_KM_match, [KM_matches], tf.int32, name = 'gather_KM_matches')
        # gathered_particles = tf.gather_nd(pd_particles, gather_KM_matches, 'final_outputs_after_KM')
        # canonical_mask = tf.py_func(self.generate_canonical_mask, [self.ph_voxels_card], tf.float32, name = 'canonical_mask')
        # final_particles = tf.multiply(gathered_particles, canonical_mask)
        # _score_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pd_score, labels = score_label))

        #### End of Advanced particle network ####

        # Generate labels for particle score
        score_label = tf.py_func(self.generate_score_label, [KM_matches, self.ph_voxels_card], tf.float32, name = 'score_label')

        # Inverse constraint
        interpolate_alpha = tf.random.normal(shape = (), mean = 0.5, stddev = 0.5)
        voxel_latents_interpolate = interpolate_alpha * voxel_latents_X + (1.0 - interpolate_alpha) * voxel_latents_Y

        voxel_latents_ssi = self.SimulatorInverse(self.Simulator(voxel_latents_interpolate, self.latent_dim, is_train = is_train, reuse = True), self.latent_dim, is_train = is_train, reuse = True)
        voxel_latents_sis = self.Simulator(self.SimulatorInverse(voxel_latents_interpolate, self.latent_dim, is_train = is_train, reuse = True), self.latent_dim, is_train = is_train, reuse = True)
        forward_backward_pred = self.SimulatorInverse(forward_prediction, self.latent_dim, is_train = is_train, reuse = True)
        backward_forward_pred = self.Simulator(backward_prediction, self.latent_dim, is_train = is_train, reuse = True)

        # Compute losses

        inverse_constraint =\
            tf.reduce_mean(self.loss_func(voxel_latents_ssi - voxel_latents_interpolate)) +\
            tf.reduce_mean(self.loss_func(voxel_latents_sis - voxel_latents_interpolate)) +\
            tf.reduce_mean(self.loss_func(forward_backward_pred - voxel_latents_X)) +\
            tf.reduce_mean(self.loss_func(backward_forward_pred - voxel_latents_Y))

        simulate_reconstruct_loss =\
            tf.reduce_mean(self.loss_func( forward_prediction - voxel_latents_Y)) +\
            tf.reduce_mean(self.loss_func(backward_prediction - voxel_latents_X))
        
        # with zero mask
        # reconstruct_loss =\
        #     tf.reduce_mean(tf.multiply(reconstruct_loss_mask, self.loss_func(decoded_feature_map_X - batch_feature_map_X))) +\
        #     tf.reduce_mean(tf.multiply(reconstruct_loss_mask, self.loss_func(decoded_feature_map_Y - batch_feature_map_Y)))
        
        # without zero mask
        reconstruct_loss =\
            tf.reduce_mean(self.loss_func(decoded_feature_map_X - batch_feature_map_X)) +\
            tf.reduce_mean(self.loss_func(decoded_feature_map_Y - batch_feature_map_Y))
        
        particle_network_loss =\
            tf.reduce_mean(self.loss_func(final_particles - self.ph_voxels))

        particle_score_loss =\
            _score_loss
        
        particle_card_loss =\
            tf.reduce_mean(self.loss_func(pd_card - self.ph_voxels_card))
        
        zero_loss =\
            tf.reduce_mean(self.loss_func(batch_feature_map_X))

        net_vars =\
            tl.layers.get_variables_with_name('Encoder', True, True) +\
            tl.layers.get_variables_with_name('Decoder', True, True) +\
            tl.layers.get_variables_with_name('Simulator', True, True) +\
            tl.layers.get_variables_with_name('SimulatorInv', True, True)
        
        particle_net_vars =\
            tl.layers.get_variables_with_name('ParticleNet', True, True) +\
            tl.layers.get_variables_with_name('ParticleDecoder', True, True)

        return\
            reconstruct_loss, simulate_reconstruct_loss,\
            particle_network_loss, particle_card_loss, particle_score_loss,\
            inverse_constraint, net_vars, particle_net_vars, zero_loss
    
    def predict_network(self):

        feature_maps_X = []
        
        # Only predict as batch_size = 1
        # Calculate decoded latent
        _x = self.particleNetwork(self.ph_X[0], self.particle_latent_dim, is_train = False, reuse = tf.AUTO_REUSE)
        feature_maps_X.append(_x)
        batch_feature_map_X = tf.stack(feature_maps_X)
        voxel_latents_X = self.Encoder(batch_feature_map_X, self.particle_latent_dim, self.latent_dim, is_train = False, reuse = tf.AUTO_REUSE)

        forward_prediction = voxel_latents_X
        for i in range(self.latent_simulate_steps):
            forward_prediction = self.Simulator(forward_prediction, self.latent_dim, is_train = False, reuse = tf.AUTO_REUSE)

        decoded_feature_map_Y = self.Decoder(forward_prediction, self.latent_dim, self.particle_latent_dim, is_train = False, reuse = tf.AUTO_REUSE)

        # Gather final forward-simulating results
        # flatten_inputs = tf.reshape(batch_feature_map_X, [16 * 16 * 16, self.particle_latent_dim]) 
        flatten_inputs = tf.reshape(decoded_feature_map_Y, [16 * 16 * 16, self.particle_latent_dim]) 
        _, flat_final_outputs, flat_cardinality = self.particleDecoder_naive(16 * 16 * 16, flatten_inputs, None, 7, is_train = False, reuse = tf.AUTO_REUSE)
        # finalOutputs - [4096, 32, 7]; cardinality - [4096]
        # final_outputs = tf.reshape(flat_final_outputs, [16, 16, 16, 7])
        # cardinality = tf.reshape(flat_cardinality, [16, 16, 16])

        return flat_cardinality, flat_final_outputs

    def build_prediction(self):

        # Network for prediction
        self.predict_cardinality, self.predict_outputs = self.predict_network()
    
    def build_model(self):

        # Train & Validation
        self.train_reconstructLoss, self.train_simLoss,\
        self.train_particleRawLoss, self.train_particleCardLoss, self.train_particleScoreLoss,\
        self.train_invLoss, self.net_vars, self.particle_vars, self.zero_loss =\
            self.build_network(True, False)

        self.val_reconstructLoss, self.val_simLoss,\
        self.val_particleRawLoss, self.val_particleCardLoss, self.val_particleScoreLoss,\
        self.val_invLoss, _, _, _ =\
            self.build_network(False, True)

        # self.train_particleLoss = self.train_particleCardLoss
        # self.val_particleLoss = self.val_particleCardLoss

        self.train_particleLoss = self.train_particleCardLoss + self.train_particleRawLoss + self.train_particleScoreLoss
        self.val_particleLoss = self.val_particleCardLoss + self.val_particleRawLoss + self.val_particleScoreLoss

        if(self.train_particle_net_on_main_phase):
            self.train_loss = self.train_reconstructLoss + self.train_simLoss + self.train_particleLoss + self.train_invLoss
            self.val_loss = self.val_reconstructLoss + self.val_simLoss + self.val_particleLoss + self.val_invLoss
        else:
            self.train_loss = self.train_reconstructLoss + self.train_simLoss
            self.val_loss = self.val_reconstructLoss + self.val_simLoss

            # self.train_loss = self.train_reconstructLoss + self.train_simLoss + self.train_invLoss
            # self.val_loss = self.val_reconstructLoss + self.val_simLoss + self.val_invLoss

        self.train_op = self.optimizer.minimize(self.train_loss, var_list = self.net_vars)
        # self.train_op = self.optimizer.minimize(self.train_loss, var_list = self.net_vars + self.particle_vars)
        self.particle_train_op = self.optimizer.minimize(self.train_particleLoss, var_list = self.particle_vars)

    def card_only_prediction(self, count):

        feature_maps_X = []

        _x = self.particleNetwork(self.ph_X[0], self.particle_latent_dim, is_train = False, reuse = tf.AUTO_REUSE)
        feature_maps_X.append(_x)
        batch_feature_map_X = tf.stack(feature_maps_X)

        results = []
        
        # initial step
        flatten_inputs = tf.reshape(batch_feature_map_X, [16 * 16 * 16, self.particle_latent_dim])
        _, _, flat_cardinality = self.particleDecoder_naive(16 * 16 * 16, flatten_inputs, None, 7, is_train = False, reuse = tf.AUTO_REUSE)
        # _, _, flat_cardinality = self.particleDecoder_semi_advanced(16 * 16 * 16, flatten_inputs, None, 7, is_train = False, reuse = tf.AUTO_REUSE)
        results.append(tf.reshape(flat_cardinality, [16, 16, 16], name = 'init_step') / 10.0)

        loss = 0

        # predicts
        for i in range(count):

            voxel_latents_X = self.Encoder(batch_feature_map_X, self.particle_latent_dim, self.latent_dim, is_train = False, reuse = tf.AUTO_REUSE)

            forward_prediction = voxel_latents_X
            for i in range(self.latent_simulate_steps):
                forward_prediction = self.Simulator(forward_prediction, self.latent_dim, is_train = False, reuse = tf.AUTO_REUSE)

            decoded_feature_map_Y = self.Decoder(forward_prediction, self.latent_dim, self.particle_latent_dim, is_train = False, reuse = tf.AUTO_REUSE)
            loss += tf.reduce_mean(self.loss_func(decoded_feature_map_Y - batch_feature_map_X))

            # cycle prediction
            batch_feature_map_X = decoded_feature_map_Y

            # Gather final forward-simulating results
            # flatten_inputs = tf.reshape(batch_feature_map_X, [16 * 16 * 16, self.particle_latent_dim]) 
            flatten_inputs = tf.reshape(decoded_feature_map_Y, [16 * 16 * 16, self.particle_latent_dim]) 
            _, _, flat_cardinality = self.particleDecoder_naive(16 * 16 * 16, flatten_inputs, None, 7, is_train = False, reuse = tf.AUTO_REUSE)
            results.append(tf.reshape(flat_cardinality, [16, 16, 16]) / 10.0)
        
        final_result = tf.stack(results)
        return final_result, loss / count
