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

from Kuhn_Munkres import KM

from time import gmtime, strftime

class model_particleConv:

    def __init__(self, size, latent_dim, batch_size, optimizer):
        
        # Size of each grid
        self.size = size
        self.latent_dim = latent_dim
        self.combine_method = tf.reduce_sum
        self.loss_func = tf.abs
        self.resSize = 1
        self.batch_size = batch_size

        # self.act = (lambda x: 0.8518565165255 * tf.exp(-2 * tf.pow(x, 2)) - 1) # normalization constant c = (sqrt(2)*pi^(3/2)) / 3, 0.8518565165255 = c * sqrt(5).
        self.act = tf.nn.elu
        self.convact = tf.nn.elu
        # self.act = tf.nn.relu

        self.wdev=0.1
        self.onorm_lambda = 0.0

        self.initial_grid_size = 6.0
        self.total_world_size = 96.0

        self.ph_X = tf.placeholder('float32', [batch_size, size, 3]) # x y z
        self.ph_Y_progress = tf.placeholder('float32', [batch_size]) # -1.0 ~ 1.0
        self.ph_Y = tf.placeholder('float32', [batch_size, size, 3])
        # self.ph_Ycard = tf.placeholder('float32', [batch_size])
        # self.ph_max_length = tf.placeholder('int32', [2])

        self.optimizer = optimizer

    # 1 of a batch goes in this function at once.
    def particleNetwork(self, input_particle, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("particleNet", reuse = reuse) as vs:

            gridCount = int(self.total_world_size // self.initial_grid_size)

            # Assume particle array ranked 2 and entries 0, 1, 2 contains x, y, z coordinates.
            particle_grid =\
                (tf.floordiv(input_particle[:, 0], self.initial_grid_size) + gridCount // 2) * (gridCount ** 2) +\
                (tf.floordiv(input_particle[:, 1], self.initial_grid_size) + gridCount // 2) *  gridCount +\
                (tf.floordiv(input_particle[:, 2], self.initial_grid_size) + gridCount // 2)
            particle_grid = tf.dtypes.cast(particle_grid, tf.int32)

            normalized_particle = input_particle
            normalized_particle = tf.mod(normalized_particle, self.initial_grid_size) - (self.initial_grid_size / 2) # FIXME: If particle data contains not only position, modulo(normalize) entries for pos only.

            n = InputLayer(normalized_particle, name = 'input')

            n = DenseLayer(n, n_units = 128, act = self.act, name = 'fc1', W_init = w_init)
            n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc2', W_init = w_init)

            return tf.reshape(tf.unsorted_segment_sum(n.outputs, particle_grid, gridCount ** 3, name = 'segSum'), [gridCount, gridCount, gridCount, output_dim], name = 'reshape') # W-D-H-C
    
    def convNetwork(self, input_data, input_channels, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("convNet", reuse = reuse) as vs:

            n = InputLayer(input_data, name = 'input')

            n = Conv3dLayer(n, shape = (3, 3, 3, input_channels, 32), strides = (1, 2, 2, 2, 1), name = 'conv1', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = self.convact, is_train = is_train, gamma_init = g_init, name = 'conv1/bn')
            # reduced 1/2 (1/2)

            n = Conv3dLayer(n, shape = (3, 3, 3, 32, 64), strides = (1, 1, 1, 1, 1), name = 'conv2', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = self.convact, is_train = is_train, gamma_init = g_init, name = 'conv2/bn')

            r2 = n

            # n = Conv3dLayer(n, shape = (3, 3, 3, 64, 128), strides = (1, 2, 2, 2, 1), name = 'conv3', W_init = w_init, b_init = b_init)
            # n = BatchNormLayer(n, act = self.convact, is_train = is_train, gamma_init = g_init, name = 'conv3/bn')

            # r8 = n

            n = Conv3dLayer(n, shape = (3, 3, 3, 64, 128), strides = (1, 2, 2, 2, 1), name = 'conv4', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = self.convact, is_train = is_train, gamma_init = g_init, name = 'conv4/bn')
            # reduced 1/2 (1/4)

            r4 = n

            # return r4, r4, r2

            # self.resSize x ResBlocks

            temp = n

            for i in range(self.resSize):
                nn = Conv3dLayer(n, shape = (3, 3, 3, 128, 128), strides = (1, 1, 1, 1, 1), name = 'res%d/conv1' % i, W_init = w_init, b_init = b_init)
                nn = BatchNormLayer(nn, act = self.convact, is_train = is_train, gamma_init = g_init, name = 'res%d/conv1/bn' % i)
                nn = Conv3dLayer(nn, shape = (1, 1, 1, 128, 128), strides = (1, 1, 1, 1, 1), name = 'res%d/conv2' % i, W_init = w_init, b_init = b_init)
                nn = BatchNormLayer(nn, act = self.convact, is_train = is_train, gamma_init = g_init, name = 'res%d/conv2/bn' % i)
                nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                n = nn

            n = Conv3dLayer(n, shape = (3, 3, 3, 128, 128), strides = (1, 1, 1, 1, 1), name = 'resout/conv1', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = self.convact, is_train = is_train, gamma_init = g_init, name = 'resout/conv1/bn')
            n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

            n = Conv3dLayer(n, shape = (3, 3, 3, 128, 128), strides = (1, 2, 2, 2, 1), name = 'conv5', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = self.convact, is_train = is_train, gamma_init = g_init, name = 'conv5/bn')
            # reduced 1/2 (1/8)

            return n, r4, r2

    # def cardinalityNetwork(self, input_grids, is_train = False, reuse = False):
    #     # TODO

    # def outputNetwork(self, input_latent, output_dim, is_train = False, reuse = False):
    #     # TODO

    def progressPredictNetwork(self, input_3dfeature, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("progressPredictNet", reuse = reuse) as vs:

            n = InputLayer(input_3dfeature, name = 'input')

            n = FlattenLayer(n, name = 'flatten')

            n = DenseLayer(n, n_units = 256, act = self.act, name = 'fc1', W_init = w_init)
            n = DenseLayer(n, n_units = 1, act = tf.identity, name = 'fc_out', W_init = w_init)

            return tf.reshape(n.outputs, [self.batch_size])

    def generate_match(self, card):

        # card: [bs]
        batch_size = card.shape[0]

        pre_mask = np.zeros((batch_size, self.size), dtype = 'f')
        mask = np.zeros((batch_size, self.size * 3), dtype = 'f')

        for b in range(batch_size):
            for i in range(int(card[b])):
                pre_mask[b, i] = 1
            # np.random.shuffle(pre_mask[b, :])

        match = np.zeros((batch_size, self.size * 3, 2), dtype = np.int32)

        index = 0

        for b in range(batch_size):
            index = 0
            for i in range(self.size):
                if pre_mask[b, i] > 0.2: # randomly picked 0.2 (just same as == 1)
                    for p in range(3):
                        match[b, index * 3 + p, 0] = b
                        match[b, index * 3 + p, 1] = i * 3 + p
                        mask[b, i * 3 + p] = 1.0
                    index += 1
            for i in range(self.size):
                if pre_mask[b, i] < 0.2:
                    for p in range(3):
                        match[b, index * 3 + p, 0] = b
                        match[b, index * 3 + p, 1] = i * 3 + p
                        mask[b, i * 3 + p] = 0.0
                    index += 1
            
        return mask, match
    
    def generate_KM_match(self, src):

        result = np.zeros((self.batch_size, self.size, 2), dtype = np.int32)

        for b in range(self.batch_size):
            for p in range(self.size):
                result[b, src[b, p]] = np.asarray([b, p]) # KM match order reversed (ph_Y -> output => output -> ph_Y)
        
        return result

    def no_return_assign(self, ref, value):
        tf.assign(ref, value)
        return 0

    def build_network(self, is_train, reuse):

        ## Collect 3D feature maps ##
        feature_maps = []

        for b in range(self.batch_size):
            feature_maps.append(self.particleNetwork(self.ph_X[b], self.latent_dim, is_train = is_train, reuse = tf.AUTO_REUSE))
        
        batch_feature_map = tf.stack(feature_maps)

        r8, r4, r2 = self.convNetwork(batch_feature_map, self.latent_dim, is_train = is_train, reuse = reuse)

        progress_predicted = self.progressPredictNetwork(r8.outputs, is_train = is_train, reuse = reuse)

        progress_loss = tf.reduce_mean(tf.square(progress_predicted - self.ph_Y_progress))

        net_vars = tl.layers.get_variables_with_name('particleNet', True, True) + tl.layers.get_variables_with_name('convNet', True, True) + tl.layers.get_variables_with_name('progressPredictNet', True, True)

        return progress_loss, net_vars
    
    def build_model(self):

        self.train_pLoss, self.trainable_vars = self.build_network(True, False)
        self.val_pLoss, _ = self.build_network(False, True)

        self.train_loss = self.train_pLoss
        self.val_loss = self.val_pLoss

        self.train_op = self.optimizer.minimize(self.train_loss, var_list=self.trainable_vars)
