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

from time import gmtime, strftime

class model_conv3d:

    def __init__(self, size, latent_dim, input_channels, optimizer):
        
        self.size = size
        self.latent_dim = latent_dim
        self.input_channels = input_channels

        self.resSize = 12
        self.loss_func = tf.abs
        self.use_mass_loss = False

        self.ph_X = tf.placeholder('float32', [None, size, size, size, 1])
        self.ph_Y = tf.placeholder('float32', [None, size, size, size, 1])

        self.optimizer = optimizer

    def encoder(self, input_data, output_dim, input_channels, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("encoder", reuse = reuse) as vs:

            n = InputLayer(input_data, name = 'input')

            n = Conv3dLayer(n, shape = (3, 3, 3, input_channels, 32), strides = (1, 2, 2, 2, 1), name = 'conv1', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'conv1/bn')
            # 16 x 16 x 16

            n = Conv3dLayer(n, shape = (3, 3, 3, 32, 64), strides = (1, 1, 1, 1, 1), name = 'conv2', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'conv2/bn')
            # 16 x 16 x 16

            r16 = n

            n = Conv3dLayer(n, shape = (3, 3, 3, 64, 128), strides = (1, 2, 2, 2, 1), name = 'conv3', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'conv3/bn')
            # 8 x 8 x 8

            r8 = n

            n = Conv3dLayer(n, shape = (3, 3, 3, 128, 128), strides = (1, 2, 2, 2, 1), name = 'conv4', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'conv4/bn')
            # 4 x 4 x 4

            r4 = n

            # 6 x ResBlocks

            temp = n

            for i in range(self.resSize):
                nn = Conv3dLayer(n, shape = (3, 3, 3, 128, 128), strides = (1, 1, 1, 1, 1), name = 'res%d/conv1' % i, W_init = w_init, b_init = b_init)
                nn = BatchNormLayer(nn, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'res%d/conv1/bn' % i)
                nn = Conv3dLayer(nn, shape = (1, 1, 1, 128, 128), strides = (1, 1, 1, 1, 1), name = 'res%d/conv2' % i, W_init = w_init, b_init = b_init)
                nn = BatchNormLayer(nn, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'res%d/conv2/bn' % i)
                nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                n = nn

            n = Conv3dLayer(n, shape = (3, 3, 3, 128, 128), strides = (1, 1, 1, 1, 1), name = 'resout/conv1', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'resout/conv1/bn')
            n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

            n = Conv3dLayer(n, shape = (3, 3, 3, 128, 128), strides = (1, 2, 2, 2, 1), name = 'conv5', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'conv5/bn')
            # 2 x 2 x 2

            flatten = FlattenLayer(n, name = 'reshape')
            n = DenseLayer(flatten, n_units = output_dim, act = tf.identity, name = 'fc', W_init = w_init)

            return n, r4, r8, r16
    
    def decoder(self, input_data, output_dim, output_channels, u_net_skips, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        size = output_dim // 16

        # Make sure we are at the right size
        assert size * 16 == output_dim

        hasUNet = False

        if len(u_net_skips) > 0:
            hasUNet = True
            r4 = u_net_skips[0]
            r8 = u_net_skips[1]
            r16 = u_net_skips[2]
            r32 = InputLayer(u_net_skips[3], name = 'skipInput')

        with tf.variable_scope("decoder", reuse = reuse) as vs:

            n = InputLayer(input_data, name = 'input')

            n = DenseLayer(n, n_units = (size ** 3) * 128, act = tf.identity, name = 'fc', W_init = w_init)
            n = ReshapeLayer(n, shape = (-1, size, size, size, 128), name = 'reshape')
            # 2 x 2 x 2

            n = DeConv3d(n, n_filter = 128, filter_size = (3, 3, 3), strides = (2, 2, 2), name = 'convt1', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'convt1/bn')
            # 4 x 4 x 4

            if hasUNet:
                n = ConcatLayer([n, r4], 4, name = 'concat/r4')

            n = DeConv3d(n, n_filter = 128, filter_size = (3, 3, 3), strides = (2, 2, 2), name = 'convt2', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'convt2/bn')
            # 8 x 8 x 8

            if hasUNet:
                n = ConcatLayer([n, r8], 4, name = 'concat/r8')

            n = Conv3dLayer(n, shape = (3, 3, 3, 256, 128), strides = (1, 1, 1, 1, 1), name = 'conv3', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'conv3/bn')
            # 8 x 8 x 8

            n = DeConv3d(n, n_filter = 64, filter_size = (3, 3, 3), strides = (2, 2, 2), name = 'convt4', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'convt4/bn')
            # 16 x 16 x 16

            if hasUNet:
                n = ConcatLayer([n, r16], 4, name = 'concat/r16')

            n = DeConv3d(n, n_filter = 32, filter_size = (3, 3, 3), strides = (2, 2, 2), name = 'convt5', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.elu, is_train = is_train, gamma_init = g_init, name = 'convt5/bn')
            # 32 x 32 x 32

            if hasUNet:
                n = ConcatLayer([n, r32], 4, name = 'concat/r32')

            n = Conv3dLayer(n, shape = (3, 3, 3, 32 + self.input_channels, output_channels), strides = (1, 1, 1, 1, 1), name = 'conv6', W_init = w_init, b_init = b_init)
            n = BatchNormLayer(n, act = tf.nn.sigmoid, is_train = is_train, gamma_init = g_init, name = 'conv6/bn')
            # 32 x 32 x 32

            return n

    def build_model(self):

        # Training
        self.encoder_net_train, self.r4, self.r8, self.r16 = self.encoder(self.ph_X, self.latent_dim, self.input_channels, is_train = True, reuse = False)
        self.decoder_net_train = self.decoder(self.encoder_net_train.outputs, self.size, self.input_channels, [self.r4, self.r8, self.r16, self.ph_X], is_train = True, reuse = False)

        self.totalMass = tf.reduce_sum(self.decoder_net_train.outputs)
        self.groundTruthMass = tf.reduce_sum(self.ph_Y)
        self.massLoss = 0
        
        if self.use_mass_loss:
            self.massLoss = 0.0001 * tf.abs(self.totalMass - self.groundTruthMass)

        self.raw_loss = tf.reduce_mean(self.loss_func(self.decoder_net_train.outputs - self.ph_Y))
        self.loss = self.raw_loss + self.massLoss

        self.train_op = self.optimizer.minimize(self.loss)

        # Validation
        self.encoder_net_val, self.r4v, self.r8v, self.r16v = self.encoder(self.ph_X, self.latent_dim, self.input_channels, is_train = False, reuse = True)
        self.decoder_net_val = self.decoder(self.encoder_net_val.outputs, self.size, self.input_channels, [self.r4v, self.r8v, self.r16v, self.ph_X], is_train = False, reuse = True)

        # Use L1 in any configureation as a evaluation
        self.val_loss = tf.reduce_mean(tf.abs(self.decoder_net_val.outputs - self.ph_Y))
