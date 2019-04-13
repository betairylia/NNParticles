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

class model_set:

    def __init__(self, size, latent_dim, batch_size, optimizer):
        
        # Size of each grid
        self.size = size
        self.latent_dim = latent_dim
        self.combine_method = tf.reduce_sum
        self.loss_func = tf.abs
        self.resSize = 1
        self.subspace_div = 16
        self.batch_size = batch_size

        # self.act = (lambda x: 0.8518565165255 * tf.exp(-2 * tf.pow(x, 2)) - 1) # normalization constant c = (sqrt(2)*pi^(3/2)) / 3, 0.8518565165255 = c * sqrt(5).
        self.act = tf.nn.elu
        # self.act = tf.nn.relu

        self.wdev=0.1
        self.onorm_lambda = 0.0

        self.ph_X = tf.placeholder('float32', [batch_size, 27, size, 7]) # x y z xx(-1, 0, 1) yy(-1, 0, 1) zz(-1, 0, 1) is_particle(0, 1)
        self.ph_Y = tf.placeholder('float32', [batch_size, size, 3])
        self.ph_Ycard = tf.placeholder('float32', [batch_size])
        self.ph_max_length = tf.placeholder('int32', [2])

        self.optimizer = optimizer

    def particleNetwork(self, input_particle, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("particleNet", reuse = reuse) as vs:

            n = InputLayer(input_particle, name = 'input')

            # n = DenseLayer(n, n_units = output_dim // 2, act = tf.identity, name = 'fc1', W_init = w_init)
            # n = BatchNormLayer(n, act = self.act, is_train = is_train, gamma_init = g_init, name = 'fc1/bn')
            # n = DenseLayer(n, n_units = output_dim, act = tf.identity, name = 'fc2', W_init = w_init)
            # n = BatchNormLayer(n, act = self.act, is_train = is_train, gamma_init = g_init, name = 'fc1/bn')

            # n = DenseLayer(n, n_units = 16, act = self.act, name = 'fc1', W_init = w_init)

            # ResBlocks

            n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc1', W_init = w_init)
            n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc2', W_init = w_init)

            # temp = n

            # # Try to use 1D conv layers instead of fc.

            # for i in range(3):
            #     # nn = DenseLayer(n, n_units = output_dim, act = tf.identity, name = 'res%d/fc1' % i, W_init = w_init)
            #     nn = SubSpaceWrapper(n, n_units = output_dim, subDiv = 2 * self.subspace_div, subConnect = 1, act = tf.identity, name = 'res%d/ss1' % i, W_init = w_init)
            #     nn = BatchNormLayer(nn, act = self.act, is_train = is_train, gamma_init = g_init, name = 'res%d/fc1/bn' % i)
            #     nn = SubSpaceWrapper(n, n_units = output_dim, subDiv = 2 * self.subspace_div, subConnect = 3, act = tf.identity, name = 'res%d/ss2' % i, W_init = w_init)
            #     nn = BatchNormLayer(nn, act = self.act, is_train = is_train, gamma_init = g_init, name = 'res%d/fc2/bn' % i)
            #     nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
            #     n = nn

            # n = DenseLayer(n, n_units = output_dim, act = tf.identity, name = 'resout/fc1', W_init = w_init)
            # n = BatchNormLayer(n, act = self.act, is_train = is_train, gamma_init = g_init, name = 'resout/fc1/bn')
            # n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

            # n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc2', W_init = w_init)

            return n

    def particleNetwork_attention(self, input_particle, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        w_init_attention = tf.random_normal_initializer(stddev=2.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("particleNet", reuse = reuse) as vs:

            n = InputLayer(input_particle, name = 'input')

            # attention = DenseLayer(n, n_units = output_dim, act = tf.nn.sigmoid, name = 'fc1_attention', W_init = w_init)
            attention = DenseLayer(n, n_units = 32, act = self.act, name = 'fc1_attention', W_init = w_init_attention)
            attention = DenseLayer(attention, n_units = 32, act = self.act, name = 'fc2_attention', W_init = w_init_attention)
            # attention = DenseLayer(attention, n_units = 32, act = self.act, name = 'fc3_attention', W_init = w_init_attention)
            attention = DenseLayer(attention, n_units = output_dim, act = tf.nn.sigmoid, name = 'fc4_attention', W_init = w_init_attention)

            n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'fc1', W_init = w_init)
            n = ElementwiseLayer([n, attention], combine_fn = tf.multiply, name = 'attention_dot')

            return n
    
    def gridAttentionMask(self, gridID, gridCount, input_latent, latent_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)

        # Attention used only with gridID
        nobatch_grid_one_hot = tf.one_hot(gridID, gridCount)
        input_grid_one_hot = tf.reshape(nobatch_grid_one_hot, [1, gridCount])

        with tf.variable_scope("gridAttention", reuse = reuse) as vs:

            n = InputLayer(input_grid_one_hot, name = 'input')
            
            n = DenseLayer(n, n_units = latent_dim, act = tf.nn.sigmoid, name = 'fc1_gAttention', W_init = w_init)

            return tf.multiply(input_latent, n.outputs, name = 'attentionMul')
    
    def mainNetwork(self, input_latent, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("mainNet", reuse = reuse) as vs:

            n = InputLayer(input_latent, name = 'input')

            return n
            # n = DenseLayer(n, n_units = output_dim, act = tf.identity, name = 'fc1', W_init = w_init)
            # n = BatchNormLayer(n, act = self.act, is_train = is_train, gamma_init = g_init, name = 'fc1/bn')

            # ResBlocks

            temp = n

            # Try to use 1D conv layers instead of fc.

            for i in range(self.resSize):
                # nn = DenseLayer(n, n_units = output_dim, act = tf.identity, name = 'res%d/fc1' % i, W_init = w_init)
                nn = SubSpaceWrapper(n, n_units = output_dim, subDiv = self.subspace_div, subConnect = 1, act = tf.identity, name = 'res%d/ss1' % i, W_init = w_init)
                nn = BatchNormLayer(nn, act = self.act, is_train = is_train, gamma_init = g_init, name = 'res%d/fc1/bn' % i)
                nn = SubSpaceWrapper(n, n_units = output_dim, subDiv = self.subspace_div, subConnect = 3, act = tf.identity, name = 'res%d/ss2' % i, W_init = w_init)
                nn = BatchNormLayer(nn, act = self.act, is_train = is_train, gamma_init = g_init, name = 'res%d/fc2/bn' % i)
                nn = ElementwiseLayer([n, nn], tf.add, name = 'res%d/add' % i)
                n = nn

            n = DenseLayer(n, n_units = output_dim, act = self.act, name = 'resout/fc1', W_init = w_init)
            # n = DenseLayer(n, n_units = output_dim, act = tf.identity, name = 'resout/fc1', W_init = w_init)
            # n = BatchNormLayer(n, act = self.act, is_train = is_train, gamma_init = g_init, name = 'resout/fc1/bn')

            n = ElementwiseLayer([n, temp], tf.add, name = 'resout/add')

            return n
    
    def cardinalityNetwork(self, input_latent, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)

        with tf.variable_scope("cardinalityNet", reuse = reuse) as vs:

            n = InputLayer(input_latent, name = 'input')

            n = DenseLayer(n, n_units = 1, act = tf.identity, name = 'fc_out', W_init = w_init)
            # n = DenseLayer(n, n_units = 16, act = tf.identity, name = 'fc_out', W_init = w_init)

            print("Card_out shape:")
            print(n.outputs.shape)

            # card = tf.reduce_mean(tf.reshape(n.outputs, [self.batch_size]), axis = 1)
            card = tf.reshape(n.outputs, [self.batch_size])

            return n, card

    def outputNetwork(self, input_latent, output_dim, is_train = False, reuse = False):

        w_init = tf.random_normal_initializer(stddev=self.wdev)

        with tf.variable_scope("outputNet", reuse = reuse) as vs:

            n = InputLayer(input_latent, name = 'input')

            n = DenseLayer(n, n_units = output_dim, act = tf.identity, name = 'fc_out', W_init = w_init)

            return n

    def generate_match(self, card):

        # card: [bs]
        batch_size = card.shape[0]

        pre_mask = np.zeros((batch_size, self.size), dtype = 'f')
        mask = np.zeros((batch_size, self.size * 3), dtype = 'f')

        for b in range(batch_size):
            for i in range(int(card[b])):
                pre_mask[b, i] = 1
            np.random.shuffle(pre_mask[b, :])

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

        ### Collect latent vectors ###

        # Stack them in to one batch per grid
        self.grids = tf.unstack(self.ph_X, axis = 1)
        p_latent = []
        for g in range(27):

            current_grid = g
            # current_grid = 13
            
            # g_latent = []
            # for s in range(1):
            #     g_latent.append(self.particleNetwork(self.ph_X[:, g, s, :], self.latent_dim, is_train = is_train, reuse = tf.AUTO_REUSE).outputs)
            # # p_latent.append(self.combine_method(tf.stack(g_latent, 1), 1))
            # p_latent.append(g_latent[0])

            ph_X_stacked = tf.concat(tf.unstack(self.grids[current_grid], axis = 0), 0)

            print("Stacked X for %d" % current_grid)
            print(ph_X_stacked)

            # output particle latents [bs * size, latent_dim]
            p_latent_stacked = self.particleNetwork(ph_X_stacked, self.latent_dim, is_train = is_train, reuse = tf.AUTO_REUSE)
            # p_latent_stacked = self.particleNetwork_attention(ph_X_stacked, self.latent_dim, is_train = is_train, reuse = tf.AUTO_REUSE)
            
            # reduced latents [bs, latent_dim]
            raw_latents = tf.reshape(p_latent_stacked.outputs, [self.batch_size, self.size, self.latent_dim])
            masked_latents = tf.multiply(raw_latents, tf.reshape(self.ph_X[:, current_grid, :, 6], [self.batch_size, self.size, 1]))

            # Grid attention mask
            raw_grid_latent = self.combine_method(masked_latents, 1, name = 'grid_%d_latent' % current_grid)
            masked_grid_latent = self.gridAttentionMask(current_grid, 27, raw_grid_latent, self.latent_dim, is_train = is_train, reuse = tf.AUTO_REUSE)

            p_latent.append(masked_grid_latent)
        
        # Final reduced latents [bs, latent_dim]
        latent = self.combine_method(tf.stack(p_latent, 1), 1, name = 'latent')
        # latent = p_latent[0]

        print("latent")
        print(latent.shape)

        ### Feed the main network ###
        main_net = self.mainNetwork(latent, self.latent_dim, is_train = is_train, reuse = reuse)
        main_outputs = main_net.outputs

        print("main_outputs")
        print(main_outputs.shape)

        ### Feed the output networks ###
        cardinality_net, cardinality = self.cardinalityNetwork(main_outputs, is_train = is_train, reuse = reuse)

        # FIXME: uncomment
        # output_net = self.outputNetwork(main_outputs, self.size * 3, is_train = is_train, reuse = reuse)
        # output = output_net.outputs

        # print("output")
        # print(output.shape)
        # FIXME

        ### Collect results ###
        # FIXME: uncomment
        # clipped_cardinality = tf.floor(cardinality)
        # clipped_cardinality = tf.minimum(clipped_cardinality, self.size)
        # clipped_cardinality = tf.maximum(clipped_cardinality, 0)
        # FIXME: uncomment
        card_mask, card_match = tf.py_func(self.generate_match, [self.ph_Ycard], [tf.float32, tf.int32], name="card_mask")

        print("card")
        print(cardinality.shape)

        data_loss = 0
        avg_position_error = 0
        final_outputs = 0

        # with tf.variable_scope("outputs", reuse = reuse):
        #     final_outputs = tf.get_variable("final_outputs", [self.batch_size, self.size, 3], dtype = tf.float32,\
        #         collections=[tf.GraphKeys.LOCAL_VARIABLES], initializer=tf.constant_initializer(0.0), trainable=False)

        # # Assign particles to correct position (left-aligned with predicted cardinality)
        # for b in range(self.batch_size):
        #     for p in range(self.size):
        #         tf.assign(final_outputs[b, p, 0:3], tf.cond(match[b, p] > 0, lambda: output[b, (match[b, p] - 1) : (match[b, p] + 2)], lambda: tf.constant([0., 0., 0.])))
        
        # FIXME: uncomment
        # masked_output = tf.multiply(output, card_mask, name = 'masked_output')
        # final_outputs = tf.gather_nd(masked_output, card_match, name = 'final_outputs')
        # final_outputs = tf.reshape(final_outputs, [self.batch_size, self.size, 3], name = 'reshaped_final_outputs')
        # # final_outputs = tf.reshape(output, [self.batch_size, self.size, 3], name = 'reshaped_final_outputs')
        
        # ### Calculate Loss ###
        # KM_matches = tf.py_func(KM, [final_outputs, self.ph_Y, self.ph_Ycard, self.ph_max_length], tf.int32, name = 'KM_matches')
        # gather_KM_matches = tf.py_func(self.generate_KM_match, [KM_matches], tf.int32, name = 'gather_KM_matches')

        # data_loss = tf.reduce_mean(self.loss_func(tf.gather_nd(final_outputs, gather_KM_matches, 'final_outputs_after_KM') - self.ph_Y))
        # avg_position_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.gather_nd(final_outputs, gather_KM_matches, 'final_outputs_after_KM') - self.ph_Y), 2)))

        # for batch_idx in range(self.batch_size):
        #     for i in range(self.size):
        #         data_loss += tf.reduce_mean(self.loss_func((final_outputs[batch_idx, i] - self.ph_Y[batch_idx, KM_matches[batch_idx, i]])))
        #         avg_position_error += tf.sqrt(tf.reduce_sum(tf.square(final_outputs[batch_idx, i, 0:3] - self.ph_Y[batch_idx, KM_matches[batch_idx, i], 0:3])))

        # data_loss = data_loss
        # avg_position_error = avg_position_error
        # FIXME

        card_loss = tf.reduce_mean(tf.square(cardinality - self.ph_Ycard))

        trainable_weights = tl.layers.get_variables_with_name('W', True, True)
        print(trainable_weights)

        ortho_loss = 0
        onorm_count = 0
        for weight in trainable_weights:
            wshape = weight.shape
            if len(wshape) == 2 and wshape[0] <= wshape[1]:
                print(wshape[0])
                onorm_count += 1
                ortho_loss += tf.reduce_sum(tf.abs(tf.matmul(weight, tf.transpose(weight)) - tf.eye(int(wshape[0]))))
        ortho_loss /= onorm_count

        if is_train:
            self.mask = card_match

        net_vars = tl.layers.get_variables_with_name('particleNet', True, True) + tl.layers.get_variables_with_name('mainNet', True, True) + tl.layers.get_variables_with_name('cardinalityNet', True, True) +  tl.layers.get_variables_with_name('outputNet', True, True)

        return data_loss, card_loss, ortho_loss, avg_position_error, final_outputs, cardinality, latent, net_vars
    
    def build_model(self):

        self.train_data_loss, self.train_card_loss, self.train_ortho_loss, self.train_error, self.output, self.card, self.latent, self.trainable_vars = self.build_network(True, False)
        _, __, _____, self.val_error, self.val_output, self.val_card, ___, ____ = self.build_network(False, True)

        # L1 regularization
        # l1_regularizer = tf.contrib.layers.l1_regularizer(scale = 0.005, scope = None)
        # weights = tf.trainable_variables()
        # regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

        self.train_loss = self.train_card_loss + self.onorm_lambda * self.train_ortho_loss
        # self.train_loss = self.train_card_loss + regularization_penalty
        # self.train_loss = self.train_data_loss + 0.3 * self.train_card_loss
        self.train_card = tf.reduce_mean(self.card)
        self.val_loss = self.val_error

        self.train_op = self.optimizer.minimize(self.train_loss, var_list=self.trainable_vars)
