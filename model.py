import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
import math

from tensorlayer.prepro import *
from tensorlayer.layers import *

# RNN Cell
def encoder_rnn_cell(n_hidden_units, stacks):

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = n_hidden_units, state_is_tuple = True)
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.7)

    return cell

def decoder_rnn_cell(n_hidden_units, stacks):

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = n_hidden_units, state_is_tuple = True)
    proj = tf.contrib.rnn.OutputProjectionWrapper(cell, 9)
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.7)

    return proj

# Grid Encoder network
def grid_enc_model(input_data, output_dim, is_train = False, reuse = False):

    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    # adv_act = activation = (lambda x: 0.8518565165255 * tf.exp(-2 * tf.pow(x, 2)) - 1) # normalization constant c = (sqrt(2)*pi^(3/2)) / 3, 0.8518565165255 = c * sqrt(5).

    with tf.variable_scope("grid_enc_network", reuse=reuse) as vs:
        
        n = InputLayer(input_data, name = 'input')
        
        # n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        # n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")

        n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")

        n = DenseLayer(n, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn3")
        # n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout3")

        # n = InputLayer(input_data, name = 'input')
        
        # n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn1")

        # n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn2")

        # n = DenseLayer(n, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn3")
        # n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout3")
    
    return n

# Grid Decoder network
def grid_dec_model(input_data, output_dim, is_train = False, reuse = False):

    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    # adv_act = activation = (lambda x: 0.8518565165255 * tf.exp(-2 * tf.pow(x, 2)) - 1) # normalization constant c = (sqrt(2)*pi^(3/2)) / 3, 0.8518565165255 = c * sqrt(5).

    with tf.variable_scope("grid_dec_network", reuse=reuse) as vs:
        
        n = InputLayer(input_data, name = 'input')
        
        # n = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        # n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")

        n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")

        c = DenseLayer(n, output_dim, act = tf.nn.tanh, W_init = w_init, b_init = b_init, name = "fc3_c")
        # c = DenseLayer(n, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3_c")
        # c = BatchNormLayer(c, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn3_c")

        h = DenseLayer(n, output_dim, act = tf.nn.tanh, W_init = w_init, b_init = b_init, name = "fc3_h")
        # h = DenseLayer(n, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3_h")
        # h = BatchNormLayer(h, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn3_h")
        # n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout3")

        # n = InputLayer(input_data, name = 'input')
        
        # n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn1")

        # n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn2")

        # n = DenseLayer(n, output_dim, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn3")
        # n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout3")
    
    return c, h

# Simulate network
def simulate_model(input_data, is_train = False, reuse = False):
    
    w_init = tf.random_normal_initializer(stddev=0.5)
    b_init = tf.random_normal_initializer(stddev=0.5)
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("simulate_network", reuse=reuse) as vs:
        
        n = InputLayer(input_data, name = 'input')
        
        n = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn1")

        n = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn2")
        # n = DropoutLayer(n, keep = 0.8, is_fix = True, is_train = is_train, name = "dropout2")

        res = n

        # for i in range(6):
        for i in range(1):
            nn = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "b%d/res_fc1" % i)
            nn = BatchNormLayer(nn, act = lrelu, is_train = is_train, gamma_init = g_init, name = "b%d/res_bn1" % i)
            nn = DenseLayer(nn, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "b%d/res_fc2" % i)
            nn = BatchNormLayer(nn, act = lrelu, is_train = is_train, gamma_init = g_init, name = "b%d/res_bn2" % i)
            nn = ElementwiseLayer([n, nn], tf.add, name = "b%d/res_add" % i)
            n = nn
        
        n = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "res_fc2")
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "res_bn2")
        n = ElementwiseLayer([n, res], tf.add, name = "res_add")

        n = DenseLayer(n, 256, act = tf.nn.tanh, W_init = w_init, b_init = b_init, name = "fc3")
        # n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        # n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = "bn3")
        # n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout3")

        # n = InputLayer(input_data, name = 'input')
        
        # n = DenseLayer(n, 512, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc1")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn1")

        # n = DenseLayer(n, 256, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc2")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn2")
        # n = DropoutLayer(n, keep = 0.8, is_fix = True, is_train = is_train, name = "dropout2")

        # n = DenseLayer(n, 128, act = tf.identity, W_init = w_init, b_init = b_init, name = "fc3")
        # n = BatchNormLayer(n, act = adv_act, is_train = is_train, gamma_init = g_init, name = "bn3")
        # n = DropoutLayer(n, keep = 0.7, is_fix = True, is_train = is_train, name = "dropout3")

    return n
