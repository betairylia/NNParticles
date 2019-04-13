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

from model import *
import dataLoad
from Kuhn_Munkres import KM

from tensorlayer.prepro import *
from tensorlayer.layers import *
from termcolor import colored, cprint

from time import gmtime, strftime

parser = argparse.ArgumentParser(description="Run the NN for particle simulation")

parser.add_argument('datapath')
parser.add_argument('outputpath')
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-ep', '--epochs', type = int, default = 200)
parser.add_argument('-ss', '--start-step', type = int, default = 200)
parser.add_argument('-bs', '--batch-size', type = int, default = 64)
parser.add_argument('-ds', '--delta-step', type = int, default = 1, help = "How many steps ds will the network predict, step i -> step (i+ds), 0 for identity test")
parser.add_argument('-vm', '--velocity-multiplier', type = float, default = 1.0, help = "Multiplies the velocity by this factor")
parser.add_argument('-maxl', '--particle-max-length', type = int, default = 200, help = "Max particles in a single grid")
parser.add_argument('-gu', '--grid-units', type = int, default = 256, help = "Latent tensor size for each grid")
parser.add_argument('-ehu', '--encoder-hidden-units', type = int, default = 256, help = "Hidden state vec length for encoder RNN")
parser.add_argument('-ecd', '--encoder-cell-depth', type = int, default = 1, help = "Encoder LSTM Cell stacks (depth)")
parser.add_argument('-dhu', '--decoder-hidden-units', type = int, default = 256, help = "Hidden state vec length for decoder RNN")
parser.add_argument('-dcd', '--decoder-cell-depth', type = int, default = 1, help = "Decoder LSTM Cell stacks (depth)")
parser.add_argument('-name', '--name', type = str, default = "NoName", help = "Name to show on tensor board")
parser.add_argument('-save', '--save', type = str, default = "None", help = "Path to store trained model")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")

args = parser.parse_args()

# Hyper parameters
n_epochs = args.epochs                          # particle data size
n_batch_size = args.batch_size                  # batch size
n_inputs = 6                                    # particle data size
n_enc_hidden_units = args.encoder_hidden_units  # encoder hidden state tensor length
n_dec_hidden_units = args.decoder_hidden_units  # decoder hidden state tensor length
n_enc_stacks = args.encoder_cell_depth          # encoder cell depth
n_dec_stacks = args.decoder_cell_depth          # decoder cell depth
n_grid_units = args.grid_units                  # Grid latent units
n_grid_length = args.particle_max_length        # Max particles in a grid
dataLoad.maxParticlesPerGrid = args.particle_max_length

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

# Log dir
logPath = "logs/" + args.name + "(" + strftime("%Y-%m-%d %H-%Mm-%Ss", gmtime()) + ")/"
os.makedirs(logPath)

def train():
    
    placeHolder_X = tf.placeholder('float32', [n_batch_size, 27, n_grid_length + 2, 9])
    placeHolder_X_size = tf.placeholder('float32', [n_batch_size, 27])
    placeHolder_Y = tf.placeholder('float32', [n_batch_size, n_grid_length + 2, 9])
    placeHolder_Y_size = tf.placeholder('float32', [n_batch_size])

    placeHolder_max_length = tf.placeholder('int32', [2])

    enc_cell = encoder_rnn_cell(n_enc_hidden_units, n_enc_stacks)
    enc_zero_state = enc_cell.zero_state(n_batch_size, dtype = tf.float32)

    # Testing
    # Collect latent vector
    test_latent_grids = []

    for i in range(27):
        with tf.variable_scope("Grid_encoder_RNN", reuse = tf.AUTO_REUSE):
            _, latent_tmp = tf.nn.dynamic_rnn(
                cell = enc_cell, 
                dtype = tf.float32,
                sequence_length = placeHolder_X_size[:, i],
                inputs = placeHolder_X[:, i, 1:, :], # crop out the "<Start>"
                initial_state = enc_zero_state)
        
        print(latent_tmp)
        
        test_latent_grids.append(grid_enc_model(latent_tmp[1], n_grid_units, is_train = False, reuse = tf.AUTO_REUSE).outputs)
    
    # Predict next step
    test_big_tensor = tf.concat(test_latent_grids, 1)

    print(test_big_tensor.shape)

    # Simulate using NN
    test_simulated_train = simulate_model(test_big_tensor, is_train = False, reuse = False)

    # Decode grid tensor
    (test_grid_train_c, test_grid_train_h) = grid_dec_model(test_simulated_train.outputs, n_dec_hidden_units, is_train = False, reuse = False)

    placeHolder_test_input = tf.placeholder('float32', [None, None, 9])
    tc = test_grid_train_c.outputs
    th = test_grid_train_h.outputs

    print(tc.shape)
    print(th.shape)

    dec_cell = decoder_rnn_cell(n_dec_hidden_units, n_dec_stacks)

    # Decode particle data
    with tf.variable_scope("Grid_decoder_RNN", reuse = tf.AUTO_REUSE):
        test_outputs, test_state = tf.nn.dynamic_rnn(
            cell = dec_cell,
            dtype = tf.float32,
            inputs = placeHolder_test_input,
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(tc, th))

    # Create session
    sess = tf.Session()

    sess.run(tf.local_variables_initializer())
    tl.layers.initialize_global_variables(sess)

    saver = tf.train.Saver()

    if args.load != "None":
        saver.restore(sess, args.load)

    epoch_idx = args.start_step - 1
    iteration = 0

    maxl_array = np.zeros((2))
    maxl_array[0] = n_grid_length
    maxl_array[1] = n_grid_length

    for epoch, content in dataLoad.gen_epochs_predict(args.datapath, args.start_step, n_batch_size, args.delta_step, args.velocity_multiplier):

        batch_idx = 0
        for _x, _x_size, _y, _y_size in epoch:

            # Initial data
            initial_RNN_input = np.zeros((n_batch_size, 1, 9))
            for i in range(n_batch_size):
                initial_RNN_input[i, 0, :] = [0, 0, 0, 0, 0, 0, 1, -1, -1]

            # First step of RNN
            feed_dict = { placeHolder_X: _x, placeHolder_X_size: _x_size, placeHolder_test_input: initial_RNN_input, placeHolder_max_length: maxl_array }            
            rnnOutput = np.zeros((n_batch_size, n_grid_length, 9))
            _rnnOutput, sc, sh = sess.run([test_outputs, tc, th], feed_dict)
            rnnOutput[:, 0, :] = _rnnOutput[:, 0, :]

            for i in range(1, n_grid_length):
                feed_dict = { placeHolder_test_input: rnnOutput[:, i-1:i, :], tc: sc, th: sh }
                _rnnOutput, s = sess.run([test_outputs, test_state], feed_dict)
                sc = s[0]
                sh = s[1]
                rnnOutput[:, i, :] = _rnnOutput[:, 0, :]

            for i in range(n_batch_size):
                dataLoad.write_content(content, epoch_idx + args.delta_step, batch_idx * n_batch_size + i, rnnOutput[i, :(int(_y_size[i]) - 1), 0:6], int(_y_size[i]) - 1)

            print(colored("Grid %04d / %04d" % (batch_idx * n_batch_size, content['gridCount']), 'green'))
            batch_idx += 1
        epoch_idx += args.delta_step

        print(colored("Step %04d / %04d" % (epoch_idx, content['stepCount']), 'yellow'))

        # if epoch_idx > (args.start_step + 10 * args.delta_step):
        #     break

    print(colored("Saving File to %s !" % (args.outputpath), 'magenta'))
    dataLoad.save_file(content, args.outputpath, args.delta_step * args.velocity_multiplier)

train()
