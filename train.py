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
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-ep', '--epochs', type = int, default = 200)
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
    placeHolder_iY = tf.placeholder('float32', [n_batch_size, n_grid_length + 2, 9])    
    placeHolder_Y_size = tf.placeholder('float32', [n_batch_size])

    placeHolder_max_length = tf.placeholder('int32', [2])

    enc_cell = encoder_rnn_cell(n_enc_hidden_units, n_enc_stacks)
    enc_zero_state = enc_cell.zero_state(n_batch_size, dtype = tf.float32)

    # Collect latent vectors
    latent_grids = []

    for i in range(27):
        with tf.variable_scope("Grid_encoder_RNN", reuse = tf.AUTO_REUSE):
            _, latent_tmp = tf.nn.dynamic_rnn(
                cell = enc_cell, 
                dtype = tf.float32,
                sequence_length = placeHolder_X_size[:, i],
                inputs = placeHolder_X[:, i, 1:, :], # crop out the "<Start>"
                initial_state = enc_zero_state)
        
        print(latent_tmp)
        
        latent_grids.append(grid_enc_model(latent_tmp[1], n_grid_units, is_train = True, reuse = tf.AUTO_REUSE).outputs)
    
    # Predict next step
    big_tensor = tf.concat(latent_grids, 1)

    print(big_tensor.shape)

    # Simulate using NN
    simulated_train = simulate_model(big_tensor, is_train = True, reuse = False)

    # Decode grid tensor
    (grid_train_c, grid_train_h) = grid_dec_model(simulated_train.outputs, n_dec_hidden_units, is_train = True, reuse = False)

    # Decode particle data
    dec_cell = decoder_rnn_cell(n_dec_hidden_units, n_dec_stacks)
    dec_zero_state = dec_cell.zero_state(n_batch_size, dtype = tf.float32)

    with tf.variable_scope("Grid_decoder_RNN", reuse = tf.AUTO_REUSE):
        outputs, state = tf.nn.dynamic_rnn(
            cell = dec_cell,
            dtype = tf.float32,
            sequence_length = placeHolder_Y_size,
            inputs = placeHolder_iY[:, :-1, :],
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(grid_train_c.outputs, grid_train_h.outputs))

    raw_loss = tf.reduce_sum(tf.abs(outputs - placeHolder_Y[:, 1:, :])) # Cut the <Start> tag away for ground truth

    ground_truth_seq = placeHolder_Y[:, 1:-1, :] # Cut <Start> and <End>
    predict_seq = outputs[:, :-1, :] # Cut <End>
    seq_length = tf.subtract(placeHolder_Y_size, 1) # Get cut seq length

    matches = tf.py_func(KM, [predict_seq, ground_truth_seq, seq_length, placeHolder_max_length], tf.int32)

    loss = 0

    avg_position_error = 0
    avg_velocity_error = 0

    for batch_idx in range(n_batch_size):
        for i in range(n_grid_length):
            loss += tf.reduce_sum(tf.square((predict_seq[batch_idx, i] - ground_truth_seq[batch_idx, matches[batch_idx, i]]) * 2.5))
            avg_position_error += tf.sqrt(tf.reduce_sum(tf.square(predict_seq[batch_idx, i, 0:3] - ground_truth_seq[batch_idx, matches[batch_idx, i], 0:3])))
            avg_velocity_error += tf.sqrt(tf.reduce_sum(tf.square(predict_seq[batch_idx, i, 3:6] - ground_truth_seq[batch_idx, matches[batch_idx, i], 3:6])))

    total_particles = tf.reduce_sum(placeHolder_Y_size)
    avg_position_error = avg_position_error / total_particles
    avg_velocity_error = avg_velocity_error / total_particles

    tf.summary.scalar('Average Position error', avg_position_error)
    tf.summary.scalar('Average Velocity error', avg_velocity_error)

    tf.summary.scalar('KM_Real_Training_Loss', loss)
    tf.summary.scalar('Training_Loss', raw_loss)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

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
    test_simulated_train = simulate_model(big_tensor, is_train = False, reuse = True)

    # Decode grid tensor
    (test_grid_train_c, test_grid_train_h) = grid_dec_model(test_simulated_train.outputs, n_dec_hidden_units, is_train = False, reuse = True)

    # Decode particle data
    with tf.variable_scope("Grid_decoder_RNN", reuse = tf.AUTO_REUSE):
        outputs, state = tf.nn.dynamic_rnn(
            cell = dec_cell,
            dtype = tf.float32,
            sequence_length = placeHolder_Y_size,
            inputs = placeHolder_iY[:, :-1, :],
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(test_grid_train_c.outputs, test_grid_train_h.outputs))

    raw_test_loss = tf.reduce_sum(tf.abs(outputs - placeHolder_Y[:, 1:, :])) # Cut the <Start> tag away for ground truth

    ground_truth_seq = placeHolder_Y[:, 1:-1, :] # Cut <Start> and <End>
    predict_seq = outputs[:, :-1, :] # Cut <End>
    seq_length = tf.subtract(placeHolder_Y_size, 1) # Get cut seq length

    matches = tf.py_func(KM, [predict_seq, ground_truth_seq, seq_length, placeHolder_max_length], tf.int32)

    test_loss = 0
    for batch_idx in range(n_batch_size):
        for i in range(n_grid_length):
            test_loss += tf.reduce_sum(tf.square((predict_seq[batch_idx, i] - ground_truth_seq[batch_idx, matches[batch_idx, i]]) * 2.5))

    # tf.summary.scalar('KM_Real_Testing_Loss', test_loss)
    # tf.summary.scalar('Testing_Loss', raw_test_loss)

    merged = tf.summary.merge_all()

    # Create session
    sess = tf.Session()

    writer = tf.summary.FileWriter(logPath, sess.graph)

    sess.run(tf.local_variables_initializer())
    tl.layers.initialize_global_variables(sess)

    saver = tf.train.Saver()

    if args.load != "None":
        saver.restore(sess, args.load)

    epoch_idx = 0
    iteration = 0

    maxl_array = np.zeros((2))
    maxl_array[0] = n_grid_length
    maxl_array[1] = n_grid_length

    for epoch_train, epoch_test in dataLoad.gen_epochs(n_epochs, args.datapath, n_batch_size, args.delta_step, args.velocity_multiplier):
        
        training_loss = 0
        training_batches = 0
        testing_loss = 0
        testing_batches = 0

        # Train
        batch_idx = 0
        for _x, _x_size, _y, _y_size in epoch_train:

            # Initial data
            initial_RNN_input = np.zeros((n_batch_size, n_grid_length + 2, 9))
            for i in range(n_batch_size):
                initial_RNN_input[i, 0, :] = [0, 0, 0, 0, 0, 0, 1, -1, -1]

            # Train network
            feed_dict = { placeHolder_X: _x, placeHolder_X_size: _x_size, placeHolder_Y: _y, placeHolder_iY: initial_RNN_input, placeHolder_Y_size: _y_size, placeHolder_max_length: maxl_array }
            
            # feed_dict.update( net.all_drop )

            _, n_loss, summary = sess.run([train_op, loss, merged], feed_dict = feed_dict)
            
            writer.add_summary(summary, iteration)

            training_loss += n_loss
            training_batches += 1

            # print(colored("Epoch %3d, Iteration %6d:\t" % (epoch_idx, batch_idx), 'cyan') + colored("loss = %.8f" % (n_loss), 'green'))

            batch_idx += 1
            iteration += 1

        # Test
        # for _x, _x_size, _y, _y_size in epoch_test:

        #     # Train network
        #     feed_dict = { placeHolder_X: _x, placeHolder_X_size: _x_size, placeHolder_Y: _y, placeHolder_Y_size: _y_size, placeHolder_max_length: maxl_array }
            
        #     # dp_dict = tl.utils.dict_to_one( net.all_drop )
        #     # feed_dict.update( dp_dict )

        #     n_loss, summary = sess.run([test_loss, merged], feed_dict = feed_dict)
            
        #     testing_loss += n_loss
        #     testing_batches += 1

        training_loss = 20 * math.log10( training_loss / training_batches )
        # testing_loss = 20 * math.log10( testing_loss / testing_batches )
        
        print(colored("**************************\n", 'red') + colored("Epoch %3d:\t" % (epoch_idx), 'cyan') + colored("\n\t\tTraining Loss = %.8f dB" % (training_loss), 'yellow') + colored("\n\t\tTesting Loss  = %.8f dB" % (testing_loss), 'green'))
    
        epoch_idx += 1

    # Save the network
    if(args.save != "None"):
        save_path = saver.save(sess, "savedModels/" + args.save + ".ckpt")
        print("Model saved in %s" % (save_path))

train()
