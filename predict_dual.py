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

from termcolor import colored, cprint

from model_dual import model_dual as model_net
import dataLoad_dual as dataLoad

from time import gmtime, strftime

import progressbar

from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser(description="Run the NN for particle simulation")

parser.add_argument('datapath')
parser.add_argument('outpath')
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-bs', '--batch-size', type = int, default = 64)
parser.add_argument('-ds', '--delta-step', type = int, default = 1, help = "How many steps ds will the network predict, step i -> step (i+ds), 0 for identity test")
parser.add_argument('-ls', '--latent-step', type = int, default = 1, help = "How many ds will the network simulate in latent space before predicting final result (decoding)")
parser.add_argument('-ss', '--start-step', type = int, default = 0, help = "Which step to start")
parser.add_argument('-conly', '--card-only', dest = 'card_only', action = 'store_const', default = False, const = True, help = "Only predicting card (use density map visualizer instead)")
parser.add_argument('-simit', '--sim-iter', type = int, default = 130, help = "How many steps (= latent steps) to simulate ( total steps = simit * ls * ds )")
parser.add_argument('-vm', '--velocity-multiplier', type = float, default = 1.0, help = "Multiplies the velocity by this factor")
parser.add_argument('-zdim', '--latent-dim', type = int, default = 256, help = "Length of the latent vector")
parser.add_argument('-res', '--res-size', type = int, default = 4, help = "Length of res layers (res block stacks)")
parser.add_argument('-maxpool', '--maxpool', dest = 'combine_method', action='store_const', default = tf.reduce_mean, const = tf.reduce_max, help = "use Max pooling instead of sum up for permutation invariance")
parser.add_argument('-size', '--size', type = int, default = 2560, help = "Total amount of particles we are going to deal with")
parser.add_argument('-vSize', '--voxel-size', type = int, default = 32, help = "Max amount of particles in a voxel")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")
parser.add_argument('-debug', '--debug', dest = "enable_debug", action = 'store_const', default = False, const = True, help = "Enable debugging")

args = parser.parse_args()

dataLoad.particleCount = args.size
dataLoad.maxParticlePerGrid = args.voxel_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

# Create the model
optimizer = tf.train.AdamOptimizer()

# model = model_net(16, args.latent_dim, args.batch_size, optimizer)
model = model_net(args.size, args.voxel_size, args.latent_dim, args.batch_size, optimizer)
model.resSize = args.res_size
model.combine_method = args.combine_method

# Headers
headers = dataLoad.read_file_header(args.datapath)
model.total_world_size = 96.0
model.initial_grid_size = model.total_world_size / 16

model.latent_simulate_steps = args.latent_step
# model.initial_grid_size = model.total_world_size / 4

# model.build_model()

# Build the model
if args.card_only:
    card_prediction, _loss = model.card_only_prediction(args.sim_iter)
else:
    model.build_prediction()

# Create session
sess = tf.Session()

if args.enable_debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

sess.run(tf.local_variables_initializer())
tl.layers.initialize_global_variables(sess)

# Save & Load
saver = tf.train.Saver()

if args.load != "None":
    saver.restore(sess, args.load)
    print("Model restored.")

# variables_names = [v.name for v in tl.layers.get_variables_with_name('', True)]
# values = sess.run(variables_names)
# cprint('Trainable vars', 'red')
# for k, v in zip(variables_names, values):
#     cprint("Variable: " + str(k), 'yellow')
#     cprint("Shape: " + str(v.shape), 'green')
#     #print(v)

batch_idx = 0
current_step = 0

# Prediction
groundTruth_content = dataLoad.read_file(args.datapath, args.delta_step * args.velocity_multiplier)
batch_X = np.zeros((1, args.size, 7))

if args.card_only:
    # 1st batch
    batch_X[0] = groundTruth_content['data'][args.start_step, :, 0:7]
    # batch_X[1] = groundTruth_content['data'][args.start_step, :, 0:7]

    results, loss = sess.run([card_prediction, _loss], feed_dict = {model.ph_X: batch_X})
    # results, loss, vloss, tloss = sess.run([card_prediction, _loss, _valloss, _trainloss], feed_dict = {model.ph_X: batch_X, model.ph_Y: batch_X})
    print("Loss = %f" % (loss))
    np.save(args.outpath, results)
    print('Please run \'python convertNpyToRBin.py %s %s\'' % (args.outpath, args.outpath.split('.')[0] + '.rbin'))
else:
    # Create result file
    content = {}

    content['gridCountX'] = 16
    content['gridCountY'] = 16
    content['gridCountZ'] = 16
    content['gridCount'] = 4096
    content['gridSize'] = 6

    content['worldLength'] = 96
    content['gravity'] = False
    content['boundary'] = False

    content['stepCount'] = args.sim_iter + 1
    content['particleCount'] = np.zeros((content['stepCount'], content['gridCount']), dtype = np.int32)
    content['data'] = np.zeros((content['stepCount'], content['gridCount'], args.voxel_size, 6))

    # 1st batch
    batch_X[0] = groundTruth_content['data'][args.start_step, :, 0:7]

    # Write 1st batch to result
    for p in range(batch_X.shape[1]):
        gridX = batch_X[0, p, 0] // content['gridSize'] + content['gridCountX'] // 2
        gridY = batch_X[0, p, 1] // content['gridSize'] + content['gridCountY'] // 2
        gridZ = batch_X[0, p, 2] // content['gridSize'] + content['gridCountZ'] // 2
        
        gridHash = int(gridX * content['gridCountY'] * content['gridCountZ'] + gridY * content['gridCountZ'] + gridZ)

        # Translate particles from world space to voxel space
        content['data'][current_step, gridHash, content['particleCount'][current_step, gridHash]] = np.asarray([\
            batch_X[0, p, 0] - (gridX - content['gridCountX'] // 2) * content['gridSize'] - content['gridSize'] / 2,\
            batch_X[0, p, 1] - (gridY - content['gridCountY'] // 2) * content['gridSize'] - content['gridSize'] / 2,\
            batch_X[0, p, 2] - (gridZ - content['gridCountZ'] // 2) * content['gridSize'] - content['gridSize'] / 2,\
            batch_X[0, p, 3],\
            batch_X[0, p, 4],\
            batch_X[0, p, 5]])
        content['particleCount'][current_step, gridHash] += 1

    current_step += 1

    # Simulation loop
    while current_step <= args.sim_iter:

        print("Step %d" % current_step)

        # Feed the data and run the net
        card, data = sess.run([model.predict_cardinality, model.predict_outputs], feed_dict = {model.ph_X: batch_X})

        # Write the data to result
        # The velocity data here was multiplied by ds*vm, and it will be divided back when writing data back to files.
        content['data'][current_step, :, :, :] = data[:, :, 0:6]
        content['particleCount'][current_step, :] = card[:]

        # Prepare next batch
        batch_X = np.zeros((1, args.size, 7))
        particle_idx = 0
        for g in range(content['gridCount']):

            gridPosX = g // (content['gridCountZ'] * content['gridCountX']) * content['gridSize'] - (content['gridCountX'] // 2 * content['gridSize'])
            gridPosY = g % (content['gridCountZ'] * content['gridCountX']) // content['gridCountZ'] * content['gridSize'] - (content['gridCountY'] // 2 * content['gridSize'])
            gridPosZ = g % content['gridCountZ'] * content['gridSize'] - (content['gridCountZ'] // 2 * content['gridSize'])

            for p in range(content['particleCount'][current_step, g]):

                # Translate particles from voxel space to world space
                batch_X[0, particle_idx, 0] = content['data'][current_step, g, p, 0] + gridPosX + (content['gridSize'] / 2) # x
                batch_X[0, particle_idx, 1] = content['data'][current_step, g, p, 1] + gridPosY + (content['gridSize'] / 2) # y
                batch_X[0, particle_idx, 2] = content['data'][current_step, g, p, 2] + gridPosZ + (content['gridSize'] / 2) # z

                batch_X[0, particle_idx, 3:6] = content['data'][current_step, g, p, 3:6]
                batch_X[0, particle_idx, 6] = 1

                particle_idx += 1

        current_step += 1

    # Save the results
    dataLoad.save_file(content, args.outpath, args.delta_step * args.velocity_multiplier)
