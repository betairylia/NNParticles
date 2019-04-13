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

from model_3dconv import model_conv3d
import dataLoad_3dconv as dataLoad

from time import gmtime, strftime

import progressbar

parser = argparse.ArgumentParser(description="Run the NN for particle simulation")

parser.add_argument('datapath')
parser.add_argument('outpath')
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-ss', '--start-step', type = int, default = 1, help = "Which step do we start from")
parser.add_argument('-ds', '--delta-step', type = int, default = 1, help = "How many steps ds will the network predict, step i -> step (i+ds), 0 for identity test")
parser.add_argument('-zdim', '--latent-dim', type = int, default = 256, help = "Length of the latent vector")
parser.add_argument('-ch', '--channels', type = int, default = 1, help = "How many channels is our data")
parser.add_argument('-size', '--size', type = int, default = 32, help = "Size of the grids (resolution) we are going to deal with")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load for prediction")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

# Set up data
dataLoad.channels = args.channels

# Create the model
model = model_conv3d(args.size, args.latent_dim, args.channels, tf.train.AdamOptimizer())

# Build the model
model.build_model()

# Create session
sess = tf.Session()

sess.run(tf.local_variables_initializer())
tl.layers.initialize_global_variables(sess)

# Save & Load
saver = tf.train.Saver()

if args.load != "None":
    saver.restore(sess, args.load)

step = 0

# Read data
groundTruth = np.load(args.datapath)
stepCount = groundTruth.shape[0]

assert(groundTruth.shape[1] == args.size)

result = np.zeros(groundTruth.shape)

# Copy steps from groundTruth
result[0:args.start_step] = groundTruth[0:args.start_step]

# Start a progress bar
bar = progressbar.ProgressBar(maxval = groundTruth.shape[0], widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

# Training process
for step in range(args.start_step, groundTruth.shape[0], args.delta_step):

    # print("----------------------\nStep %d" % step)
    
    _x = np.zeros((1, args.size, args.size, args.size))
    _x[0] = result[step - args.delta_step]

    _x.shape = (1, args.size, args.size, args.size, 1)

    # print("_x = %f" % np.sum(_x))

    result_array = sess.run(model.decoder_net_train.outputs, feed_dict = {model.ph_X: _x})
    # result_array = sess.run(model.decoder_net_val.outputs, feed_dict = {model.ph_X: _x})

    # print("result = [%f]" % np.sum(result_array))
    # print(result_array.shape)

    result_array.shape = (1, args.size, args.size, args.size)

    for s in range(step, min(step + args.delta_step, groundTruth.shape[0])):
        # print("filling %d" % s)
        result[s] = result_array[0]

    # input('Press any key...')
    
    bar.update(step)
bar.finish()

# Save the prediction
np.save(args.outpath, result)
