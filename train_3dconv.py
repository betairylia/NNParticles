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
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-ep', '--epochs', type = int, default = 200)
parser.add_argument('-bs', '--batch-size', type = int, default = 64)
parser.add_argument('-ds', '--delta-step', type = int, default = 1, help = "How many steps ds will the network predict, step i -> step (i+ds), 0 for identity test")
parser.add_argument('-zdim', '--latent-dim', type = int, default = 256, help = "Length of the latent vector")
parser.add_argument('-res', '--res-size', type = int, default = 8, help = "Length of res layers (res block stacks)")
parser.add_argument('-lr', '--learning-rate', type = float, default = 0.001, help = "learning rate")
parser.add_argument('-beta1', '--beta1', type = float, default = 0.9, help = "beta1")
parser.add_argument('-beta2', '--beta2', type = float, default = 0.999, help = "beta2")
parser.add_argument('-l2', '--l2-loss', dest = 'loss_func', action='store_const', default = tf.abs, const = tf.square, help = "use L2 Loss")
parser.add_argument('-mass', '--mass-loss', dest = 'use_mass_loss', action='store_const', default = False, const = True, help = "use total mass regulaizer")
parser.add_argument('-ch', '--channels', type = int, default = 1, help = "How many channels is our data")
parser.add_argument('-size', '--size', type = int, default = 32, help = "Size of the grids (resolution) we are going to deal with")
parser.add_argument('-log', '--log', type = str, default = "logs_gridSearch", help = "Path to log dir")
parser.add_argument('-name', '--name', type = str, default = "NoName", help = "Name to show on tensor board")
parser.add_argument('-save', '--save', type = str, default = "None", help = "Path to store trained model")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

logPath = os.path.join(args.log, args.name + "(" + strftime("%Y-%m-%d %H-%Mm-%Ss", gmtime()) + ")/")

# Set up data
dataLoad.channels = args.channels

# Create the model
model = model_conv3d(args.size, args.latent_dim, args.channels, tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2))
model.resSize = args.res_size
model.loss_func = args.loss_func
model.use_mass_loss = args.use_mass_loss

# Build the model
model.build_model()

# Summary the variables
tras = tf.summary.scalar('Training Loss', model.raw_loss)
vals = tf.summary.scalar('Validation Loss', model.val_loss)

merged_train = tf.summary.merge([tras])
merged_val = tf.summary.merge([vals])

# Create session
sess = tf.Session()

train_writer = tf.summary.FileWriter(logPath + '/train', sess.graph)
val_writer = tf.summary.FileWriter(logPath + '/validation', sess.graph)

sess.run(tf.local_variables_initializer())
tl.layers.initialize_global_variables(sess)

# Save & Load
saver = tf.train.Saver()

if args.load != "None":
    saver.restore(sess, args.load)

batch_idx_train = 0
batch_idx_test = 0
epCount = dataLoad.fileCount(args.datapath)
epoch_idx = 0
validation_gap = 4

stepFactor = (epCount - (epCount // validation_gap)) / (epCount // validation_gap)

# Training process
for epoch, is_train in dataLoad.gen_epochs(args.epochs, args.datapath, args.batch_size, args.delta_step, validation_gap):
    
    epoch_idx += 1
    print(colored("Epoch %3d" % (epoch_idx), 'yellow'))
    
    if is_train == True:
        for _x, _y in epoch:
            _, loss, summary = sess.run([model.train_op, model.loss, tras], feed_dict = {model.ph_X: _x, model.ph_Y: _y})
            train_writer.add_summary(summary, batch_idx_train)
            batch_idx_train += 1
    
    else:
        for _x, _y in epoch:
            loss, summary = sess.run([model.val_loss, vals], feed_dict = {model.ph_X: _x, model.ph_Y: _y})
            val_writer.add_summary(summary, round(batch_idx_test * stepFactor))
            batch_idx_test += 1

# Save the network
if(args.save != "None"):
    save_path = saver.save(sess, "savedModels/" + args.save + ".ckpt")
    print("Model saved in %s" % (save_path))
