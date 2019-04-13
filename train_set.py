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

from model_set import model_set
import dataLoad_set as dataLoad

from time import gmtime, strftime

import progressbar

from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser(description="Run the NN for particle simulation")

parser.add_argument('datapath')
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-ep', '--epochs', type = int, default = 200)
parser.add_argument('-bs', '--batch-size', type = int, default = 64)
parser.add_argument('-ds', '--delta-step', type = int, default = 1, help = "How many steps ds will the network predict, step i -> step (i+ds), 0 for identity test")
parser.add_argument('-vm', '--velocity-multiplier', type = float, default = 1.0, help = "Multiplies the velocity by this factor")
parser.add_argument('-zdim', '--latent-dim', type = int, default = 256, help = "Length of the latent vector")
parser.add_argument('-subsp', '--sub-spaces', type = int, default = 16, help = "Number of subspace divisions")
parser.add_argument('-res', '--res-size', type = int, default = 8, help = "Length of res layers (res block stacks)")
parser.add_argument('-lr', '--learning-rate', type = float, default = 0.001, help = "learning rate")
parser.add_argument('-beta1', '--beta1', type = float, default = 0.9, help = "beta1")
parser.add_argument('-beta2', '--beta2', type = float, default = 0.999, help = "beta2")
parser.add_argument('-l2', '--l2-loss', dest = 'loss_func', action='store_const', default = tf.abs, const = tf.square, help = "use L2 Loss")
parser.add_argument('-maxpool', '--maxpool', dest = 'combine_method', action='store_const', default = tf.reduce_mean, const = tf.reduce_max, help = "use Max pooling instead of sum up for permutation invariance")
parser.add_argument('-mass', '--mass-loss', dest = 'use_mass_loss', action='store_const', default = False, const = True, help = "use total mass regulaizer")
parser.add_argument('-onorm', '--ortho-norm', dest = 'use_onorm', action='store_const', default = False, const = True, help = "use orthogonal weight regulaization")
parser.add_argument('-ch', '--channels', type = int, default = 1, help = "How many channels is our data")
parser.add_argument('-size', '--size', type = int, default = 32, help = "Size of the grids (resolution) we are going to deal with")
parser.add_argument('-log', '--log', type = str, default = "logs", help = "Path to log dir")
parser.add_argument('-name', '--name', type = str, default = "NoName", help = "Name to show on tensor board")
parser.add_argument('-save', '--save', type = str, default = "None", help = "Path to store trained model")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")
parser.add_argument('-debug', '--debug', dest = "enable_debug", action = 'store_const', default = False, const = True, help = "Enable debugging")

args = parser.parse_args()

dataLoad.maxParticlesPerGrid = args.size

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

logPath = os.path.join(args.log, args.name + "(" + strftime("%Y-%m-%d %H-%Mm-%Ss", gmtime()) + ")/")

# Set up data
dataLoad.channels = args.channels

# Create the model
# optimizer = tf.train.MomentumOptimizer(learning_rate = args.learning_rate, momentum = args.beta1)
optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2)

model = model_set(args.size, args.latent_dim, args.batch_size, optimizer)
model.resSize = args.res_size
model.subspace_div = args.sub_spaces
model.loss_func = args.loss_func
model.combine_method = args.combine_method
model.use_mass_loss = args.use_mass_loss

if args.use_onorm:
    model.onorm_lambda = 0.1

# Build the model
model.build_model()

# Summary the variables
tras = tf.summary.scalar('Training Loss', model.train_loss)
traos = tf.summary.scalar('Training Ortho Loss', model.train_ortho_loss)
# trads = tf.summary.scalar('Training Data Loss', model.train_data_loss)
tracs = tf.summary.scalar('Training Card Loss', model.train_card_loss)
tracd = tf.summary.scalar('Training Cardinality', model.train_card)
# trade = tf.summary.scalar('Avg Training distance error', model.train_error)
vals = tf.summary.scalar('Avg Validation distance error', model.val_error)

# merged_train = tf.summary.merge([tras, trads, tracs, traos, tracd, trade])
merged_train = tf.summary.merge([tras, traos, tracs, tracd])
merged_val = tf.summary.merge([vals])

# Create session
sess = tf.Session()

if args.enable_debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # variables_names = [v.name for v in tf.trainable_variables()]
    # values = sess.run(variables_names)
    # cprint('Trainable vars', 'red')
    # for k, v in zip(variables_names, values):
    #     cprint("Variable: " + str(k), 'yellow')
    #     cprint("Shape: " + (v.shape), 'green')
    #     #print(v)

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

epoch_idx = 0
iteration = 0

maxl_array = np.zeros((2))
maxl_array[0] = args.size
maxl_array[1] = args.size

stepFactor = 9

# Training process
for epoch_train, epoch_validate in dataLoad.gen_epochs(args.epochs, args.datapath, args.batch_size, args.delta_step, args.velocity_multiplier):
    
    epoch_idx += 1
    print(colored("Epoch %03d" % (epoch_idx), 'yellow'))
    
    # Train
    for _x, _y, _y_size in epoch_train:
        feed_dict = { model.ph_X: _x, model.ph_Y: _y[:, :, 0:3], model.ph_Ycard: _y_size, model.ph_max_length: maxl_array }
        _, n_loss, summary = sess.run([model.train_op, model.train_loss, merged_train], feed_dict = feed_dict)
        train_writer.add_summary(summary, batch_idx_train)
        batch_idx_train += 1

        print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("   Train   It %08d" % batch_idx_train, 'magenta'))

    # Test
    # for _x, _y, _y_size in epoch_validate:
    #     feed_dict = { model.ph_X: _x, model.ph_Y: _y[:, :, 0:3], model.ph_Ycard: _y_size, model.ph_max_length: maxl_array }
    #     n_loss, summary = sess.run([model.val_loss, merged_val], feed_dict = feed_dict)
    #     val_writer.add_summary(summary, round(batch_idx_test * stepFactor))
    #     batch_idx_test += 1

    #     print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("Validation It %08d" % batch_idx_test, 'magenta'))

# Save the network
if(args.save != "None"):
    save_path = saver.save(sess, "savedModels/" + args.save + ".ckpt")
    print("Model saved in %s" % (save_path))
