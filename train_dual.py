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
parser.add_argument('-gpu', '--cuda-gpus')
parser.add_argument('-ep', '--epochs', type = int, default = 200)
parser.add_argument('-preep', '--pretrain-epochs', type = int, default = 10)
parser.add_argument('-bs', '--batch-size', type = int, default = 64)
parser.add_argument('-ds', '--delta-step', type = int, default = 1, help = "How many steps ds will the network predict, step i -> step (i+ds), 0 for identity test")
parser.add_argument('-vm', '--velocity-multiplier', type = float, default = 1.0, help = "Multiplies the velocity by this factor")
parser.add_argument('-zdim', '--latent-dim', type = int, default = 256, help = "Length of the latent vector")
parser.add_argument('-res', '--res-size', type = int, default = 4, help = "Length of res layers (res block stacks)")
parser.add_argument('-ignrpn', '--ignore-particle-net', dest = "ignore_pnet", action = 'store_const', default = False, const = True, help = "Enable debugging")
parser.add_argument('-lr', '--learning-rate', type = float, default = 0.001, help = "learning rate")
parser.add_argument('-beta1', '--beta1', type = float, default = 0.9, help = "beta1")
parser.add_argument('-beta2', '--beta2', type = float, default = 0.999, help = "beta2")
parser.add_argument('-l2', '--l2-loss', dest = 'loss_func', action='store_const', default = tf.abs, const = tf.square, help = "use L2 Loss")
parser.add_argument('-maxpool', '--maxpool', dest = 'combine_method', action='store_const', default = tf.reduce_mean, const = tf.reduce_max, help = "use Max pooling instead of sum up for permutation invariance")
parser.add_argument('-size', '--size', type = int, default = 2560, help = "Total amount of particles we are going to deal with")
parser.add_argument('-vSize', '--voxel-size', type = int, default = 32, help = "Max amount of particles in a voxel")
parser.add_argument('-log', '--log', type = str, default = "logs", help = "Path to log dir")
parser.add_argument('-name', '--name', type = str, default = "NoName", help = "Name to show on tensor board")
parser.add_argument('-save', '--save', type = str, default = "None", help = "Path to store trained model")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")
parser.add_argument('-debug', '--debug', dest = "enable_debug", action = 'store_const', default = False, const = True, help = "Enable debugging")

args = parser.parse_args()

dataLoad.particleCount = args.size
dataLoad.maxParticlePerGrid = args.voxel_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

logPath = os.path.join(args.log, args.name + "(" + strftime("%Y-%m-%d %H-%Mm-%Ss", gmtime()) + ")/")

# Create the model
# optimizer = tf.train.MomentumOptimizer(learning_rate = args.learning_rate, momentum = args.beta1)
optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2)

# model = model_net(16, args.latent_dim, args.batch_size, optimizer)
model = model_net(args.size, args.voxel_size, args.latent_dim, args.batch_size, optimizer)
model.resSize = args.res_size
model.loss_func = args.loss_func
model.combine_method = args.combine_method
model.train_particle_net_on_main_phase = not args.ignore_pnet

# Headers
headers = dataLoad.read_file_header(dataLoad.get_fileNames(args.datapath)[0])
model.total_world_size = 96.0
model.initial_grid_size = model.total_world_size / 16

# model.initial_grid_size = model.total_world_size / 4

# Build the model
model.build_model()

# Summary the variables
tras = tf.summary.scalar('Training Loss', model.train_loss)
trass = tf.summary.scalar('Training Simulate Loss', model.train_simLoss)
trars = tf.summary.scalar('Training Reconstruction Loss', model.train_reconstructLoss)
traps = tf.summary.scalar('Training Particle Loss', model.train_particleLoss)
ptraps = tf.summary.scalar('Pre-training Particle Loss', model.train_particleLoss)
ptraprs = tf.summary.scalar('Pre-training Particle Raw Loss', model.train_particleRawLoss)
ptrapcs = tf.summary.scalar('Pre-training Particle Card Loss', model.train_particleCardLoss)
ptrapss = tf.summary.scalar('Pre-training Particle Score Loss', model.train_particleScoreLoss)
trais = tf.summary.scalar('Training Inverse Loss', model.train_invLoss)
trazs = tf.summary.scalar('Training Voxels norm', model.zero_loss)
vals = tf.summary.scalar('Validation Loss', model.val_loss)

merged_pretrain = tf.summary.merge([ptraps, ptraprs, ptrapcs, ptrapss])

if not args.ignore_pnet:
    merged_train = tf.summary.merge([tras, trass, trars, traps, trais, trazs])
else:
    merged_train = tf.summary.merge([tras, trass, trars, trais, trazs])
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
maxl_array[0] = args.voxel_size
maxl_array[1] = args.voxel_size

epCount = dataLoad.fileCount(args.datapath)
validation_gap = 4
stepFactor = (epCount - (epCount // validation_gap)) / (epCount // validation_gap)

# Pre-training process
for epoch, is_train in dataLoad.gen_epochs(args.pretrain_epochs, args.datapath, args.batch_size, args.delta_step, args.velocity_multiplier, validation_gap):

    epoch_idx += 1
    print(colored("Pre-train Epoch %03d" % (epoch_idx), 'yellow'))
    
    # Train
    for _x, _y, _y_progress, _v, _vc, _vp in epoch:
        feed_dict = { model.ph_X: _x, model.ph_Y: _y, model.ph_Y_progress: _y_progress, model.ph_voxels: _v, model.ph_voxels_card: _vc, model.ph_voxels_pos: _vp, model.ph_max_length: maxl_array }
        _, n_loss, summary = sess.run([model.particle_train_op, model.train_particleLoss, merged_pretrain], feed_dict = feed_dict)
        train_writer.add_summary(summary, batch_idx_train)
        batch_idx_train += 1

        print(colored("Pre-Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("   Train   It %08d" % batch_idx_train, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))

epoch_idx = 0
batch_idx_train = 0

# Training process
for epoch, is_train in dataLoad.gen_epochs(args.epochs, args.datapath, args.batch_size, args.delta_step, args.velocity_multiplier, validation_gap):
    
    epoch_idx += 1
    print(colored("Epoch %03d" % (epoch_idx), 'yellow'))
    
    # Train
    if is_train == True:
        for _x, _y, _y_progress, _v, _vc, _vp in epoch:
            feed_dict = { model.ph_X: _x, model.ph_Y: _y, model.ph_Y_progress: _y_progress, model.ph_voxels: _v, model.ph_voxels_card: _vc, model.ph_voxels_pos: _vp, model.ph_max_length: maxl_array }
            _, n_loss, summary = sess.run([model.train_op, model.train_loss, merged_train], feed_dict = feed_dict)
            train_writer.add_summary(summary, batch_idx_train)
            batch_idx_train += 1

            print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("   Train   It %08d" % batch_idx_train, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))
    # Test
    else:
        for _x, _y, _y_progress, _v, _vc, _vp in epoch:
            feed_dict = { model.ph_X: _x, model.ph_Y: _y, model.ph_Y_progress: _y_progress, model.ph_voxels: _v, model.ph_voxels_card: _vc, model.ph_voxels_pos: _vp, model.ph_max_length: maxl_array }
            n_loss, summary = sess.run([model.val_loss, merged_val], feed_dict = feed_dict)
            val_writer.add_summary(summary, round(batch_idx_test * stepFactor))
            batch_idx_test += 1

            print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("Validation It %08d" % batch_idx_test, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))

# Save the network
if(args.save != "None"):
    save_path = saver.save(sess, "savedModels/" + args.save + ".ckpt")
    print("Model saved in %s" % (save_path))
