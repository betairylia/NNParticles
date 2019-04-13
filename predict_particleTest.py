# python train_particleTest.py -gpu 2 -ep 20 -bs 128 -vSize 22 -vm 10 -zdim 30 -hdim 64 -enc plain -dec plain -log log_particleTest -name Plain_Plain_bs128_z30h64_gs8_gm22 MDSets/2560_smallGrid/

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

from model_particlesTest import model_particles as model_net
# import dataLoad_particleTest as dataLoad                        # Legacy method, strongly disagree with i.i.d. distribution among batch(epoch)es.
import dataLoad_particleTest_combinednpy as dataLoad            # New method, shuffle & mixed randomly

from time import gmtime, strftime

import progressbar

from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser(description="Run the NN for particle simulation")

parser.add_argument('datapath')
parser.add_argument('-gpu', '--cuda-gpus')

parser.add_argument('-ep', '--epochs', type = int, default = 20)
parser.add_argument('-bs', '--batch-size', type = int, default = 16)
parser.add_argument('-vLen', '--voxel-length', type = int, default = 96, help = "Size of voxel (0 to use voxel length stored in file)")
parser.add_argument('-vSize', '--voxel-size', type = int, default = 2560, help = "Max amount of particles in a voxel")
parser.add_argument('-vm', '--velocity-multiplier', type = float, default = 1.0, help = "Multiplies the velocity by this factor")

parser.add_argument('-zdim', '--latent-dim', type = int, default = 512, help = "Length of the latent vector")
parser.add_argument('-hdim', '--hidden-dim', type = int, default = 64, help = "Length of the hidden vector inside network")
parser.add_argument('-cdim', '--cluster-dim', type = int, default = 128, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-ccnt', '--cluster-count', type = int, default = 256, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-knnk', '--nearest-neighbor', type = int, default = 16, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-enc', '--encoder', type = str, default = "plain_ln", help = "Selected encoder architechure")
parser.add_argument('-dec', '--decoder', type = str, default = "fold_ln", help = "Selected decoder architechure")

parser.add_argument('-lr', '--learning-rate', type = float, default = 0.0003, help = "learning rate")
parser.add_argument('-beta1', '--beta1', type = float, default = 0.9, help = "beta1")
parser.add_argument('-beta2', '--beta2', type = float, default = 0.999, help = "beta2")
parser.add_argument('-l2', '--l2-loss', dest = 'loss_func', action='store_const', default = tf.abs, const = tf.square, help = "use L2 Loss")
parser.add_argument('-maxpool', '--maxpool', dest = 'combine_method', action='store_const', default = tf.reduce_mean, const = tf.reduce_max, help = "use Max pooling instead of sum up for permutation invariance")
parser.add_argument('-adam', '--adam', dest = 'adam', action='store_const', default = False, const = True, help = "Use Adam optimizer")

parser.add_argument('-log', '--log', type = str, default = "logs", help = "Path to log dir")
parser.add_argument('-name', '--name', type = str, default = "NoName", help = "Name to show on tensor board")
parser.add_argument('-save', '--save', type = str, default = "None", help = "Path to store trained model")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")
parser.add_argument('-debug', '--debug', dest = "enable_debug", action = 'store_const', default = False, const = True, help = "Enable debugging")

args = parser.parse_args()

dataLoad.maxParticlesPerGrid = args.voxel_size
if args.voxel_length == 0:
    dataLoad.overrideGrid = False
else:
    dataLoad.overrideGrid = True
    dataLoad.overrideGridSize = args.voxel_length

if args.name == "NoName":
    args.name = "[NPY][NoCard][1st2ndmomentEdges(edgeMask,[u;v;edg])][NoPosInVertFeature] E(%s)-D(%s)-%d^3(%d)g%dh%dz-bs%dlr%f-%s" % (args.encoder, args.decoder, args.voxel_length, args.voxel_size, args.hidden_dim, args.latent_dim, args.batch_size, args.learning_rate, 'Adam' if args.adam else 'mSGD')

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

# Create the model
if args.adam:
    optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2)
else:
    optimizer = tf.train.MomentumOptimizer(learning_rate = args.learning_rate, momentum = args.beta1)

# model = model_net(16, args.latent_dim, args.batch_size, optimizer)
model = model_net(args.voxel_size, args.latent_dim, args.batch_size, optimizer)
model.particle_hidden_dim = args.hidden_dim
model.loss_func = args.loss_func
model.combine_method = args.combine_method
model.encoder_arch = args.encoder
model.decoder_arch = args.decoder
model.knn_k = args.nearest_neighbor
model.cluster_feature_dim = args.cluster_dim
model.cluster_count = args.cluster_count

# Headers
# headers = dataLoad.read_file_header(dataLoad.get_fileNames(args.datapath)[0])
model.total_world_size = 96.0
model.initial_grid_size = model.total_world_size / 16

# model.initial_grid_size = model.total_world_size / 4

# Build the model
val_loss, rec_particles, rec_fold_particles, pool_poses = model.build_predict()

# Create session
sess = tf.Session()

sess.run(tf.local_variables_initializer())
tl.layers.initialize_global_variables(sess)

# Save & Load
saver = tf.train.Saver()

if args.load != "None":
    saver.restore(sess, args.load)
    print("Model Loaded.")

batch_idx_train = 0
batch_idx_test = 0

epoch_idx = 0
iteration = 0

maxl_array = np.zeros((2))
maxl_array[0] = args.voxel_size
maxl_array[1] = args.voxel_size

epCount = dataLoad.fileCount(args.datapath)
stepFactor = 9
totalIterations = 100

bs = args.batch_size
N = args.voxel_size
pRange = 3

groundTruth = np.zeros((totalIterations * bs, N, pRange))
reconstruct = np.zeros((totalIterations * bs, N, pRange))
foldOnly    = np.zeros((totalIterations * bs, N, pRange))
pool1       = np.zeros((totalIterations * bs, N, pRange))
pool2       = np.zeros((totalIterations * bs, N, pRange))
pool3       = np.zeros((totalIterations * bs, N, pRange))
pools = [pool1, pool2, pool3]

for epoch_train, epoch_validate in dataLoad.gen_epochs(args.epochs, args.datapath, args.batch_size, args.velocity_multiplier):

    epoch_idx += 1
    print(colored("Epoch %03d" % (epoch_idx), 'yellow'))

    # Train
    for _x, _x_size in epoch_train:
        feed_dict = { model.ph_X: _x, model.ph_card: _x_size, model.ph_max_length: maxl_array }
        n_loss, p, fp, ppa = sess.run([val_loss, rec_particles, rec_fold_particles, pool_poses], feed_dict = feed_dict)

        sidx = batch_idx_train * bs
        eidx = (batch_idx_train + 1) * bs
        batch_idx_train += 1

        groundTruth[sidx:eidx, :, :] = _x[:, :, 0:pRange]
        reconstruct[sidx:eidx, :, :] =  p[:, :, 0:pRange]
        foldOnly   [sidx:eidx, :, :] = fp[:, :, 0:pRange]

        for pidx in range(min(3, len(ppa))):
            pcnt = ppa[pidx].shape[1]
            pools[pidx][sidx:eidx, :pcnt, :] = ppa[pidx][:, :pcnt, :pRange]

        print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("   Train   It %08d" % batch_idx_train, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))

        if batch_idx_train >= totalIterations:
            break
    
    if batch_idx_train >= totalIterations:
        break

if not os.path.exists(os.path.join('PAE_Reconstruction', args.name)):
    os.makedirs(os.path.join('PAE_Reconstruction', args.name))

np.save(os.path.join('PAE_Reconstruction', args.name, 'gt.npy'), groundTruth) 
np.save(os.path.join('PAE_Reconstruction', args.name, 'rc.npy'), reconstruct) 
np.save(os.path.join('PAE_Reconstruction', args.name, 'rf.npy'), foldOnly) 
np.save(os.path.join('PAE_Reconstruction', args.name, 'p1.npy'), pool1) 
np.save(os.path.join('PAE_Reconstruction', args.name, 'p2.npy'), pool2) 
np.save(os.path.join('PAE_Reconstruction', args.name, 'p3.npy'), pool3) 

