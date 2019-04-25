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

import model_graph as model
from model_graph import model_particles as model_net
# import dataLoad_particleTest as dataLoad                        # Legacy method, strongly disagree with i.i.d. distribution among batch(epoch)es.
import dataLoad_graph as dataLoad            # New method, shuffle & mixed randomly

from time import gmtime, strftime

import progressbar

from tensorflow.python import debug as tf_debug

from tensorflow.python.client import timeline

parser = argparse.ArgumentParser(description="Run the NN for particle simulation")

parser.add_argument('datapath')
parser.add_argument('-gpu', '--cuda-gpus')

parser.add_argument('-ep', '--epochs', type = int, default = 20)
parser.add_argument('-bs', '--batch-size', type = int, default = 16)
parser.add_argument('-vLen', '--voxel-length', type = int, default = 96, help = "Size of voxel (0 to use voxel length stored in file)")
parser.add_argument('-vSize', '--voxel-size', type = int, default = 2560, help = "Max amount of particles in a voxel")
parser.add_argument('-vm', '--velocity-multiplier', type = float, default = 1.0, help = "Multiplies the velocity (input[..., 3:]) by this factor")
parser.add_argument('-norm', '--normalize', type = float, default = 1.0, help = "stddev of input data")

parser.add_argument('-zdim', '--latent-dim', type = int, default = 512, help = "Length of the latent vector")
parser.add_argument('-hdim', '--hidden-dim', type = int, default = 64, help = "Length of the hidden vector inside network")
parser.add_argument('-cdim', '--cluster-dim', type = int, default = 128, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-ccnt', '--cluster-count', type = int, default = 256, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-odim', '--output-dim', type = int, default = 6, help = "What kind of data should we output?")
parser.add_argument('-knnk', '--nearest-neighbor', type = int, default = 16, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-loop', '--loop-sim', type = int, default = 5, help = "Loop simulation sim count")

parser.add_argument('-lr', '--learning-rate', type = float, default = 0.0003, help = "learning rate")
parser.add_argument('-beta1', '--beta1', type = float, default = 0.9, help = "beta1")
parser.add_argument('-beta2', '--beta2', type = float, default = 0.999, help = "beta2")
parser.add_argument('-l2', '--l2-loss', dest = 'loss_func', action='store_const', default = tf.abs, const = tf.square, help = "use L2 Loss")
parser.add_argument('-maxpool', '--maxpool', dest = 'combine_method', action='store_const', default = tf.reduce_mean, const = tf.reduce_max, help = "use Max pooling instead of sum up for permutation invariance")
parser.add_argument('-adam', '--adam', dest = 'adam', action='store_const', default = False, const = True, help = "Use Adam optimizer")
parser.add_argument('-fp16', '--fp16', dest = 'dtype', action='store_const', default = tf.float32, const = tf.float16, help = "Use FP16 instead of FP32")
parser.add_argument('-nloop', '--no-loop', dest = 'doloop', action='store_const', default = True, const = False, help = "Don't loop simulation regularization")
parser.add_argument('-nsim', '--no-sim', dest = 'dosim', action='store_const', default = True, const = False, help = "Don't do Simulation")

parser.add_argument('-log', '--log', type = str, default = "logs", help = "Path to log dir")
parser.add_argument('-name', '--name', type = str, default = "NoName", help = "Name to show on tensor board")
parser.add_argument('-save', '--save', type = str, default = "None", help = "Path to store trained model")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")
parser.add_argument('-debug', '--debug', dest = "enable_debug", action = 'store_const', default = False, const = True, help = "Enable debugging")
parser.add_argument('-prof', '--profile', dest = "profile", action = 'store_const', default = False, const = True, help = "Enable profiling (at step 10)")
# parser.add_argument('-prof', '--profile', type = str, default = "None", help = "Path to store profiling timeline (at step 100)")

args = parser.parse_args()

dataLoad.maxParticlesPerGrid = args.voxel_size
if args.voxel_length == 0:
    dataLoad.overrideGrid = False
else:
    dataLoad.overrideGrid = True
    dataLoad.overrideGridSize = args.voxel_length

if args.name == "NoName":
    args.name = "[NPY][NoCard][1st2ndmomentEdges(edgeMask,[u;v;edg])][NoPosInVertFeature] E(%s)-D(%s)-%d^3(%d)g%dh%dz-bs%dlr%f-%s" % ("graph", "graph", args.voxel_length, args.voxel_size, args.hidden_dim, args.latent_dim, args.batch_size, args.learning_rate, 'Adam' if args.adam else 'mSGD')

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

logPath = os.path.join(args.log, args.name + "(" + strftime("%Y-%m-%d %H-%Mm-%Ss", gmtime()) + ")/")

model.default_dtype = args.dtype
# model.default_dtype = tf.float32

# Create the model
if args.adam:
    optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2)
else:
    optimizer = tf.train.MomentumOptimizer(learning_rate = args.learning_rate, momentum = args.beta1)

if args.dtype == tf.float16:
    loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(16.0, 32)
    # loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(96.0)
    optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# model = model_net(16, args.latent_dim, args.batch_size, optimizer)
model = model_net(args.voxel_size, args.latent_dim, args.batch_size, optimizer, args.output_dim)
model.particle_hidden_dim = args.hidden_dim
model.loss_func = args.loss_func
model.combine_method = args.combine_method
model.knn_k = args.nearest_neighbor
model.cluster_feature_dim = args.cluster_dim
model.cluster_count = args.cluster_count
model.doSim = args.dosim
model.doLoop = args.dosim and args.doloop
model.loops = args.loop_sim
model.normalize = args.normalize

# Headers
# headers = dataLoad.read_file_header(dataLoad.get_fileNames(args.datapath)[0])
model.total_world_size = 96.0
model.initial_grid_size = model.total_world_size / 16

# model.initial_grid_size = model.total_world_size / 4

# Build the model
model.build_model()

# Summary the variables
ptraps = tf.summary.scalar('Particle Loss', model.train_particleLoss)
ptraprs = tf.summary.scalar('Particle Reconstruct Loss', model.train_particleRecLoss)
ptrapss = tf.summary.scalar('Particle Simulation Loss', model.train_particleSimLoss)
# vals = tf.summary.scalar('Validation Loss', model.val_particleLoss)
ptrapfs = tf.summary.scalar('Particle HQPool Loss', model.train_HQPLoss)
ptrapps = tf.summary.scalar('Particle Pool Align Loss', model.train_PALoss)

# merged_train = tf.summary.merge([ptraps, ptraprs, ptrapcs, tps, tls])
# merged_val = tf.summary.merge([vals])
merged_train = tf.summary.merge_all()

# Create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

if args.profile:
    builder = tf.profiler.ProfileOptionBuilder
    prof_opts = builder(builder.time_and_memory()).order_by('micros').build()
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

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

save_path = "savedModels/" + args.name + "/"

# Save & Load
saver = tf.train.Saver()

if args.load != "None":
    saver.restore(sess, args.load)

if args.load == "auto" or args.load == "Auto":
    latest_ckpt = tf.train.latest_checkpoint(save_path)
    if latest_ckpt is not None:
        saver.restore(sess, latest_ckpt)

batch_idx_train = 0
batch_idx_test = 0

epoch_idx = 0
iteration = 0

maxl_array = np.zeros((2))
maxl_array[0] = args.voxel_size
maxl_array[1] = args.voxel_size

epCount = dataLoad.fileCount(args.datapath)
stepFactor = 9

for epoch_train, epoch_validate in dataLoad.gen_epochs(args.epochs, args.datapath, args.batch_size, args.velocity_multiplier, True, args.output_dim):

    epoch_idx += 1
    print(colored("Epoch %03d" % (epoch_idx), 'yellow'))

    # Train
    for _x, _x_size in epoch_train:
        
        if batch_idx_train == 10 and args.profile:
            print(colored("Profiling in progress...", 'yellow'))

            with tf.contrib.tfprof.ProfileContext('prof/%s' % args.name, trace_steps = [], dump_steps = []) as pctx:
                if args.dosim and args.doloop:
                    feed_dict = { model.ph_X: _x[0], model.ph_Y: _x[1], model.ph_L: _x[2], model.ph_card: _x_size, model.ph_max_length: maxl_array }
                elif args.dosim:
                    feed_dict = { model.ph_X: _x[0], model.ph_Y: _x[1], model.ph_card: _x_size, model.ph_max_length: maxl_array }
                else:
                    feed_dict = { model.ph_X: _x[0], model.ph_card: _x_size, model.ph_max_length: maxl_array }

                pctx.trace_next_step()
                pctx.dump_next_step()

                _, n_loss, summary = sess.run([model.train_op, model.train_particleLoss, merged_train], feed_dict = feed_dict) #, options = run_options, run_metadata = run_metadata)
                train_writer.add_summary(summary, batch_idx_train)
                batch_idx_train += 1

                pctx.profiler.profile_operations(options = prof_opts)

                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open (os.path.join('profiles/', args.profile + '.json'), 'w') as f:
                    # f.write(chrome_trace)
        
        else:
            
            if args.dosim and args.doloop:
                feed_dict = { model.ph_X: _x[0], model.ph_Y: _x[1], model.ph_L: _x[2], model.ph_card: _x_size, model.ph_max_length: maxl_array }
            elif args.dosim:
                feed_dict = { model.ph_X: _x[0], model.ph_Y: _x[1], model.ph_card: _x_size, model.ph_max_length: maxl_array }
            else:
                feed_dict = { model.ph_X: _x[0], model.ph_card: _x_size, model.ph_max_length: maxl_array }

            _, n_loss, summary = sess.run([model.train_op, model.train_particleLoss, merged_train], feed_dict = feed_dict)
            train_writer.add_summary(summary, batch_idx_train)
            batch_idx_train += 1

        print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("   Train   It %08d" % batch_idx_train, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))

        if batch_idx_train % (15000 // args.batch_size) == 0:
            save_path = saver.save(sess, save_path + args.save + ".ckpt", global_step = batch_idx_train)
            print("Checkpoint saved in %s" % (save_path))

    # Test
    # for _x, _x_size in epoch_validate:
    #     feed_dict = { model.ph_X: _x, model.ph_card: _x_size, model.ph_max_length: maxl_array }
    #     n_loss, summary = sess.run([model.val_particleLoss, merged_val], feed_dict = feed_dict)
    #     val_writer.add_summary(summary, round(batch_idx_test * stepFactor))
    #     batch_idx_test += 1

    #     print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("Validation It %08d" % batch_idx_test, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))
    
    # Save the network
    # if args.save != "None" and epoch_idx % 2 == 0:
    #     save_path = saver.save(sess, "savedModels/" + args.save + "_latest.ckpt")
    #     print("Temporal checkpoint saved in %s" % (save_path))

# Save the network
if(args.save != "None"):
    save_path = saver.save(sess, "savedModels/" + args.save + ".ckpt")
    print("Model saved in %s" % (save_path))
