# python train_particleTest.py -gpu 2 -ep 20 -bs 128 -vSize 22 -vm 10 -zdim 30 -hdim 64 -enc plain -dec plain -log log_particleTest -name Plain_Plain_bs128_z30h64_gs8_gm22 MDSets/2560_smallGrid/

import tensorflow as tf
import numpy as np
import scipy
import time
import math
import argparse
import random
import sys
import os
import json

from termcolor import colored, cprint

import model_graph_sim_latent as model
from model_graph_sim_latent import model_particles as model_net

import dataLoad_graph_sim_latent as dataLoad            # New method, shuffle & mixed randomly

from time import gmtime, strftime

import progressbar

from tensorflow.python import debug as tf_debug

from tensorflow.python.client import timeline
from tensorflow.contrib.tensorboard.plugins import projector

import gc

parser = argparse.ArgumentParser(description="Run the NN for particle simulation")

parser.add_argument('datapath')
parser.add_argument('outpath')
parser.add_argument('-gpu', '--cuda-gpus')

parser.add_argument('-ep', '--epochs', type = int, default = 80)
parser.add_argument('-bs', '--batch-size', type = int, default = 8)
parser.add_argument('-vLen', '--voxel-length', type = int, default = 96, help = "Size of voxel (0 to use voxel length stored in file)")
parser.add_argument('-vSize', '--voxel-size', type = int, default = 2048, help = "Max amount of particles in a voxel")
parser.add_argument('-vm', '--velocity-multiplier', type = float, default = 1.0, help = "Multiplies the velocity (input[..., 3:]) by this factor")
parser.add_argument('-norm', '--normalize', type = float, default = 1.0, help = "stddev of input data")

parser.add_argument('-zdim', '--latent-dim', type = int, default = 128, help = "Length of the latent vector")
parser.add_argument('-hdim', '--hidden-dim', type = int, default = 64, help = "Length of the hidden vector inside network")
parser.add_argument('-cdim', '--cluster-dim', type = int, default = 13, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-ccnt', '--cluster-count', type = int, default = 32, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-odim', '--output-dim', type = int, default = 3, help = "What kind of data should we output?")
parser.add_argument('-knnk', '--nearest-neighbor', type = int, default = 16, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-loop', '--loop-sim', type = int, default = 5, help = "Loop simulation sim count")

parser.add_argument('-lr', '--learning-rate', type = float, default = 0.0003, help = "learning rate")
parser.add_argument('-beta1', '--beta1', type = float, default = 0.9, help = "beta1")
parser.add_argument('-beta2', '--beta2', type = float, default = 0.999, help = "beta2")
parser.add_argument('-l2', '--l2-loss', dest = 'loss_func', action='store_const', default = tf.abs, const = tf.square, help = "use L2 Loss")
parser.add_argument('-maxpool', '--maxpool', dest = 'combine_method', action='store_const', default = tf.reduce_mean, const = tf.reduce_max, help = "use Max pooling instead of sum up for permutation invariance")
parser.add_argument('-adam', '--adam', dest = 'adam', action='store_const', default = True, const = False, help = "Use Adam optimizer")
parser.add_argument('-fp16', '--fp16', dest = 'dtype', action='store_const', default = tf.float32, const = tf.float16, help = "Use FP16 instead of FP32")

parser.add_argument('-log', '--log', type = str, default = "logs", help = "Path to log dir")
parser.add_argument('-name', '--name', type = str, default = "NoName", help = "Name to show on tensor board")
parser.add_argument('-nameAE', '--nameAE', type = str, default = "NoName", help = "Name of AE model to load")
parser.add_argument('-preview', '--previewName', type = str, default = "unnamed", help = "Name for save preview point clouds")
parser.add_argument('-save', '--save', type = str, default = "model", help = "Path to store trained model")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")
parser.add_argument('-conf', '--config', type = str, default = "None", help = "Config overwrite file")
parser.add_argument('-debug', '--debug', dest = "enable_debug", action = 'store_const', default = False, const = True, help = "Enable debugging")
parser.add_argument('-prof', '--profile', dest = "profile", action = 'store_const', default = False, const = True, help = "Enable profiling (at step 10)")
# parser.add_argument('-prof', '--profile', type = str, default = "None", help = "Path to store profiling timeline (at step 100)")

parser.add_argument('-conv', '--conv', type = str, default = "c", help = "c, concat, attention")
parser.add_argument('-convd', '--conv_dim', type = int, default = 2, help = "d in cconv")
parser.add_argument('-loss', '--loss-metric', type = str, default = "EMDUB", help = "chamfer, EMDUB, sinkhorn")
parser.add_argument('-cgen', '--conditional-generator', type = str, default = "AdaIN", help = "AdaIN, concat, final_selection")
parser.add_argument('-modelnorm', '--model-norm', type = str, default = "None", help = "None, BrN, LN, IN")
parser.add_argument('-maxpconv', '--max-pool-conv', dest = "max_pool_conv", action = 'store_const', default = False, const = True, help = 'Enable max pool conv instead of mean (sum)')
parser.add_argument('-density', '--density-estimation', dest = 'density_estimation', action = 'store_const', default = False, const = True, help = 'Use estimated density (reciprocal) as initial point feature')

parser.add_argument('-fss', '--file-sim-steps', type = int, default = 380, help = "How many sim steps per run")

args = parser.parse_args()

def write_models(array, meta, dirc, name):
    if not os.path.exists(dirc):
        os.makedirs(dirc)
    
    with open(os.path.join(dirc, name), 'w') as model_file:
        for pi in range(array.shape[0]):
            for ci in range(array.shape[1]):
                model_file.write('%f ' % array[pi, ci])
            if meta is not None:
                for mi in range(len(meta)):
                    pCount = array.shape[0] // meta[mi]
                    model_file.write('%d ' % (pi // pCount))
            model_file.write('\n')

dataLoad.maxParticlesPerGrid = args.voxel_size
if args.voxel_length == 0:
    dataLoad.overrideGrid = False
else:
    dataLoad.overrideGrid = True
    dataLoad.overrideGridSize = args.voxel_length

if args.name == "NoName":
    args.name = "[NPY][NoCard][1st2ndmomentEdges(edgeMask,[u;v;edg])][NoPosInVertFeature] E(%s)-D(%s)-%d^3(%d)g%dh%dz-bs%dlr%f-%s" % ("graph", "graph", args.voxel_length, args.voxel_size, args.hidden_dim, args.latent_dim, args.batch_size, args.learning_rate, 'Adam' if args.adam else 'mSGD')

if args.previewName == 'unnamed':
    args.previewName = args.name

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_gpus

logPath = os.path.join(args.log, args.name + "(" + strftime("%Y-%m-%d %H-%Mm-%Ss", gmtime()) + ")/")

AE_load_path = "savedModels/" + args.nameAE + "/"
save_path = "savedModels/" + args.name + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(args.outpath):
    os.makedirs(args.outpath)

# Generate or load model configs

model_config = None

if args.load == "auto" or args.load == "Auto":
    print("Loading model config from %s" % os.path.join(save_path, 'config.json'))
    with open(os.path.join(save_path, 'config.json'), 'r') as jsonFile:
        model_config = json.load(jsonFile)
elif args.config != 'None':
    print("Loading model config from %s" % args.config)
    with open(args.config, 'r') as jsonFile:
        model_config = json.load(jsonFile)
else:
    print("Loading model config from %s" % os.path.join(AE_load_path, 'config.json'))
    with open(os.path.join(AE_load_path, 'config.json'), 'r') as jsonFile:
        model_config = json.load(jsonFile)

if model_config == None:

    model_config =\
    {
        'useVector': False,                             # pending
        'conv': args.conv,                              # pending
        'convd': args.conv_dim,                            # OK
        'loss': args.loss_metric,                              # OK
        'maxpoolconv': args.max_pool_conv,              # OK
        'density_estimate': args.density_estimation,    # pending
        'normalization': args.model_norm,               # OK

        'ccnt': args.cluster_count,
        'cdim': args.cluster_dim,
        'steps': args.loop_sim,

        'encoder': {
            'blocks' : 3,
            'particles_count' : [2048, 512, args.cluster_count],
            'conv_count' : [2, 0, 0],
            'res_count' : [0, 2, 3],
            'kernel_size' : [args.nearest_neighbor, args.nearest_neighbor, args.nearest_neighbor],
            'bik' : [0, 48, 96],
            'channels' : [args.hidden_dim // 2, args.hidden_dim * 2, max(args.latent_dim, args.hidden_dim * 4)],
        },
        'decoder': {
            'blocks' : 1,
            'pcnt' : [2048], # particle count
            'generator' : [5], # Generator depth
            'maxLen' : [0.05],
            'nConv' : [0],
            'nRes' : [0],
            'hdim' : [args.hidden_dim // 3],
            'fdim' : [512], # dim of features used for folding
            'gen_hdim' : [512],
            'knnk' : [args.nearest_neighbor // 2],
            'genStruct' : args.conditional_generator,
            'genFeatures' : True,
        },
        'simulator': {
            'layers': [256],
            'knnk': args.nearest_neighbor,
            'GRU': False,
            'GRU_hd': 256,
            'IN': True,
        },
        'stages': [[0, 0]]
    }

elif 'simulator' not in model_config: # add to AE config

    model_config['ccnt'] = args.cluster_count
    model_config['cdim'] = args.cluster_dim
    model_config['steps'] = args.loop_sim

    model_config['simulator'] = {
        'layers': [256],
        'knnk': args.nearest_neighbor,
        'GRU': False,
        'GRU_hd': 256,
        'IN': True,
    }

with open(os.path.join(save_path, 'config.json'), 'w') as jsonFile:
    json.dump(model_config, jsonFile)

print("Model config saved as %s" % os.path.join(save_path, 'config.json'))

model.default_dtype = args.dtype
# with tf.variable_scope('summaries') as ss:
#     model.summary_scope = ss
# model.default_dtype = tf.float32

# Create the model
if args.adam:
    optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2, epsilon=1e-8)
else:
    optimizer = tf.train.MomentumOptimizer(learning_rate = args.learning_rate, momentum = args.beta1)

if args.dtype == tf.float16:
    loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(16.0, 32)
    # loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(96.0)
    optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

train_val_path = os.path.join(args.datapath, 'train_val/')
test_path = os.path.join(args.datapath, 'test/')

data_header = dataLoad.get_headers(train_val_path, test_path)
_, _, normalize, test_lf, test_rf, test_normalize = data_header

# model = model_net(16, args.latent_dim, args.batch_size, optimizer)
model = model_net(args.voxel_size, args.latent_dim, args.batch_size, optimizer, args.output_dim, model_config)
model.particle_hidden_dim = args.hidden_dim
model.loss_func = args.loss_func
model.combine_method = args.combine_method
model.knn_k = args.nearest_neighbor
model.cluster_feature_dim = args.cluster_dim
model.cluster_count = args.cluster_count
model.doSim = True
model.doLoop = True
model.loops = args.loop_sim

train_loops = args.loop_sim

model.normalize = normalize
if normalize == {}:
    print("No normalization descriptor found ... ")
    model.normalize = {'mean': 0.0, 'std': args.normalize}

# Headers
# headers = dataLoad.read_file_header(dataLoad.get_fileNames(args.datapath)[0])
model.total_world_size = 96.0
model.initial_grid_size = model.total_world_size / 16

# model.initial_grid_size = model.total_world_size / 4

# Build the model
model.build_model()
model.build_test_output([1, 2048, 6], test_normalize, True)

# Summary the variables
ptraps = tf.summary.scalar('Training Loss', model.train_loss)
vals = tf.summary.scalar('Validation Loss', model.val_loss, collections = None)

merged_train = tf.summary.merge([ptraps])
merged_val = tf.summary.merge([vals])

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
sess.run(tf.global_variables_initializer())

# Save & Load
saver = tf.train.Saver()

if args.load == "auto" or args.load == "Auto":
    latest_ckpt = tf.train.latest_checkpoint(save_path)
    if latest_ckpt is not None:
        saver.restore(sess, latest_ckpt)
        print(colored("Check point loaded: %s" % latest_ckpt, 'red'))
elif args.load != "None":
    saver.restore(sess, args.load)
else: # Load AE
    AEsaver = tf.train.Saver(tf.trainable_variables('net/ParticleDecoder'))
    latest_ckpt = tf.train.latest_checkpoint(AE_load_path)
    if latest_ckpt is not None:
        AEsaver.restore(sess, latest_ckpt)
        print(colored("AE Check point loaded: %s" % latest_ckpt, 'red'))

batch_idx_train = 0
batch_idx_test = 0

epoch_idx = 0
iteration = 0

epochs = dataLoad.gen_epochs(args.epochs, data_header, args.batch_size, 0.3, args.loop_sim, args.file_sim_steps)

# TODO: loss weights
loss_weights = np.ones((args.file_sim_steps))
one_weights = np.ones((args.file_sim_steps))
loss_weight_mask = np.power(0.95, [i for i in range(train_loops)])

test_raws, test_lats = dataLoad.get_one_test_file(test_rf[0], test_lf[0], 0, args.file_sim_steps)

if not os.path.exists('./previews/%s/' % args.previewName):
        os.makedirs('./previews/%s/' % args.previewName)

dataLoad.save_npy_to_GRBin(test_raws[0], './previews/%s/gt.grbin' % args.previewName)
dataLoad.save_npy_to_GRBin(test_lats[0], './previews/%s/latent.grbin' % args.previewName)

sess.graph.finalize()

while True:
    
    print("gc.garbage:")
    print(gc.garbage)
    del gc.garbage[:]
    
    batch_train, batch_validate = next(epochs, [None, None])
    epoch_idx += 1

    if batch_train == None:
        break

    print(colored("Epoch %03d" % (epoch_idx), 'yellow'))

    # Training loop
    while True:

        _x, _x_steps = next(batch_train, [None, None])
        if _x is None:
            break

        _x_lweight = loss_weights[_x_steps] * loss_weight_mask
        # print(_x_lweight)

        if batch_idx_train == 10 and args.profile:
            raise NotImplementedError
        
        else:
            
            # feed_dict = { model.ph_X: _x, model.ph_stepweights: _x_lweight }

            # _, n_loss, summary = sess.run([model.train_op, model.train_loss, merged_train], feed_dict = feed_dict)
            # train_writer.add_summary(summary, batch_idx_train)
            
            batch_idx_train += 1

        # print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("It %08d" % batch_idx_train, 'magenta') + ' - ' + colored("Train =%7.4f" % (n_loss), 'green'), end = ' ')

        if batch_idx_train % 20 == 0:
            # _vx, _vx_steps = next(batch_validate, [None, None])
            # _vx_lweight = one_weights[_vx_steps] * loss_weight_mask
            
            # feed_dict = { model.ph_X: _vx, model.ph_stepweights: _vx_lweight }
            
            # n_loss, summary = sess.run([model.val_loss, merged_val], feed_dict = feed_dict)
            # val_writer.add_summary(summary, batch_idx_test * 20)
            batch_idx_test += 1
        else:
            n_loss = 0.0

        print(colored("(val =%7.4f)" % n_loss, 'blue'))

        if batch_idx_train % 1000 == 0:
            cprint("Test simulation in progess ... ", 'cyan')
            test_sim_rec, test_loss = model.get_test_output(test_lats[:, 0, :, :], test_raws, sess, verbose = True)
            np.save(os.path.join(args.outpath, 'rec.npy'), test_sim_rec[0])
            np.save(os.path.join(args.outpath, 'gt.npy'), test_raws[0])
            np.save(os.path.join(args.outpath, 'loss.npy'), test_loss)
            sys.exit(0)
            # dataLoad.save_npy_to_GRBin(test_sim_rec[0], './previews/%s/test-%d-rc.grbin' % (args.previewName, batch_idx_train))

        if batch_idx_train % (16000 // args.batch_size) == 0:
            sav = saver.save(sess, save_path + args.save + ".ckpt", global_step = batch_idx_train)
            print("Checkpoint saved in %s" % (sav))

# Save the network
if(args.save != "None"):
    save_path = saver.save(sess, save_path + "/final.ckpt")
    print("Model saved in %s" % (save_path))
