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

import model_graph_final as model
from model_graph_final import model_particles as model_net

# import model_graph as model
# from model_graph import model_particles as model_net

# import dataLoad_particleTest as dataLoad                        # Legacy method, strongly disagree with i.i.d. distribution among batch(epoch)es.
import dataLoad_graph as dataLoad            # New method, shuffle & mixed randomly

from time import gmtime, strftime

import progressbar

from tensorflow.python import debug as tf_debug

from tensorflow.python.client import timeline
from tensorflow.contrib.tensorboard.plugins import projector

parser = argparse.ArgumentParser(description="Run the NN for particle simulation")

parser.add_argument('datapath')
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
parser.add_argument('-nloop', '--no-loop', dest = 'doloop', action='store_const', default = False, const = True, help = "Don't loop simulation regularization")
parser.add_argument('-nsim', '--no-sim', dest = 'dosim', action='store_const', default = False, const = True, help = "Don't do Simulation")

parser.add_argument('-log', '--log', type = str, default = "logs", help = "Path to log dir")
parser.add_argument('-name', '--name', type = str, default = "NoName", help = "Name to show on tensor board")
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
parser.add_argument('-lvec', '--vector-latent', dest = "use_vector", action = 'store_const', default = False, const = True, help = 'Use latent vector instead of graph')
parser.add_argument('-deep', '--deep-model', dest = "deep", action = 'store_const', default = False, const = True, help = 'Use deep encoder model')

parser.add_argument('-latent', '--latent-code', dest = 'latent_code', action='store_const', default = False, const = True, help = "Store latent code instead of reconstruction results")

args = parser.parse_args()

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

save_path = "savedModels/" + args.name + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Generate or load model configs

# args.outpath = os.path.join(args.outpath, args.name)
# if not os.path.exists(args.outpath):
#     os.makedirs(args.outpath)

model_config = None

if args.load != "auto":
    print("Please use -load auto for any case. THANKS!!")
    raise NotImplementedError

if args.load == "auto" or args.load == "Auto":
    print("Loading model config from %s" % os.path.join(save_path, 'config.json'))
    with open(os.path.join(save_path, 'config.json'), 'r') as jsonFile:
        model_config = json.load(jsonFile)

elif args.config != 'None':
    print("Loading model config from %s" % args.config)
    with open(args.config, 'r') as jsonFile:
        model_config = json.load(jsonFile)

if model_config == None:
    model_config =\
    {
        'useVector': args.use_vector,                   # OK
        'conv': args.conv,                              # OK
        'convd': args.conv_dim,                         # OK
        'loss': args.loss_metric,                       # OK
        'maxpoolconv': args.max_pool_conv,              # OK
        'density_estimate': args.density_estimation,    # OK
        'normalization': args.model_norm,               # OK
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
            'generator' : [6 if args.use_vector else 5], # Generator depth
            'maxLen' : [0.0 if args.use_vector else 0.05],
            'nConv' : [0],
            'nRes' : [0],
            'hdim' : [args.hidden_dim // 3],
            'fdim' : [512], # dim of features used for folding
            'gen_hdim' : [512],
            'knnk' : [args.nearest_neighbor // 2],
            'genStruct' : args.conditional_generator,
            'genFeatures' : True,
        },
        'stages': [[0, 0]]
    }

    if args.deep == True:
        model_config['encoder'] = {
            'blocks': 5,
            'particles_count': [2048, 768, 256, 96, args.cluster_count],
            'conv_count': [2, 1, 1, 0, 0],
            'res_count': [0, 1, 2, 3, 4],
            'kernel_size': [args.nearest_neighbor, args.nearest_neighbor, args.nearest_neighbor, args.nearest_neighbor, args.nearest_neighbor],
            'bik': [0, 48, 48, 48, 48],
            'channels': [args.hidden_dim // 2, args.hidden_dim, args.hidden_dim * 2, args.hidden_dim * 3, max(args.latent_dim, args.hidden_dim * 4)],
        }

model.default_dtype = args.dtype

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

_, _, normalize = dataLoad.get_fileNames(args.datapath)

# model = model_net(16, args.latent_dim, args.batch_size, optimizer)
model = model_net(args.voxel_size, args.latent_dim, args.batch_size, optimizer, args.output_dim, model_config)
model.particle_hidden_dim = args.hidden_dim
model.loss_func = args.loss_func
model.combine_method = args.combine_method
model.knn_k = args.nearest_neighbor
model.cluster_feature_dim = args.cluster_dim
model.cluster_count = args.cluster_count
model.doSim = args.dosim
model.doLoop = args.dosim and args.doloop
model.loops = args.loop_sim

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

# Create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

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

save_path = "savedModels/" + args.name + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save & Load
saver = tf.train.Saver()

if args.load == "auto" or args.load == "Auto":
    latest_ckpt = tf.train.latest_checkpoint(save_path)
    if latest_ckpt is not None:
        saver.restore(sess, latest_ckpt)
        print("Check point loaded: %s" % latest_ckpt)
elif args.load != "None":
    saver.restore(sess, args.load)

# prepare data
bs = 2048
grid_count  = 32
grid_size   = 0.15
kernel_name = 'net/ParticleEncoder/enc0/conv_first/gconv'
channels = model_config['encoder']['channels'][0]
# channels = 512
full_kernel = True

grid_lspc = np.linspace(-grid_size, grid_size, grid_count)
gX, gY, gZ = np.meshgrid(grid_lspc, grid_lspc, grid_lspc)
grid_data = np.stack((gX, gY, gZ), axis = -1)
grid_data = np.reshape(grid_data, (-1, 3))

fCh = model_config['convd']
ph = tf.placeholder(args.dtype, [bs, 3])
_kernel = model.getKernelEmbeddings(ph, channels, fCh, None, kernel_name, full_kernel)
totalCnt = grid_count ** 3

result_kernel = np.zeros((totalCnt, channels, _kernel.get_shape()[2]), np.float16)
batch_feed = np.zeros((bs, 3))

for bid in range(math.ceil(totalCnt / bs)):

    print("%8d / %8d" % (bid * bs, totalCnt))

    batch_start = bid * bs
    batch_end   = (bid + 1) * bs
    batch_end   = min(batch_end, totalCnt)

    batch_feed[:, :] = 0
    batch_feed[0:batch_end - batch_start, :] = grid_data[batch_start:batch_end, :]
    _res = sess.run(_kernel, feed_dict = {ph: batch_feed})
    result_kernel[batch_start:batch_end, :, :] = _res[0:batch_end - batch_start, :, :]

result_kernel = np.reshape(result_kernel, (grid_count, grid_count, grid_count, channels, _kernel.get_shape()[2]))
np.save('activation_kernel/kernels/%s_%s_%s.npy' % (args.name, kernel_name.replace('/', '_'), 'full' if full_kernel else 'spatial'), result_kernel)
