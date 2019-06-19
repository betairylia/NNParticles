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

from termcolor import colored, cprint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
sys.path.append(os.path.join(ROOT_DIR, '../'))

import model_rawFC as model_net

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
parser.add_argument('-preview', '--previewName', type = str, default = "unnamed", help = "Name for save preview point clouds")
parser.add_argument('-save', '--save', type = str, default = "model", help = "Path to store trained model")
parser.add_argument('-load', '--load', type = str, default = "None", help = "File to load to continue training")
parser.add_argument('-debug', '--debug', dest = "enable_debug", action = 'store_const', default = False, const = True, help = "Enable debugging")
parser.add_argument('-prof', '--profile', dest = "profile", action = 'store_const', default = False, const = True, help = "Enable profiling (at step 10)")
# parser.add_argument('-prof', '--profile', type = str, default = "None", help = "Path to store profiling timeline (at step 100)")

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

# Create the model
ph_X = tf.placeholder(tf.float32, [args.batch_size, 5120, 4])
normalized_X = ph_X[:, :, 0:3] / args.normalize

with tf.variable_scope('net', reuse = False):
    train_rec, train_ep = model_net.get_model(ph_X, True, 0.98)
with tf.variable_scope('net', reuse = True):
    val_rec, val_ep = model_net.get_model(ph_X, False, 0.98)

train_loss, train_ep = model_net.get_loss(train_rec, normalized_X, train_ep)
val_loss, val_ep = model_net.get_loss(val_rec, normalized_X, val_ep)
optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate, beta1 = args.beta1, beta2 = args.beta2, epsilon=1e-8)
train_op = optimizer.minimize(train_loss)

val_rec_out = val_rec * args.normalize

# Summary the variables
tras = tf.summary.scalar('Training Loss', train_loss)
vals = tf.summary.scalar('Validation Loss', val_loss)

merged_train = tf.summary.merge([tras])
merged_val = tf.summary.merge([vals])

# Create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

if args.enable_debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

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

batch_idx_train = 0
batch_idx_test = 0

epoch_idx = 0
iteration = 0

epCount = dataLoad.fileCount(args.datapath)
stepFactor = 9

epochs = dataLoad.gen_epochs(args.epochs, args.datapath, args.batch_size, args.velocity_multiplier, True, args.output_dim)

while True:
    batch_train, batch_validate = next(epochs, [None, None])
    epoch_idx += 1

    if batch_train == None:
        break

    print(colored("Epoch %03d" % (epoch_idx), 'yellow'))

    # Training loop
    while True:

        _x, _x_size = next(batch_train, [None, None])
        if _x == None:
            break

        if batch_idx_train == 10 and args.profile:
            raise NotImplementedError
        
        else:
            
            feed_dict = { ph_X: _x[0] }
            _, n_loss, summary = sess.run([train_op, train_loss, merged_train], feed_dict = feed_dict)
            train_writer.add_summary(summary, batch_idx_train)
            
            batch_idx_train += 1

        print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("It %08d" % batch_idx_train, 'magenta') + ' - ', end = '')
        print(colored("Train =%7.4f" % (i, n_loss), 'green'), end = ' ')

        _vx, _vx_size = next(batch_validate, [None, None])
        
        feed_dict = { ph_X: _vx[0] }
        
        if batch_idx_test % 100 == 0:

            n_loss, summary, _rec = sess.run([val_loss, merged_val, val_rec_out[0, :, :]], feed_dict = feed_dict)
            _gt = _vx[0][0, :, 0:3]
            
            val_writer.add_summary(summary, batch_idx_test)
            
            write_models(_rec, None, './previews/%s' % args.previewName, 'validation-%d-rec.asc' % batch_idx_test)
            write_models(_gt, None, './previews/%s' % args.previewName, 'validation-%d-gt.asc' % batch_idx_test)

        else:
            
            n_loss, summary = sess.run([val_loss, merged_val], feed_dict = feed_dict)
            val_writer.add_summary(summary, batch_idx_test)
        
        batch_idx_test += 1

        print(colored("(val =%7.4f)" % n_loss, 'blue'))

        if batch_idx_train % (16000 // args.batch_size) == 0:
            sav = saver.save(sess, save_path + args.save + ".ckpt", global_step = batch_idx_train)
            print("Checkpoint saved in %s" % (sav))

# Save the network
if(args.save != "None"):
    save_path = saver.save(sess, "savedModels/" + args.save + ".ckpt")
    print("Model saved in %s" % (save_path))
