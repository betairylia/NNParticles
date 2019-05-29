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

import model_graph_sim as model
from model_graph_sim import model_particles as model_net
# import dataLoad_particleTest as dataLoad                        # Legacy method, strongly disagree with i.i.d. distribution among batch(epoch)es.
import dataLoad_graph as dataLoad            # New method, shuffle & mixed randomly

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
parser.add_argument('-vm', '--velocity-multiplier', type = float, default = 1.0, help = "Multiplies the velocity (input[..., 3:]) by this factor")
parser.add_argument('-norm', '--normalize', type = float, default = 1.0, help = "stddev of input data")

parser.add_argument('-zdim', '--latent-dim', type = int, default = 512, help = "Length of the latent vector")
parser.add_argument('-hdim', '--hidden-dim', type = int, default = 64, help = "Length of the hidden vector inside network")
parser.add_argument('-cdim', '--cluster-dim', type = int, default = 32, help = "How many neighbors should be considered in the graph network")
parser.add_argument('-ccnt', '--cluster-count', type = int, default = 64, help = "How many neighbors should be considered in the graph network")
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
normalized_X = model.ph_X / args.normalize
normalized_Y = model.ph_Y / args.normalize
cpos, cfea, poolX, evalsX = model.build_predict_Enc(normalized_X, True, False)
if args.dosim:
    cpos_Y, cfea_Y, poolY, evalsY = model.build_predict_Enc(normalized_Y, False, True)

pRange = 3
outDim = args.output_dim

ph_cpos = tf.placeholder(args.dtype, [args.batch_size, args.cluster_count, pRange])
ph_cfea = tf.placeholder(args.dtype, [args.batch_size, args.cluster_count, args.cluster_dim])
if args.dosim:
    spos, sfea = model.build_predict_Sim(ph_cpos, ph_cfea, False, False)
prec, precf, ___l = model.build_predict_Dec(ph_cpos, ph_cfea, normalized_X[:, :, 0:outDim], True, False,  outDim = outDim)
if args.dosim:
    rec , recf, loss = model.build_predict_Dec(   spos,    sfea, normalized_Y[:, :, 0:outDim], False, True, outDim = outDim)

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

save_path = "savedModels/" + args.name + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save & Load
saver = tf.train.Saver()

# You should load a trained model
# sess.run(tf.local_variables_initializer())
# # tl.layers.initialize_global_variables(sess)
# sess.run(tf.global_variables_initializer())

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

maxl_array = np.zeros((2))
maxl_array[0] = args.voxel_size
maxl_array[1] = args.voxel_size

epCount = dataLoad.fileCount(args.datapath)
stepFactor = 9

bs = args.batch_size
N = args.voxel_size
totalIterations = 90

groundTruth = np.zeros((totalIterations * bs, N, outDim))
groundTrutX = np.zeros((totalIterations * bs, N, outDim))
reconstruct = np.zeros((totalIterations * bs, N, outDim))
# fold        = np.zeros((totalIterations * bs, N, outDim))

# clusters_X  = np.zeros((totalIterations * bs, args.cluster_count, pRange))
# clusters_Y  = np.zeros((totalIterations * bs, args.cluster_count, pRange))

totalIterations -= 1

# pgt = tf.placeholder('float32', [8, N, pRange])
# prc = tf.placeholder('float32', [8, N, pRange])
# pl = model.chamfer_metric(pgt / args.normalize, prc / args.normalize, pRange, tf.abs) * 40.0

# pools = [np.zeros((totalIterations * bs, model.pCount[i+1], pRange)) for i in range(model.pool_count)]
# evals = [np.zeros((totalIterations * bs, model.pCount[i]  , pRange + 1)) for i in range(model.pool_count)]

for epoch_train, epoch_validate in dataLoad.gen_epochs(args.epochs, args.datapath, args.batch_size, args.velocity_multiplier, False, args.output_dim):

    epoch_idx += 1
    print(colored("Epoch %03d" % (epoch_idx), 'yellow'))

    # Train
    ecnt = 0
    for _x, _x_size in epoch_train:

        # print(args.dosim)

        # Initial batch - compute latent clusters
        if ecnt == 0 and args.dosim:
            _cpos, _cfea, eX = sess.run([cpos, cfea, evalsX], feed_dict = { model.ph_X: _x[0] })
            _spos, _sfea = _cpos, _cfea
            _rec, n_loss = sess.run([prec, ___l], feed_dict = { ph_cpos: _spos, ph_cfea: _sfea, model.ph_card: _x_size, model.ph_X: _x[0], model.ph_max_length: maxl_array })

            groundTruth[0, :, :] = _x[0][:, :, 0:outDim]
            reconstruct[0, :, :] = _rec[:, :, 0:outDim]
            batch_idx_train += 1
            
            print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("   Train   It %08d" % batch_idx_train, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))

        ecnt += 1
        
        if args.dosim:
            # Simulation
            _spos, _sfea, _rec, n_loss = sess.run([spos, sfea, rec, loss], feed_dict = { ph_cpos: _spos, ph_cfea: _sfea, model.ph_card: _x_size, model.ph_Y: _x[1], model.ph_max_length: maxl_array })
            
            # Get encoded features & clusters
            # _cpos_x, _cfea_x = sess.run([  cpos,   cfea], feed_dict = { model.ph_X: _x[0] })
            # _cpos_y, _cfea_y = sess.run([cpos_Y, cfea_Y], feed_dict = { model.ph_Y: _x[1] })
        else:
            # Just do auto-encoder
            _cpos, _cfea, pX, eX = sess.run([cpos, cfea, poolX, evalsX], feed_dict = {model.ph_X: _x[0]})
            # print(_cfea)
            _rec, _recf, n_loss = sess.run([prec, precf, ___l], feed_dict = { ph_cpos: _cpos, ph_cfea: _cfea, model.ph_card: _x_size, model.ph_X: _x[0], model.ph_max_length: maxl_array })

        sidx = batch_idx_train * bs
        eidx = (batch_idx_train + 1) * bs
        batch_idx_train += 1

        print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("   Train   It %08d" % batch_idx_train, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))

        groundTrutX[sidx:eidx, :, :] = _x[0][:, :, 0:outDim]
        if args.dosim:
            groundTruth[sidx:eidx, :, :] = _x[1][:, :, 0:outDim]
        reconstruct[sidx:eidx, :, :] = _rec[:, :, 0:outDim]
        # fold[sidx:eidx, :, :] = _recf[:, :, 0:outDim]

        if not args.dosim:
            for i in range(len(pX)):
                pass
                # pools[i][sidx:eidx] = pX[i]
                # evals[i][sidx:eidx] = eX[i]

        # if args.dosim:
            # clusters_X[ sidx:eidx, :, :] = _cpos_x[:, :, 0:pRange] * 48.0
            # clusters_Y[ sidx:eidx, :, :] = _cpos_y[:, :, 0:pRange] * 48.0
        # else:
            # clusters_X[ sidx:eidx, :, :] = _cpos[:, :, 0:pRange] * args.normalize

        if batch_idx_train >= (totalIterations + 1):
            break

    # Test
    # for _x, _x_size in epoch_validate:
    #     feed_dict = { model.ph_X: _x, model.ph_card: _x_size, model.ph_max_length: maxl_array }
    #     n_loss, summary = sess.run([model.val_particleLoss, merged_val], feed_dict = feed_dict)
    #     val_writer.add_summary(summary, round(batch_idx_test * stepFactor))
    #     batch_idx_test += 1

    #     print(colored("Ep %04d" % epoch_idx, 'yellow') + ' - ' + colored("Validation It %08d" % batch_idx_test, 'magenta') + ' - ' + colored(" Loss = %03.4f" % n_loss, 'green'))

outpath = os.path.join('MDSets/results/ShapeNet_CC', args.name)

if not os.path.exists(outpath):
    os.makedirs(outpath)

np.save(os.path.join(outpath, 'rc.npy'), reconstruct)
# np.save(os.path.join(outpath, 'rf.npy'), fold)
np.save(os.path.join(outpath, 'gX.npy'), groundTrutX)
# if args.dosim:
    # np.save(os.path.join(outpath, 'eX.npy'), eX[0])
# else:
    # for i in range(len(pools)):
        # np.save(os.path.join(outpath, 'p%d.npy' % i), pools[i])
        # np.save(os.path.join(outpath, 'e%d.npy' % i), evals[i])
# np.save(os.path.join(outpath, 'cX.npy'), clusters_X)

if args.dosim:
    np.save(os.path.join(outpath, 'gt.npy'), groundTruth)
    # np.save(os.path.join(outpath, 'cY.npy'), clusters_Y)

# Generate CC ASC files
if outDim > 3:
    with open(os.path.join(outpath, 'gt.asc'), 'w') as fgt:
        for i in range(N):
            fgt.write("%f %f %f %f %f %f\n" % (groundTruth[0, i, 0], groundTruth[0, i, 1], groundTruth[0, i, 2], groundTruth[0, i, 3], groundTruth[0, i, 4], groundTruth[0, i, 5]))

    with open(os.path.join(outpath, 'rc.asc'), 'w') as fgt:
        for i in range(N):
            fgt.write("%f %f %f %f %f %f\n" % (reconstruct[0, i, 0], reconstruct[0, i, 1], reconstruct[0, i, 2], reconstruct[0, i, 3], reconstruct[0, i, 4], reconstruct[0, i, 5]))

# print("Loss check")
# print(sess.run(pl, feed_dict = {pgt: groundTrutX[0:8], prc: reconstruct[0:8]}))
# print(reconstruct)

