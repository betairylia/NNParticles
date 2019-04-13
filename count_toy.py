import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.prepro import *
from tensorlayer.layers import *
from termcolor import colored, cprint

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

particle_slots = 256
batch_size = 64
act = tf.nn.elu
# act = tf.identity
iterations = 5000

netGraph = [[], [], []]

def gen_batch(particle_prob, noise_scale, noise_mean):
    
    batch_X = np.zeros((batch_size, particle_slots, 7), dtype = np.float32)

    batch_X[:, :, 0] = np.random.binomial(1, particle_prob, (batch_size, particle_slots))
    batch_Y = np.sum(batch_X[:, :, 0], axis = 1)
    batch_X[:, :, 1:4] = np.random.normal(noise_mean, noise_scale, (batch_size, particle_slots, 3))
    batch_X[:, :, 5:8] = 0

    return batch_X, batch_Y

phX = tf.placeholder('float32', [batch_size, particle_slots, 7])
phY = tf.placeholder('float32', [batch_size])

w = tf.Variable(tf.random_normal([7, 1], mean = 0.0, stddev = 0.0))
b = tf.Variable(tf.random_normal([1], mean = 0.0, stddev = 0.0))

phX_stacked = tf.concat(tf.unstack(phX, axis = 0), 0)
latents_stacked = act(tf.add(tf.matmul(phX_stacked, w), b))
# latents_stacked = act(tf.matmul(phX_stacked, w))
print(latents_stacked.shape)
# latent = tf.reduce_sum(tf.reshape(latents_stacked, [batch_size, particle_slots, 1]), 1)

raw_latents = tf.reshape(latents_stacked, [batch_size, particle_slots, 1])
masked_latents = tf.multiply(raw_latents, tf.reshape(phX[:, :, 0], [batch_size, particle_slots, 1]))
latent_grid = tf.reduce_sum(masked_latents, 1)
latent = tf.reduce_sum(tf.stack([latent_grid], 1), 1)

loss = tf.reduce_mean(tf.square(tf.reshape(latent, [batch_size]) - phY))
# train_op = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_op = tf.train.MomentumOptimizer(0.00001, 0.98).minimize(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(iterations):
    
    # _X, _Y = gen_batch(0.5, 0.0, 0.0)
    _X, _Y = gen_batch(np.random.uniform(), 1.0, 0.0)
    # print(_X)
    # print(_Y)
    _, _loss, _w, _b = sess.run([train_op, loss, w, b], feed_dict = {phX: _X, phY: _Y})

    print(colored("Iteration %05d : " % i, 'yellow') + colored("Loss = %.4f" % _loss, 'green'))

    netGraph[0].append(_w[0, 0])
    netGraph[1].append(_w[1, 0])
    netGraph[2].append(_b[0])

colorMap = cm.YlOrBr
colors = []
for i in range(iterations):
    colors.append(colorMap(0.2 + 0.8 * (i / iterations)))

# fig = pyplot.figure()
# ax = Axes3D(fig)

# ax.scatter(netGraph[0], netGraph[1], netGraph[2], c = colors, zorder = 2)
# ax.plot(netGraph[0], netGraph[1], netGraph[2], c = [0.3, 0.3, 0.3], ls = '-', zorder = 1)
# pyplot.show

# plt.scatter(netGraph[0], netGraph[1], c = colors, zorder = 2)
# plt.plot(netGraph[0], netGraph[1], c = [0.3, 0.3, 0.3], ls = '-', zorder = 1)
# plt.show()
# plt.clf()

# plt.scatter(netGraph[1], netGraph[2], c = colors, zorder = 2)
# plt.plot(netGraph[1], netGraph[2], c = [0.3, 0.3, 0.3], ls = '-', zorder = 1)
# plt.show()
# plt.clf()

pyplot.figure(figsize=(8, 8))

pyplot.scatter(netGraph[0], netGraph[2], c = colors, zorder = 2)
pyplot.plot(netGraph[0], netGraph[2], c = [0.3, 0.3, 0.3], ls = '-', zorder = 1)

axes = pyplot.gca()
axes.set_xlim([-0.2, 1.0])
axes.set_ylim([-0.2, 1.0])

pyplot.show()
pyplot.clf()
