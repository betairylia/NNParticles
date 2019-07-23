#!/usr/bin/env python
# coding: utf-8

# In[87]:


import tensorflow as tf
import ot
import ot.gpu


# In[5]:


import os
os.listdir()


# In[9]:


import sys
sys.path.append('..')

from external.structural_losses.tf_approxmatch import approx_match, match_cost
from external.sampling.tf_sampling import farthest_point_sample, prob_sample


# In[16]:


import numpy as np
import matplotlib.pylab as pl
from matplotlib import pyplot as plt
import ot.plot
from sklearn.datasets import make_moons, make_circles


# In[37]:


def get_ref(n):
    X, y = make_moons(n_samples = n * 2, shuffle = False, noise = 0.1)
    ref = X[:n] # Reference samples
    return ref

def get_rec(n, good_ratio):
    good_count = int(n * good_ratio)
    bad_count  = n - good_count

    good_rec = np.random.normal([0.0, 0.0], [0.3, 0.3], size = (good_count, 2))
    bad_rec  = np.random.normal([1.0,-1.0], [1.0, 0.1], size = ( bad_count, 2))
    rec = np.concatenate([good_rec, bad_rec], axis = 0)
    
    return rec


# In[101]:


n = 32
ref = get_ref(n)
rec_1 = get_rec(n, 0.2)
rec_2 = get_rec(n, 0.9)
rec = get_rec(n, 0.5)


# In[102]:


plt.scatter(ref[:, 0], ref[:, 1], color = 'black', marker = 'x')
plt.scatter(rec_1[:, 0], rec_1[:, 1], color = 'blue')
plt.scatter(rec_2[:, 0], rec_2[:, 1], color = 'red')
plt.scatter(rec[:, 0], rec[:, 1], color = 'green')


# In[47]:


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[123]:


def chamfer(n, ref, rec):
    plan = np.zeros((n, n))
    dist = np.sqrt(np.sum(np.square(np.reshape(ref, (1, n, 2)) - np.reshape(rec, (n, 1, 2))), axis = -1))
    np.fill_diagonal(dist, 10.0)
    
    min_ref = np.argmin(dist, axis = 1)
    min_rec = np.argmin(dist, axis = 0)
    
    for i, j in zip(range(n), min_rec):
        plan[i, j] = 1.0
        pass
        
    for i, j in zip(min_ref, range(n)):
        plan[i, j] = 1.0
        pass
    
    plan /= plan.sum()
    
#     plt.imshow(plan)
    
    return plan, (plan * dist).sum()

def sinkhorn(n, ref, rec):
    dist = np.sqrt(np.sum(np.square(np.reshape(ref, (1, n, 2)) - np.reshape(rec, (n, 1, 2))), axis = -1))
    a = np.ones((n,)) / n
    b = np.ones((n,)) / n
    plan = ot.sinkhorn(a, b, dist / dist.max(), 2e-3)
#     plt.imshow(plan)
    return plan, (plan * dist).sum()

def exactEMD(n, ref, rec):
    dist = np.sqrt(np.sum(np.square(np.reshape(ref, (1, n, 2)) - np.reshape(rec, (n, 1, 2))), axis = -1))
    a = np.ones((n,)) / n
    b = np.ones((n,)) / n
    plan = ot.emd(a, b, dist)
#     plt.imshow(plan)
    print(plan.sum())
    return plan, (plan * dist).sum()

def aEMD(n, ref, rec):
    ref_3d = np.zeros((1, n, 3))
    rec_3d = np.zeros((1, n, 3))
    ref_3d[0, :, 0:2] = ref
    rec_3d[0, :, 0:2] = rec
    
    ph_ref = tf.placeholder(tf.float32, [1, n, 3])
    ph_rec = tf.placeholder(tf.float32, [1, n, 3])
    match = approx_match(ph_ref, ph_rec)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    
    plan = sess.run(match, feed_dict = {ph_ref: ref_3d, ph_rec: rec_3d})[0, :, :]
    plan /= plan.sum()
#     plt.imshow(plan)
    
    dist = np.sqrt(np.sum(np.square(np.reshape(ref, (1, n, 2)) - np.reshape(rec, (n, 1, 2))), axis = -1))
        
    return plan, (plan * dist).sum()


# In[127]:


# recs = [rec_1, rec_2, rec]
recs = [rec]
fcnt = 0
for rc in recs:
    pds = [chamfer(n, ref, rc), sinkhorn(n, rec, rc), exactEMD(n, rec, rc), aEMD(n, rec, rc)]
    name = ['Chamfer pseudo-', 'Sinkhorn ', 'exact Earth\'s Mover ', 'Aunction-based EM ']

    for p in range(len(pds)):
        plan, dist = pds[p]
        title = name[p] + 'distance, value = ' + '%6f' % dist
        
        pl.figure(fcnt)
        fcnt += 1
        ot.plot.plot2D_samples_mat(ref, rc, plan, color = [.5, .5, 1])
        pl.plot(ref[:, 0], ref[:, 1], 'ob', label = 'Reference points')
        pl.plot(rc[:, 0],  rc[:, 1],  'xr', label = 'Reconstructed points')
        pl.legend(loc = 0)
        pl.title(title)

        pl.show()

# In[ ]:




