import numpy as np
import sys
import os

srcdir = sys.argv[1]

def get_files(main_dir):

    files = []

    for filename in os.listdir(main_dir):
        if filename.split('.')[-1] == 'npy':
            files.append(os.path.join(main_dir, filename))

    return files

files = get_files(srcdir)

first = np.load(files[0])
shape = first.shape

# [N, 5120, dim] or [N, 5120, 3, dim]
N = len(files)
dim = shape[-1]
means = np.zeros((N, dim))
varis = np.zeros((N, dim))
pcnts = np.zeros((N, ))

mean = np.zeros((dim,))
std = np.zeros((dim,))
cnt = 0.0

np.set_printoptions(precision = 4, suppress = True, sign = ' ')

for i in range(len(files)):
   
    print("%4d / %4d - %s" % (i, len(files), files[i]))

    cur_file = np.load(files[i])
    pcnts[i] = cur_file.shape[0]

    if len(cur_file.shape) == 4:
        ax = (0, 1, 2)
    else:
        ax = (0, 1)

    means[i, :] = np.mean(cur_file, axis = ax)
    varis[i, :] =  np.var(cur_file, axis = ax)

    print(means[i, :])
    print(np.sqrt(varis[i, :]))

    del cur_file

for i in range(len(files)):
    mean = mean + pcnts[i] * means[i, :]
    cnt += pcnts[i]

mean = mean / float(cnt)

for i in range(len(files)):
    std = std + pcnts[i] * (varis[i, :] + np.power((means[i, :] - mean), 2.0))

std = std / float(cnt)
std = np.sqrt(std)

print("Mean")
print(mean)
print("Std")
print(std)

np.save(os.path.join(srcdir, 'mean.npy'), mean)
np.save(os.path.join(srcdir, 'stddev.npy'), std)

