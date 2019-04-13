import numpy as np

import os
import sys
import struct
import time
import random

import progressbar

srcPath = sys.argv[1]

if len(sys.argv) > 2:
    dstPath = sys.argv[2]
else:
    dstPath = '.'.join(srcPath.split('.')[:-1] + ['grbin'])

data = np.load(srcPath)

assert(len(data.shape) == 3)

if data.shape[2] > 3:
    data = data[:, :, 0:3]

file = open(dstPath, 'wb')

print('Writing file ' + dstPath)

file.write(int(data.shape[0]).to_bytes(4, byteorder = 'little', signed = False))
file.write(int(data.shape[1]).to_bytes(4, byteorder = 'little', signed = False))
file.write(int(data.shape[2]).to_bytes(4, byteorder = 'little', signed = False))

bar = progressbar.ProgressBar(maxval = data.shape[0], widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

for step in range(data.shape[0]):
    for x in range(data.shape[1]):
        for y in range(data.shape[2]):
            file.write(struct.pack('f', data[step, x, y]))
    bar.update(step)
bar.finish

file.close()
