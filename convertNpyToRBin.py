import numpy as np

import os
import sys
import struct
import time
import random

import progressbar

srcPath = sys.argv[1]
dstPath = sys.argv[2]

data = np.load(srcPath)

assert(len(data.shape) == 4)

file = open(dstPath, 'wb')

print('Writing file ' + dstPath)

file.write(int(data.shape[0]).to_bytes(4, byteorder = 'little', signed = False))
file.write(int(data.shape[1]).to_bytes(4, byteorder = 'little', signed = False))
file.write(int(data.shape[2]).to_bytes(4, byteorder = 'little', signed = False))
file.write(int(data.shape[3]).to_bytes(4, byteorder = 'little', signed = False))

bar = progressbar.ProgressBar(maxval = data.shape[0], widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

for step in range(data.shape[0]):
    for x in range(data.shape[1]):
        for y in range(data.shape[2]):
            for z in range(data.shape[3]):
                file.write(struct.pack('f', data[step, x, y, z]))
    bar.update(step)
bar.finish

file.close()
