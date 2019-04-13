import numpy as np
import scipy
import time
import math
import argparse
import random
import sys
import os

import dataLoad
import progressbar

from termcolor import colored, cprint

from time import gmtime, strftime

parser = argparse.ArgumentParser(description="Convert data to density maps")

parser.add_argument('datapath')
parser.add_argument('outpath')
parser.add_argument('-size', '--data-size', type = int, default = 32)
parser.add_argument('-sigma', '--particle-sigma', type = float, default = 1.0)
parser.add_argument('-count', '--particle-sample-count', type = int, default = 16)
parser.add_argument('-maxl', '--particle-max-length', type = int, default = 200, help = "Max particles in a single grid")

args = parser.parse_args()

dataLoad.maxParticlesPerGrid = args.particle_max_length

files = dataLoad.get_fileNames(args.datapath)

for pFile in files:
    
    content = dataLoad.read_file(pFile, 1.0)

    steps = content['stepCount']
    gCount = content['gridCount']
    dataArray = np.zeros((steps, args.data_size, args.data_size, args.data_size))

    dm_gridSize = (content['gridCountX'] * content['gridSize']) / args.data_size

    print(colored("Start converting file", 'green'))
    
    # Start a progress bar
    bar = progressbar.ProgressBar(maxval = content['stepCount'], widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for step in range(steps):

        for grid in range(gCount):
            
            (gridX, gridY, gridZ) = dataLoad.unpackGridHash(content, grid)
            pCount = content['particleCount'][step, grid]

            for particle in range(pCount):
                
                particlePos = content['data'][step, grid, particle, 0:3]
                particlePos += [gridX * content['gridSize'], gridY * content['gridSize'], gridZ * content['gridSize']]
                particlePos += [0.5 * content['gridSize'], 0.5 * content['gridSize'], 0.5 * content['gridSize']]

                for samples in range(args.particle_sample_count):
                    
                    particlePosCpy = np.copy(particlePos)
                    particlePosCpy += np.random.normal(0.0, args.particle_sigma, particlePosCpy.shape)

                    pX = int( particlePosCpy[0] // dm_gridSize ) % args.data_size
                    pY = int( particlePosCpy[1] // dm_gridSize ) % args.data_size
                    pZ = int( particlePosCpy[2] // dm_gridSize ) % args.data_size

                    dataArray[step, pX, pY, pZ] += 1.0 / args.particle_sample_count

        bar.update(step)
    
    bar.finish()
    
    rawName = pFile.split('\\')[-1].split('.')[0]
    np.save(os.path.join(args.outpath, rawName), dataArray)

print(colored("Complete.", 'yellow'))
