import numpy as np

import os
import sys
import struct
import time
import random
import ntpath

import progressbar

FILE_PATH = 'MDSets/2560_BigGrid/'
outpath = '/mnt/93196856-2b8a-4c7b-a37b-6affb441b6e0/Datasets/LJP_timeorder/'
maxParticlesPerGrid = 2560
particles = 2560

skip_steps = 5
combines = 20

maxtotalfiles = 1000
shuffle = False

if not os.path.exists(outpath):
    os.makedirs(outpath)

def packGridHash(content, x, y, z):

    x = x % content['gridCountX']
    y = y % content['gridCountY']
    z = z % content['gridCountZ']

    return y * content['gridCountZ'] * content['gridCountX'] + x * content['gridCountZ'] + z

def read_file_header(filename):

    file = open(filename, 'rb')
    file_content = {}

    print('Reading file ' + filename)

    # Read file header
    file_content['gridCountX'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    file_content['gridCountY'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    file_content['gridCountZ'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

    file_content['gridCount'] = file_content['gridCountX'] * file_content['gridCountY'] * file_content['gridCountZ']

    file_content['gridSize'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

    tmp = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    if tmp == 0:
        file_content['gravity'] = False
    else:
        file_content['gravity'] = True

    tmp = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    if tmp == 0:
        file_content['boundary'] = False
    else:
        file_content['boundary'] = True

    file_content['stepCount'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

    return file_content

def read_file_override(filename, vM = 1.0):

    file = open(filename, 'rb')
    file_content = {}

    print('Reading file ' + filename)

    # Read file header
    file_content['gridCountX'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    file_content['gridCountY'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    file_content['gridCountZ'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    file_content['originalGridCount'] = file_content['gridCountX'] * file_content['gridCountY'] * file_content['gridCountZ']

    file_content['gridSize'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

    file_content['worldSize'] = file_content['gridCountX'] * file_content['gridSize']
    overrideGridSize = file_content['worldSize'] # Single grid (use all particles at once)

    tmp = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    if tmp == 0:
        file_content['gravity'] = False
    else:
        file_content['gravity'] = True

    tmp = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    if tmp == 0:
        file_content['boundary'] = False
    else:
        file_content['boundary'] = True

    file_content['stepCount'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

    # Override part
    file_content['gridSize'] = overrideGridSize
    file_content['gridCountX'] = file_content['worldSize'] // file_content['gridSize']
    file_content['gridCountY'] = file_content['worldSize'] // file_content['gridSize']
    file_content['gridCountZ'] = file_content['worldSize'] // file_content['gridSize']

    file_content['gridCount'] = file_content['gridCountX'] * file_content['gridCountY'] * file_content['gridCountZ']
    print("overrided grid count: %d" % file_content['gridCount'])

    # Start a progress bar
    bar = progressbar.ProgressBar(maxval = file_content['stepCount'], widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    # create data buffer

    # Data format: 6 - [locationX, locationY, locationZ, velocityX, velocityY, velocityZ]
    file_content['data'] = np.zeros((file_content['stepCount'], file_content['gridCount'], maxParticlesPerGrid, 6), dtype = np.float32)

    file_content['particleCount'] = np.zeros((file_content['stepCount'], file_content['gridCount']), dtype = np.int32)
    file_content['maxParticles'] = 0
    pCount = np.zeros((1,), dtype = np.int32)

    # Read file content
    for step in range(file_content['stepCount']):

        current_step = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

        for grid in range(file_content['originalGridCount']):

            pCount[0] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
            # file_content['particleCount'][current_step, grid] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
            # print(pCount)

            byte_data = file.read(24 * pCount[0])

            # Read particles
            for particle in range(pCount[0]):

                (pX, pY, pZ, vX, vY, vZ) = struct.unpack('6f', byte_data[particle * 24 : particle * 24 + 24])
                pX = (pX + file_content['worldSize'] / 2)
                pZ = (pZ + file_content['worldSize'] / 2)
                gX = int(pX // file_content['gridSize'])
                gY = int(pY // file_content['gridSize'])
                gZ = int(pZ // file_content['gridSize'])

                # grid it belongs to
                true_grid = packGridHash(file_content, gX, gY, gZ)
                # print(true_grid)

                gridPosX = gX * file_content['gridSize']
                gridPosY = gY * file_content['gridSize']
                gridPosZ = gZ * file_content['gridSize']

                # TODO: Data normalization
                curIdx = file_content['particleCount'][current_step, true_grid]
                if curIdx >= maxParticlesPerGrid:
                    curIdx = maxParticlesPerGrid - 1

                file_content['data'][current_step, true_grid, curIdx, 0] = pX - gridPosX - file_content['gridSize'] / 2
                file_content['data'][current_step, true_grid, curIdx, 1] = pY - gridPosY - file_content['gridSize'] / 2
                file_content['data'][current_step, true_grid, curIdx, 2] = pZ - gridPosZ - file_content['gridSize'] / 2

                file_content['data'][current_step, true_grid, curIdx, 3] = vX * vM
                file_content['data'][current_step, true_grid, curIdx, 4] = vY * vM
                file_content['data'][current_step, true_grid, curIdx, 5] = vZ * vM

                file_content['particleCount'][current_step, true_grid] += 1

                # np.set_printoptions(edgeitems = 16, suppress = True, precision = 2)
                # print("#%5d in g%6d(%+2d, %+2d, %+2d): %+3.2f %+3.2f %+3.2f -> %s" % (file_content['particleCount'][current_step, true_grid], true_grid, gX, gY, gZ, pX, pY, pZ, str(file_content['data'][current_step, true_grid, curIdx, :])))

                if(file_content['particleCount'][current_step, true_grid] > file_content['maxParticles']):
                    file_content['maxParticles'] = file_content['particleCount'][current_step, true_grid]

        bar.update(current_step)
    
    bar.finish()

    if file_content['maxParticles'] > maxParticlesPerGrid:
        print("Overflow - %d particles in a grid !" % (file_content['maxParticles']))

    return file_content

def get_fileNames(main_dir):
    
    files = []

    for filename in os.listdir(main_dir):
        if filename.split('.')[-1] == 'mdset':
            files.append(os.path.join(main_dir, filename))
    
    return files

# Get all files
files = get_fileNames(FILE_PATH)
files = files[0:maxtotalfiles]
totalCount = len(files)
npyCount = totalCount // combines

sample_header = read_file_header(files[0])

simsteps = (sample_header['stepCount']) // skip_steps
readcnt = 0
fcnt = 0

for i in range(npyCount):
    
    available_sims = combines
    particle_array = []

    # for j in range(0):
    for j in range(available_sims):

        fidx = i * combines + j
        if fidx >= totalCount:
            break

        particle_array.append(read_file_override(files[fidx])['data'][::skip_steps, 0, :, :])
        # file_contents.append({})
        readcnt += 1
        print("Read %d / %d" % (readcnt, totalCount))

    # Build array
    for j in range(available_sims // combines):

        # print('j %2d' % j)
        final_arr = np.concatenate(particle_array, axis = 0)
        if shuffle:
            final_arr = np.random.permutation(final_arr)
        np.save(os.path.join(outpath, 'combined_%d.npy' % fcnt), final_arr)
        print("Output file %s" % ('combined_%d.npy' % fcnt))
        fcnt += 1
    
