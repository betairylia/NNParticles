import numpy as np
import os
import sys
import struct
import time
import random

import progressbar

maxParticlesPerGrid = 200
overrideGrid = True
overrideGridSize = 48

# np.set_printoptions(suppress=True)

def get_fileNames(main_dir):
    
    files = []

    for filename in os.listdir(main_dir):
        if filename.split('.')[-1] == 'npy':
            files.append(os.path.join(main_dir, filename))
    
    return files

def fileCount(path):
    return len(get_fileNames(path))

def gen_batch(content, batch_size, vM, is_Train = True, shuffle = True):
    
    if is_Train == True:
        start = 0.0
        # end = 0.9
        end = 1.0
    else:
        # start = 0.9
        start = 1.0
        end = 1.0
    
    assert content['gridSize'] == maxParticlesPerGrid

    batch_X = np.zeros((batch_size, maxParticlesPerGrid, 7))
    batch_Y = np.zeros((batch_size, maxParticlesPerGrid, 7))
    batch_L = np.zeros((batch_size, maxParticlesPerGrid, 7))
    batch_X_size = np.zeros((batch_size))
    batch_idx = 0
    avgCard = 0

    # shuffle the steps
    steps = list(range(content['stepCount']))

    # in order to get validation set "pure" from training set, shuffling with a fixed order (maybe it is not that pure so whatever)
    if shuffle == True:
        random.Random(8246).shuffle(steps)
    # steps = steps[0:(len(steps) // 4)]
    
    for step in steps:
        
        batch_X[batch_idx, :, 0:6] = content['data'][step, 0, :, 0:6]
        batch_Y[batch_idx, :, 0:6] = content['data'][step, 1, :, 0:6]
        batch_L[batch_idx, :, 0:6] = content['data'][step, 2, :, 0:6]
        
        batch_X[batch_idx, :, 3:6] *= vM
        batch_Y[batch_idx, :, 3:6] *= vM
        batch_L[batch_idx, :, 3:6] *= vM

        # FIXME: How to store grid cards?
        batch_X[batch_idx, :, 6] = 1
        batch_Y[batch_idx, :, 6] = 1
        batch_L[batch_idx, :, 6] = 1

        batch_X_size[batch_idx] = maxParticlesPerGrid
        avgCard += batch_X_size[batch_idx]

        # Count batch
        batch_idx += 1
        if batch_idx >= batch_size:
            batch_idx = 0

            print("Avg card = %6.2f" % (avgCard / batch_size), end = ' ')
            avgCard = 0

            yield [batch_X, batch_Y, batch_L], batch_X_size
                
def gen_batch_predict(content, batch_size, currentStep, step_count):
    
    batch_X = np.zeros((batch_size, 27, maxParticlesPerGrid, 6))
    batch_Y = np.zeros((batch_size, maxParticlesPerGrid, 6))
    batch_Y_size = np.zeros((batch_size))
    batch_idx = 0
    
    for grid in range(int(content['gridCount'])):

        (gridX, gridY, gridZ) = unpackGridHash(content, grid)

        # Fill training data from 3x3x3 neighboors
        for xx in range(-1, 2):
            for yy in range(-1, 2):
                for zz in range(-1, 2):
                    offsetHash = (xx + 1) * 3 * 3 + (yy + 1) * 3 + (zz + 1)
                    batch_X[batch_idx, offsetHash, :, 0:3] = content['data'][currentStep, packGridHash(content, gridX + xx, gridY + yy, gridZ + zz), :, 0:3]
                    batch_X[batch_idx, offsetHash, :, 3:6] = np.asarray([xx, yy, zz])

        batch_Y[batch_idx, :, 0:3] = content['data'][currentStep + step_count, grid, :, 0:3]
        batch_Y_size[batch_idx] = content['particleCount'][currentStep + step_count, grid]

        # Count batch
        batch_idx += 1
        if batch_idx >= batch_size:
            batch_idx = 0
            yield batch_X, batch_Y, batch_Y_size
    
    batch_idx = 0
    yield batch_X, batch_Y, batch_Y_size

def gen_epochs(n_epochs, path, batch_size, vM, shuffle = True):

    files = get_fileNames(path)
    # for i in range(len(files)):
    #     read_file(files[i])

    for i in range(n_epochs):
    # for i in range(n_epochs * len(files)):
        print("Reading data...")
        data = np.load(files[i % len(files)]) # [step, 3, gridCount, 6(channels)]
        content = {'data': data, 'stepCount': data.shape[0], 'gridSize': data.shape[2]}
        # content = read_file(files[0], step_count * vM)
        yield gen_batch(content, batch_size, vM, is_Train = True, shuffle = shuffle), gen_batch(content, batch_size, vM, is_Train = False, shuffle = shuffle)

def gen_epochs_predict(path, start_step, batch_size, step_count, vM):

    content = read_file_predict(path, start_step, step_count * vM)

    i = start_step - 1
    while i < (content['stepCount'] - step_count):
        yield gen_batch_predict(content, batch_size, i, step_count), content
        i += step_count

def write_content(content, step, gridHash, particleArray, particleCount):
    if(gridHash < content['gridCount']):
        content['data'][step, gridHash, 1:particleCount+1, 0:6] = particleArray[0:particleCount, 0:6]

def save_file(content, filename, vM):
    file = open(filename, 'wb')

    print('Writing file ' + filename)

    # File header
    file.write(content['gridCountX'].to_bytes(4, byteorder = 'little', signed = False))
    file.write(content['gridCountY'].to_bytes(4, byteorder = 'little', signed = False))
    file.write(content['gridCountZ'].to_bytes(4, byteorder = 'little', signed = False))

    file.write(content['gridSize'].to_bytes(4, byteorder = 'little', signed = False))

    if content['gravity'] == True:
        file.write((1).to_bytes(4, byteorder = 'little', signed = False))
    else:
        file.write((0).to_bytes(4, byteorder = 'little', signed = False))

    if content['boundary'] == True:
        file.write((1).to_bytes(4, byteorder = 'little', signed = False))
    else:
        file.write((0).to_bytes(4, byteorder = 'little', signed = False))

    file.write(content['stepCount'].to_bytes(4, byteorder = 'little', signed = False))

    # Start a progress bar
    bar = progressbar.ProgressBar(maxval = content['stepCount'], widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i in range(content['stepCount']):

        # Current step
        file.write(i.to_bytes(4, byteorder = 'little', signed = False))

        # Grids
        for grid in range(content['gridCount']):
            
            file.write(content['particleCount'][i, grid].tobytes())

            gridPosX = grid % (content['gridCountZ'] * content['gridCountX']) // content['gridCountZ'] * content['gridSize'] - (content['gridCountX'] // 2 * content['gridSize'])
            gridPosY = grid // (content['gridCountZ'] * content['gridCountX']) * content['gridSize']
            gridPosZ = grid % content['gridCountZ'] * content['gridSize'] - (content['gridCountZ'] // 2 * content['gridSize'])

            for particle in range(content['particleCount'][i, grid]):
                
                x = content['data'][i, grid, particle, 0] + gridPosX + (content['gridSize'] / 2)
                y = content['data'][i, grid, particle, 1] + gridPosY + (content['gridSize'] / 2)
                z = content['data'][i, grid, particle, 2] + gridPosZ + (content['gridSize'] / 2)

                vX = content['data'][i, grid, particle, 3] / vM
                vY = content['data'][i, grid, particle, 4] / vM
                vZ = content['data'][i, grid, particle, 5] / vM

                file.write(struct.pack('6f', x, y, z, vX, vY, vZ))
            
        bar.update(i)
    
    bar.finish()

    return
