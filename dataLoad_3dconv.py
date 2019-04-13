import numpy as np
import os
import sys
import struct
import time
import random

import progressbar

channels = 1

def get_fileNames(main_dir):
    
    files = []

    for filename in os.listdir(main_dir):
        if filename.split('.')[-1] == 'npy':
            files.append(os.path.join(main_dir, filename))
    
    return files

def gen_batch(content, batch_size, step_count):
    
    steps_total = content.shape[0]

    if channels == 1 and len(content.shape) == 4:
        content = np.reshape(content, (-1, content.shape[1], content.shape[2], content.shape[3], 1))

    assert content.shape[4] == channels

    batch_X = np.zeros((batch_size, content.shape[1], content.shape[2], content.shape[3], content.shape[4]))
    batch_Y = np.zeros((batch_size, content.shape[1], content.shape[2], content.shape[3], content.shape[4]))

    # shuffle the steps
    steps = list(range(steps_total - step_count))
    random.shuffle(steps)
    steps = steps[0:batch_size * (len(steps) // batch_size)]

    batch_idx = 0

    # Fill the batches
    for step in steps:

        batch_X[batch_idx] = content[step]
        batch_Y[batch_idx] = content[step + step_count]

        # Count batch
        batch_idx += 1
        if batch_idx >= batch_size:
            batch_idx = 0
            yield batch_X, batch_Y

def fileCount(path):
    return len(get_fileNames(path))

def gen_epochs(n_epochs, path, batch_size, step_count, valLoop):

    files = get_fileNames(path)

    for i in range(n_epochs):

        content = np.load(files[i % len(files)])
        
        # Validation
        if ((i % len(files)) % valLoop == (valLoop - 1)): 
            yield (gen_batch(content, batch_size, step_count), False)

        # Train
        else:
            yield (gen_batch(content, batch_size, step_count), True)

# TODO: Change following functions for 3d convolution version
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
                
                x = content['data'][i, grid, particle + 1, 0] + gridPosX + (content['gridSize'] / 2)
                y = content['data'][i, grid, particle + 1, 1] + gridPosY + (content['gridSize'] / 2)
                z = content['data'][i, grid, particle + 1, 2] + gridPosZ + (content['gridSize'] / 2)

                vX = content['data'][i, grid, particle + 1, 3] / vM
                vY = content['data'][i, grid, particle + 1, 4] / vM
                vZ = content['data'][i, grid, particle + 1, 5] / vM

                file.write(struct.pack('6f', x, y, z, vX, vY, vZ))
            
        bar.update(i)
    
    bar.finish()

    return
