import numpy as np
import os
import sys
import struct
import time
import random

import progressbar

particleCount = 2560
maxParticlePerGrid = 30

# np.set_printoptions(suppress=True)

def get_fileNames(main_dir):
    
    files = []

    for filename in os.listdir(main_dir):
        if filename.split('.')[-1] == 'mdset':
            files.append(os.path.join(main_dir, filename))
    
    return files

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

    file_content['worldLength'] = float(file_content['gridSize'] * file_content['gridCountX'])

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

def read_file(filename, vM = 1.0):

    file = open(filename, 'rb')
    file_content = {}

    print('Reading file ' + filename)

    # Read file header
    file_content['gridCountX'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    file_content['gridCountY'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
    file_content['gridCountZ'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

    file_content['gridCount'] = file_content['gridCountX'] * file_content['gridCountY'] * file_content['gridCountZ']

    file_content['gridSize'] = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

    file_content['worldLength'] = float(file_content['gridSize'] * file_content['gridCountX'])

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

    # Start a progress bar
    bar = progressbar.ProgressBar(maxval = file_content['stepCount'], widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    # create data buffer

    # Data format: 6 - [locationX, locationY, locationZ, velocityX, velocityY, velocityZ, identitybit]
    file_content['data'] = np.zeros((file_content['stepCount'], particleCount, 7))

    currentIndex = 0

    # Read file content
    for step in range(file_content['stepCount']):

        current_step = int.from_bytes(file.read(4), byteorder = 'little', signed = False)
        currentIndex = 0

        for grid in range(file_content['gridCount']):

            gridParticleCount = int.from_bytes(file.read(4), byteorder = 'little', signed = False)

            byte_data = file.read(24 * gridParticleCount)

            # Read particles
            for particle in range(gridParticleCount):

                # Position
                (file_content['data'][current_step, currentIndex, 0], file_content['data'][current_step, currentIndex, 1],\
                 file_content['data'][current_step, currentIndex, 2], file_content['data'][current_step, currentIndex, 3],\
                 file_content['data'][current_step, currentIndex, 4], file_content['data'][current_step, currentIndex, 5]) =\
                struct.unpack('6f', byte_data[particle * 24 : particle * 24 + 24])

                file_content['data'][current_step, currentIndex, 3] *= vM
                file_content['data'][current_step, currentIndex, 4] *= vM
                file_content['data'][current_step, currentIndex, 5] *= vM

                file_content['data'][current_step, currentIndex, 6] = 1

                file_content['data'][current_step, currentIndex, 1] -= (file_content['worldLength'] / 2) # Center Y position to 0

                currentIndex += 1

        bar.update(current_step)
    
    bar.finish()

    return file_content

def fileCount(path):
    return len(get_fileNames(path))
    
def gen_batch(content, batch_size, step_count):
    
    # batch_X = np.zeros((batch_size, 16, 3))
    # batch_Y = np.zeros((batch_size, 16, 3))

    batch_X = np.zeros((batch_size, particleCount, 7))
    batch_Y = np.zeros((batch_size, particleCount, 7))
    batch_Y_progress = np.zeros((batch_size, ))

    batch_voxels = np.zeros((8 * batch_size, maxParticlePerGrid, 7))
    batch_voxels_card = np.zeros((8 * batch_size))
    batch_voxels_pos = np.zeros((8 * batch_size, 4), dtype = np.int32)

    batch_idx = 0

    # shuffle the steps
    steps = list(range(content['stepCount'] - step_count))
    random.shuffle(steps)
    # steps = steps[0:(len(steps) // 4)]

    print(steps[0])

    for step in steps:

        # batch_X[batch_idx, :, :] = content['data'][step, 0:16, 0:3]
        # batch_Y[batch_idx, :, :] = content['data'][step + step_count, 0:16, 0:3]

        batch_X[batch_idx, :, :] = content['data'][step, :, 0:7]
        batch_Y[batch_idx, :, :] = content['data'][step + step_count, :, 0:7]
        batch_Y_progress[batch_idx] = 2.0 * (step / content['stepCount']) - 1.0

        # Choose from X
        pid = np.random.randint(0, particleCount, size = [8,]) # select random particle
        p = np.zeros((8, 3), dtype = np.int32)
        voxelSize = content['worldLength'] / 16.0
        p[:, 0] = content['data'][step, pid, 0] // voxelSize
        p[:, 1] = content['data'][step, pid, 1] // voxelSize
        p[:, 2] = content['data'][step, pid, 2] // voxelSize # get corresponding voxel
        
        batch_voxels[(8 * batch_idx):(8 * (batch_idx + 1)), :, :] = 0 # reset dirty array
        batch_voxels_card[(8 * batch_idx):(8 * (batch_idx + 1))] = 0

        batch_voxels_pos[(8 * batch_idx):(8 * (batch_idx + 1)), 0] = batch_idx
        batch_voxels_pos[(8 * batch_idx):(8 * (batch_idx + 1)), 1] = p[:, 0] + 8
        batch_voxels_pos[(8 * batch_idx):(8 * (batch_idx + 1)), 2] = p[:, 1] + 8
        batch_voxels_pos[(8 * batch_idx):(8 * (batch_idx + 1)), 3] = p[:, 2] + 8
        
        for particle in range(particleCount):
            
            pX = content['data'][step, particle, 0:7] # particle from X
            gpX = pX[0:3] // voxelSize # voxel of particle from X

            pX[0:3] = np.mod(pX[0:3], voxelSize) - (voxelSize / 2)
            
            for ii in range(8):
                if int(gpX[0]) == p[ii, 0] and int(gpX[1]) == p[ii, 1] and int(gpX[2]) == p[ii, 2]: # our target grid
                    batch_voxels[8 * batch_idx + ii, int(batch_voxels_card[8 * batch_idx + ii]), :] = pX
                    batch_voxels_card[8 * batch_idx + ii] += 1
                        
            # pY = content['data'][step + step_count, particle, 0:6] # particle from Y
            # gpY = pY[0:3] % (content['worldLength'] / 16.0) # grid of particle from Y
            
            # for ii in range(4,8):
            #     if gpY[0] * (content['gridCountX'] ** 2) + gpY[1] * content['gridCountX'] + gpY[2] == p[ii]: # our target grid
            #         batch_voxels[8 * batch_idx + ii, batch_voxels_card[8 * batch_idx + ii]] = pY
            #         batch_voxels_card[8 * batch_idx + ii] += 1

        # Count batch
        batch_idx += 1
        if batch_idx >= batch_size:
            batch_idx = 0
            yield batch_X, batch_Y, batch_Y_progress, batch_voxels, batch_voxels_card, batch_voxels_pos

def gen_epochs(n_epochs, path, batch_size, step_count, vM, valLoop):

    files = get_fileNames(path)

    for i in range(n_epochs):

        content = read_file(files[i % len(files)], step_count * vM)
        
        # Validation
        if ((i % len(files)) % valLoop == (valLoop - 1)): 
            yield (gen_batch(content, batch_size, step_count), False)

        # Train
        else:
            yield (gen_batch(content, batch_size, step_count), True)

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

            gridPosX = grid // (content['gridCountZ'] * content['gridCountX']) * content['gridSize'] - (content['gridCountX'] // 2 * content['gridSize'])
            gridPosY = grid % (content['gridCountZ'] * content['gridCountX']) // content['gridCountZ'] * content['gridSize']
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
