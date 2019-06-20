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
    val = None
    norm = {}

    for filename in os.listdir(main_dir):
        if filename.split('.')[-1] == 'npy':
            if filename == 'val.npy':
                val = os.path.join(main_dir, filename)
            elif filename == 'mean.npy':
                norm['mean'] = np.load(os.path.join(main_dir, filename)).astype(np.float32)
            elif filename == 'stddev.npy':
                norm['std'] = np.load(os.path.join(main_dir, filename)).astype(np.float32)
            else:
                files.append(os.path.join(main_dir, filename))
    
    if val == None:
        val = files[len(files) - 1]
        del files[len(files) - 1]

    if len(files) <= 0:
        files.append(val)
        val = None

    return files, val, norm

def fileCount(path):
    return len(get_fileNames(path))

def gen_batch(content, batch_size, vM, is_Train = True, shuffle = True):
     
    assert content['gridSize'] == maxParticlesPerGrid or print("content - %s (%s); vSize - %d" % (str(content['gridSize']), str(content['data'].shape), maxParticlesPerGrid))

    batch_X = np.zeros((batch_size, maxParticlesPerGrid, content['dim'] + 1))
    batch_Y = np.zeros((batch_size, maxParticlesPerGrid, content['dim'] + 1))
    batch_L = np.zeros((batch_size, maxParticlesPerGrid, content['dim'] + 1))
    batch_X_size = np.zeros((batch_size))
    batch_idx = 0
    avgCard = 0

    noSim = False
    if content['data'].ndim == 3:
        noSim = True

    # shuffle the steps
    steps = list(range(content['stepCount']))

    # in order to get validation set "pure" from training set, shuffling with a fixed order (maybe it is not that pure so whatever)
    if shuffle == True:
        random.shuffle(steps)
        # random.Random(8246).shuffle(steps)
    # steps = steps[0:(len(steps) // 4)]
    
    if is_Train:
        for step in steps:
            
            if noSim == False:
                
                # print("NoSim == False <!>")
                # print("Content['data']: %s" % str(content['data'].shape))
                # print("Slice: %s" % str(content['data'][step, 0, :, 0:content['dim']].shape))

                batch_X[batch_idx, :, 0:content['dim']] = content['data'][step, 0, :, 0:content['dim']]
                batch_Y[batch_idx, :, 0:content['dim']] = content['data'][step, 1, :, 0:content['dim']]
                batch_L[batch_idx, :, 0:content['dim']] = content['data'][step, 2, :, 0:content['dim']]

                batch_X[batch_idx, :, 3:content['dim']] *= vM        
                batch_Y[batch_idx, :, 3:content['dim']] *= vM
                batch_L[batch_idx, :, 3:content['dim']] *= vM

                # FIXME: How to store grid cards?
                batch_X[batch_idx, :, content['dim']] = 1
                batch_Y[batch_idx, :, content['dim']] = 1
                batch_L[batch_idx, :, content['dim']] = 1

                batch_X_size[batch_idx] = maxParticlesPerGrid
                avgCard += batch_X_size[batch_idx]

            else:
                # print("Content['data']: %s" % str(content['data'].shape))
                # print("Slice: %s" % str(content['data'][step, :, 0:content['dim']].shape))

                batch_X[batch_idx, :, 0:content['dim']] = content['data'][step, :, 0:content['dim']]
                batch_X[batch_idx, :, 3:content['dim']] *= vM
                batch_X[batch_idx, :, content['dim']] = 1
                batch_X_size[batch_idx] = maxParticlesPerGrid
                avgCard += batch_X_size[batch_idx]

            # Count batch
            batch_idx += 1
            if batch_idx >= batch_size:
                batch_idx = 0

                print("%6.2f" % (avgCard / batch_size), end = ' ')
                avgCard = 0

                yield [batch_X, batch_Y, batch_L], batch_X_size
    else:
        i = 0
        while True:
            
            step = steps[i % len(steps)]
            i += 1

            if noSim == False:
                batch_X[batch_idx, :, 0:content['dim']] = content['data'][step, 0, :, 0:content['dim']]
                batch_Y[batch_idx, :, 0:content['dim']] = content['data'][step, 1, :, 0:content['dim']]
                batch_L[batch_idx, :, 0:content['dim']] = content['data'][step, 2, :, 0:content['dim']]

                batch_X[batch_idx, :, 3:content['dim']] *= vM        
                batch_Y[batch_idx, :, 3:content['dim']] *= vM
                batch_L[batch_idx, :, 3:content['dim']] *= vM

                # FIXME: How to store grid cards?
                batch_X[batch_idx, :, content['dim']] = 1
                batch_Y[batch_idx, :, content['dim']] = 1
                batch_L[batch_idx, :, content['dim']] = 1

                batch_X_size[batch_idx] = maxParticlesPerGrid
                # avgCard += batch_X_size[batch_idx]

            else:
                batch_X[batch_idx, :, 0:content['dim']] = content['data'][step, :, 0:content['dim']]
                batch_X[batch_idx, :, 3:content['dim']] *= vM
                batch_X[batch_idx, :, content['dim']] = 1
                batch_X_size[batch_idx] = maxParticlesPerGrid
                # avgCard += batch_X_size[batch_idx]

            # Count batch
            batch_idx += 1
            if batch_idx >= batch_size:
                batch_idx = 0

                # print("Avg card = %6.2f" % (avgCard / batch_size), end = ' ')
                # avgCard = 0

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

def gen_epochs(n_epochs, path, batch_size, vM, shuffle = True, dim = 0):

    files, val, norm = get_fileNames(path)
    # for i in range(len(files)):
    #     read_file(files[i])

    print("Training set:")
    print(files)
    print('*=*=*')
    print('Validation set:')
    print(val)
    print('*=*=*')
    print('Normalization descriptor:')
    print(norm)

    if val is not None:
        print("Loading validation set...")
        data_val = np.load(val)
        if (data_val.shape[1] if data_val.ndim == 3 else data_val.shape[2]) >= maxParticlesPerGrid:
            if data_val.ndim == 3:
                data_new = np.zeros((data_val.shape[0], maxParticlesPerGrid, data_val.shape[2]))
            else:
                data_new = np.zeros((data_val.shape[0], data_val.shape[1], maxParticlesPerGrid, data_val.shape[3]))

            for i in range(data_val.shape[0]):
                if data_val.ndim == 3:
                    data_new[i] = np.random.permutation(data_val[i])[:maxParticlesPerGrid]
                else:
                    for j in range(data_val.shape[1]):
                        data_new[i, j] = np.random.permutation(data_val[i, j])[:maxParticlesPerGrid]

            del data_val
            data_val = data_new
            del data_new

        if data_val.ndim == 3:
            content_val = {'data': data_val, 'stepCount': data_val.shape[0], 'gridSize': data_val.shape[1], 'dim': data_val.shape[2]}
        else:
            content_val = {'data': data_val, 'stepCount': data_val.shape[0], 'gridSize': data_val.shape[2], 'dim': data_val.shape[3]}
        if dim > 0: content_val['dim'] = dim
    else:
        content_val = {}

    for i in range(n_epochs):
    # for i in range(n_epochs * len(files)):
        print("Reading data...")
        data = np.load(files[i % len(files)]) # [step, 3, gridCount, 6(channels)]
        if (data.shape[1] if data.ndim == 3 else data.shape[2]) >= maxParticlesPerGrid:
            if data.ndim == 3:
                data_new = np.zeros((data.shape[0], maxParticlesPerGrid, data.shape[2]))
            else:
                data_new = np.zeros((data.shape[0], data.shape[1], maxParticlesPerGrid, data.shape[3]))
            
            for i in range(data.shape[0]):
                if data.ndim == 3:
                    data_new[i] = np.random.permutation(data[i])[:maxParticlesPerGrid]
                else:
                    for j in range(data.shape[1]):
                        data_new[i, j] = np.random.permutation(data[i, j])[:maxParticlesPerGrid]

            del data
            data = data_new
            del data_new

        if data.ndim == 3:
            content = {'data': data, 'stepCount': data.shape[0], 'gridSize': data.shape[1], 'dim': data.shape[2]}
        else:
            content = {'data': data, 'stepCount': data.shape[0], 'gridSize': data.shape[2], 'dim': data.shape[3]}
        if dim > 0: content['dim'] = dim
        # content = read_file(files[0], step_count * vM)
        yield gen_batch(content, batch_size, vM, is_Train = True, shuffle = shuffle), gen_batch(content_val, batch_size, vM, is_Train = False, shuffle = shuffle)

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
