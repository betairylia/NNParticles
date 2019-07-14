import numpy as np
import os
import sys
import struct
import time
import random
import math

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

def get_fileNames_predict(main_dir):
    
    files = []
    norm = {}

    for filename in os.listdir(main_dir):
        if filename.split('.')[-1] == 'npy':
            if filename == 'mean.npy':
                norm['mean'] = np.load(os.path.join(main_dir, filename)).astype(np.float32)
            elif filename == 'stddev.npy':
                norm['std'] = np.load(os.path.join(main_dir, filename)).astype(np.float32)
            else:
                files.append(os.path.join(main_dir, filename))
    
    return files, norm

def fileCount(path):
    return len(get_fileNames(path))

def gen_batch(data, batch_size, ratio, train_steps = 30, shuffle = True):
    
    fileCnt, steps, N, C = data.shape
    avilable_steps = steps - train_steps + 1

    non_stop = False
    if ratio <= 0.0:
        non_stop = True

    contents  = []
    for i in range(fileCnt):
        for j in range(avilable_steps):
            contents.append([i, j])

    totalIter = fileCnt * avilable_steps

    if shuffle:
        random.shuffle(contents)

    if not non_stop:
        totalIter = math.floor(totalIter * ratio) // batch_size * batch_size
        assert totalIter >= 0
        contents = contents[:totalIter]

    i = 0
    
    batch_X = np.zeros((batch_size, train_steps, N, C))
    batch_steps = np.zeros((batch_size, train_steps), dtype=np.int32)
    
    while True:
        cur_content = contents[i]
        bid = i % batch_size

        for ds in range(train_steps):
            f = contents[i][0]
            s = contents[i][1] + ds
            batch_X[bid, ds, :, :] = data[f, s, :, :]
            batch_steps[bid, ds] = s

        i += 1
        if i % batch_size == 0:
            yield batch_X, batch_steps
        
        if i >= totalIter and not non_stop:
            break

def get_headers(path, test_path):

    files, val, norm = get_fileNames(path)
    tl_files, _ = get_fileNames_predict(os.path.join(test_path, 'latent/'))
    tr_files, tr_norm = get_fileNames_predict(os.path.join(test_path, 'raw/'))

    return files, val, norm, tl_files, tr_files, tr_norm

def to_sep_sims(data, single_file_sim = 380):

    total_steps = data.shape[0]
    N = data.shape[1]
    C = data.shape[2]

    return data.reshape((-1, single_file_sim, N, C))

def gen_epochs(n_epochs, header, batch_size, ratio = 0.3, train_steps = 30, sim_steps = 380):

    files, val, norm, tl_files, tr_files, tr_norm = header

    print("Training set:")
    print(files)
    print('*=*=*')
    print('Validation set:')
    print(val)
    print('*=*=*')
    print('Test set:')
    print(list(zip(tl_files, tr_files)))
    print('*=*=*')
    print('Normalization descriptor (latent):')
    print(norm)
    print('*=*=*')
    print('Normalization descriptor (raw):')
    print(tr_norm)

    if val is not None:
        print("Loading validation set...")
        data_val = to_sep_sims(np.load(val), single_file_sim = sim_steps) # [sims, steps, N, C]
    else:
        data_val = {}

    for i in range(n_epochs):
    # for i in range(n_epochs * len(files)):
        print("Reading data...")
        data = to_sep_sims(np.load(files[i % len(files)]), single_file_sim = sim_steps) # [sims, steps, N, C]
        yield\
            gen_batch(data, batch_size, ratio, train_steps, shuffle = True),\
            gen_batch(data_val, batch_size, -1.0, train_steps, shuffle = True)

def get_one_test_file(raw_file, latent_file, sim_file_idx, sim_steps = 380, raw_particles = 2048):

    raw = to_sep_sims(np.load(raw_file), single_file_sim = sim_steps)
    
    raw = raw[sim_file_idx:(sim_file_idx+1), :, :, :]
    raw = np.transpose(raw, (2, 0, 1, 3))
    raw = np.random.permutation(raw)[:raw_particles]
    raw = np.transpose(raw, (1, 2, 0, 3))
    
    lat = to_sep_sims(np.load(latent_file), single_file_sim = sim_steps)

    return raw, lat[sim_file_idx:(sim_file_idx+1), :, :, :]

def save_npy_to_GRBin(data, dstPath):
    
    assert(len(data.shape) == 3)
    if data.shape[2] > 3:
        data = data[:, :, 0:3]
    
    fileGRBin = open(dstPath, 'wb')

    print('Writing file ' + dstPath)

    fileGRBin.write(int(data.shape[0]).to_bytes(4, byteorder = 'little', signed = False))
    fileGRBin.write(int(data.shape[1]).to_bytes(4, byteorder = 'little', signed = False))
    fileGRBin.write(int(data.shape[2]).to_bytes(4, byteorder = 'little', signed = False))

    for step in range(data.shape[0]):
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                fileGRBin.write(struct.pack('f', data[step, x, y]))

    fileGRBin.close()

    print('Complete!')
