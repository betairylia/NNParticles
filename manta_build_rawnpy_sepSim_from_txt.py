import numpy as np

import os
import sys
import struct
import time
import random
import ntpath

import progressbar

FILE_PATH = '/home/betairya/RP_CG/mantaflow/manta/build/dataset_fluids/raws'
outpath = 'MDSets/fluids_sepSim/'
particles = 5120

step_count = 380
start_step = 0
combines = 20

maxtotalfiles = 1000

if not os.path.exists(outpath):
    os.makedirs(outpath)

def read_file_override(filename, vM = 1.0):

    file_content = {}
    print('Reading files ' + str(filename))

    file_content['data'] = np.zeros((step_count, particles // 2, 6), np.float32)

    bar = progressbar.ProgressBar(maxval = step_count * 2, widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    with open(filename[0], 'r') as posFile:
        
        lines = posFile.readlines()
        assert len(lines) == (particles + 2) * step_count

        l = 0
        for s in range(step_count):
            l += 1
            for p in range(particles):
                if p % 2 == 0:
                    l += 1
                    continue
                arr = lines[l].split(' ')[0][1:-1]
                pos_arr = arr.split(',')
                file_content['data'][s, p // 2, 0:3] = np.asarray([float(pos_arr[0]), float(pos_arr[1]), float(pos_arr[2])])
                l += 1
            l += 1
            bar.update(s)
    
    with open(filename[1], 'r') as velFile:
        
        lines = velFile.readlines()
        assert len(lines) == (particles + 2) * step_count

        l = 0
        for s in range(step_count):
            l += 1
            for p in range(particles):
                if p % 2 == 0:
                    l += 1
                    continue
                arr = lines[l].split(']')[0][1:]
                vel_arr = arr.split(',')
                file_content['data'][s, p // 2, 3:6] = np.asarray([float(vel_arr[0]), float(vel_arr[1]), float(vel_arr[2])])
                l += 1
            l += 1
            bar.update(s + step_count)

    bar.finish()

    return file_content

def get_fileNames(main_dir):
    
    files = []
    cnt = 0

    for filename in os.listdir(main_dir):
        cnt += 1

    for i in range(cnt // 2):
        files.append(['', ''])

    for filename in os.listdir(main_dir):
        splits = ntpath.basename(filename).split('_')
        idx = int(splits[0])
        if splits[-1] == 'pos.txt':
            files[idx][0] = os.path.join(main_dir, filename)
        if splits[-1] == 'vel.txt':
            files[idx][1] = os.path.join(main_dir, filename)
    
    return files

# Get all files
files = get_fileNames(FILE_PATH)
files = files[0:maxtotalfiles]
totalCount = len(files)
npyCount = totalCount // combines

simsteps = step_count
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

        particle_array.append(read_file_override(files[fidx])['data'])
        # file_contents.append({})
        readcnt += 1
        print("Read %d / %d" % (readcnt, totalCount))

    # Build array
    for j in range(available_sims // combines):

        # print('j %2d' % j)
        final_arr = np.concatenate(particle_array, axis = 0)
        np.save(os.path.join(outpath, 'combined_%d.npy' % fcnt), final_arr)
        print("Output file %s" % ('combined_%d.npy' % fcnt))
        fcnt += 1
    
