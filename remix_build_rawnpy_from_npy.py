import numpy as np

import os
import sys
import struct
import time
import random
import ntpath

import progressbar

FILE_PATH = '/home/betairya/RP_CG/mantaflow/manta/build/dataset_fluids/raws'
# outpath = '/mnt/Datasets/MDSets/LSsim_combined_2560_nonstop/'
# outpath = 'MDSets/LSsim_combined_2560_tst/'
# outpath = 'MDSets/LSsim_combined_2560_validation_ss600/'
# outpath = '/media/betairya/Data Disk/Datasets/MDSet_sim_5_loop30/'
outpath = 'MDSets/fluid_1_l28'
particles = 5120

step_count = 380
start_step = 0
singlefile_sim_count = 4
combines = 2

maxtotalfiles = 1000
shuffle_steps = False

if not os.path.exists(outpath):
    os.makedirs(outpath)

def read_file_override(filename, vM = 1.0):

    file_content = {}
    print('Reading file ' + str(filename))

    file_content['data'] = np.load(filename)

    return file_content

def get_fileNames(main_dir):
    
    files = []
    cnt = 0

    for filename in os.listdir(main_dir):
        splits = ntpath.basename(filename).split('.')
        if splits[-1] == 'pos.txt':
            files[idx][0] = os.path.join(main_dir, filename)
        if splits[-1] == 'vel.txt':
            files[idx][1] = os.path.join(main_dir, filename)
    
    return files

# Get all files
files = get_fileNames(FILE_PATH)
files = files[0:maxtotalfiles]
totalCount = len(files)
npyCount = totalCount // singlefile_sim_count

# sample_header = read_file_header(files[0])
# assert singlefile_sim_count * singlefile_sim_steps == sample_header['stepCount']

ss = 1
if skip_steps == True:
    ss = single_sim_steps

simsteps = step_count - start_step - (long_sim_loops * single_sim_steps)
simsteps = simsteps // ss
readcnt = 0
fcnt = 0

print("Total #Files = %4d, Checking file count..." % totalCount)
assert totalCount % singlefile_sim_count == 0

print("Checking combines...")
assert singlefile_sim_count % combines == 0

print("Legal simulation steps count = %d, Checking..." % simsteps)
assert simsteps % singlefile_sim_count == 0

for i in range(npyCount):
    
    available_sims = min(singlefile_sim_count, totalCount - (i * singlefile_sim_count))
    # particle_array = [np.zeros([simsteps, 3, maxParticlesPerGrid, 6]) for k in range(available_sims)]
    file_contents = []

    # for j in range(0):
    for j in range(available_sims):

        fidx = i * singlefile_sim_count + j
        if fidx >= totalCount:
            break

        file_contents.append(read_file_override(files[fidx]))
        # file_contents.append({})
        readcnt += 1
        print("Read %d / %d" % (readcnt, totalCount))

        file_contents[j]['steporder'] = list(range(start_step, start_step + simsteps * ss, ss))
        
        if shuffle_steps == True:
            random.shuffle(file_contents[j]['steporder'])

    # build all indices
    file_sim_step = simsteps // available_sims
    indices = [list(range(available_sims)) for k in range(available_sims)]
    for idxidx in range(available_sims):
        random.shuffle(indices[idxidx])
    
    # print(indices)
    # input('')

    # Build array
    for j in range(available_sims // combines):

        # print('j %2d' % j)
        particle_array = [np.zeros([simsteps, 3, particles, 6]) for c in range(combines)]
        
        for c in range(combines):

            # print('c %2d' % c)
                
            for k in range(available_sims):
                cur_file_content = file_contents[indices[j*combines+c][k]]
                # print('j*c+c = \t%4d' % (j*combines+c))
                # print('fsims = \t%4d' % file_sim_step)
                # print(cur_file_content['steporder'][((j*combines+c)*file_sim_step):((j*combines+c+1)*file_sim_step)])
                # input('')
                order = np.asarray(cur_file_content['steporder'][((j*combines+c)*file_sim_step):((j*combines+c+1)*file_sim_step)])
                order_Y = order + single_sim_steps
                order_L = order + (long_sim_loops * single_sim_steps)
                particle_array[c][(k*file_sim_step):((k+1)*file_sim_step), 0, :, :] = cur_file_content['data'][order, 0, :, 0:6]
                particle_array[c][(k*file_sim_step):((k+1)*file_sim_step), 1, :, :] = cur_file_content['data'][order_Y, 0, :, 0:6]
                particle_array[c][(k*file_sim_step):((k+1)*file_sim_step), 2, :, :] = cur_file_content['data'][order_L, 0, :, 0:6]

        final_arr = np.concatenate(particle_array, axis = 0)
        np.save(os.path.join(outpath, 'combined_%d.npy' % fcnt), final_arr)
        print("Output file %s" % ('combined_%d.npy' % fcnt))
        fcnt += 1
    
