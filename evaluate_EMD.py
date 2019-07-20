import numpy as np
import ot
import os
import sys
from tqdm import tqdm

np.random.seed(18754)
numParticles = 2048

ref_dir = 'MDSets/ShapeNet_CC_Official/test/'
rec_dir_prefix = 'paper_results/'
model_name = sys.argv[1]
rec_dir = os.path.join(rec_dir_prefix, model_name + '/')
rec_loss_dir = os.path.join(rec_dir, 'loss/')

load_from_file = True
random_gen = False
optimal_pc = False

if model_name == 'random':
    random_gen = True
    load_from_file = False
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)

if model_name == 'optimal':
    optimal_pc = True
    load_from_file = False
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)

if not os.path.exists(rec_loss_dir):
    os.makedirs(rec_loss_dir)

def EMD(ref, rec):
    
    ref = np.random.permutation(ref)[:2048]
    rec = np.random.permutation(rec)[:2048]

    a = np.ones((2048,)) / 2048.
    b = np.ones((2048,)) / 2048.

    row = np.reshape(ref, (2048, 1, 3))
    col = np.reshape(rec, (1, 2048, 3))

    M = row - col
    M = np.square(M)
    M = np.sum(M, axis = 2)
    M = np.sqrt(M)

    loss = ot.emd2(a, b, M)
    return loss

def get_fileNames(main_dir):
    
    files = []

    for filename in os.listdir(main_dir):
        if 'mean' in filename or 'stddev' in filename:
            continue
        if filename.split('.')[-1] == 'npy':
            files.append(os.path.join(main_dir, filename))
    
    print(files)
    return files

rfFiles = get_fileNames(ref_dir)
totalCnt = 10259

t = tqdm(total = totalCnt)
totalLoss = 0.0
cnt = 0

for rf in rfFiles:
    
    ref_npy = np.load(rf)

    if load_from_file == True:
        rec_npy = np.load(os.path.join(rec_dir, os.path.basename(rf)))
    if random_gen == True:
        mean_std = [np.load(os.path.join(ref_dir, 'mean.npy')), np.load(os.path.join(ref_dir, 'stddev.npy'))]

    # print(ref_npy.shape[0])
    # print(rec_npy.shape[0])
    # assert ref_npy.shape[0] == rec_npy.shape[0]
    
    chunk_loss = []

    for i in range(min(ref_npy.shape[0], (rec_npy.shape[0] if load_from_file else ref_npy.shape[0]))):
        
        _ref = ref_npy[i]
        
        if load_from_file == True:
            _rec = rec_npy[i]
        if random_gen == True:
            _rec = np.random.normal(np.reshape(mean_std[0], (1, 3)), np.reshape(mean_std[1], (1, 3)), size = (2048, 3))
        if optimal_pc == True:
            _r = np.random.permutation(_ref)
            _ref = _r[: 2048]
            _rec = _r[-2048:]

        loss = EMD(_ref, _rec)
        chunk_loss.append(loss)
        totalLoss += loss
        cnt += 1
        t.update(1)

        t.set_postfix(cur = loss, avg = totalLoss / cnt, refresh = True)
    
    np.save(os.path.join(rec_loss_dir, os.path.basename(rf)), np.array(chunk_loss))

print("Total file count: %d" % cnt)
print("Expected total file count: %d" % totalCnt)

print(totalLoss / cnt)
with open(os.path.join(rec_loss_dir, 'avgEMD.txt'), 'w') as f:
    f.write("%f" % (totalLoss / float(cnt)))
