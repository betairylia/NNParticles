import numpy as np
import ot
import os
import sys
from tqdm import tqdm

np.random.seed(18754)
numParticles = 2048

ref_dir = ''
rec_dir_prefix = ''
model_name = sys.argv[1]
rec_dir = os.path.join(rec_dir_prefix, model_name + '/')
rec_loss_dir = os.path.join(rec_loss_dir, 'loss/')

if not os.path.exists(rec_loss_dir):
    os.makedirs(rec_loss_dir)

def EMD(ref, rec):
    
    ref = np.random.permutation(ref)[:2048]
    rec = np.random.permutation(ref)[:2048]

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
        if filename.split('.')[-1] == 'npy':
            files.append(os.path.join(main_dir, filename))
    
    return files

rfFiles = get_fileNames(ref_dir)
totalCnt = 10259

t = tqdm(total = totalCnt)
totalLoss = 0.0
cnt = 0

for rf in rfFiles:
    
    ref_npy = np.load(rf)
    rec_npy = np.load(os.path.join(rec_path, os.path.basename(rf)))
    assert ref_npy.shape[0] == rec_npy.shape[0] or "Bad npy files!"
    chunk_loss = []

    for i in range(ref_npy.shape[0]):
        loss = EMD(ref_npy[i], rec_npy[i])
        chunk_loss.append(loss)
        totalLoss += loss
        cnt += 1
        t.update(1)

        t.set_postfix(curr_loss = loss, avg_loss = totalLoss / cnt, refresh = False)
    
    np.save(os.path.join(rec_loss_dir, os.path.basename(rf)), np.array(chunk_loss))

print("Total file count: %d" % cnt)
print("Expected total file count: %d" % totalCnt)

print(totalLoss / cnt)
with open(os.path.join(rec_loss_dir, 'avgEMD.txt'), 'w') as f:
    f.write("%f" % totalLoss / cnt)
