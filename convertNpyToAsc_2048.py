import numpy as np
import sys
from tqdm import tqdm

fName = sys.argv[1]
outPrefix = sys.argv[2]

arr = np.load(fName)

numSamples = arr.shape[0]
N = arr.shape[1]
D = arr.shape[2]

for i in tqdm(range(numSamples)):
    with open(outPrefix + '%d.asc' % i, 'w') as f:
        _r = np.random.permutation(arr[i])[:2048]
        for p in range(2048):
            for c in range(D):
                f.write('%f ' % _r[p, c])
            f.write('\n')

