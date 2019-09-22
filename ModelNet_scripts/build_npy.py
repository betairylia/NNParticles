import os
import numpy as np
import random
import json

N = 5120

splits = {'train': [], 'test': []}
sid  = 0
pid  = 0

class_dict = {}

outdir = './npys/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    os.makedirs(os.path.join(outdir, 'label'))

for subdir, dirs, files in os.walk('.'):
    for fp in files:
        if fp.strip().split('.')[-1] == 'asc':
            spl = subdir.split('/')[-1]
            cls = subdir.split('/')[-2]
            if cls not in class_dict:
                class_dict[cls] = len(class_dict)
            cls = class_dict[cls]
            fp = os.path.join(subdir, fp)
            with open(fp, 'r') as ascfile:
                data = np.zeros((N, 3), dtype = np.float32)
                pid = 0
                for line in ascfile.readlines():
                    tmp = line.split(' ')
                    tmp = [float(tmpp) for tmpp in tmp]
                    data[pid, :] = np.asarray(tmp)
                    pid += 1
            splits[spl].append({'pc': data, 'class': cls})
            sid += 1
            print("\r%5d: %5s %20s %3d" % (sid, spl, subdir.split('/')[-2], cls), end = '')

random.shuffle(splits['train'])
random.shuffle(splits['test'])
splits['val'] = splits['train'][:512]
splits['train'] = splits['train'][512:]

json.dump(list(class_dict.keys()), open('classes.json', 'w'))

for split in splits:
    pcs = [sample['pc'] for sample in splits[split]]
    pcs = np.stack(pcs, axis = 0)
    cls = [sample['class'] for sample in splits[split]]
    cls = np.asarray(cls, dtype = np.uint8)

    np.save(os.path.join(outdir, '%s.npy' % split), pcs)
    np.save(os.path.join(outdir, 'label', '%s.npy' % split), cls)

