import sys
import os
from coutFile import *
import numpy as np
from pyntcloud import PyntCloud

samples = None
mute = CoutToFile(sys.stderr, 'tmp.txt')

cnt = 0

for subdir, dirs, files in os.walk('.'):
    for fp in files:
        if fp.strip().split('.')[-1] == 'obj':
            fp = os.path.join(subdir, fp)
            # with mute:
            #     v, f, n = pcu.read_obj(fp)
            # samples, _ = pcu.sample_mesh_random(v, f, np.array([], dtype = v.dtype), num_samples = 5120)
            # os.system('cloudcompare.CloudCompare -SILENT -AUTO_SAVE OFF -O %s -CLEAR_NORMALS -C_EXPORT_FMT ASC -SAMPLE_MESH POINTS %d -SS RANDOM %d -NO_TIMESTAMP -SAVE_CLOUDS > /dev/null' % (fp, 5120 * 3, 5120))
            # print(fp)
            # f = open(fp.strip()[:-4] + "_SAMPLED_POINTS_SUBSAMPLED.asc", 'r')
            # samples -= samples.mean(axis = 0)
            # samples /= samples.std(axis = 0)
            # with open(fp + '.asc', 'w') as fout:
            #     for i in range(len(samples)):
            #         fout.write('%f %f %f\n' % (samples[i][0], samples[i][1], samples[i][2]))
            # os.system("cat %s.asc" % fp)
            # sys.exit()
            
            try:
                cloud = PyntCloud.from_file(fp)
                sampled_cloud = cloud.get_sample("mesh_random", n = 5120)
                samples = sampled_cloud.to_numpy()
                
                samples -= samples.mean(axis = 0)
                samples /= samples.std()
                with open(fp + '.asc', 'w') as fout:
                    print(fp + '.asc')
                    for i in range(len(samples)):
                        fout.write('%f %f %f\n' % (samples[i][0], samples[i][1], samples[i][2]))
                
                cnt += 1
                print("\r%6d" % cnt, end = '')
            except:
                os.system("rm %s" % fp + '.asc')
                print("File generation failed, delete file and skipping")
            # break

