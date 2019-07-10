import os
import sys
import numpy as np
from tqdm import tqdm
import csv
import random
import json

asc_path = '/home/betairya/RP_ML/NNParticles/MDSets/ShapeNet_CC/models/obj/ascs/'
split_file = '/home/betairya/Downloads/all.csv'
out_path = '/home/betairya/RP_ML/NNParticles/MDSets/ShapeNet_CC_Official'

splits = {}
class_dict = {}

NPoints = 5120



def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)



def output_npy(split_array, target_path, chunkSize = 5000):
    
    visit_order = list(range(len(split_array)))
    random.shuffle(visit_order)

    chunk_cnt = 0
    chunk_contents = []
    chunk_classes = []

    for i in tqdm(range(len(visit_order))):
        content = split_array[visit_order[i]]
        cloud_array = np.zeros((NPoints, 3))

        if not os.path.isfile(os.path.join(asc_path, content['name'])):
            continue

        with open(os.path.join(asc_path, content['name']), 'r') as cloudFile:
            cloudLines = cloudFile.readlines()
            assert len(cloudLines) == NPoints

            pIdx = 0
            for cl in cloudLines:
                XYZ = cl.split(' ')
                cloud_array[pIdx, 0] = float(XYZ[0])
                cloud_array[pIdx, 1] = float(XYZ[1])
                cloud_array[pIdx, 2] = float(XYZ[2])
                pIdx += 1

        chunk_contents.append(cloud_array)
        chunk_classes.append(content['class'])

        if chunkSize > 0 and len(chunk_contents) >= chunkSize:
            
            mkdir(target_path)
            
            chunk_cnt += 1
            chunk_npy = np.stack(chunk_contents, axis = 0)
            np.save(os.path.join(target_path, 'chunk%d.npy' % chunk_cnt), chunk_npy)

            mkdir(os.path.join(target_path, 'classes/'))

            classes = np.asarray(chunk_classes)
            np.save(os.path.join(target_path, 'classes/', 'chunk%d.npy' % chunk_cnt), classes)

            chunk_contents = []
    
    if len(chunk_contents) > 0:
        
        mkdir(target_path)
            
        chunk_cnt += 1
        chunk_npy = np.stack(chunk_contents, axis = 0)
        np.save(os.path.join(target_path, 'chunk%d.npy' % chunk_cnt), chunk_npy)

        mkdir(os.path.join(target_path, 'classes/'))

        classes = np.asarray(chunk_classes)
        np.save(os.path.join(target_path, 'classes/', 'chunk%d.npy' % chunk_cnt), classes)

        chunk_contents = []



with open(split_file, newline = '') as csvfile:
    
    reader = csv.DictReader(csvfile)
    
    print("Reading csv file, building up index arrays...")
    for row in tqdm(reader):

        if row['subSynsetId'] not in class_dict:
            class_dict[row['subSynsetId']] = len(class_dict)
        if row['split'] not in splits:
            splits[row['split']] = []
        
        splits[row['split']].append({'name': '%s_%s_SAMPLED_POINTS_SUBSAMPLED.asc' % (row['synsetId'], row['modelId']), 'class': class_dict[row['subSynsetId']]})



print('Making training split...')
# output_npy(splits['train'], os.path.join(out_path, 'train/'))

print('Making validation split...')
# output_npy(splits['val'], os.path.join(out_path, 'val/'), chunkSize = -1)

print('Making test split...')
# output_npy(splits['test'], os.path.join(out_path, 'test/'))

print(class_dict)

inverse_class_dict = []
for subsId in class_dict:
    inverse_class_dict.append('')
for subsId in class_dict:
    inverse_class_dict[class_dict[subsId]] = subsId

with open(os.path.join(out_path, 'class_labels.json'), 'w') as jsonFile:
    json.dump(inverse_class_dict, jsonFile)
