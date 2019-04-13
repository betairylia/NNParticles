import dataLoad_set as dataLoad
import argparse

dataLoad.maxParticlesPerGrid = 2560

parser = argparse.ArgumentParser(description="Check max particles in a single grid for a given data folder")
parser.add_argument('datapath')
args = parser.parse_args()

files = dataLoad.get_fileNames(args.datapath)

maxParticles = 0

for file in files:
    content = dataLoad.read_file_header(file)
    print(content)

    if content['maxParticles'] > maxParticles:
        maxParticles = content['maxParticles']

print("\nMax particles = %d\n" % maxParticles)
