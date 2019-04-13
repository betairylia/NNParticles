import dataLoad
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Check max particles in a single grid for a given data folder")
parser.add_argument('datapath')
args = parser.parse_args()

files = dataLoad.get_fileNames(args.datapath)

maxParticles = 0

for file in files:
    content = dataLoad.read_file(file)

    velocity = np.zeros((3,))
    velocity_mag = np.zeros((content['stepCount'],))
    cnt = 0

    for step in range(int(content['stepCount'])):

        velocity = np.zeros((3,))
        cnt = 0
        print("Step: ", step)

        for grid in range(int(content['gridCount'])):
            for particle in range(int(content['particleCount'][step, grid])):
                velocity += content['data'][step, grid, particle, 3:6]
                cnt += 1
        
        velocity /= cnt
        print("Mean velocity: ", velocity)
        print("Mean velocity mag: ", np.sqrt(np.sum(np.square(velocity))))

        velocity_mag[step] = np.sqrt(np.sum(np.square(velocity)))

    break

plt.plot(velocity_mag)
plt.show()
