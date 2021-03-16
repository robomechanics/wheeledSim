# wheeledRobotSimPybullet
## Overview
Maintainer: Sean J. Wang, sjw2@andrew.cmu.edu
## Installation
To install, clone repository and use pip to install

    git clone https://github.com/robomechanics/wheeledSim.git
    cd wheeledSim
    pip install .
    
### Required Dependencies
- [Python 3]
- [PyBullet](pybullet.org) Physics simulation engine
- [NumPy](numpy.org)
- [SciPy](scipy.org)
- [noise](pypi.org/project/noise)

### Optional Dependencies
- [PyTorch](pytorch.org)
- [matplotlib]

## Basic Usage
run simController.py for an example
### simController
The simController class handles controlling the robot and generation of new terrains. It can also be used to get the robot's state and sensor data (LiDAR or heightmap)
### RandomRockyTerrain
The RandomRockyTerrain class handles generation of random terrains. It generates terrains by first creating a Voronoi partition to create random blocks, smooths out edges, then adds random perlin noise.
Look at examples/terrainGeneration.py for example
### robot controllers
The simController can handle different types of wheeled robots. Currently, only one is included which is the Clifford Robot.