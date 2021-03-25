# wheeledSim
## Overview
Maintainer: Sean J. Wang, sjw2@andrew.cmu.edu
## Installation
To install, first install dependencies. Clone repository and use pip to install

    git clone https://github.com/robomechanics/wheeledSim.git
    cd wheeledSim
    pip install .

### Required Dependencies
This package is built using Python 3. The following packages are required.
- [PyBullet](https://pybullet.org) Physics simulation engine
- [NumPy](https://numpy.org)
- [SciPy](https://scipy.org)
- [noise](https://pypi.org/project/noise)

### Optional Dependencies
The following packages are only necessary to run some examples.
- [PyTorch](pytorch.org)
- [matplotlib](https://matplotlib.org/)

## Examples
### cliffordExample.py
Example of simulating the Clifford robot. Random drive actions are taken. As Clifford is driving, sensor data is measured and plotted.

### dataGather_terrainMap.py
Example of running multiple simulations of the Clifford robot in parallel. Random drive actions are taken. Data (robot starting state, action, robot ending state)for each timestep, as well as the global terrain map for each trajectory is saved. All data is then loaded as a PyTorch dataset.

## Basic Functionality
### simController
The simController class handles simulation updates, robot controls, and generation of new terrains. Other functionality includes generating robot state and sensor data (LiDAR or heightmap), checking for failure, and generating smooth random actions.

### RandomRockyTerrain
The RandomRockyTerrain class handles generation of random terrains. It generates terrains by first creating a Voronoi partition to create random blocks, smooths out edges, then adds random perlin noise.
Look at examples/terrainGeneration.py for example

### robot controllers
The simController can handle different types of wheeled robots. Currently, only one is included which is the Clifford Robot.

## Examples
