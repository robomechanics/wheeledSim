import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from wheeledSim.parallelSimDataset import gatherData
from wheeledSim.trajectoryDataset import trajectoryDataset
from wheeledSim.robotStateTransformation import robotStateTransformation


"""
This script runs multiple simulations of clifford in parallel.
In each simulation, clifford is driven using random actions (sin of ou noise).
The trajectories, actions, sensing state, global terrain map is saved.
At the end of this script, data is loaded and can be used for training.
"""

if __name__ == '__main__':
    """parameters for parallel processing """
    numParallelSims = 2 # number of parallel sims. Increase if computer has enough threads
    trajectoryLength = 64 # length of trajectories
    numTrajectoriesPerSim = 2 # how many trajectories gathered for each sim
    startNewFile = True # should we delete old data and gather data from scratch?

    """simulation/robot parameters (consistent between simulations)"""
    # parameters for simulation
    simParams = {"timeStep":1./500.,
                "stepsPerControlLoop":50,
                "numSolverIterations":300,
                "gravity":-10,
                "contactBreakingThreshold":0.0001,
                "contactSlop":0.0001,
                "moveThreshold":0.1,
                "maxStopMoveLength":25}
    # parameters for generating global terrain map
    terrainMapParams = {"mapWidth":300, # width of matrix
                    "mapHeight":300, # height of matrix
                    "widthScale":0.1, # each pixel corresponds to this distance
                    "heightScale":0.1}
    # parameters for generating terrain obstacles
    terrainParams = {"AverageAreaPerCell":1.0,
                    "cellPerlinScale":5.0,
                    "cellHeightScale":0.6,
                    "smoothing":0.7,
                    "perlinScale":2.5,
                    "perlinHeightScale":0.1,
                    "flatRadius":0.5,
                    "blendRadius":0.5}
    # parameters for robot
    robotParams = {"maxThrottle":20,
                    "maxSteerAngle":0.5,
                    "susOffset":-0.00,
                    "susLowerLimit":-0.01,
                    "susUpperLimit":0.00,
                    "susDamping":10,
                    "susSpring":500,
                    "traction":1.25,
                    "masScale":1.0}
    # parameters for sensing
    senseParams = {"senseDim":[5,5], # width (meter) and height (meter) of terrain map
                    "senseResolution":[300,300], #(num pixels wide x num pixels high)
                    "sensorPose":[[0,0,0],[0,0,0,1]], # pose of sensor relative to body
                    "senseType":-1} # we use -1 so that sensing data isn't generated during simulation

    """ data saving folder"""
    dataRootDir = 'data/'
    if not os.path.isdir(dataRootDir):
        os.mkdir(dataRootDir)
    dataRootDir = dataRootDir + 'test1/'
    if not os.path.isdir(dataRootDir):
        os.mkdir(dataRootDir)
    """ save simulation parameters in data folder """
    np.save(dataRootDir+'allSimParams.npy',[simParams,robotParams,terrainMapParams,terrainParams,senseParams])

    """ run parallel simulation and save data """
    # following function will load simulation parameters from saved file
    gatherData(numParallelSims,numTrajectoriesPerSim,trajectoryLength,dataRootDir,startNewFile)

    """
    Now the data folder should contain ".pt" files that contain data for each trajectory
    There should also be a "meta.csv" that contains names of each ".pt" file and the length of each trajectory
    """

    """ Load up trajectory into dataset"""
    csv_file_name = "meta.csv"
    sampleLength = trajectoryLength # sample full trajectories. Could also sample part of trajectory.
    startMidTrajectory = False # use false to sample from start of each trajectory
    staticDataIndices = [4] # Index 4 corresponds to global terrain map which is static for each trajectory.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data = trajectoryDataset(csv_file_name,dataRootDir,sampleLength=sampleLength,startMidTrajectory=startMidTrajectory,staticDataIndices=staticDataIndices,device=device)
    samples = iter(DataLoader(data,shuffle=True,batch_size=1))

    """ sample one trajectory """
    sample = next(samples)
    # loop through trajectory
    batchNum = 0
    for t in range(sample[0].shape[1]):
        print("time: " + str(t))
        print("starting state:\n" + str(sample[0][batchNum,t,:]))
        print("sensing from simulation:\n" +str(sample[1][batchNum,t,:]))
        print("action taken:\n" + str(sample[2][batchNum,t,:]))
        print("ending state:\n" + str(sample[3][batchNum,t,:]))
        print("\n\n\n")
    
    """
    Note that sensing from simulation was empty since senseParams['senseType']==-1
    local terrain maps not generated during simulation
    We can generate local terrain maps from global map and state of robot
    """
    print("global terrain map shape")
    print(sample[4].shape)
    # generate local terrain map for all timesteps
    states = robotStateTransformation(sample[0],terrainMap=sample[4],terrainMapParams=terrainMapParams,senseParams=senseParams)
    localMaps = states.getHeightMap(useChannel=True) # use channel will add channel dimension
    print("shape local terrain map for 1st time step")
    timeStep = 0
    print(localMaps[batchNum,timeStep,:].shape)