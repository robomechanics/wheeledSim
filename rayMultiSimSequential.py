import pybullet as p
import time
from cliffordRobot import Clifford
from simController import simController as simController
from replayBuffer import sequentialReplayBuffer
import ray
import numpy as np

"""
This script runs simulation in parallel to quickly gather training data. Uses sequentialReplayBuffer to store data
NOTE: sequentialReplayBuffer IS DEPRECATED. RUN rayMultiSimDataset.py INSTEAD!!!
"""

ray.init()

@ray.remote
class singleProcess:
    def __init__(self,index):
        self.index = index
    def setup(self,numTrajectoriesPerSim,trajectoryLength,simulationParamsIn={},cliffordParamsIn={},terrainMapParamsIn={},terrainParamsIn={},senseParamsIn={}):
        # start sim
        physicsClientId = p.connect(p.DIRECT)
        # initialize clifford robot
        robot = Clifford(params=cliffordParams,physicsClientId=physicsClientId)
        # initialize simulation controller
        self.sim = simController(robot,simulationParamsIn=simulationParamsIn,senseParamsIn=senseParamsIn,terrainMapParamsIn=terrainMapParamsIn,terrainParamsIn=terrainParamsIn,physicsClientId=physicsClientId)
        data = self.sim.controlLoopStep([0,0])
        bufferLength=numTrajectoriesPerSim*trajectoryLength
        self.replayBuffer = sequentialReplayBuffer(bufferLength,data[0],data[1])
        self.trajectoryLength=trajectoryLength
        self.numTrajectoriesPerSim=numTrajectoriesPerSim
    def gatherSimData(self):
        sTime = time.time()
        while not self.replayBuffer.bufferFilled:
            # while haven't gathered enough data
            self.sim.newTerrain()
            self.sim.resetRobot()
            while self.replayBuffer.getCurrentSequenceLength() < self.trajectoryLength:
                # while current trajectory isn't long enough
                # step simulation and add data
                data = self.sim.controlLoopStep(self.sim.randomDriveAction())
                if data[2]: # simulation failed, need to reset
                    self.replayBuffer.purgeCurrentSequence() # delete current sequence
                    break # restart trjectory
                else:
                    # otherwise, add data
                    self.replayBuffer.addData(data[0],data[1])
            self.replayBuffer.endTrajectory() # start new trajectory
            runTime = (time.time()-sTime)/3600
            numTrajectoriesGathered = len(self.replayBuffer.sequenceStartIndices)-1
            if numTrajectoriesGathered > 0:
                print("sim: " + str(self.index) + ", numTrajectories: " + str(numTrajectoriesGathered) + ", " + 
                        "time elapsed: " + "%.2f"%runTime + " hours, " + 
                        "estimated time left: " + "%.2f"%(float(self.numTrajectoriesPerSim-numTrajectoriesGathered)*runTime/float(numTrajectoriesGathered)) + "hours")
        return self.replayBuffer.bufferIndex
    def combineData(self,otherProcess):
        print("combining sim " + str(ray.get(otherProcess.outputIndex.remote())) + " to sim " + str(self.sim.physicsClientId))
        self.replayBuffer.inheritData(ray.get(otherProcess.outputReplayBuffer.remote()))
    def saveData(self,saveDataPrefix):
        print("saving")
        self.replayBuffer.saveDataPrefix = saveDataPrefix
        self.replayBuffer.saveData()
    def outputIndex(self):
        return self.index
    def outputReplayBuffer(self):
        return self.replayBuffer

if __name__=="__main__":
    numParallelSims = 8
    #totalNumTrajectories = 1024#4096
    #numTrajectoriesPerSim = int(np.ceil(totalNumTrajectories/numParallelSims))
    numTrajectoriesPerSim = 1024
    trajectoryLength = 64
    # load all simulation parameters
    [simParams,cliffordParams,terrainMapParams,terrainParams,senseParams] = np.load('exampleAllSimParams.npy',allow_pickle=True)
    # start all parallel simulations
    processes = [singleProcess.remote(i) for i in range(numParallelSims)]
    for process in processes:
        ray.get(process.setup.remote(numTrajectoriesPerSim,trajectoryLength,simulationParamsIn=simParams,cliffordParamsIn=cliffordParams,terrainMapParamsIn=terrainMapParams,terrainParamsIn=terrainParams,senseParamsIn=senseParams))
    print("finished initialization")
    results = [processes[i].gatherSimData.remote() for i in range(numParallelSims)]
    ray.get(results)
    for i in range(1,numParallelSims):
        ray.get(processes[0].combineData.remote(processes[i]))
    ray.get(processes[0].saveData.remote(saveDataPrefix))