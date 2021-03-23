import pybullet as p
import time
import torch
from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.simController import simController as simController
import concurrent.futures
import numpy as np
import csv
from os import path

class singleProcess:
    def __init__(self,index):
        self.index = index
    def setup(self,numTrajectoriesPerSim,trajectoryLength,root_dir,simulationParamsIn={},cliffordParamsIn={},terrainMapParamsIn={},terrainParamsIn={},senseParamsIn={}):
        print("setup sim " +str(self.index))
        # save parameters
        self.trajectoryLength = trajectoryLength
        self.numTrajectoriesPerSim = numTrajectoriesPerSim
        self.root_dir = root_dir
        # start sim
        physicsClientId = p.connect(p.DIRECT)
        # initialize clifford robot
        robot = Clifford(params=cliffordParamsIn,physicsClientId=physicsClientId)
        # initialize simulation controller
        self.sim = simController(robot,simulationParamsIn=simulationParamsIn,senseParamsIn=senseParamsIn,terrainMapParamsIn=terrainMapParamsIn,terrainParamsIn=terrainParamsIn,physicsClientId=physicsClientId)
        stateAction,newState,terminateFlag = self.sim.controlLoopStep([0,0])
        self.fileCounter = 0
        self.filenames = []
        self.trajectoryLengths = []
    def newTrajectoryData(self,stateAction,newState):
        self.trajectoryData = []
        for i in range(len(stateAction)):
            self.trajectoryData.append(torch.from_numpy(np.array(stateAction[i])).unsqueeze(0).float())
        for i in range(len(newState)):
            self.trajectoryData.append(torch.from_numpy(np.array(newState[i])).unsqueeze(0).float())
        self.trajectoryData.append(torch.from_numpy(np.array(self.sim.terrain.gridZ)).float())
    def addSampleToTrajData(self,stateAction,newState):
        for i in range(len(stateAction)):
            self.trajectoryData[i] = torch.cat((self.trajectoryData[i],torch.from_numpy(np.array(stateAction[i])).unsqueeze(0).float()),dim=0)
        for i in range(len(newState)):
            self.trajectoryData[i+len(stateAction)] = torch.cat((self.trajectoryData[i+len(stateAction)],torch.from_numpy(np.array(newState[i])).unsqueeze(0).float()),dim=0)
    def saveTrajectory(self):
        filename = 'sim'+str(self.index)+'_'+str(self.fileCounter)+'.pt'
        while path.exists(self.root_dir+filename):
            self.fileCounter+=1
            filename = 'sim'+str(self.index)+'_'+str(self.fileCounter)+'.pt'
        self.filenames.append(filename)
        self.trajectoryLengths.append(self.trajectoryData[0].shape[0])
        torch.save(self.trajectoryData,self.root_dir+filename)
    def gatherSimData(self):
        sTime = time.time()
        while len(self.filenames) < self.numTrajectoriesPerSim:
            # while haven't gathered enough data
            # reset simulation start new trajectory
            self.sim.newTerrain()
            self.sim.resetRobot()
            stateAction,newState,terminateFlag = self.sim.controlLoopStep(self.sim.randomDriveAction())
            self.newTrajectoryData(stateAction,newState)
            while not terminateFlag:
                # while robot isn't stuck, step simulation and add data
                stateAction,newState,terminateFlag = self.sim.controlLoopStep(self.sim.randomDriveAction())
                self.addSampleToTrajData(stateAction,newState)
                if self.trajectoryData[0].shape[0] >= self.trajectoryLength:
                    # if trajectory is long enough, save trajectory and start new one
                    self.saveTrajectory()
                    break
            # print estimated time left
            if len(self.filenames) > 0:
                runTime = (time.time()-sTime)/3600
                print("sim: " + str(self.index) + ", numTrajectories: " + str(len(self.filenames)) + ", " + 
                        "time elapsed: " + "%.2f"%runTime + " hours, " + 
                        "estimated time left: " + "%.2f"%(float(self.numTrajectoriesPerSim-len(self.filenames))*runTime/float(len(self.filenames))) + "hours")
        return self.filenames,self.trajectoryLengths
    def outputIndex(self):
        return self.index

def gatherData(numParallelSims,numTrajectoriesPerSim,trajectoryLength,rootDir,startNewFile):
    # load all simulation parameters
    [simParams,cliffordParams,terrainMapParams,terrainParams,senseParams] = np.load(rootDir+'allSimParams.npy',allow_pickle=True)
    # start all parallel simulations
    processes = [singleProcess(i) for i in range(numParallelSims)]
    for process in processes:
        process.setup(numTrajectoriesPerSim,trajectoryLength,root_dir=rootDir,
            simulationParamsIn=simParams,cliffordParamsIn=cliffordParams,terrainMapParamsIn=terrainMapParams,terrainParamsIn=terrainParams,senseParamsIn=senseParams)
    print("finished initialization")
    executor = concurrent.futures.ProcessPoolExecutor()
    results = [executor.submit(process.gatherSimData) for process in processes]
    concurrent.futures.wait(results,return_when=concurrent.futures.ALL_COMPLETED)
    # write metadata csv file
    if startNewFile:
        csvFile = open(rootDir+'meta.csv', 'w', newline='')
    else:
        csvFile = open(rootDir+'meta.csv', 'a', newline='')
    csvWriter = csv.writer(csvFile,delimiter=',')
    if startNewFile:
        csvWriter.writerow(['filenames','trajectoryLengths'])
    for result in results:
        fileNames = result.result()[0]
        trajLengths = result.result()[1]
        for i in range(len(fileNames)):
            csvWriter.writerow([fileNames[i],trajLengths[i]])
    csvFile.flush()
