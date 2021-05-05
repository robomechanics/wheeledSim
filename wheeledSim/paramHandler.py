import numpy as np
from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.simController import simController

class paramHandler:
    def __init__(self, allParamsFile = None, physicsClientId=0, robotType = Clifford,
                robotParams = {}, simParams = {}, terrainMapParams = {}, terrainParams = {}, senseParams = {}, explorationParams = {}):
        self.robotParams = {}
        self.simParams = {}
        self.terrainMapParams = {}
        self.terrainParams = {}
        self.senseParams = {}
        self.explorationParams = {}
        # load parameters from file if provided
        if not allParamsFile is None:
            allParamsFile = np.load(allParamsFile,allow_pickle=True)
            self.robotParams.update(allParams[0])
            self.simParams.update(allParams[1])
            self.terrainMapParams.update(allParams[2])
            self.terrainParams.update(allParams[3])
            self.senseParams.update(allParams[4])
            self.explorationParams.update(allParams[5])
        # update any parameters based on input
        self.robotParams.update(robotParams)
        self.simParams.update(simParams)
        self.terrainMapParams.update(terrainMapParams)
        self.terrainParams.update(terrainParams)
        self.senseParams.update(senseParams)
        self.explorationParams.update(explorationParams)
        robot = robotType(params=self.robotParams,physicsClientId=physicsClientId)
        self.sim = simController(robot,simulationParamsIn=self.simParams,senseParamsIn=self.senseParams,terrainMapParamsIn=self.terrainMapParams,
            terrainParamsIn=self.terrainParams,explorationParamsIn=self.explorationParams,physicsClientId=physicsClientId)
        self.robotParams = robot.params
        self.simParams = self.sim.simulationParams
        self.terrainMapParams = self.sim.terrain.terrainMapParams
        self.terrainParams = self.sim.terrain.terrainParams
        self.senseParams = self.sim.senseParams
        self.explorationParams = self.sim.randDrive.explorationParams
    def saveParams(self, allParamsFile = 'allSimParams.npy'):
        np.save(allParamsFile,[self.robotParams,self.simParams,self.terrainMapParams,self.terrainParams,self.senseParams,self.explorationParams])
    def simulate(self,trajLength=100):
        for i in range(trajLength):
            stateAction,newState,terminateFlag = self.sim.controlLoopStep(self.sim.randomDriveAction())
            if terminateFlag:
                break