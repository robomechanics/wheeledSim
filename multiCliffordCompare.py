import pybullet as p
import time
import pybullet_data
from RandomRockyTerrain import RandomRockyTerrain
from cliffordFixed import Clifford
from simController import simController
import numpy as np
import matplotlib.pyplot as plt

def ignoreCollisions(botID1,botID2):
    nJoints1 = p.getNumJoints(botID1)
    nJoints2 = p.getNumJoints(botID2)
    for i in range(-1,nJoints1):
        for j in range(-1,nJoints2):
            p.setCollisionFilterPair(botID1,botID2,i,j,0)

if __name__=="__main__":
    physicsClientId = p.connect(p.GUI,options="--opengl3")#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    moveThreshold = 0.1
    maxStopMoveLength = 25
    cliffordParam1 = {"maxThrottle":30,
                            "maxSteerAngle":0.5,
                            "susOffset":-0.0,
                            "susLowerLimit":-0.01,
                            "susUpperLimit":0.003,
                            "susDamping":100,
                            "susSpring":50000,
                            "traction":1.0,
                            "massScale":1}
    cliffordParam2 = {"maxThrottle":40,
                            "maxSteerAngle":0.75,
                            "susOffset":-0.002,
                            "susLowerLimit":-0.02,
                            "susUpperLimit":0.00,
                            "susDamping":1,
                            "susSpring":75,
                            "traction":1.25,
                            "massScale":1.5}
    terrainParams = {"AverageAreaPerCell":1,
                    "cellPerlinScale":5,
                    "cellHeightScale":0.9,
                    "smoothing":0.7,
                    "perlinScale":2.5,
                    "perlinHeightScale":0.1}
    cliffordEasyNovice = {"maxThrottle":30,
                            "maxSteerAngle":0.5,
                            "susOffset":-0.001,
                            "susLowerLimit":-0.02,
                            "susUpperLimit":0.00,
                            "susDamping":100,
                            "susSpring":50000,
                            "traction":1.25,
                            "masScale":1.0}
    #terrainParams = {"AverageAreaPerCell":1,
    #                "cellPerlinScale":5,
    #                "cellHeightScale":0.35,
    #                "smoothing":0.7,
    #                "perlinScale":2.5,
    #                "perlinHeightScale":0.1}
    cliffordParam1 = cliffordEasyNovice
    clifford1 = simController(timeStep=1./500.,stepsPerControlLoop=50,numSolverIterations=300,physicsClientId=physicsClientId,
                            cliffordParams = cliffordParam1,terrainParams=terrainParams,moveThreshold = moveThreshold,maxStopMoveLength = maxStopMoveLength)
    sameSimComp = False
    if sameSimComp:
        clifford2 = simController(timeStep=1./500.,stepsPerControlLoop=50,numSolverIterations=300,physicsClientId=physicsClientId,
                                cliffordParams = cliffordParam2,existingTerrain=clifford1.terrain)
        ignoreCollisions(clifford1.clifford.cliffordID,clifford2.clifford.cliffordID)
        clifford1.resetClifford(doFall=False)
        clifford2.resetClifford()
        for i in range (1000):
            randDrive = clifford1.randomDriveAction()
            randDrive = [0.1,0]
            clifford1.controlLoopStepInit(randDrive)
            clifford2.controlLoopStepInit(randDrive)
            clifford1.controlLoopSimPlay()
        p.disconnect
    else:
        trial1X = []
        trial1Y = []
        trial2X = []
        trial2Y = []
        clifford1.newTerrain()
        gridZ = np.copy(clifford1.terrain.gridZ)
        clifford1.resetClifford()
        for i in range(10000000):
            data = clifford1.controlLoopStep([0,0])#clifford1.randomDriveAction())
            trial1X.append(data[3][0])
            trial1Y.append(data[3][1])
            if data[2]:
                print('fail')
                #break
        input("press enter to continue")
        clifford1.clifford.changeColor([0.1,0.6,0.1,1])
        input("press enter to continue")
        p.resetSimulation()
        clifford2 = simController(timeStep=1./500.,stepsPerControlLoop=50,numSolverIterations=300,physicsClientId=physicsClientId,
                                cliffordParams = cliffordParam2,terrainParams=terrainParams,moveThreshold = moveThreshold,maxStopMoveLength = maxStopMoveLength)
        clifford2.newTerrain(copyGridZ=gridZ)
        clifford2.resetClifford()
        for i in range(len(trial1X)):
            data = clifford2.controlLoopStep(clifford2.randomDriveAction())
            trial2X.append(data[3][0])
            trial2Y.append(data[3][1])
            if data[2]:
                print('fail')
                break
        for i in range(len(trial1X)):
            plt.clf()
            plt.plot(trial1X[0:i],trial1Y[0:i],label="robot1")
            plt.plot(trial2X[0:i],trial2Y[0:i],label="robot2")
            plt.xlim((-10,10))
            plt.ylim((-10,10))
            plt.legend()
            plt.pause(0.1)
