import pybullet as p
import numpy as np
from wheeledSim.simController import simController
from wheeledRobots.clifford.cliffordRobot import Clifford

if __name__=="__main__":
    """start pyBullet"""
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

    """initialize clifford robot"""
    cliffordParams={"maxThrottle":20, # dynamical parameters of clifford robot
                    "maxSteerAngle":0.5,
                    "susOffset":-0.00,
                    "susLowerLimit":-0.01,
                    "susUpperLimit":0.00,
                    "susDamping":10,
                    "susSpring":500,
                    "traction":1.25,
                    "massScale":1.0}
    robot = Clifford(params=cliffordParams,physicsClientId=physicsClient)

    """initialize simulation controls (terrain, robot controls, sensing, etc.)"""
    # physics engine parameters
    simParams = {"timeStep":1./500.,
                "stepsPerControlLoop":50,
                "numSolverIterations":300,
                "gravity":-10,
                "contactBreakingThreshold":0.0001,
                "contactSlop":0.0001,
                "moveThreshold":0.1,
                "maxStopMoveLength":25}
    # random terrain generation parameters
    terrainMapParams = {"mapWidth":300, # width of matrix
                    "mapHeight":300, # height of matrix
                    "widthScale":0.1, # each pixel corresponds to this distance
                    "heightScale":0.1}
    terrainParams = {"AverageAreaPerCell":1.0,
                    "cellPerlinScale":5,
                    "cellHeightScale":0.6, # parameters for generating terrain
                    "smoothing":0.7,
                    "perlinScale":2.5,
                    "perlinHeightScale":0.1}
    # robot sensor parameters
    heightMapSenseParams = {} # use base params for heightmap
    lidarDepthParams = {"senseDim":[2.*np.pi,np.pi/4.], # angular width and height of lidar sensing
                    "lidarAngleOffset":[0,0], # offset of lidar sensing angle
                    "lidarRange":120, # max distance of lidar sensing
                    "senseResolution":[512,16], # resolution of sensing (width x height)
                    "removeInvalidPointsInPC":False, # remove invalid points in point cloud
                    "senseType":1,
                    "sensorPose":[[0,0,0.3],[0,0,0,1]]} # pose of sensor relative to body
    lidarPCParams = lidarDepthParams.copy()
    lidarPCParams["senseType"] = 2
    noSenseParams = {"senseType":-1}
    senseParams = noSenseParams # use this kind of sensing
    
    # initialize simulation controller
    sim = simController(robot,simulationParamsIn=simParams,senseParamsIn=senseParams,terrainMapParamsIn=terrainMapParams,terrainParamsIn=terrainParams,physicsClientId=physicsClient)
    # save simulation parameters for future reuse (sim params, robot params, terrain map params, terrain params, sensing params)
    #np.save('exampleAllSimParams.npy',[sim.simulationParams,robot.params,sim.terrain.terrainMapParams,sim.terrainParams,sim.senseParams])
    plotSensorReadings = False # plot sensor reading during simulation?
    if plotSensorReadings:
        import matplotlib.pyplot as plt
        fig = plt.figure()
    # simulate trajectory of length 100
    for i in range(10000):
        # step simulation
        data = sim.controlLoopStep(sim.randomDriveSinusoid())# sim.randomDriveAction())
        if data[2]: # simulation failed, restartsim
            sim.newTerrain()
            sim.resetRobot()
        else:
            if plotSensorReadings:
                sensorData = data[0][1]
                plt.clf()
                if sim.senseParams["senseType"] == 2: #point cloud
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(sensorData[0,:],sensorData[1,:],sensorData[2,:],s=0.1,c='r',marker='o')
                    ax.set_xlim([-5,5])
                    ax.set_ylim([-5,5])
                    ax.set_zlim([-5,5])
                else: # 2d map
                    ax = fig.add_subplot()
                    ax.imshow(sensorData,aspect='auto')
                #plt.show()
                plt.draw()
                plt.pause(0.001)
    # end simulation
    p.disconnect()
