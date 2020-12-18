import pybullet as p
import time
import pybullet_data
from RandomRockyTerrain import RandomRockyTerrain
import numpy as np
import os
# This class adds a clifford (wheeled off road) robot to a given PyBullet simulation
class Clifford:
    def __init__(self,sdfRootPath=None,physicsClientId=0, params={}):
        if sdfRootPath is None:
            sdfRootPath = os.path.abspath(os.path.join(os.path.realpath(__file__),'../'))
        # this is the folder that the file clifford.sdf is in (relative to the folder this script is in)
        self.sdfPath = os.path.abspath(os.path.join(sdfRootPath,'clifford.sdf'))
        # define which PyBullet simulation to use
        self.physicsClientId=physicsClientId
        # Default parameters to use with Clifford
        self.params = {"maxThrottle":50,
                        "maxSteerAngle":0.5,
                        "susOffset":-0.01,
                        "susLowerLimit":-0.005,
                        "susUpperLimit":0.008,
                        "susDamping":10,
                        "susSpring":100,
                        "traction":1.5,
                        "massScale":1.0,
                        "tireMassScale":1.0}
        # change default params if defined
        for param in self.params:
            if param in params:
                self.params[param] = params[param]
        # set up Clifford robot in simulation
        self.importClifford()

    def importClifford(self):
        # load sdf file (this file defines open chain clifford. need to add constraints to make closed chain)
        self.cliffordID = p.loadSDF(self.sdfPath,physicsClientId=self.physicsClientId)[0]
        # define number of joints of clifford robot
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        initialJointStates = p.getJointStatesMultiDof(self.cliffordID,range(nJoints),physicsClientId=self.physicsClientId)
        self.initialJointPositions = [initialJointStates[i][0] for i in range(nJoints)]
        self.initialJointVelocities = [initialJointStates[i][1] for i in range(nJoints)]
        self.buildModelDict()
        linkFrame2Joint ={}
        linkFrame2Joint['upperSpring'] = [0,0,0]
        linkFrame2Joint['outer'] = [0.23,0,0]
        linkFrame2Joint['inner'] = [0.195,0,0]
        self.addClosedChainConstraint('brsupper',linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('blsupper',linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('frsupper',linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('flsupper',linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('bri',linkFrame2Joint['inner'])
        self.addClosedChainConstraint('bli',linkFrame2Joint['inner'])
        self.addClosedChainConstraint('fri',linkFrame2Joint['inner'])
        self.addClosedChainConstraint('fli',linkFrame2Joint['inner'])
        self.addClosedChainConstraint('blo',linkFrame2Joint['outer'])
        self.addClosedChainConstraint('flo',linkFrame2Joint['outer'])
        self.loosenModel()
        self.changeTraction()
        self.changeColor()
        self.reset() 
    def reset(self,pose=[[0,0,0.3],[0,0,0,1]]):
        p.resetBasePositionAndOrientation(self.cliffordID, pose[0],pose[1],physicsClientId=self.physicsClientId)
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        p.resetJointStatesMultiDof(self.cliffordID,range(nJoints),self.initialJointPositions,self.initialJointVelocities,physicsClientId=self.physicsClientId)
    def buildModelDict(self):
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        self.jointNameToID = {}
        self.linkNameToID = {}
        for i in range(nJoints):
            JointInfo = p.getJointInfo(self.cliffordID,i,physicsClientId=self.physicsClientId)
            self.jointNameToID[JointInfo[1].decode('UTF-8')] = JointInfo[0]
            self.linkNameToID[JointInfo[12].decode('UTF-8')] = JointInfo[0]
        measuredJointNames = ['frslower2upper','flslower2upper','brslower2upper','blslower2upper','axle2frwheel','frwheel2tire','flwheel2tire','brwheel2tire','blwheel2tire']
        self.measuredJointIDs = [self.jointNameToID[name] for name in measuredJointNames]
        self.motorJointsIDs = [self.jointNameToID[name] for name in measuredJointNames[-4:]]
        springJointNames = ['brslower2upper','blslower2upper','frslower2upper','flslower2upper']
        self.springJointIDs = [self.jointNameToID[name] for name in springJointNames]
    def changeColor(self,color=None):
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        if color is None:
            color = [0.6,0.1,0.1,1]
        for i in range(-1,nJoints):
            p.changeVisualShape(self.cliffordID,i,rgbaColor=color,specularColor=color,physicsClientId=self.physicsClientId)
        tires = ['frtire','fltire','brtire','bltire']
        for tire in tires:
            p.changeVisualShape(self.cliffordID,self.linkNameToID[tire],rgbaColor=[0.15,0.15,0.15,1],specularColor=[0.15,0.15,0.15,1],physicsClientId=self.physicsClientId)
    def loosenModel(self):
        nJoints = p.getNumJoints(self.cliffordID,physicsClientId=self.physicsClientId)
        tireIndices = [self.linkNameToID[name] for name in ['frtire','fltire','brtire','bltire']]
        for i in range(nJoints):
            if len(p.getJointStateMultiDof(bodyUniqueId=self.cliffordID,jointIndex=i,physicsClientId=self.physicsClientId)[0]) == 4:
                p.setJointMotorControlMultiDof(bodyUniqueId=self.cliffordID,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=[0,0,0,1],
                                                positionGain=0,
                                                velocityGain=0,
                                                force=[0,0,0],physicsClientId=self.physicsClientId)
                dynamicsData = p.getDynamicsInfo(self.cliffordID,i,physicsClientId=self.physicsClientId)
                massScale = self.params["massScale"]
                if i in tireIndices:
                    massScale = massScale*self.params['tireMassScale']
                newMass = dynamicsData[0]*massScale
                newInertia = [dynamicsData[2][j]*massScale for j in range(len(dynamicsData[2]))]
                p.changeDynamics(self.cliffordID,i,mass = newMass, localInertiaDiagonal=newInertia,linearDamping=0.2,angularDamping=0.2,restitution=0,physicsClientId=self.physicsClientId)
        springJointNames = ['brslower2upper','blslower2upper','frslower2upper','flslower2upper']
        springConstant = 0
        springDamping = 0
        maxSpringForce = 0
        for name in springJointNames:
            JointInfo = p.getJointInfo(self.cliffordID,self.jointNameToID[name],physicsClientId=self.physicsClientId)
            p.setJointMotorControlArray(bodyUniqueId=self.cliffordID,
                                    jointIndices=[self.jointNameToID[name]],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0],
                                    positionGains=[springConstant],
                                    velocityGains=[springDamping],
                                    forces=[maxSpringForce],physicsClientId=self.physicsClientId)
            p.changeDynamics(self.cliffordID,self.jointNameToID[name],jointLowerLimit=self.params["susLowerLimit"],jointUpperLimit=self.params["susUpperLimit"],physicsClientId=self.physicsClientId)
            #print(p.getDynamicsInfo(self.cliffordID,self.jointNameToID[name],physicsClientId=self.physicsClientId))
    def addClosedChainConstraint(self,linkName,linkFrame2Joint):
        linkState = p.getLinkState(self.cliffordID,self.linkNameToID[linkName],physicsClientId=self.physicsClientId)
        bodyState = p.getBasePositionAndOrientation(self.cliffordID,physicsClientId=self.physicsClientId)
        world2joint = p.multiplyTransforms(linkState[4],linkState[5],linkFrame2Joint,[0,0,0,1])
        body2world = p.invertTransform(bodyState[0],bodyState[1])
        body2joint = p.multiplyTransforms(body2world[0],body2world[1],world2joint[0],world2joint[1])
        linkcm2world = p.invertTransform(linkState[0],linkState[1])
        linkcm2joint = p.multiplyTransforms(linkcm2world[0],linkcm2world[1],world2joint[0],world2joint[1])
        c = p.createConstraint(self.cliffordID,-1,self.cliffordID,self.linkNameToID[linkName],p.JOINT_POINT2POINT,[0,0,0],body2joint[0],linkcm2joint[0],physicsClientId=self.physicsClientId)
        #c = p.createConstraint(self.cliffordID,
        #                       self.jointNameToID['axle2frwheel'],
        #                       self.cliffordID,
        #                       self.jointNameToID['axle2flwheel'],
        #                       jointType=p.JOINT_GEAR,
        #                       jointAxis=[0, 1, 0],
        #                       parentFramePosition=[0, 0, 0],
        #                       childFramePosition=[0, 0, 0],physicsClientId=self.physicsClientId)
        p.changeConstraint(c, gearRatio=-1, maxForce=10000,physicsClientId=self.physicsClientId)

    def changeTraction(self,newTraction=None):
        if newTraction!=None:
            self.params["traction"] = newTraction
        tires = ['frtire','fltire','brtire','bltire']
        for tire in tires:
            p.changeDynamics(self.cliffordID,self.linkNameToID[tire],lateralFriction=self.params["traction"],physicsClientId=self.physicsClientId)

    def updateSpringForce(self):
        springStates = p.getJointStates(self.cliffordID,self.springJointIDs,physicsClientId=self.physicsClientId)
        susForces = []
        for springState in springStates:
            posErr = self.params["susOffset"] -springState[0]
            velErr = -springState[1]
            susForces.append(posErr*self.params["susSpring"]+velErr*self.params["susDamping"])#-10)
        p.setJointMotorControlArray(self.cliffordID,self.springJointIDs,p.TORQUE_CONTROL,forces=susForces,physicsClientId=self.physicsClientId)

    def drive(self,driveSpeed):
        maxForce = 10
        p.setJointMotorControlArray(self.cliffordID,self.motorJointsIDs,p.VELOCITY_CONTROL,
                                    targetVelocities=[driveSpeed*self.params["maxThrottle"]]*4,
                                    forces=[maxForce]*4,
                                    physicsClientId=self.physicsClientId)
    def steer(self,angle):
        maxForce = 10000
        p.setJointMotorControl2(bodyUniqueId=self.cliffordID, 
        jointIndex=self.jointNameToID['axle2frwheel'], 
        controlMode=p.POSITION_CONTROL,
        maxVelocity = 10,
        targetPosition = angle*self.params["maxSteerAngle"],
        force = maxForce,physicsClientId=self.physicsClientId)
        
        p.setJointMotorControl2(bodyUniqueId=self.cliffordID, 
        jointIndex=self.jointNameToID['axle2flwheel'], 
        controlMode=p.POSITION_CONTROL,
        maxVelocity = 10,
        targetPosition = angle*self.params["maxSteerAngle"],
        force = maxForce,physicsClientId=self.physicsClientId)
        
        #p.setJointMotorControl2(bodyUniqueId=self.cliffordID, 
        #jointIndex=self.jointNameToID['axle2flwheel'], 
        #controlMode=p.POSITION_CONTROL,
        #targetPosition = angle,
        #force = maxForce,physicsClientId=self.physicsClientId)
    def getBaseVelocity_body(self):
        gwb = p.getBasePositionAndOrientation(self.cliffordID,physicsClientId=self.physicsClientId)
        Rbw = p.invertTransform(gwb[0],gwb[1])[1]
        Vw = p.getBaseVelocity(self.cliffordID,physicsClientId=self.physicsClientId)
        v_b = p.multiplyTransforms([0,0,0],Rbw,Vw[0],[0,0,0,1])[0]
        w_b = p.multiplyTransforms([0,0,0],Rbw,Vw[1],[0,0,0,1])[0]
        return list(v_b)+list(w_b)
    def getPositionOrientation(self):
        pwb,Rwb = p.getBasePositionAndOrientation(self.cliffordID,physicsClientId=self.physicsClientId)
        #forwardDir = p.multiplyTransforms([0,0,0],Rwb,[1,0,0],[0,0,0,1])[0]
        #headingAngle = np.arctan2(forwardDir[1],forwardDir[0])
        #Rbw = p.invertTransform([0,0,0],Rwb)[1]
        #upDir = p.multiplyTransforms([0,0,0],Rbw,[0,0,1],[0,0,0,1])[0]
        #tiltAngles = [np.arccos(upDir[2])]
        #tiltAngles.append(np.arctan2(upDir[1],upDir[0]))
        return (pwb,Rwb)#,headingAngle,tiltAngles)
    def measureJoints(self):
        jointStates = p.getJointStates(self.cliffordID,self.measuredJointIDs,physicsClientId=self.physicsClientId)
        positionReadings = [jointStates[i][0] for i in range(5)]
        velocityReadings = [jointState[1] for jointState in jointStates]
        measurements = positionReadings + velocityReadings
        return measurements


if __name__=="__main__":
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    timeStep = 1./240.
    p.setTimeStep(timeStep)
    p.setPhysicsEngineParameter(numSolverIterations=500)
    #planeId = p.loadURDF("plane.urdf")
    mapWidth = 300
    mapHeight = 300
    terrain = RandomRockyTerrain(mapWidth,mapHeight,0.1,0.1)
    terrain.generate(AverageAreaPerCell=10,cellHeightScale=0.5,perlinHeightScale=0.05,smoothing=1)
    clifford = Clifford()
    for i in range (100):
        clifford.updateSpringForce()
        p.stepSimulation()
    #clifford.drive(10)
    steerAng = 0.5
    #clifford.steer(steerAng)
    startTime = time.time()
    for i in range (50000):
        clifford.updateSpringForce()
        p.stepSimulation()
        #print((i*timeStep)/(time.time()-startTime))
        if i%60==0:
            #posOrientation = clifford.getPositionOrientation()
            #print(posOrientation)
            print(clifford.measureJoints())
            #rHeightMap = (terrain.robotHeightMap(posOrientation[0],posOrientation[1],4,4,0.5))
    print("done")
    p.disconnect()