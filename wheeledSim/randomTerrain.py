import pybullet as p
import numpy as np
from scipy.ndimage import gaussian_filter
from noise import pnoise1,pnoise2
from scipy.interpolate import interp2d
from scipy.interpolate import griddata

class terrain(object):
    """
    parent class for random terrain generation
    """
    def __init__(self,terrainMapParamsIn={},physicsClientId=0):
        self.physicsClientId = physicsClientId
        # base parameters for map used to generate terrain
        self.terrainMapParams = {"mapWidth":300, # width of matrix
                        "mapHeight":300, # height of matrix
                        "widthScale":0.1, # each pixel corresponds to this distance
                        "heightScale":0.1}
        self.terrainMapParams.update(terrainMapParamsIn)
        # store parameters
        self.mapWidth = self.terrainMapParams["mapWidth"] 
        self.mapHeight = self.terrainMapParams["mapHeight"]
        self.meshScale = [self.terrainMapParams["widthScale"],self.terrainMapParams["heightScale"],1]
        self.mapSize = [(self.mapWidth-1)*self.meshScale[0],(self.mapHeight-1)*self.meshScale[1]] # dimension of terrain (meters x meters)
        # define x and y of map grid
        self.gridX = np.linspace(-self.mapSize[0]/2.0,self.mapSize[0]/2.0,self.mapWidth)
        self.gridY = np.linspace(-self.mapSize[1]/2.0,self.mapSize[1]/2.0,self.mapHeight)
        self.gridX,self.gridY = np.meshgrid(self.gridX,self.gridY)
        self.terrainShape = [] # used to replace terrain shape if it already exists
        self.terrainBody = []
        self.color = [0.82,0.71,0.55,1]
    def copyGridZ(self,gridZIn):
        self.gridZ=np.copy(copyGridZ)
        self.updateTerrain()
    def updateTerrain(self):
        # delete previous terrain if exists
        if isinstance(self.terrainShape,int):
            p.removeBody(self.terrainBody,physicsClientId=self.physicsClientId)
            self.terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD,meshScale = self.meshScale, heightfieldData=self.gridZ.reshape(-1),
                                                        numHeightfieldRows=self.mapWidth, numHeightfieldColumns=self.mapHeight,
                                                        replaceHeightfieldIndex = self.terrainShape,physicsClientId=self.physicsClientId)
        else:
            self.terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD,meshScale = self.meshScale, heightfieldData=self.gridZ.reshape(-1), 
                                                        numHeightfieldRows=self.mapWidth, numHeightfieldColumns=self.mapHeight,
                                                        physicsClientId=self.physicsClientId)
            self.terrainOffset = (np.max(self.gridZ)+np.min(self.gridZ))/2.
        self.terrainBody  = p.createMultiBody(0, self.terrainShape,physicsClientId=self.physicsClientId)
        # position terrain correctly
        p.resetBasePositionAndOrientation(self.terrainBody,[-self.meshScale[0]/2.,-self.meshScale[1]/2.,self.terrainOffset], [0,0,0,1],physicsClientId=self.physicsClientId)
        # change to brown terrain
        p.changeVisualShape(self.terrainBody, -1, textureUniqueId=-1,rgbaColor=self.color,physicsClientId=self.physicsClientId)
        # change contact parameters of terrain
        p.changeDynamics(self.terrainBody,-1,collisionMargin=0.01,restitution=0,contactStiffness=30000,contactDamping=1000,physicsClientId=self.physicsClientId)
    # return terrain map relative to a Pose
    def sensedHeightMap(self,sensorPose,mapDim,mapResolution):
        maxRadius = np.sqrt((mapDim[0]/2.)**2+(mapDim[1]/2.)**2)
        vecX = self.gridX.reshape(-1)-sensorPose[0][0]
        vecY = self.gridY.reshape(-1)-sensorPose[0][1]
        indices = np.all(np.stack((np.abs(vecX)<=(maxRadius+self.meshScale[0]),np.abs(vecY)<=(maxRadius+self.meshScale[1]))),axis=0)
        vecX = vecX[indices]
        vecY = vecY[indices]
        vecZ = self.gridZ.reshape(-1)[indices]
        forwardDir = np.array(p.multiplyTransforms([0,0,0],sensorPose[1],[1,0,0],[0,0,0,1])[0][0:2])
        headingAngle = np.arctan2(forwardDir[1],forwardDir[0])
        relativeX = vecX*np.cos(headingAngle)+vecY*np.sin(headingAngle)
        relativeY = -vecX*np.sin(headingAngle)+vecY*np.cos(headingAngle)
        rMapX = np.linspace(-mapDim[0]/2.,mapDim[0]/2.,mapResolution[0])
        rMapY = np.linspace(-mapDim[1]/2.,mapDim[1]/2.,mapResolution[1])
        points = np.stack((relativeX,relativeY)).transpose()
        rMapX,rMapY = np.meshgrid(rMapX,rMapY)
        return griddata(points, vecZ, (rMapX,rMapY))-sensorPose[0][2]
    # find maximum terrain height within a circle around a position
    def maxLocalHeight(self,position,radius):
        vecX = self.gridX.reshape(-1)-position[0]
        vecY = self.gridY.reshape(-1)-position[1]
        indices = vecX*vecX+vecY*vecY<radius
        vecZ = self.gridZ.reshape(-1)[indices]
        return np.expand_dims(np.max(vecZ), axis=0)

class randomSloped(terrain):
    """
    This class generates flat terrain with random slope
    """
    def __init__(self,terrainMapParamsIn={},physicsClientId=0):
        super().__init__(terrainMapParamsIn,physicsClientId)
        self.terrainParams = {"gmm_centers":[0],
                            "gmm_vars":[1],
                            "gmm_weights":[1]}
    def generate(self,terrainParamsIn={}):
        self.terrainParams.update(terrainParamsIn)
        index = np.random.choice(len(self.terrainParams['gmm_weights']),p=self.terrainParams['gmm_weights'])
        slope = np.random.normal(self.terrainParams['gmm_centers'][index],self.terrainParams['gmm_vars'][index])
        self.gridZ = self.gridX*slope
        self.updateTerrain()

class randomRockyTerrain(terrain):
    """
    This class handles the generation of random rocky terrain
    """
    # initialize terrain object
    def __init__(self,terrainMapParamsIn={},physicsClientId=0):
        super().__init__(terrainMapParamsIn,physicsClientId)
        self.terrainParams = {"AverageAreaPerCell":1,
                            "cellPerlinScale":5,
                            "cellHeightScale":0.9,
                            "smoothing":0.7,
                            "perlinScale":2.5,
                            "perlinHeightScale":0.1,
                            "flatRadius":1,
                            "blendRadius":0.5}
    # generate new terrain. (Delete old terrain if exists)
    def generate(self,terrainParamsIn={},copyBlockHeight=None,goal=None):
        self.terrainParams.update(terrainParamsIn)
        # generate random blocks
        if copyBlockHeight is None:
            numCells = int(float(self.mapSize[0])*float(self.mapSize[1])/float(self.terrainParams["AverageAreaPerCell"]))
            blockHeights = self.randomSteps(self.gridX.reshape(-1),self.gridY.reshape(-1),numCells,self.terrainParams["cellPerlinScale"],self.terrainParams["cellHeightScale"])
            blockHeights = gaussian_filter(blockHeights.reshape(self.gridX.shape), sigma=self.terrainParams["smoothing"])
        else:
            blockHeights=copyBlockHeight
        # add more small noise
        smallNoise = self.perlinNoise(self.gridX.reshape(-1),self.gridY.reshape(-1),self.terrainParams["perlinScale"],self.terrainParams["perlinHeightScale"])
        smallNoise = smallNoise.reshape(self.gridX.shape)
        self.gridZ = (blockHeights+smallNoise)
        # params for flat areas (start and goal)
        if hasattr(self.terrainParams['flatRadius'],'__len__'):
            startFlatRadius = self.terrainParams['flatRadius'][0]
            goalFlatRadius = startFlatRadius
            if len(self.terrainParams['flatRadius']) > 1:
                goalFlatRadius = self.terrainParams['flatRadius'][1]
        else:
            startFlatRadius = self.terrainParams['flatRadius']
            goalFlatRadius = startFlatRadius
        if hasattr(self.terrainParams['blendRadius'],'__len__'):
            startBlendRadius = self.terrainParams['blendRadius'][0]
            goalBlendRadius = startBlendRadius
            if len(self.terrainParams['blendRadius']) > 1:
                goalBlendRadius = self.terrainParams['blendRadius'][1]
        else:
            startBlendRadius = self.terrainParams['blendRadius']
            goalBlendRadius = startBlendRadius
        # make center flat for initial robot start position
        distFromOrigin = np.sqrt(self.gridX*self.gridX + self.gridY*self.gridY)
        flatIndices = distFromOrigin<startFlatRadius
        if flatIndices.any():
            flatHeight = np.mean(self.gridZ[flatIndices])
            self.gridZ[flatIndices] = flatHeight
            distFromFlat = distFromOrigin - startFlatRadius
            blendIndices = distFromFlat < startBlendRadius
            self.gridZ[blendIndices] = flatHeight + (self.gridZ[blendIndices]-flatHeight)*distFromFlat[blendIndices]/startBlendRadius
        # make goal flat
        if not goal is None:
            distFromGoal = np.sqrt((self.gridX-goal[0])**2+(self.gridY-goal[1])**2)
            flatIndices = distFromGoal<goalFlatRadius
            if flatIndices.any():
                flatHeight = np.mean(self.gridZ[flatIndices])
                self.gridZ[flatIndices] = flatHeight
                distFromFlat = distFromGoal - goalFlatRadius
                blendIndices = distFromFlat < goalBlendRadius
                self.gridZ[blendIndices] = flatHeight + (self.gridZ[blendIndices]-flatHeight)*distFromFlat[blendIndices]/goalBlendRadius
        self.gridZ = self.gridZ-np.min(self.gridZ)
        self.updateTerrain()
    def randomSteps(self,xPoints,yPoints,numCells,cellPerlinScale,cellHeightScale):
        centersX = np.random.uniform(size=numCells,low=np.min(xPoints),high=np.max(xPoints))
        centersY = np.random.uniform(size=numCells,low=np.min(yPoints),high=np.max(yPoints))
        # remove centers too close to origin
        indicesToKeep = (centersX**2+centersY**2)>4
        centersX = np.append(centersX[indicesToKeep],0)
        centersY = np.append(centersY[indicesToKeep],0)
        centersZ = self.perlinNoise(centersX,centersY,cellPerlinScale,cellHeightScale)
        numCells = len(centersZ)
        xPointsMatrix = np.matmul(np.matrix(xPoints).transpose(),np.ones((1,numCells)))
        yPointsMatrix = np.matmul(np.matrix(yPoints).transpose(),np.ones((1,numCells)))
        centersXMatrix = np.matmul(np.matrix(centersX).transpose(),np.ones((1,len(xPoints)))).transpose()
        centersYMatrix = np.matmul(np.matrix(centersY).transpose(),np.ones((1,len(yPoints)))).transpose()
        xDiff = xPointsMatrix - centersXMatrix
        yDiff = yPointsMatrix - centersYMatrix
        distMatrix = np.multiply(xDiff,xDiff)+np.multiply(yDiff,yDiff)
        correspondingCell = np.argmin(distMatrix,axis=1)
        return centersZ[correspondingCell]
    def perlinNoise(self,xPoints,yPoints,perlinScale,heightScale):
        randomSeed = np.random.rand(2)*1000
        return np.array([pnoise2(randomSeed[0]+xPoints[i]*perlinScale,randomSeed[1]+yPoints[i]*perlinScale) for i in range(len(xPoints))])*heightScale