import pybullet as p
import time
import numpy as np
import sys
sys.path.append('../') # include directory with simulation files
from RandomRockyTerrain import RandomRockyTerrain

if __name__ == '__main__':
    # start pybullet simulation
    physicsClientId = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setGravity(0,0,-10)
    # define world terrain map parameters
    mapParams = {"mapWidth":300, # width of terrain map
                "mapHeight":300, # height of terrain map
                "widthScale":0.1, # Width of each pixel in map (meters)
                "heightScale":0.1 # Width of each pixel in map (meters)
                }
    # if any params are missing, default params will be used
    # alternatively, can pass mapParams={} for all default
    # create terrain object
    terrain = RandomRockyTerrain(terrainMapParamsIn=mapParams,physicsClientId=physicsClientId)
    # generate terrain
    terrainParams = {# first few params define block cells
                    "AverageAreaPerCell":1,
                    # height of cells defined by perlin noise
                    "cellPerlinScale":5, # this define width scale of perlin noise
                    "cellHeightScale":0.9, # this defines height scale of perlin noise
                    "smoothing":0.7, # smoothing on blocks. 0 smoothing will have sharp edges
                    # next two parameters define added perlin noise
                    "perlinScale":2.5, # width scale of perlin noise
                    "perlinHeightScale":0.1, # height scale of perlin noise
                    # next two parameters define flat ground region
                    "flatRadius":1, # this distance from the origin will be flat ground
                    "blendRadius":0.5 # distance of blend from flat ground to nonflat ground
                    }
    terrain.generate(terrainParamsIn=terrainParams)
    input("press enter for next terrain")

    # Alternatively, terrain can be manually created
    print('manually created terrain')
    terrainHeight = np.clip(terrain.gridX/2.0+terrain.gridY/2.0,-2,2)
    terrain.generate(copyGridZ = terrainHeight)
    input("press enter for next terrain")

    # block ground (no smoothing, no small perlin noise)
    print('block ground')
    terrainParams = {"AverageAreaPerCell":0.3,
                    "cellPerlinScale":5,
                    "cellHeightScale":0.9,
                    "smoothing":0,
                    "perlinScale":0,
                    "perlinHeightScale":0,
                    "flatRadius":0,
                    "blendRadius":0
                    }
    terrain.generate(terrainParamsIn=terrainParams)
    input("press enter for next terrain")

    # lower cell height scale
    print('lower cell height scale')
    terrainParams["cellHeightScale"]=0.5
    terrain.generate(terrainParamsIn=terrainParams)
    input("press enter for next terrain")

    # lower cell perlin scale (make neighboring cells more similar in height)
    print('lower cell perlin scale (make neighboring cells more similar in height)')
    terrainParams["cellPerlinScale"]=0.5
    terrain.generate(terrainParamsIn=terrainParams)
    input("press enter for next terrain")

    # add smoothing
    print('add smoothing')
    terrainParams["smoothing"]=0.5
    terrain.generate(terrainParamsIn=terrainParams)
    input("press enter for next terrain")

    # add flat ground in center
    print('add flat ground in center')
    terrainParams["flatRadius"]=1
    terrainParams["blendRadius"]=0.5
    terrain.generate(terrainParamsIn=terrainParams)
    input("press enter for next terrain")
