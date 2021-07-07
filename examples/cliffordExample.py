import pybullet as p
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
    
    # initialize simulation controller
    sim = simController(robot,physicsClientId=physicsClient)
    for i in range(10000):
        action = sim.randomDriveAction()
        stateAction,newState,isTerm = sim.controlLoopStep(action)
        if isTerm: # simulation failed, restart sim
            sim.newTerrain()
            sim.resetRobot()
    # end simulation
    p.disconnect()
