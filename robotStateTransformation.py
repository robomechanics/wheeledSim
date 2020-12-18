import torch
class robotStateTransformation(object):
    """
    This class handles the state data of the robot. It records the current state of the robot,
    and transforms the robot state based on predicted movements. It also can output data in a
    format for neural network predictions
    """
    def __init__(self,startState,numParticles = 1):
        self.currentState = startState.repeat_interleave(numParticles,dim=0)
        self.originalDimPrefix = self.currentState.shape[0:len(self.currentState.shape)-1]
        self.device = self.currentState.device
    def clone(self):
        output = robotStateTransformation(self.currentState.clone())
        output.originalDimPrefix = self.originalDimPrefix
        return output
    def updateState(self,prediction):
        self.currentState = self.currentState.view(-1,self.currentState.shape[-1])
        prediction = prediction.view(-1,prediction.shape[-1])
        relativePos = prediction[:,0:3]
        relativeOrien = prediction[:,3:7]
        relativeOrien = relativeOrien/torch.sqrt((relativeOrien**2).sum(dim=1)).unsqueeze(1)
        newTwist = prediction[:,7:13]
        newJointState = prediction[:,13:]
        newPosition,newOrientation = self.transformMul(self.currentState[:,0:3],self.currentState[:,3:7],relativePos,relativeOrien)
        self.currentState = torch.cat((newPosition,newOrientation,newTwist,newJointState),dim=1)
        self.currentState = self.currentState.view(self.originalDimPrefix+self.currentState.shape[-1:])
        return self.currentState
    def getPredictionInput(self):
        self.currentState = self.currentState.view(-1,self.currentState.shape[-1])
        Rwb = self.currentState[:,3:7]
        Rbw = self.qinv(Rwb)
        upDirWorld = torch.zeros(self.currentState.shape[0],3).to(self.currentState.device)
        upDirWorld[:,2] = 1.
        upDirRobot = self.qrot(Rbw,upDirWorld)
        twist = self.currentState[:,7:13]
        jointState = self.currentState[:,13:]
        predictionInput = torch.cat((upDirRobot,twist,jointState),dim=1)
        self.currentState = self.currentState.view(self.originalDimPrefix+self.currentState.shape[-1:])
        return predictionInput.view(self.originalDimPrefix+predictionInput.shape[-1:])
    def getRelativeState(self,absoluteState):
        self.currentState = self.currentState.view(-1,self.currentState.shape[-1])
        absoluteState = absoluteState.view(-1,absoluteState.shape[-1])
        Rbw = self.qinv(self.currentState[:,3:7])
        pbw = self.qrot(Rbw,-self.currentState[:,0:3])
        relativePos,relativeOrien = self.transformMul(pbw,Rbw,absoluteState[:,0:3],absoluteState[:,3:7])
        relativeState = torch.cat((relativePos,relativeOrien,absoluteState[:,7:]),dim=1)
        self.currentState = self.currentState.view(self.originalDimPrefix+self.currentState.shape[-1:])
        return relativeState.view(self.originalDimPrefix+relativeState.shape[-1:])
    def getPosHeading(self):
        self.currentState = self.currentState.view(-1,self.currentState.shape[-1])
        xVec = torch.zeros(self.currentState.shape[0],3).to(self.currentState.device)
        xVec[:,0] = 1.
        xVec = self.qrot(self.currentState[:,3:7],xVec)
        heading = torch.atan2(xVec[:,1],xVec[:,0]).unsqueeze(1)
        posHeading = torch.cat((self.currentState[:,0:3],heading),dim=1)
        self.currentState = self.currentState.view(self.originalDimPrefix+self.currentState.shape[-1:])
        return posHeading.view(self.originalDimPrefix+posHeading.shape[-1:])
    def getHeightMap(self,terrainMap,terrainMapParams,senseParams):
        # define map pixel locations relative to robot
        pixelXRelRobot=torch.linspace(-senseParams['senseDim'][0]/2,senseParams['senseDim'][0]/2,senseParams['senseResolution'][0],device=self.device)
        pixelYRelRobot=torch.linspace(-senseParams['senseDim'][1]/2,senseParams['senseDim'][1]/2,senseParams['senseResolution'][1],device=self.device)
        pixelXRelRobot,pixelYRelRobot=torch.meshgrid(pixelXRelRobot,pixelYRelRobot)
        pixelLocRobot=torch.cat((pixelXRelRobot.unsqueeze(-1),pixelYRelRobot.unsqueeze(-1)),dim=-1).reshape(-1,2).transpose(0,1)
        # rotate and translate pixel locations to get world position
        posHeading = self.getPosHeading()
        posHeading = posHeading.view(-1,posHeading.shape[-1])
        rotationRow1 = torch.cat((torch.cos(posHeading[:,3]).unsqueeze(-1),torch.sin(posHeading[:,3]).unsqueeze(-1)),dim=-1)
        rotationRow2 = torch.cat((-torch.sin(posHeading[:,3]).unsqueeze(-1),torch.cos(posHeading[:,3]).unsqueeze(-1)),dim=-1)
        rotations = torch.cat((rotationRow1.unsqueeze(-1),rotationRow2.unsqueeze(-1)),dim=-1)
        translations = posHeading[:,0:2]
        # rotate and translate pixels
        pixelLocRobot = pixelLocRobot.unsqueeze(0).repeat(posHeading.shape[0],1,1)
        pixelLocWorld = torch.bmm(rotations,pixelLocRobot)+translations.unsqueeze(-1)
        # scale to terrainMapSize
        pixelLocWorld[:,0,:] = pixelLocWorld[:,0,:]/(terrainMapParams['mapWidth']-1)/terrainMapParams['widthScale']*2.
        pixelLocWorld[:,1,:] = pixelLocWorld[:,1,:]/(terrainMapParams['mapHeight']-1)/terrainMapParams['heightScale']*2.
        pixelLocWorld = pixelLocWorld.transpose(1,2).reshape(-1,pixelXRelRobot.shape[0],pixelXRelRobot.shape[1],2)
        heightMaps = torch.from_numpy(terrainMap).float().to(self.device).unsqueeze(0).unsqueeze(0).repeat(pixelLocWorld.shape[0],1,1,1)
        heightMaps = torch.nn.functional.grid_sample(heightMaps,pixelLocWorld,align_corners=True).squeeze(dim=1)
        heightMaps = heightMaps-posHeading[:,2].unsqueeze(1).unsqueeze(1).repeat(1,heightMaps.shape[1],heightMaps.shape[2])
        heightMaps = heightMaps.transpose(1,2)
        return heightMaps.view(self.originalDimPrefix+heightMaps.shape[-2:])
    def transformMul(self,p1,q1,p2,q2):
        qout = self.qmul(q1,q2)
        pout = p1+self.qrot(q1,p2)
        return (pout,qout)
    def qmul(self,q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4

        original_shape = q.shape
        
        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        x = terms[:, 3, 0] + terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1]
        y = terms[:, 3, 1] + terms[:, 0, 2] + terms[:, 1, 3] - terms[:, 2, 0]
        z = terms[:, 3, 2] - terms[:, 0, 1] + terms[:, 1, 0] + terms[:, 2, 3]
        w = terms[:, 3, 3] - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2]
        qout = torch.stack((x, y, z, w), dim=1).view(original_shape)
        return qout
    def qrot(self,q, v):
        qinv_ = self.qinv(q)
        p = torch.cat((v,torch.zeros((v.shape[0],1),device=v.device,dtype=v.dtype)),dim=1)
        p_prime = self.qmul(self.qmul(q,p),qinv_)
        return p_prime[:,:3]
    def qinv(self,q):
        assert q.shape[-1] == 4
        if torch.isnan(torch.sum(q)).item():
            "print(bad qinv input)"
        if (torch.mean((q**2).sum(dim=1))) < 0.001:
            print("VERY SMALL quaternion_qinv")
            stop
        qout = torch.cat((-q[:,0:3],q[:,3:4]),dim=1)/(q**2).sum(dim=1).unsqueeze(1)
        if torch.isnan(torch.sum(qout)).item():
            "print(bad qinv output)"
        return(qout)