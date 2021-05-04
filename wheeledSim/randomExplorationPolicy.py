import numpy as np

class boundedExplorationNoise(object):
    def __init__(self, explorationParamsIn={}):
        self.explorationParams = {"offset":[0,0],
                                "k":[0.1,0.1],
                                "var":[0.5,0.5],
                                "smoothingAlpha":0.8,
                                "actionScale":1}
        self.explorationParams.update(explorationParamsIn)
        self.k = self.explorationParams["k"]
        self.var = self.explorationParams["var"]
        self.smoothingAlpha = self.explorationParams["smoothingAlpha"]
        self.xOffset = np.array(self.explorationParams["offset"])
        self.x = self.xOffset
        self.smoothed = self.x
        self.actionScale = self.explorationParams["actionScale"]
    def reset(self):
        pass
    def next(self):
        dx = self.k*(self.xOffset-self.x) + np.random.normal(0,self.var)
        self.x = self.x+dx
        self.smoothed = self.smoothingAlpha*self.smoothed + (1-self.smoothingAlpha)*self.x
        return np.tanh(self.smoothed)*self.actionScale

class fixedRandomAction(object):
    def __init__(self,explorationParamsIn={}):
        self.explorationParams = {"gmm_centers":[[0.5,0.5],[0.5,-0.5]],
                                "gmm_vars":[[0.1,0.1],[0.1,0.1]],
                                "gmm_weights":[0.5,0.5]}
        self.explorationParams.update(explorationParamsIn)
        self.gmm_centers = self.explorationParams['gmm_centers']
        self.gmm_vars = self.explorationParams['gmm_vars']
        self.gmm_weights = self.explorationParams['gmm_weights']
        self.reset()
    def reset(self):
        index = np.random.choice(len(self.gmm_weights),p=self.gmm_weights)
        invalidAction=True
        while invalidAction:
            self.action = np.random.normal(self.gmm_centers[index],self.gmm_vars[index])
            invalidAction = np.any(self.action > 1) or np.any(self.action < -1)
    def next(self):
        return self.action

if __name__ == '__main__':
    a = fixedRandomAction()
    samplesX = []
    samplesY = []
    for i in range(10000):
        a.reset()
        sample = a.next()
        samplesX.append(sample[0])
        samplesY.append(sample[1])
    import matplotlib.pyplot as plt
    plt.scatter(samplesX,samplesY)
    plt.show()