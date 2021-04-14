import numpy as np
import matplotlib.pyplot as plt

class boundedExplorationNoise(object):
    def __init__(self, offset, k, var,smoothingAlpha = 0):
        self.k = k
        self.var = var
        self.smoothingAlpha = smoothingAlpha
        self.xOffset = np.arctanh(offset)
        self.x = self.xOffset
        self.smoothed = self.x
    def next(self):
        dx = self.k*(self.xOffset-self.x) + np.random.normal(0,self.var)
        self.x = self.x+dx
        self.smoothed = self.smoothingAlpha*self.smoothed + (1-self.smoothingAlpha)*self.x
        return np.tanh(self.smoothed)

if __name__ == '__main__':
    noise = boundedExplorationNoise([0,0],[0.1,0.1],[0.5,0.5],0.8)
    for j in range(10):
        noise1 = []
        noise2 = []
        for i in range(64):
            generatedNoise = noise.next()
            noise1.append(generatedNoise[0])
            noise2.append(generatedNoise[1])
        plt.figure(1)
        plt.clf()
        plt.plot(noise1)
        plt.plot(noise2)
        plt.show()
