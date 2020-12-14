import numpy as np
import matplotlib.pyplot as plt


class ouNoise(object):
    def __init__(self,offset=np.array([0.2,0]),\
        damp = np.array([0.0125,0.001]),\
        var=np.array([0.025,0.05]),\
        filtAlpha=np.array([0.8,0.99]),\
        lowerlimit=-np.ones(2),\
        upperlimit=np.ones(2)):
        self.offset = np.array(offset)
        self.damp = np.array(damp)
        self.var = np.array(var)
        self.filtAlpha = filtAlpha
        self.lowerlimit = lowerlimit
        self.upperlimit = upperlimit
        self.noise = np.copy(self.offset)
        self.noiseFilt = np.copy(self.noise)
    def genNoise(self):
        dNoise = self.damp*(self.offset-self.noise) + np.random.normal(0,self.var)
        self.noise = self.noise + dNoise
        self.noiseFilt = self.noiseFilt*self.filtAlpha+(1-self.filtAlpha)*self.noise
        self.noiseFilt = np.maximum(self.lowerlimit,np.minimum(self.upperlimit,self.noiseFilt))
        return np.copy(self.noiseFilt)
    def multiGenNoise(self,numGen,returnAll = False):
        allNoises = []
        for i in range(numGen):
            allNoises.append(self.genNoise())
        if returnAll:
            return allNoises
        else:
            return allNoises[-1]
    def flipOffset(self):
        self.offset = -self.offset

if __name__ == '__main__':
    noise = ouNoise()
    noise1 = []
    noise2 = []
    for i in range(100):
        generatedNoise = noise.multiGenNoise(50)
        noise1.append(generatedNoise[0])
        noise2.append(generatedNoise[1])
    plt.figure(1)
    plt.plot(noise1)
    plt.plot(noise2)
    plt.show()