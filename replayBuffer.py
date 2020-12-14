import torch
import numpy as np
from os import path
import time
import matplotlib.pyplot as plt

"""
replay buffer that can be used to store and sample data for training
NOTE: THESE CLASSES ARE DEPRECATED!!! USE trajectoryDataset INSTEAD!!!!!
"""

class ReplayBuffer(object):
	def __init__(self,bufferLength=0,sampleNNInput=[],sampleNNOutput=[],saveDataPrefix='',loadDataPrefix='',chooseCPU = False):
		if chooseCPU:
			self.device='cpu'
		else:
			self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.saveDataPrefix = saveDataPrefix
		self.loadDataPrefix = loadDataPrefix
		self.bufferLength = bufferLength
		self.bufferIndex = 0
		self.bufferFilled = False
		self.inputData = []
		self.outputData = []
		for data in sampleNNInput:
			self.inputData.append(torch.zeros((self.bufferLength,)+np.array(data).shape,device=self.device))
		for data in sampleNNOutput:
			self.outputData.append(torch.zeros((self.bufferLength,)+np.array(data).shape,device=self.device))
	def purgeData(self):
		self.bufferIndex = 0
		self.bufferFilled = False
	def addData(self,nnInput,nnOutputGroundTruth):
		inputData,outputData = self.processData(nnInput,nnOutputGroundTruth)
		for i in range(len(inputData)):
			self.inputData[i][self.bufferIndex,:] = inputData[i].detach().clone()
		for i in range(len(outputData)):
			self.outputData[i][self.bufferIndex,:] = outputData[i].detach().clone()
		self.bufferIndex+=1
		if self.bufferIndex == self.bufferLength:
			self.bufferIndex = 0
			self.bufferFilled = True
	def processData(self,nnInput,nnOutputGroundTruth):
		inputData = []
		outputData = []
		for data in nnInput:
			inputData.append(torch.from_numpy(np.array(data)).to(self.device).unsqueeze(0).float())
		for data in nnOutputGroundTruth:
			outputData.append(torch.from_numpy(np.array(data)).to(self.device).unsqueeze(0).float())
		return inputData,outputData
	def getData(self,batchSize=1):
		returnedData=[]
		index = 0
		maxIndex = self.bufferLength if self.bufferFilled else self.bufferIndex
		while index < maxIndex:
			endIndex = np.min([index+batchSize,maxIndex])
			returnedData.append([[self.inputData[i][index:endIndex,:] for i in range(0,len(self.inputData))],
								[self.outputData[i][index:endIndex,:] for i in range(0,len(self.outputData))]])
			index = endIndex
		return returnedData
	def getRandBatch(self,batchSize=1,device='',percentageRange=[0,1]):
		if len(device)==0:
			device = self.device
		maxIndex = self.bufferLength if self.bufferFilled else self.bufferIndex
		minIndex = int(maxIndex*percentageRange[0])
		maxIndex = int(maxIndex*percentageRange[1])
		indices = minIndex+np.random.choice(maxIndex-minIndex,batchSize) if (maxIndex-minIndex) > batchSize else np.arange(minIndex,maxIndex)
		#indices = np.random.randint(0,maxIndex,batchSize)
		returnedData =[[self.inputData[i][indices,:].to(device) for i in range(0,len(self.inputData))],
						[self.outputData[i][indices,:].to(device) for i in range(0,len(self.outputData))]]
		return returnedData
	def saveData(self):
		for i in range(len(self.inputData)):
			torch.save(self.inputData[i],self.saveDataPrefix+"simInputData"+str(i)+".pt")
		for i in range(len(self.outputData)):
			torch.save(self.outputData[i],self.saveDataPrefix+"simOutputData"+str(i)+".pt")
	def loadData(self,matchLoadSize=False):
		if matchLoadSize:
			self.inputData = []
			self.outputData = []
			i = 0
			while path.exists(self.loadDataPrefix+"simInputData"+str(i)+".pt"):
				self.inputData.append(torch.load(self.loadDataPrefix+"simInputData"+str(i)+".pt").to(self.device))
				i+=1
			i = 0
			while path.exists(self.loadDataPrefix+"simOutputData"+str(i)+".pt"):
				self.outputData.append(torch.load(self.loadDataPrefix+"simOutputData"+str(i)+".pt").to(self.device))
				i+=1
			self.bufferIndex = self.inputData[0].shape[0]
			self.bufferFilled = True
			self.bufferLength = self.bufferIndex
		else:
			data = torch.load(self.loadDataPrefix+"simInputData0.pt").to(self.device)
			self.bufferIndex = np.min(self.bufferLength,data.shape[0])
			self.bufferFilled = False if self.bufferIndex<self.bufferLength else True
			for i in range(len(self.inputData)):
				self.inputData[i][0:self.bufferIndex,:] = torch.load(self.loadDataPrefix+"simInputData"+str(i)+".pt").to(self.device)[0:self.bufferIndex,:]
			for i in range(len(self.outputData)):
				self.outputData[i][0:self.bufferIndex,:] = torch.load(self.loadDataPrefix+"simOutputData"+str(i)+".pt").to(self.device)[0:self.bufferIndex,:]
			print("data loaded buffer filled: " + str(self.bufferFilled) + " buffer index: " + str(self.bufferIndex))
	def inheritData(self,otherBuffer):
		while len(self.inputData)>0:
			del self.inputData[0]
		while len(self.outputData)>0:
			del self.outputData[0]
		torch.cuda.empty_cache()
		data = otherBuffer.getRandBatch(self.bufferLength)
		self.inputData = [data[0][i].to(self.device) for i in range(len(data[0]))]
		self.outputData = [data[1][i].to(self.device) for i in range(len(data[1]))]
		self.bufferIndex = self.inputData[0].shape[0]
		if self.bufferIndex == self.bufferLength:
			self.bufferFilled = True

class sequentialReplayBuffer(object):
	def __init__(self,bufferLength=0,sampleNNInput=[],sampleNNOutput=[],saveDataPrefix='',loadDataPrefix='',device = 'cpu'):
		# sets up new replay buffer with correct size
		self.device = device
		self.saveDataPrefix = saveDataPrefix
		self.loadDataPrefix = loadDataPrefix
		self.bufferLength = bufferLength
		self.inputData = []
		self.outputData = []
		self.lastSequenceLength = -1
		for data in sampleNNInput:
			self.inputData.append(torch.zeros((self.bufferLength,)+np.array(data).shape,device=self.device))
		for data in sampleNNOutput:
			self.outputData.append(torch.zeros((self.bufferLength,)+np.array(data).shape,device=self.device))
		self.purgeData()
	def purgeData(self): # delete everything in replay buffer
		self.bufferIndex = 0
		self.bufferFilled = False
		self.sequenceStartIndices = np.array([0])
		self.lastSequenceLength = -1
	def addData(self,nnInput,nnOutputGroundTruth,isSequenceEnd=False): # add new sample
		self.lastSequenceLength = -1
		if self.bufferFilled:
			self.deleteFirstData()
		inputData,outputData = self.processData(nnInput,nnOutputGroundTruth)
		for i in range(len(inputData)):
			self.inputData[i][self.bufferIndex,:] = inputData[i]
		for i in range(len(outputData)):
			self.outputData[i][self.bufferIndex,:] = outputData[i]
		self.bufferIndex+=1
		if isSequenceEnd:
			self.endTrajectory()
		if self.bufferIndex == self.bufferLength:
			self.bufferFilled = True
	def endTrajectory(self):
		self.sequenceStartIndices = np.unique(np.append(self.sequenceStartIndices,self.bufferIndex))
	def processData(self,nnInput,nnOutputGroundTruth): # format data to add to replay buffer
		inputData = []
		outputData = []
		for data in nnInput:
			if torch.is_tensor(data):
				inputData.append(data.to(self.device))
			else:
				inputData.append(torch.from_numpy(np.array(data)).to(self.device).unsqueeze(0).float())
		for data in nnOutputGroundTruth:
			if torch.is_tensor(data):
				outputData.append(data.to(self.device))
			else:
				outputData.append(torch.from_numpy(np.array(data)).to(self.device).unsqueeze(0).float())
		return inputData,outputData
	def deleteFirstData(self): # deletes very first sample to make room in replay buffer
		for i in range(len(self.inputData)):
			self.inputData[i][range(self.bufferLength-1),:] = self.inputData[i][range(1,self.bufferLength),:]
		for i in range(len(self.outputData)):
			self.outputData[i][range(self.bufferLength-1),:] = self.outputData[i][range(1,self.bufferLength),:]
		self.bufferIndex-=1
		self.sequenceStartIndices = np.unique(np.maximum(self.sequenceStartIndices-1,0))
		self.bufferFilled = False
	def getRandSequence(self,batchSize = 1,device='',percentageRange = [0.,1.]): # get single randomly selected sequence
		if len(device)==0:
			device = self.device
		if batchSize > self.bufferIndex:
			batchSize = self.bufferIndex
		randIndexOrder = np.random.choice(len(self.sequenceStartIndices),len(self.sequenceStartIndices),replace=False)
		data2pull = np.array([])
		batchStartIndices = np.array([0])
		while len(data2pull)<batchSize:
			startIndex = self.sequenceStartIndices[randIndexOrder[0]]
			if randIndexOrder[0]<len(self.sequenceStartIndices)-1:
				endIndex = self.sequenceStartIndices[randIndexOrder[0]+1]
			else:
				endIndex = self.bufferIndex
			data2pull = np.append(data2pull,np.arange(startIndex,endIndex))
			batchStartIndices = np.append(batchStartIndices,len(data2pull))
			randIndexOrder = randIndexOrder[range(1,len(randIndexOrder))]
		data2pull = data2pull[range(batchSize)]
		batchStartIndices = np.unique(np.append(batchStartIndices[range(len(batchStartIndices)-1)],len(data2pull)))
		returnedData =[[self.inputData[i][data2pull,:].to(device) for i in range(0,len(self.inputData))],
						[self.outputData[i][data2pull,:].to(device) for i in range(0,len(self.outputData))],
						batchStartIndices]
		return returnedData
	def getRandSequenceFixedLength(self,numSequences,sequenceLength,device='',percentageRange=[0.,1.]): # returns sequence data of fixed length as list of individual sequence data
		if len(device)==0:
			device = self.device
		if self.lastSequenceLength != sequenceLength:
			self.validStartIndices = []
			for i in range(len(self.sequenceStartIndices)-1):
				self.validStartIndices = self.validStartIndices+list(range(self.sequenceStartIndices[i],self.sequenceStartIndices[i+1]-sequenceLength))
			self.validStartIndices = np.array(self.validStartIndices)
			self.lastSequenceLength = sequenceLength
		startIndex = self.validStartIndices[self.validStartIndices >= int(self.bufferIndex*percentageRange[0])]
		startIndex = startIndex[startIndex < int(self.bufferIndex*percentageRange[1])-sequenceLength]
		startIndex = np.random.choice(startIndex,numSequences,replace=False)
		output=[]
		for i in range(sequenceLength):
			output.append([[self.inputData[j][startIndex+i,:].to(device) for j in range(0,len(self.inputData))],
						[self.outputData[j][startIndex+i,:].to(device) for j in range(0,len(self.outputData))]])
		return output
	def getRandSequenceFixedLengthInterleaved(self,sequenceLength,batchSize,device='',percentageRange=[0.,1.]): # returns sequence data of fixed length interleaved
		if len(device)==0:
			device = self.device
		if self.lastSequenceLength != sequenceLength:
			self.validStartIndices = []
			for i in range(len(self.sequenceStartIndices)-1):
				self.validStartIndices = self.validStartIndices+list(range(self.sequenceStartIndices[i],self.sequenceStartIndices[i+1]-sequenceLength))
			self.validStartIndices = np.array(self.validStartIndices)
			self.lastSequenceLength = sequenceLength
		startIndex = self.validStartIndices[self.validStartIndices >= int(self.bufferIndex*percentageRange[0])]
		startIndex = startIndex[startIndex < int(self.bufferIndex*percentageRange[1])-sequenceLength]
		startIndex = np.random.choice(startIndex,batchSize,replace=False)
		indices = torch.from_numpy(startIndex).repeat_interleave(sequenceLength) + torch.tensor(range(sequenceLength)).repeat(batchSize)
		#indices = torch.from_numpy(startIndex).repeat(sequenceLength) + torch.tensor(range(sequenceLength)).repeat_interleave(batchSize)
		output = [[self.inputData[i][indices,:].to(device) for i in range(len(self.inputData))],
				[self.outputData[i][indices,:].to(device) for i in range(len(self.outputData))]]
		return output
	def saveData(self): # save data to files
		for i in range(len(self.inputData)):
			torch.save(self.inputData[i][range(0,self.bufferIndex),:],self.saveDataPrefix+"sequentialSimInputData"+str(i)+".pt")
		for i in range(len(self.outputData)):
			torch.save(self.outputData[i][range(0,self.bufferIndex),:],self.saveDataPrefix+"sequentialSimOutputData"+str(i)+".pt")
		np.save(self.saveDataPrefix+"sequenceStartIndices.npy",self.sequenceStartIndices)
	def loadData(self,fixBufferLength = False): # load data from files
		if fixBufferLength:
			tempBuffer = sequentialReplayBuffer(loadDataPrefix=self.loadDataPrefix,device = self.device)
			tempBuffer.loadData(fixBufferLength = False)
			self.purgeData()
			self.inheritData(tempBuffer)
			del tempBuffer
		else:
			self.inputData = []
			self.outputData = []
			i = 0
			while path.exists(self.loadDataPrefix+"sequentialSimInputData"+str(i)+".pt"):
				self.inputData.append(torch.load(self.loadDataPrefix+"sequentialSimInputData"+str(i)+".pt").to(self.device))
				i+=1
			i = 0
			while path.exists(self.loadDataPrefix+"sequentialSimOutputData"+str(i)+".pt"):
				self.outputData.append(torch.load(self.loadDataPrefix+"sequentialSimOutputData"+str(i)+".pt").to(self.device))
				i+=1
			self.bufferIndex = self.inputData[0].shape[0]
			self.bufferFilled = True
			self.bufferLength = self.bufferIndex
			self.sequenceStartIndices = np.load(self.loadDataPrefix+"sequenceStartIndices.npy")
	def inheritData(self,otherBuffer,varyBufferLength=True): # add data from another replay buffer
		self.sequenceStartIndices = np.unique(np.append(self.sequenceStartIndices,self.bufferIndex))
		if varyBufferLength and (self.bufferIndex + otherBuffer.bufferIndex > self.bufferLength):
			self.bufferFilled = False
			toAdd = self.bufferIndex + otherBuffer.bufferIndex - self.bufferLength
			self.bufferLength = self.bufferLength + toAdd
			for i in range(len(self.inputData)):
				self.inputData[i] = torch.cat((self.inputData[i],torch.zeros((toAdd,)+tuple(self.inputData[i].shape[1:]),device=self.device)),dim=0)
			for i in range(len(self.outputData)):
				self.outputData[i] = torch.cat((self.outputData[i],torch.zeros((toAdd,)+tuple(self.outputData[i].shape[1:]),device=self.device)),dim=0)
		for i in range(otherBuffer.bufferIndex):
			newInput = [otherBuffer.inputData[j][i,:] for j in range(len(otherBuffer.inputData))]
			newOutput = [otherBuffer.outputData[j][i,:] for j in range(len(otherBuffer.outputData))]
			self.addData(newInput,newOutput,(i+1) in otherBuffer.sequenceStartIndices)
	def getCurrentSequenceLength(self): # get length of current sequence length
		return self.bufferIndex-self.sequenceStartIndices[-1]
	def purgeCurrentSequence(self): # delete current sequence. Deletes last sequence if current sequence is empty
		if self.getCurrentSequenceLength() == 0:
			if len(self.sequenceStartIndices) > 1:
				self.sequenceStartIndices = self.sequenceStartIndices[0:-1]
		else:
			self.bufferIndex = self.sequenceStartIndices[-1]
		self.bufferFilled = False

if __name__ == '__main__':
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load replay buffer
    cpuReplayBuffer = sequentialReplayBuffer(loadDataPrefix='simData/',saveDataPrefix='simData/')
    cpuReplayBuffer.loadData()
    dataBatch = cpuReplayBuffer.getRandSequenceFixedLength(10,5,percentageRange=[0.,0.1])




