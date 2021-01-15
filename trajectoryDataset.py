import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import csv

class trajectoryDataset(Dataset):
    """
    This class stores trajectory data for the robot. It can be used with torch's dataloaders to return
    samples for training
    """
    def __init__(self, csv_file, root_dir,sampleLength=None, startMidTrajectory=False,staticDataIndices = [],device='cpu'):
        self.to(device)
        self.root_dir = root_dir
        csvData = pd.read_csv(root_dir+csv_file)
        self.trajFiles = np.asarray(csvData.iloc[:,0])
        self.trajLengths = np.asarray(csvData.iloc[:,1])
        if sampleLength is None:
            sampleLength = np.min(self.trajLengths)
        self.setSampleLength(sampleLength,startMidTrajectory)
        self.preloaded=False
        self.staticDataIndices = staticDataIndices

    def to(self,device):
        """ choose device (cpu or gpu) that the data should be loaded to"""
        self.device = device

    def preloadToDevice(self):
        """
        Load data to device in the beginning for faster return of samples. In practice, this function usually
        can't be called due to limited memory
        """
        self.preloadedData = [torch.load(self.root_dir+trajFile,map_location=self.device) for trajFile in self.trajFiles]
        self.preloaded = True

    def __len__(self):
        return self.sampleIndices[-1]

    def __getitem__(self, idx):
        trajToUse = np.arange(self.sampleIndices.shape[0])[self.sampleIndices>idx].min()
        if self.startMidTrajectory:
            startIndex = idx if trajToUse==0 else idx-self.sampleIndices[trajToUse-1]
        else:
            startIndex = 0
        endIndex = startIndex+self.sampleLength
        if self.preloaded:
            loadedData = self.preloadedData[trajToUse]
        else:
            loadedData = torch.load(self.root_dir+self.trajFiles[trajToUse],map_location=self.device)
        output = tuple()
        for i in range(len(loadedData)):
            if i in self.staticDataIndices:
                output = output + (loadedData[i],)
            else:
                output = output + (loadedData[i][startIndex:endIndex,:],)
        return output#tuple(loadedData[i][startIndex:endIndex,:] for i in range(len(loadedData)))

    def setSampleLength(self,length,startMidTrajectory=None):
        """ Set trajectory length of samples """
        self.sampleLength = length
        if not startMidTrajectory is None:
            self.startMidTrajectory = startMidTrajectory
        if self.startMidTrajectory:
            samplesPerTraj = self.trajLengths+1-length
            samplesPerTraj[samplesPerTraj<0] = 0
        else:
            samplesPerTraj = np.zeros(self.trajLengths.shape)
            samplesPerTraj[self.trajLengths>=length] = 1
        self.sampleIndices = np.cumsum(samplesPerTraj).astype(int)

if __name__=="__main__":
    # write sample data
    root_dir = 'data0/'
    csv_file_name = 'meta.csv'
    csvFile = open(root_dir+csv_file_name, 'w', newline='')
    csvWriter = csv.writer(csvFile,delimiter=',')
    csvWriter.writerow(['filename','trajectoryLengths'])
    for i in range(1,10):
        filename = 'data'+str(i)+'.pt'
        torch.save((torch.ones(i,2)*i,torch.ones(i,3)*i,torch.ones(i,4)*i),root_dir+filename)
        csvWriter.writerow([filename,i])
    csvFile.flush()
    # example of sampling trajectories from dataset
    tempData = trajectoryDataset(csv_file_name,root_dir,sampleLength=1,startMidTrajectory=True)
    from torch.utils.data import Dataloader
    a = DataLoader(tempData,batch_size=1)
    print(next(iter(a)))