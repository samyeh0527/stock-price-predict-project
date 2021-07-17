# split into train and test sets
from LoadDataset import *

class Split():
    def __init__(self,dataset,testVaild,trainVaild):
        self.dataset = dataset
        self.testVaild = testVaild 
        self.trainVaild = trainVaild



    def TrainAndTest(self):
        train_size = int(len(self.dataset) * 0.9)
        test_size = len(self.dataset) - train_size


        train, test = self.dataset[0:train_size,:], self.dataset[train_size:len(self.dataset),:]
        
        testVaild = self.testVaild[train_size:len(self.dataset),:]
        trainVaild = self.trainVaild[:train_size,:]
        print(train.shape)
        print(test.shape)
        return train, test,testVaild,trainVaild




if __name__ == '__main__':
    run = Split()
    start = run.TrainAndTest()