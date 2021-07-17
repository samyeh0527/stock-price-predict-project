import numpy as np



class CreateDataset():
    def __init__(self,look_back,dataset) -> None:
        self.dataset = dataset
        self.look_back =2
    # convert an array of values into a dataset matrix
    def create_dataset(self):
        dataX, dataY = [], []
        for i in range(len(self.dataset)-self.look_back-1):
            a = self.dataset[i:(i+self.look_back),:]
            dataX.append(a)
            dataY.append(self.dataset[i + self.look_back,:])
        TrainX = np.array(dataX)
        Train_Y = np.array(dataY)

        return TrainX, Train_Y 







    
if __name__ == '__main__':
    pass

