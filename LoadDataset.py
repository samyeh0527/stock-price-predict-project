from os import read
import numpy
from numpy.core.numeric import tensordot
import  pandas as pd
from pandas import read_csv

class LoadData():
    
    def __init__(self,path1,path2) :
        self.path1 = path1
        self.path2 = path2
    def load_path(self):
        dataframe = read_csv(self.path1,usecols=[5])
        dataframe_2 = read_csv(self.path2,usecols=[5])
        dataframe["Adj Close2"] =dataframe_2
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        testVaild=dataset
        trainVaild=dataset
        # print(dataset)
        return testVaild,trainVaild,dataset



if __name__ == '__main__':
    test = LoadData('D:/2344/2344TW.csv','D:/2344/4919TW.csv')
    testVaild,trainVaild = test.load_path()
    print("testVaild ",testVaild,"\n\ntrainVaild  ",trainVaild)