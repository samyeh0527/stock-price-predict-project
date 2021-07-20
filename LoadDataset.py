from os import read
import numpy
from numpy.core.numeric import tensordot
import  pandas as pd
from pandas import read_csv
from Fakedataset import *

class LoadData():
    
    def __init__(self,path1,path2) :
        self.path1 = path1
        self.path2 = path2
    def load_path(self):
        """
        Note!!!

        
        the function is Add fake data to dataset  and load dataset 
        
        """
        dataframe = read_csv(self.path1,usecols=[5])
        data_to_list = dataframe["Adj Close"].tolist()
        data_to_list.append(Fakedataset(data_to_list).Add_fake_oneday_median())
        dataframe = pd.DataFrame(data_to_list)


        dataframe_2 = read_csv(self.path2,usecols=[5])
        data_to_list = dataframe_2["Adj Close"].tolist()
        data_to_list.append(Fakedataset(data_to_list).Add_fake_oneday_median())
        dataframe_2 = pd.DataFrame(data_to_list)

        #concat two Dataframe 
        dataframe[""] =dataframe_2
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        testVaild=dataset
        trainVaild=dataset
        return testVaild,trainVaild,dataset






if __name__ == '__main__':
    testVaild,trainVaild,dataset = LoadData('D:/2344/2344TW.csv','D:/2344/4919TW.csv').load_path()
    

