
import os
from typing import ValuesView

from keras.engine import training
from CreateDataset import *
from LoadDataset import LoadData
from NormalizeMult import NormalizeMult
from SplitTrainAndTest import Split
from Attention_model import Attention_model
from FNormalizeMult import FNormalizeMult
from Visualization import visualization
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

"""
            1.---CreateDataset.py               =>  {input:}    dataset,look_bac
            |                                   =>  {ootput:}   TrainX,  Train_Y 
            |
            2.---LoadDataset.py                 =>  {input:}    path1,path2
            |                                   =>  {output:}   testVaild,  trainVaild,  dataset
            |
main.py-----3.---SplitTrainAndTest.py           =>  {input:}    dataset,    testVaild,  trainVaild
            |                                       {output:}   train,  test,  testVaild,   trainVaild
            |
            4.---Normalize.py                   =>  {iunput:}   data
            |                                       {output:}   data,   normalize
            |
            5.---Attention_model.py             =>  {input:}    Dims,   TimeSteps
            |                                       {output:}   model
            |
            |
            6.---Fnormalize.py                  =>  {input:}    data,   normalize      
            |                                       {output:}   data
            |
            7.---Attention_function.py          =>  {input:}    inputs
            |                                       {output:}   output_attention_mul
            |
            8.---Visualization.py               =>  {input:}    Name,originalvalue,predict
                                                    {output:}   visualization image
"""



class main():
    np.random.seed(1377)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    def __init__(self) -> None:
        self.path1 = 'D:/2344/2344TW.csv'
        self.path2 = 'D:/2344/4919TW.csv'
        self.look_back = 2
        self.TimeSteps = self.look_back
        self.Dims = 2
        self.epochs = 1800
        self.batch_size= 268
        


    def run(self):

        testVaild,trainVaild ,dataset = LoadData(self.path1,self.path2).load_path()
        #print(testVaild,trainVaild,dataset)
        print(self.batch_size)
        split = Split(testVaild,trainVaild ,dataset)
        train , test, testVaild , trainVaild  = split.TrainAndTest()
        #print(train.shape , test.shape, testVaild.shape , trainVaild.shape)

        # normalize the dataset
       

        dataset,da_normalize =NormalizeMult(dataset).normalizemult()

        train,tr_normalize = NormalizeMult(train).normalizemult()
    
        test,te_normalize = NormalizeMult(test).normalizemult()

        createdataset =CreateDataset(self.look_back,train)
        trainX, trY = createdataset.create_dataset()
        #print(trainX.shape, trY.shape)
        
        createdataset =CreateDataset(self.look_back,test)
        testX, teY = createdataset.create_dataset()
        
        print(testX.shape, teY.shape)
        trainY,testY =trY[:,0],teY[:,0]

        attention_mode = Attention_model(self.Dims,self.TimeSteps)
        moedl_ = attention_mode.attention_model()
        moedl_.summary()
        moedl_.compile(loss='mean_squared_error', optimizer='adam')
        View_history = moedl_.fit(trainX, trainY, epochs=self.epochs, batch_size =self.batch_size, verbose=0,validation_data=(testX, testY))
        
        #make predictions
        trainPredict = moedl_.predict(trainX)
        testPredict = moedl_.predict(testX)
        

        trainPredict_FNormalizeMult = FNormalizeMult(trainPredict,tr_normalize).FNormalizeMult_()
        testPredict_FNormalizeMult = FNormalizeMult(testPredict,te_normalize).FNormalizeMult_()
        

        
        # #calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        print('*  Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY, testPredict))
        print('*  Test Score: %.2f RMSE' % (testScore))
        teV=testVaild[:-(self.look_back+1),0] 
        trV=trainVaild[:-(self.look_back+1),0] 
        print("D A C BiGRU")
        print('*  look_back=',self.look_back)

        Datavisualization = visualization('tev',teV,testPredict_FNormalizeMult).visualization()
        testScore = math.sqrt(mean_squared_error(teV, testPredict_FNormalizeMult))
        print('*  FNormalizeMult Test Score: %.2f RMSE' % (testScore))
        Datavisualization = visualization('trV',trV,trainPredict_FNormalizeMult).visualization()
        trainScore = math.sqrt(mean_squared_error(trV, trainPredict_FNormalizeMult))
        print('*  FNormalizeMult Train Score: %.2f RMSE' % (trainScore))
        #損失值
        plt.plot(View_history.history['loss'])
        plt.plot(View_history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
if __name__ == '__main__':
    run = main()
    start = run.run()