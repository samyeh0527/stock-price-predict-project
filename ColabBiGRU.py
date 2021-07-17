
from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply,Concatenate,BatchNormalization,GRU
from keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.utils.vis_utils import plot_model
from keras import optimizers
import numpy
import  pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(7)
starttime = datetime.datetime.now()

def NormalizeMult(data):
    #normalize 
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')
    normalize = normalize.reshape(data.shape[1],2)
    # print(normalize.shape)
    for i in range(0,data.shape[1]):
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data,normalize

#反正規
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data
def attention_function(inputs):
   #inputs.shape = (batch_size, TimeSteps, Dims) 
    
    TimeSteps = K.int_shape(inputs)[1]
    a = Permute((2, 1))(inputs)
    a = Dense(TimeSteps, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    # element * wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

# convert an array of values into a dataset matrix
def create_dataset2(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back,:])
    TrainX = numpy.array(dataX)
    Train_Y = numpy.array(dataY)

    return TrainX, Train_Y 
# fix random seed for reproducibility
numpy.random.seed(1377)
# load the dataset
# dataframe = read_csv('D:/23/OilwithGold.csv')
# dataframe = dataframe.drop(['Date'], axis = 1)
dataframe = read_csv('D:/2344/2344TW.csv',usecols=[5])
dataframe_2 = read_csv('D:/2344/4919TW.csv',usecols=[5])
dataframe["Adj Close2"] =dataframe_2

dataset = dataframe.values
dataset = dataset.astype('float32')
testVaild=dataset
trainVaild=dataset


# split into train and test sets
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
testVaild = testVaild[train_size:len(dataset),:]
trainVaild = trainVaild[:train_size,:]

print(train.shape)
print(test.shape)
# normalize the dataset
dataset,da_normalize = NormalizeMult(dataset)
train,tr_normalize = NormalizeMult(train)
test,te_normalize = NormalizeMult(test)

look_back = 2
TimeSteps=look_back
Dims=2
trainX, trY = create_dataset2(train, look_back)
testX, teY = create_dataset2(test, look_back)

print(trainX.shape,trY.shape)
print(testX.shape,teY.shape)
trainY,testY =trY[:,0],teY[:,0]

# print(trainY.shape)
# print(testY.shape)


def attention_model():
    inputs = Input(shape=(TimeSteps, Dims))
    x = Conv1D(filters = 128, kernel_size = 1, activation = 'relu')(inputs)  
    attention = attention_function(x)
    BiGRU_out = Bidirectional(GRU(64, return_sequences=True,activation="relu"))(attention)
    Batch_Normalization = BatchNormalization()(BiGRU_out)
    Drop_out = Dropout(0.1)(Batch_Normalization)
    attention = attention_function(Drop_out)
    Batch_Normalization = BatchNormalization()(attention)
    Drop_out = Dropout(0.1)(Batch_Normalization)
    Flatten_ = Flatten()(Drop_out)
    output=Dropout(0.1)(Flatten_)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=[inputs], outputs=output)
    return model

m = attention_model()
m.summary()
m.compile(loss='mean_squared_error', optimizer='adam')
history = m.fit(trainX, trainY, epochs=1800, batch_size=128, verbose=1,validation_data=(testX, testY))

# make predictions
trainPredict = m.predict(trainX)
testPredict = m.predict(testX)
#FNormalize
trainPredict_FNormalizeMult= FNormalizeMult(trainPredict,tr_normalize)
testPredict_FNormalizeMult= FNormalizeMult(testPredict,te_normalize)

#calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('*  Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('*  Test Score: %.2f RMSE' % (testScore))
teV=testVaild[:-(look_back+1),0] 
trV=trainVaild[:-(look_back+1),0] 
print("D A C BiGRU")
print('*  look_back=',look_back)



# shift test predictions for plotting

                                         
plt.plot(teV)       
plt.plot(testPredict_FNormalizeMult,'r--')
plt.ylabel('price')
plt.xlabel('Date')
plt.savefig('teV')
plt.show()
testScore = math.sqrt(mean_squared_error(teV, testPredict_FNormalizeMult))
print('*  FNormalizeMult Test Score: %.2f RMSE' % (testScore))
plt.plot(trV)
plt.plot(trainPredict_FNormalizeMult,'r--')
plt.ylabel('price')
plt.xlabel('Date')
plt.savefig('trV')
plt.show()
trainScore = math.sqrt(mean_squared_error(trV, trainPredict_FNormalizeMult))
print('*  FNormalizeMult Train Score: %.2f RMSE' % (trainScore))



#損失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()