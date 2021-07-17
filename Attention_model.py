from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply,Concatenate,BatchNormalization,GRU
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from keras.layers import merge
from keras.layers.core import *
from Attention_function import attention_function

class Attention_model():
    def __init__(self,Dims,TimeSteps):
        self.TimeSteps =TimeSteps
        self.Dims =Dims

    def attention_model(self):
        inputs = Input(shape=(self.TimeSteps, self.Dims))
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

if __name__ == '__main__':
    test = Attention_model(2,2)
    result = test.attention_model()