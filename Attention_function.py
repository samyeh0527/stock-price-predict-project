from keras.layers import Multiply
from keras.layers.core import *

def attention_function(inputs, single_attention_vector=False):
    #inputs.shape = (batch_size, TimeSteps, Dims) 
        
        TimeSteps = K.int_shape(inputs)[1]
        input_dim = K.int_shape(inputs)[2]
        a = Permute((2, 1))(inputs)
        a = Dense(TimeSteps, activation='softmax')(a)
        if single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1))(a)
            a = RepeatVector(input_dim)(a)

        a_probs = Permute((2, 1))(a)
        # element * wise
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul
