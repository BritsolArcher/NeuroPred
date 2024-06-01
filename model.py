'''
Adapted from https://github.com/GEHAH/NeuroPred-CLQ
'''
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import LSTMCell
from keras.layers import RNN
from tensorflow.keras.experimental import PeepholeLSTMCell
from keras.layers import Dense, Input, Dropout
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.models import Model

from layers import MultiHeadAttention,ScaledDotProductAttention
import numpy as np

my_seed = 42
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)



def ourmodel(model_name):
    in_put = Input(shape=(97, 150))
    a = Convolution1D(128, 3, activation='relu', padding='valid')(in_put)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    c = Dropout(0.2)(c)
    if model_name == 'GRU':
        d = GRU(128, return_sequences=True)(c)
    elif model_name == 'LSTM':
        d = LSTM(128,return_sequences=True)(c)
    elif model_name == 'BiGRU':
        d = Bidirectional(GRU(128,return_sequences=True))(c)
    elif model_name == 'BiLSTM':
        d = Bidirectional(LSTM(128,return_sequences=True))(c)
    elif model_name == 'Peephole-LSTM':
        d = RNN(PeepholeLSTMCell(128),return_sequences=True)(c)
    elif model_name == 'CNN':
        d = c
    
    d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
                           return_multi_attention=False, name='Multi-Head-Attention')(d)
    # d = ScaledDotProductAttention(name='Scaled-Attention')(d)
    d = Flatten()(d)
    e = Dense(128, activation='relu', name='FC3')(d)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC1')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])