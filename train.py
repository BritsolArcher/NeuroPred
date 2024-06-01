'''
Adapted from https://github.com/GEHAH/NeuroPred-CLQ
'''
from tensorflow.keras.optimizers import Adam

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical as labelEncoding   


from sklearn.metrics import (confusion_matrix, classification_report, matthews_corrcoef, precision_score, roc_curve, auc)
from sklearn.model_selection import (StratifiedKFold, KFold, train_test_split)
from scipy import interp
from model import ourmodel
import numpy as np

my_seed = 42
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)


model_name = 'BiLSTM'
model = ourmodel(model_name)
model.summary()

data1 = np.load('data/X.npz')
X1 = data1['x_train']
X2 = data1['x_test']
y_1 = pd.read_csv('data/Process_data/train/y_train.csv').to_numpy()
y1 = labelEncoding(y_1, dtype=int)
y_2 = pd.read_csv('data/Process_data/test/y_test.csv').to_numpy()
y2 = labelEncoding(y_2, dtype=int)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()

    config.gpu_options.allow_growth = True

    session = tf.compat.v1.Session(config=config)


    setEpochNumber = 150  
    setBatchSizeNumber = 32  
    ####################################################


    names = ['first']
    name = names[0]
    
    model = ourmodel(model_name)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    filepath = 'NeuroPred_model/%sModel.tf' % (name)


    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='max')
    callbacks_list = [checkpoint]
    back = EarlyStopping(monitor='accuracy', patience=20, verbose=1, mode='auto')
    results = model.fit(x=[X1],y=y1,
                            batch_size=setBatchSizeNumber, epochs=setEpochNumber,
                            verbose=1,
                            callbacks=[callbacks_list, back])

    model.save('NeuroPred_model/Model_%s.h5'%model_name)                       
    
    # end-for
