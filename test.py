'''
Adapted from https://github.com/GEHAH/NeuroPred-CLQ
'''
import pandas as pd
from tensorflow.keras.utils import to_categorical as labelEncoding 
from sklearn.metrics import average_precision_score, f1_score,recall_score
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, roc_curve, auc,matthews_corrcoef,precision_score,accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp

from model import ourmodel
import numpy as np
my_seed = 42
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)


data1 = np.load('data/X.npz')
#X1 = data1['x_train']
X2 = data1['x_test']
y_1 = pd.read_csv('data/Process_data/train/y_train.csv').to_numpy()
y1 = labelEncoding(y_1, dtype=int)
y_2 = pd.read_csv('data/Process_data/test/y_test.csv').to_numpy()
y2 = labelEncoding(y_2,dtype=int)

model_name = 'BiLSTM'

model = ourmodel(model_name)
model.load_weights("NeuroPred_model/Model_%s.h5"%(model_name))

Accuracy = []
Sensitivity = []
Specificity = []
Precision = []
MCC = []


# Performance Metices:
Yactual = y_2
Yp = model.predict([X2])
v = Yp
Yp = Yp.argmax(axis=1)

accuracy = accuracy_score(Yactual,Yp)
Accuracy.append(accuracy)

CM = confusion_matrix(y_pred=Yp, y_true=Yactual)
print(CM)
np.savetxt("result/%s/CM/CM_%s.csv"%(model_name,model_name), CM, delimiter=",")
TN, FP, FN, TP = CM.ravel()

MCC.append(matthews_corrcoef(y_true=Yactual, y_pred=Yp))
Sensitivity.append(TP / (TP + FN))
Specificity.append(TN / (TN + FP))
Precision.append(precision_score(y_true=Yactual, y_pred=Yp))

TPR = []
meanFPR = np.linspace(0, 1, 100)

# ROC Curve
fpr, tpr, _ = roc_curve(Yactual, v[:, 1])
np.savetxt("result/%s/ROC/fpr_%s.csv"%(model_name,model_name), fpr, delimiter=",")
np.savetxt("result/%s/ROC/tpr_%s.csv"%(model_name,model_name), tpr, delimiter=",")
TPR.append(interp(meanFPR, fpr, tpr))
rocauc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold  (AUC = %0.4f)' % (rocauc))
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig('result/%s/rocauc.png'%model_name)

print('AUC:', rocauc)
print('Accuracy:', Accuracy)
print('Sensitivity: ', Sensitivity)
print('Specificity5: ', Specificity)
print('MCC:', MCC)
print('Precision: ', Precision)