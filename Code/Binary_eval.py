import os
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import DenseNet121, VGG16
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import scikitplot as skplt
#import matplotlib.pyplot as plt
import pickle

import cv2, glob, os
import numpy as np
import pandas as pd

#conv_base.summary()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))


model=Sequential()
model.add(conv_base)

model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

import matplotlib.pyplot as plt

import cv2, glob, os
import numpy as np
import pandas as pd

input_path_0 = "/home/mukesh/Documents/Datasets/MESSIDOR/0_1"
input_path_1 = "/home/mukesh/Documents/Datasets/MESSIDOR/1_1"
input_path_2 = "/home/mukesh/Documents/Datasets/MESSIDOR/2_1"
input_path_3 = "/home/mukesh/Documents/Datasets/MESSIDOR/3_1"
input_path_4 = "/home/mukesh/Documents/Datasets/MESSIDOR/4_1"
# import random
# random.shuffle(input_path_0)
test_files_0=glob.glob(os.path.join(input_path_0, "*.jpg"))
test_files_1=glob.glob(os.path.join(input_path_1, "*.jpg"))
test_files_2=glob.glob(os.path.join(input_path_2, "*.jpg"))
test_files_3=glob.glob(os.path.join(input_path_3, "*.jpg"))
test_files_4=glob.glob(os.path.join(input_path_4, "*.jpg"))

print(len(test_files_0))
print(len(test_files_1))
print(len(test_files_2))
print(len(test_files_3))
print(len(test_files_4))

model.load_weights('weights_new_after_30_sess.120-0.05.hdf5')

test_files=[]
test_files+=test_files_1
test_files+=test_files_2
test_files+=test_files_3
test_files+=test_files_4
print(len(test_files))

import random
random.shuffle(test_files_0)
random.shuffle(test_files)

PT = []
GT = []
predict=[]

tp=0
fn=0
i=0

threshold=0.2
examples_1=len(test_files_0)
examples_2=len(test_files)

while(i < examples_1):

    img = cv2.imread(test_files_0[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = model.predict(img)[0][0]
    predict.append(pre)
    val = 1 if pre > threshold else 0

    print(i,"0", pre)

    PT.append(val)
    GT.append(0)

    i+=1
i=0

while(i < examples_2):

    img = cv2.imread(test_files[i]) / 255.0
    #img = preprocess(test_files[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = model.predict(img)[0][0]
    predict.append(pre)
    val = 1 if pre > threshold else 0

    print(i,"1", pre)

    PT.append(val)
    GT.append(1)

    i+=1

with open('Ground_Truths.pkl', 'wb') as f1:
    pickle.dump(GT, f1)

with open('Predictes_values.pkl', 'wb') as f2:
    pickle.dump(predict, f2)

"""
#print(tp,fn)

i=0
fp=0
tn=0
while(i<100):
    img_1=cv2.imread(test_files_0[i])
    img_2=cv2.imread(test_files_0[i+1])
    list_new=[]
    img_1=np.asarray(img_1).astype(np.float16)
    img_2=np.asarray(img_2).astype(np.float16)

    img_1/=255.0
    img_2/=255.0

    list_new=[]

    list_new.append(img_1)
    list_new.append(img_2)
    img_1=np.asarray(img_1).astype(np.float16)
    img_2=np.asarray(img_2).astype(np.float16)

    ques=np.asarray(list_new).astype(np.float16)
    val=model.predict(ques)
    if(val[0]>=0.6):
        fp+=1
    else:
        tn+=1
    if(val[1]>=0.6):
        fp+=1
    else:
        tn+=1
    print(val[0],fp,tn)
    print(val[1],fp,tn)
    i+=2

print("tp = ",tp)
print("fn = ",fn)
print("tn = ",tn)
print("fp = ",fp)
"""

def get_scores(TN, FP, FN, TP):

    sensitivity = TP / (TP + FN)
    print("Sensitivity: ", round(sensitivity, 4))

    specificity = TN / (TN + FP)
    print("Specificity: ", round(specificity, 4))

    error_rate = (FP + FN) / (TP + FN + TN + FP)
    print("Error_rate: ", round(error_rate, 4))

    accuracy = 1 - error_rate
    print("Accuracy: ", round(accuracy, 4))

    precision = TP / (TP + FP)
    print("Precision: ", round(precision, 4))

    dice_coeff = (2 * TP) / (2 * TP + FP + FN)
    print("Dice_Coeff: ", round(dice_coeff, 4))

    jaccard = dice_coeff / (2 - dice_coeff)
    print("Jaccard: ", round(jaccard, 4))

    f_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    print("F_score: ", round(f_score, 4))


tn, fp, fn, tp = confusion_matrix(GT, PT, [0, 1]).ravel()
fpr,tpr,threshold=roc_curve(GT,predict)
# skplt.metrics.plot_roc_curve(GT, predict)
auc = roc_auc_score(GT, predict)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
print(fpr)
print(tpr)
print(threshold)
get_scores(tn, fp, fn, tp)
print("AUC : ",auc)
