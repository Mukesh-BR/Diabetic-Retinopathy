import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import DenseNet121, VGG16
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.utils.multiclass import unique_labels
import seaborn as sn
#import matplotlib.pyplot as plt
import pickle

import cv2, glob, os
import numpy as np
import pandas as pd

#conv_base.summary()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(896,896,3))
#conv_base.summary()


model=Sequential()
model.add(conv_base)

model.add(Conv2D(128,(3,1),padding='same'))
model.add(Conv2D(128,(1,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128,(3,1),padding='same'))
model.add(Conv2D(128,(1,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D())

model.add(Conv2D(128,(3,1),padding='same'))
model.add(Conv2D(128,(1,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128,(3,1),padding='same'))
model.add(Conv2D(128,(1,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(4,activation='softmax'))
model.summary()

model.load_weights('/home/mukesh/Documents/IISc/bestweights.hdf5')
#model.summary()

import matplotlib.pyplot as plt

import cv2, glob, os
import numpy as np
import pandas as pd

# def predicted(list_new):
#     print(list_new)
#     v=max(list_new)
#     print(v)
#     if(v==list_new[0]):
#         return 0
#     elif(v==list_new[1]):
#         return 1
#     elif(v==list_new[2]):
#         return 2
#     elif(v==list_new[3]):
#         return 3
#     else:
#         return 4
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm*=100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#
def predicted(list_new):
    print(list_new)
    #v=max(list_new)
    #print(v)
    if(list_new[2]>0.2):
        return 2

    if(list_new[3]>0.15):
        return 3

    if(list_new[1]>0.1):
        return 1

    return 0

# def predicted(list_new):
#     #print(list_new)
#     #v=max(list_new)
#     #print(v)
#     # if(list_new[3]>0.2):
#     #     return 3
#     if(list_new[2]>0.1):
#         return 2
#     if(list_new[1]>0.5):
#         return 1
#     return 0



input_path_0 = "/home/mukesh/Documents/Datasets/Messidor/0_2"
input_path_1 = "/home/mukesh/Documents/Datasets/Messidor/1_2"
input_path_2 = "/home/mukesh/Documents/Datasets/Messidor/2_2"
#input_path_3 = "/home/mukesh/Documents/Datasets/MESSIDOR/4_class_test_2/3_4_2"
#input_path_4 = "/home/mukesh/Documents/Datasets/iDRiD/Test/4_1"
# import random
# random.shuffle(input_path_0)
test_files_0=glob.glob(os.path.join(input_path_0, "*.jpeg"))
test_files_1=glob.glob(os.path.join(input_path_1, "*.jpeg"))
test_files_2=glob.glob(os.path.join(input_path_2, "*.jpeg"))
#test_files_3=glob.glob(os.path.join(input_path_3, "*.jpg"))
#test_files_4=glob.glob(os.path.join(input_path_4, "*.jpeg"))
'''
input_path_0 = "/home/mukesh/Documents/Datasets/iDRiD/Train/4_class_test/0_1_2"
input_path_1 = "/home/mukesh/Documents/Datasets/iDRiD/Train/4_class_test/1_2_2"
input_path_2 = "/home/mukesh/Documents/Datasets/iDRiD/Train/4_class_test/2_3_2"
input_path_3 = "/home/mukesh/Documents/Datasets/iDRiD/Train/4_class_test/3_4_2"
#input_path_4 = "/home/mukesh/Documents/Datasets/iDRiD/Test/4_1"
# import random
# random.shuffle(input_path_0)
test_files_0=glob.glob(os.path.join(input_path_0, "*.jpg"))
test_files_1=glob.glob(os.path.join(input_path_1, "*.jpg"))
test_files_2=glob.glob(os.path.join(input_path_2, "*.jpg"))
test_files_3=glob.glob(os.path.join(input_path_3, "*.jpg"))
#test_files_4=glob.glob(os.path.join(input_path_4, "*.jpeg"))
'''
print(len(test_files_0))
print(len(test_files_1))
print(len(test_files_2))
#print(len(test_files_3))
#print(len(test_files_4))


test_files=[]
test_files+=test_files_1
test_files+=test_files_2
#test_files+=test_files_3
#test_files+=test_files_4
print(len(test_files))

import random

random.shuffle(test_files_0)
random.shuffle(test_files_1)
random.shuffle(test_files_2)

PT = []
GT = []
predict=[]

tp=0
fn=0
i=0

threshold=0.2
examples_0=len(test_files_0)
examples_1=len(test_files_1)
examples_2=len(test_files_2)
#examples_3=len(test_files_3)
#examples_4=len(test_files_4)

while(i < examples_0):

    img = cv2.imread(test_files_0[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = predicted(model.predict(img)[0])
    predict.append(pre)
    #val = 1 if pre > threshold else 0

    print(i,"0", pre)

    PT.append(pre)
    GT.append(0)

    i+=1
i=0

while(i < examples_1):

    img = cv2.imread(test_files_1[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = predicted(model.predict(img)[0])
    predict.append(pre)
    #val = 1 if pre > threshold else 0

    print(i,"1", pre)

    PT.append(pre)
    GT.append(1)

    i+=1
i=0

while(i < examples_2):

    img = cv2.imread(test_files_2[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = predicted(model.predict(img)[0])
    predict.append(pre)
    #val = 1 if pre > threshold else 0

    print(i,"2", pre)

    PT.append(pre)
    GT.append(2)

    i+=1
i=0

'''
while(i < examples_3):

    img = cv2.imread(test_files_3[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = predicted(model.predict(img)[0])
    predict.append(pre)
    #val = 1 if pre > threshold else 0

    print(i,"3", pre)

    PT.append(pre)
    GT.append(3)

    i+=1
i=0
'''
#
# while(i < examples_4):
#
#     img = cv2.imread(test_files_4[i]) / 255.0
#
#     #img = preprocess(test_files_0[i], 112) / 255.0
#
#     img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
#
#     pre = predicted(model2.predict(img)[0])
#     predict.append(pre)
#     #val = 1 if pre > threshold else 0
#
#     print(i,"4", pre)
#
#     PT.append(pre)
#     GT.append(4)
#
#     i+=1
# i=0



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


mat=confusion_matrix(GT, PT,[0,1,2,3,4])
# print(mat)
# plt.matshow(mat)
# plt.savefig('confusion.png')
# fpr,tpr,threshold=roc_curve(GT,predict)
# skplt.metrics.plot_roc_curve(GT, predict)
# auc = roc_auc_score(GT, predict)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()
# print(fpr)
# print(tpr)
# print(threshold)
# get_scores(tn, fp, fn, tp)
# print("AUC : ",auc)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names=[0,1,2]
plot_confusion_matrix(GT, PT, classes=class_names,
                      title='Confusion matrix, without normalization',normalize=True)

# Plot normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()
#plt.savefig('confusion_4_class.png')
