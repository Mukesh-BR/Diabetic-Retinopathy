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

from sklearn.utils.multiclass import unique_labels

#import matplotlib.pyplot as plt
import cv2, glob
import numpy as np


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

#conv_base.summary()
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

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(896,896,3))


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

model.load_weights('./bestweights.hdf5')





input_path_0 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\0"
input_path_1 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\1"
input_path_2 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\2"
input_path_3 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\3"
input_path_4 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\4"

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

PT = []
GT = []
predict=[]

tp=0
fn=0
i=0

threshold=0.1
examples_0=len(test_files_0)
examples_1=len(test_files_1)
examples_2=len(test_files_2)
examples_3=len(test_files_3)
examples_4=len(test_files_4)

while(i < examples_0):

    img = cv2.imread(test_files_0[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = model.predict(img)[0][0]
    predict.append(1-pre)
    val = 0 if pre > threshold else 1

    print(i,"0", pre)

    PT.append(val)
    GT.append(0)

    i+=1
i=0

while(i < examples_1):

    img = cv2.imread(test_files_1[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = model.predict(img)[0][0]
    predict.append(1-pre)
    val = 0 if pre > threshold else 1

    print(i,"1", pre)

    PT.append(val)
    GT.append(1)

    i+=1
i=0

while(i < examples_2):

    img = cv2.imread(test_files_2[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = model.predict(img)[0][0]
    predict.append(1-pre)
    val = 0 if pre > threshold else 1

    print(i,"2", pre)

    PT.append(val)
    GT.append(1)

    i+=1
i=0
#
while(i < examples_3):

    img = cv2.imread(test_files_3[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = model.predict(img)[0][0]
    predict.append(1-pre)
    val = 0 if pre > threshold else 1

    print(i,"3", pre)

    PT.append(val)
    GT.append(1)

    i+=1
i=0

#
while(i < examples_4):

    img = cv2.imread(test_files_4[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = model.predict(img)[0][0]
    predict.append(1-pre)
    val = 0 if pre > threshold else 1

    print(i,"4", pre)

    PT.append(val)
    GT.append(1)

    i+=1
i=0





print("Idrid test Database NDR")
tn, fp, fn, tp = confusion_matrix(GT, PT, [0, 1]).ravel()
fpr,tpr,threshold=roc_curve(GT,predict)
# skplt.metrics.plot_roc_curve(GT, predict)
auc = roc_auc_score(GT, predict)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#print(fpr)
#print(tpr)
print("Threshold : ",threshold)
get_scores(tn, fp, fn, tp)
print("AUC : ",auc)
