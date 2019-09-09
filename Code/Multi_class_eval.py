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
from sklearn.utils.multiclass import unique_labels
import seaborn as sn
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

x = Dense(5,activation='softmax')(model.layers[-2].output)
#o = Activation('sigmoid', name='loss')(x)

in_img = tensorflow.keras.layers.Input(shape=(224, 224, 3))

model2 = Model(inputs=model.input, outputs=x)
model2.summary()

#model.summary()

import matplotlib.pyplot as plt

import cv2, glob, os
import numpy as np
import pandas as pd

def estimate_radius(img):
    mx = img[img.shape[0] // 2,:,:].sum(1)
    rx = (mx > mx.mean() / 10).sum() / 2

    my = img[:,img.shape[1] // 2,:].sum(1)
    ry = (my > my.mean() / 10).sum() / 2

    return (ry, rx)

def subtract_gaussian_blur(img):
    # http://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html
    # http://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    gb_img = cv2.GaussianBlur(img, (0, 0), 5)

    return cv2.addWeighted(img, 4, gb_img, -4, 128)

def remove_outer_circle(a, p, r):
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)

    return a * b + 128 * (1 - b)


def crop_img(img, h, w):
        h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
        w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0

        crop_img = img[h_margin:h + h_margin,w_margin:w + w_margin,:]

        return crop_img

def place_in_square(img, r, h, w):
    new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    new_img += 128
    new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img

    return new_img
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



def predicted(list_new):
    #print(list_new)
    v=max(list_new)
    #print(v)
    if(list_new[4]>0.15):
        return 4
    if(list_new[3]>0.2):
        return 3
    if(list_new[2]>0.1):
        return 2
    if(list_new[0]>0.98):
        return 0
    else:
        return 1

def preprocess1(file):
    image=cv2.imread(file)
    b,g,r=cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(g)
    return cl1

def preprocess(f, r, debug_plot=False):
    try:
        img = cv2.imread(f)

        ry, rx = estimate_radius(img)

        img=preprocess1(f)
        img=cv2.merge((img,img,img))
        if debug_plot:
            plt.figure()
            plt.imshow(img)

        resize_scale = r / max(rx, ry)
        w = min(int(rx * resize_scale * 2), r * 2)
        h = min(int(ry * resize_scale * 2), r * 2)

        img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale,interpolation=cv2.INTER_LANCZOS4)

        img = crop_img(img, h, w)
        #print("crop_img", np.mean(img), np.std(img))

        if debug_plot:
            plt.figure()
            plt.imshow(img)

        #img = subtract_gaussian_blur(img)
        #img = remove_outer_circle(img, 0.9, r)
        img = place_in_square(img, r, h, w)

        if debug_plot:
            plt.figure()
            plt.imshow(img)
            print(img.shape)
        #print("Done")
        list_name=f.split('/')
        #path_new="/content/gdrive/My Drive/IISc DR Dataset/kaggle/image_classes/1_1"
        #print(path_new+'/'+list_name[-1])
        #status=cv2.imwrite(path_new+'/'+list_name[-1],img)
        #print(status)
        return img

    except Exception as e:
        print("file {} exception {}".format(f, e))

    return None

input_path_0 = "/home/mukesh/Documents/IISc/DATA/0_1"
input_path_1 = "/home/mukesh/Documents/IISc/DATA/1_1"
input_path_2 = "/home/mukesh/Documents/IISc/DATA/2_1"
input_path_3 = "/home/mukesh/Documents/IISc/DATA/3_1"
input_path_4 = "/home/mukesh/Documents/IISc/DATA/4_1"
# import random
# random.shuffle(input_path_0)
test_files_0=glob.glob(os.path.join(input_path_0, "*.jpeg"))
test_files_1=glob.glob(os.path.join(input_path_1, "*.jpeg"))
test_files_2=glob.glob(os.path.join(input_path_2, "*.jpeg"))
test_files_3=glob.glob(os.path.join(input_path_3, "*.jpeg"))
test_files_4=glob.glob(os.path.join(input_path_4, "*.jpeg"))

print(len(test_files_0))
print(len(test_files_1))
print(len(test_files_2))
print(len(test_files_3))
print(len(test_files_4))

model2.load_weights('Weights_multi.034-0.312.hdf5')

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
examples_0=len(test_files_0)
examples_1=len(test_files_1)
examples_2=len(test_files_2)
examples_3=len(test_files_3)
examples_4=len(test_files_4)

while(i < examples_0):

    img = cv2.imread(test_files_0[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = predicted(model2.predict(img)[0])
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

    pre = predicted(model2.predict(img)[0])
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

    pre = predicted(model2.predict(img)[0])
    predict.append(pre)
    #val = 1 if pre > threshold else 0

    print(i,"2", pre)

    PT.append(pre)
    GT.append(2)

    i+=1
i=0

while(i < examples_3):

    img = cv2.imread(test_files_3[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = predicted(model2.predict(img)[0])
    predict.append(pre)
    #val = 1 if pre > threshold else 0

    print(i,"3", pre)

    PT.append(pre)
    GT.append(3)

    i+=1
i=0


while(i < examples_4):

    img = cv2.imread(test_files_4[i]) / 255.0

    #img = preprocess(test_files_0[i], 112) / 255.0

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    pre = predicted(model2.predict(img)[0])
    predict.append(pre)
    #val = 1 if pre > threshold else 0

    print(i,"4", pre)

    PT.append(pre)
    GT.append(4)

    i+=1
i=0



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
class_names=[0,1,2,3,4]
plot_confusion_matrix(GT, PT, classes=class_names,
                      title='Confusion matrix, without normalization',normalize=False)

# Plot normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()
plt.savefig('confusion.png')
