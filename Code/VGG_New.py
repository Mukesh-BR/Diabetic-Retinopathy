import os
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
#from tensorflow.keras.engine.input_layer import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import DenseNet121, VGG16
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import requests
from keras.callbacks import Callback
import slack


conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(896,896,3))
conv_base.summary()


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

# model.add(Conv2D(256,(3,3),padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Conv2D(256,(3,3),padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(4,activation='softmax'))
model.summary()

import keras
import random

train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        validation_split=0.2)

#test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/home/mukesh/Documents/Datasets/Train_Kaggle_Split/4_class_test',
        target_size=(896,896),
        batch_size=4,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        '/home/mukesh/Documents/Datasets/Train_Kaggle_Split/4_class_test',
        target_size=(896,896),
        batch_size=4,
        class_mode='categorical',
        subset='validation')

experiment_id="Python_4_"
SGD_LR=tensorflow.keras.optimizers.Adadelta(lr=0.005)


#NotifyCB = Notify()

callbacks = [
    tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=25, verbose=1),
    tensorflow.keras.callbacks.ModelCheckpoint("New_VGG.{epoch:03d}-{loss:.3f}.hdf5", monitor='loss', verbose=1, mode='auto'),
    tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6),
    tensorflow.keras.callbacks.TensorBoard(log_dir='./log_Trial', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    #NotifyCB
]

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """


    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

#weights = np.array([ 0.30888676,  1.65486389, 10.04971098, 17.01174168])
classes=[0,1,2,3]
cl_ex=[0 for i in range(28043)] +[ 1 for i in range(13094)] +[ 2 for i in range(8277)]+[3 for i in range(7628)]
weights=np.asarray(compute_class_weight('balanced',classes,cl_ex))
print(weights)

model.compile(loss=weighted_categorical_crossentropy(weights),optimizer=SGD_LR,metrics=['accuracy'])
model.load_weights('New_VGG.026-0.376.hdf5')

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=len(train_generator),
                              validation_data = validation_generator,
                              verbose=1,
                              epochs=100,
                              initial_epoch = 26,
                              #class_weight=class_weight_multi,
                              callbacks=callbacks)

