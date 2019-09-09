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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import requests
from keras.callbacks import Callback
import slack

import matplotlib.pyplot as plt
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

import cv2


# model.compile(loss='categorical_crossentropy',optimizer=SGD_LR,metrics=['accuracy'])
model.load_weights('New_VGG.020-0.315.hdf5')

# history = model.fit_generator(generator=train_generator,
#                               steps_per_epoch=len(train_generator),
#                               validation_data = validation_generator,
#                               verbose=1,
#                               epochs=100,
#                               initial_epoch = 20,
#                               #class_weight=class_weight_multi,
#                               callbacks=callbacks)

img=cv2.imread('/home/mukesh/Documents/Datasets/Train_Kaggle_Split/4_class_test/4_1/9_left.jpeg')/255.0
img1=cv2.imread('/home/mukesh/Documents/Datasets/Train_Kaggle_Split/4_class_test/4_1/9_left.jpeg')
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
print(img.shape)
print(model.predict(img))

output_class = model.output[:, 1]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('max_pooling2d_1')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(output_class, last_conv_layer.output)[0]
print(grads.shape)
# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([img])

# We multiply each channel in the feature map array
# # by "how important this channel is" with regard to the elephant class
heatmap = np.mean(conv_layer_output_value, axis=-1)
for i in range(len(pooled_grads_value)):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
#plt.show()
plt.savefig('heatmap.png')


# We use cv2 to load the original image
#img=cv2.imread('/home/mukesh/Documents/Datasets/Train_Kaggle_Split/4_class_test/2_1/158_right.jpeg')
# We resize the heatmap to have the same size as the original image
img=img1
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 1.0
superimposed_img=superimposed_img.astype('uint8')
print(superimposed_img.shape)
# Save the image to disk
status=cv2.imwrite('9_left.png', superimposed_img)
print(status)