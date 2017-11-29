# from __future__ import print_function
import numpy as np
import config
import MyMnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import os
from sklearn.utils import shuffle

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 38
nb_epoch = 12

# Sets source wieghts and net
src = "C:\\NNModels\\3"
dst = "C:\\NNModels\\4"
wieghts_name = 'wieghts.h5'
json_name = 'nn.json'

# number of convolutional filters to use
nb_filters = 32

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 3

# Reads images and lables
img, lbl = MyMnist.read()
train_amt = 2000
test_amt = lbl.size - train_amt

# the data, shuffled and split between tran and test sets
lbl_img = zip(img, lbl)
lbl_img = shuffle(lbl_img)
lbl = np.asarray([t[1] for t in lbl_img])
img = np.asarray([t[0] for t in lbl_img]).reshape((lbl.size, config.height * config.width))

train_img = np.asarray(img[:train_amt,:])
test_img = np.asarray(img[train_amt:,:])
train_lbl = np.asarray(lbl[:train_amt], 'uint8')
test_lbl = np.asarray(lbl[train_amt:], 'uint8')

# Reshape data to model needs
train_img = np.asarray(train_img, 'float32').reshape((int(train_img.shape[0]) ,1, config.height,config.width))
test_img = np.asarray(test_img, 'float32').reshape((int(test_img.shape[0]),1, config.height,config.width))
train_img /= 255
test_img /= 255

print('train shape:', train_img.shape)
print(train_img.shape[0], 'train samples')
print(test_img.shape[0], 'test samples')

# convert class vectors to binary class matrices
train_lbl = np_utils.to_categorical(train_lbl, nb_classes)
test_lbl = np_utils.to_categorical(test_lbl, nb_classes)

model = Sequential()

# Builds network
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, config.width, config.height)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Loads wieghts
model.load_weights(os.path.join(src, wieghts_name))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(train_img, train_lbl, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(test_img, test_lbl))
score = model.evaluate(test_img, test_lbl, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Saves data
json = model.to_json()
open(os.path.join(dst, json_name), 'w').write(json)
model.save_weights(os.path.join(dst, wieghts_name))