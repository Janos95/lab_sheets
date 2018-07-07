from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras import backend as K

import numpy as np

import hog

import numpy as np

import scipy.io

import keras
import keras.backend as K

from sklearn.preprocessing import MinMaxScaler


def load_ped_data():
    ped_data = scipy.io.loadmat("data/pca_ped_25x50.mat")
    ped_images = ped_data['ped_train_int_25x50'][:,1:]
    garb_images = ped_data['garb_train_int_25x50'][:,1:]
    ped_test_images = ped_data['ped_test_int_25x50'][:,1:]
    garb_test_images = ped_data['garb_test_int_25x50'][:,1:]
    del ped_data

    y_train = np.concatenate((np.ones(len(ped_images)), np.zeros(len(garb_images))))
    y_test = np.concatenate((np.ones(len(ped_test_images)), np.zeros(len(garb_test_images)))) 

    x_train = np.vstack((ped_images, garb_images)).astype('float32')
    x_test = np.vstack((ped_test_images, garb_test_images)).astype('float32')

    x_train = x_train.reshape(-1, 25, 50, 1)
    x_test = x_test.reshape(-1, 25, 50, 1)
    return  x_train, y_train, x_test, y_test

# dimensions of our images.
img_width, img_height = 25, 50
nb_train_samples = 3000
nb_validation_samples = 1000
epochs = 50
batch_size = 16

input_image = Input(shape=(img_width,img_height,1), name='input1')
input_hog = Input(shape=(360,), name='input2')

x = Conv2D(16, kernel_size=(5,5))(input_image)
#x = BatchNormalization()(x)
x = Activation('elu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
#x = Dropout(0.3)(x)

x = Conv2D(32, kernel_size=(5,5))(x)
#x = BatchNormalization()(x)
x = Activation('elu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
#x = Dropout(0.3)(x)
x = Flatten()(x)

#concatenate x and angle 
x = concatenate([x, input_hog])

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[input_image, input_hog], outputs=out)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

x_train, y_train, x_test, y_test = load_ped_data()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

test_datagen = ImageDataGenerator(
        rescale=1./255
    )


def generator(flow):
    while 1:
        x_train, y_train = flow.next()
        x_hogs = np.zeros(shape=(x_train.shape[0], 360))
        for i in range(x_hogs.shape[0]):
            x_hogs[i] = hog.extract(x_train[i].reshape(img_width,img_height))
        yield(({'input1': x_train, 'input2': x_hogs}, y_train))

flow_train = generator(train_datagen.flow(x_train, y_train, batch_size=batch_size))
flow_test = generator(test_datagen.flow(x_test, y_test, batch_size=batch_size))


model.fit_generator(
        flow_train,
        steps_per_epoch=3000 // batch_size,
        epochs=50,
        validation_data=flow_test,
        validation_steps=1000 // batch_size)

model.save_weights('checkpoint.h5')  