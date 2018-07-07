from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras import backend as K

import numpy as np

from load_ped import load_ped_data

import hog

# dimensions of our images.
img_width, img_height = 50, 100

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 3000
nb_validation_samples = 1000
epochs = 50
batch_size = 16

input_image = Input(shape=(50,100,1), name='input1')
input_hog = Input(shape=(1980,), name='input2')

x = Conv2D(16, kernel_size=(3,3))(input_image)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.3)(x)

x = Conv2D(32, kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.3)(x)
x = Flatten()(x)

#concatenate x and angle 
x = concatenate([x, input_hog])

x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[input_image, input_hog], outputs=out)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

x_train_rgb, x_train, y_train, x_test_rgb, x_test, y_test = load_ped_data()
hogs_train = np.zeros(shape=(x_train.shape[0], 1980))
hogs_test = np.zeros(shape=(x_test.shape[0], 1980))
for i in range(x_train.shape[0]):
    hogs_train[i] = hog.extract(x_train[i])
for i in range(x_test.shape[0]):
    hogs_test[i] = hog.extract(x_test[i])
x_train = x_train.reshape(-1,50,100,1)
x_test = x_test.reshape(-1,50,100,1)

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
        x_hogs = np.zeros(shape=(x_train.shape[0], 1980))
        for i in range(x_hogs.shape[0]):
            x_hogs[i] = hog.extract(x_train[i].reshape(50,100))
        yield(({'input1': x_train, 'input2': x_hogs}, y_train))

flow_train = generator(train_datagen.flow(x_train, y_train, batch_size=batch_size))
flow_test = generator(test_datagen.flow(x_test, y_test, batch_size=batch_size))


model.fit_generator(
        flow_train,
        steps_per_epoch=3000 // batch_size,
        epochs=50,
        validation_data=flow_test,
        validation_steps=1000 // batch_size)
model.save_weights('first_try.h5')  