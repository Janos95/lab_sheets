import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications
from keras import backend as K

from load_ped import load_ped_data

import hog


# dimensions of our images.
img_width, img_height = 25, 50

nb_train_samples = 3000
nb_validation_samples = 1000
epochs = 50
batch_size = 32

assert(K.image_data_format() == 'channels_last')
input_shape = (img_width, img_height, 1)


def compute_features():
    print("Loading pedestrian data")
    x_train_rgb, x_train, y_train, x_test_rgb, x_test, y_test = load_ped_data()
    print(x_train.shape)
    #print(x_train)
    #datagen = ImageDataGenerator()

    # build the VGG16 network
    print("Loading VGG16 pretrained model")
    model = applications.VGG16(include_top=False, weights='imagenet')

    #train_datagen = ImageDataGenerator(
    #    rotation_range=20,
    #    shear_range=0.2,
    #    zoom_range=0.2,
    #    horizontal_flip=True
    #    )

    print("computing vgg16 features and storing them!")
    bottleneck_features_train = model.predict(x_train_rgb)
    print(bottleneck_features_train.shape)
    np.save(open('bottleneck_features_train.npy', 'wb'),bottleneck_features_train)
    bottleneck_features_test = model.predict(x_test_rgb)
    np.save(open('bottleneck_features_test.npy', 'wb'),bottleneck_features_test)


    print("Computing hog features and saving them")
    hogs_train = np.zeros(shape=(x_train.shape[0], 1980))
    hogs_test = np.zeros(shape=(x_test.shape[0], 1980))
    for i in range(x_train.shape[0]):
        hogs_train[i] = hog.extract(x_train[i])
    for i in range(x_test.shape[0]):
        hogs_test[i] = hog.extract(x_test[i])

    np.save(open('hog_features_train.npy', 'wb'),
        hogs_train)
    np.save(open('hog_features_validation.npy', 'wb'),
        hogs_test)

    X_train = np.zeros(shape=(hogs_train.shape[0], hogs_train[0].shape[2]+bottleneck_features_train[1]*bottleneck_features_train[2]*bottleneck_features_train[3]))
    X_test = np.zeros(shape=(hogs_test.shape[0], hogs_test[0].shape[2]+bottleneck_features_test[1]*bottleneck_features_test[2]*bottleneck_features_test[3]))
    for i in range(X_train.shape[0]):
        X_train[i] = np.append(bottleneck_features_train[i].reshape(-1), hogs_train[i])
    for i in range(X_test.shape[0]):
        X_test[i] = np.append(bottleneck_features_test[i].reshape(-1), hogs_train[i])


    return X_train, y_train, X_test, y_test




def train_top_model(x_train, y_train, x_test, y_test):

    model = Sequential()
    model.add(Flatten(input_shape=x.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x, y,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights('~/')



#train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
#train_labels = np.array(
#    [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

x_train, y_train, x_test, y_test = compute_features()

train_top_model(x_train, y_train, x_test, y_test)
