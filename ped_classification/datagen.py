from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint

import numpy as np

import hog

import numpy as np

import scipy.io

import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import multiprocessing as mp

from sklearn.preprocessing import MinMaxScaler

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.18, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


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
#input_hog = Input(shape=(360,), name='input2')

x = Conv2D(16, kernel_size=(3,3))(input_image)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.3)(x)

x = Conv2D(32, kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.3)(x)

x = Conv2D(64, kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.3)(x)

x = Flatten()(x)

#x = concatenate([x, input_hog])

x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)
#model = Model(inputs=[input_image, input_hog], outputs=out)
model = Model(inputs=input_image, outputs=out)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

x_train, y_train, x_test, y_test = load_ped_data()

train_datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

test_datagen = ImageDataGenerator()


def generator(flow):
    while 1:
        x_train, y_train = flow.next()
        x_hogs = np.zeros(shape=(x_train.shape[0], 360))
        for i in range(x_hogs.shape[0]):
            x_hogs[i] = hog.extract(x_train[i].reshape(img_width,img_height))
        yield(({'input1': x_train, 'input2': x_hogs}, y_train))

flow_train = generator(train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True))
flow_test = generator(test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=True))

callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.20, verbose=1),
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint('checkpoint.h5', monitor='val_loss', save_best_only=True, verbose=0),
]
model.fit_generator(
        flow_train,
        steps_per_epoch=3000 // batch_size,
        epochs=200,
        validation_data=flow_test,
        validation_steps=1000 // batch_size,
        class_weight = {0:1 , 1:3},
        callbacks=callbacks)


#get_feature_output = K.function([model.layers[0].input, model.layers[17].input],
#                                  [model.layers[18].output])
get_feature_output = K.function([model.layers[0].input],
                                  [model.layers[16].output])


hog_train = np.zeros(shape=(x_train.shape[0], 360))
hog_test = np.zeros(shape=(x_test.shape[0], 360))
for i in range(x_train.shape[0]):
    hog_train[i] = hog.extract(x_train[i].reshape(25,50))
for i in range(x_test.shape[0]):
    hog_test[i] = hog.extract(x_test[i].reshape(25,50))


features_train = np.append(get_feature_output([x_train, hog_train])[0], hog_train, axis=1)
features_test = np.append(get_feature_output([x_test, hog_test])[0], hog_test, axis=1)

print(features_train.shape)
print(features_test.shape)

#features_train = get_feature_output([x_train, hog_train])[0]
#features_test = get_feature_output([x_test, hog_test])[0]

pca = PCA(features_train.shape[1])
features_train_transformed = pca.fit_transform(features_train)
features_test_transformed = pca.transform(features_test)

def fitSVM(i):
    print("Fitting model using {} principal components".format(i))
    clf = LinearSVC(random_state=0, dual=False)
    clf.fit(features_train_transformed[:,:i],y_train) 
    y_pred = clf.predict(features_test_transformed[:,:i])
    score_test = accuracy_score(y_test, y_pred)
    return (score_test, i)

pool = mp.Pool(processes=4)
results = pool.map(fitSVM, range(10,features_train.shape[1],25))

results.sort(key=lambda triple: triple[1])

print(results)