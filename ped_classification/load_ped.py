import numpy as np

import scipy.io

import keras
import keras.backend as K

from sklearn.preprocessing import MinMaxScaler

import cv2


def load_ped_data(rgb=False):
	ped_data = scipy.io.loadmat("data/pca_ped_25x50.mat")
	ped_images = ped_data['ped_train_int_25x50'][:,1:]
	garb_images = ped_data['garb_train_int_25x50'][:,1:]
	ped_test_images = ped_data['ped_test_int_25x50'][:,1:]
	garb_test_images = ped_data['garb_test_int_25x50'][:,1:]
	del ped_data

	y_train = np.concatenate((np.ones(len(ped_images)), np.zeros(len(garb_images))))
	y_test = np.concatenate((np.ones(len(ped_test_images)), np.zeros(len(garb_test_images)))) 

	temp_train = np.vstack((ped_images, garb_images)).astype('float32')
	temp_test = np.vstack((ped_test_images, garb_test_images)).astype('float32')
	
	img_rows, img_cols = 50, 100

	temp_train = temp_train.reshape(-1, 25, 50)
	temp_test = temp_test.reshape(-1, 25, 50)
	
	x_train = np.zeros(shape=(temp_train.shape[0], img_rows, img_cols)).astype('float32')
	x_train_rgb = np.zeros(shape=(temp_train.shape[0], img_rows, img_cols, 3)).astype('float32')
	x_test = np.zeros(shape=(temp_test.shape[0], img_rows, img_cols)).astype('float32')
	x_test_rgb = np.zeros(shape=(temp_test.shape[0], img_rows, img_cols, 3)).astype('float32')

	for i in range(x_train.shape[0]):
		x_train[i] = cv2.resize(temp_train[i], (img_cols,img_rows))
		x_train_rgb[i] = cv2.cvtColor(x_train[i], cv2.COLOR_GRAY2RGB)
	for i in range(x_test.shape[0]):
		x_test[i] = cv2.resize(temp_test[i], (img_cols,img_rows))
		x_test_rgb[i] = cv2.cvtColor(x_test[i],cv2.COLOR_GRAY2RGB)

	return  x_train_rgb, x_train, y_train, x_test_rgb, x_test, y_test