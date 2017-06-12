from __future__ import division
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import pandas as pd
from vix_prediction import build_the_model, convertSeriesToMatrix, subtract_the_mean
from clean_sp_data import get_ordered_series

def get_matrices(sp, vx):
	'''
	Takes two pandas series,
		Constructs 20-day arrays,
		Normalizes the series,
	Returns two numpy arrays
	'''
	sequence_length = 20

	sp_arrays = convertSeriesToMatrix(sp, sequence_length)
	vx_arrays = convertSeriesToMatrix(vx, sequence_length)

	sp_matrix, sp_mean = subtract_the_mean(sp_arrays)

	vx_matrix, vx_mean = subtract_the_mean(vx_arrays)

	return sp_matrix, vx_matrix

def get_train_test_split(sp, vx):
	train_row = int(round(0.9 * sp.shape[0]))
	sp_train = sp[:train_row, :]
	vx_train = vx[:train_row, :]

	np.random.shuffle(sp_train)
	np.random.shuffle(vx_train)

	x_train = sp_train[:, :-1]
	y_train = vx_train[:, -1]

	x_test = sp[train_row:, :-1]
	y_test = vx[train_row:, -1]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	return x_train, y_train, x_test, y_test

def evaluate(model, test_x, test_y):
	test_mse = model.evaluate(test_x, test_y, verbose=1)
	# print '\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y))

def plot(model, test_x, test_y):

	predicted_values = model.predict(test_x)

	print predicted_values


	num_test_samples = len(predicted_values)

	predicted_values = np.reshape(predicted_values, (num_test_samples,1))

	fig = plt.figure()
	plt.plot(test_y)
	plt.plot(predicted_values)
	plt.xlabel('Date')
	plt.ylabel('VIX')
	plt.show()

def predict(sp, vx):

	sp, vx = get_matrices(sp, vx)

	train_x, train_y, test_x, test_y = get_train_test_split(sp, vx)

	model = build_the_model("S&P")

	model.fit(train_x, train_y,batch_size=512, nb_epoch=50, validation_split=0.05, verbose=1)

	# evaluate(model, test_x, test_y)

	plot(model, test_x, test_y)

	return model


if __name__ == "__main__":

	np.random.seed(1234)

	sp = get_ordered_series("SP")

	vx = get_ordered_series("VIX")

	m = predict(sp, vx)