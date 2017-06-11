'''
This program aims to predict VOLATILITY S&P 500 (^VIX) time series using LTSM.

The data set:
Historical data for VOLATILITY S&P 500 (^VIX) from Jan. 02, 2005 to Sep. 26, 2016, which can downloaded from
https://ca.finance.yahoo.com/q/hp?a=&b=&c=&d=8&e=27&f=2016&g=d&s=%5Evix&ql=1
'''
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import pandas as pd
from clean_sp_data import get_merged_data, read_sp, get_ordered_series

# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]

    vectorSeries = list(vectorSeries.as_matrix()[1:])

    for i in range(len(vectorSeries) - sequence_length+1):

    	subset = vectorSeries[i : i + sequence_length]

        matrix.append(subset)

    return matrix

# vector to store the time series
def store_as_vector(path):
	vector_vix = []
	with open(path_to_vix) as f:
	    next(f) # skip the header row
	    for line in f:
	        fields = line.split(',')
	        vector_vix.append(float(fields[6]))
	return vector_vix

def plot_and_save(y_test, shifted_value, predicted_values, asset):
	# plot the results
	fig = plt.figure()
	plt.plot(y_test + shifted_value)
	plt.plot(predicted_values + shifted_value)
	plt.xlabel('Date')
	plt.ylabel('{}'.format(asset))
	plt.show()
	fig.savefig('output_prediction.jpg', bbox_inches='tight')

	# save the result into txt file
	test_result = zip(predicted_values, y_test) + shifted_value
	np.savetxt('output_result.txt', test_result)


def build_the_model():
	# build the model
	model = Sequential()
	# layer 1: LSTM
	model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
	model.add(Dropout(0.2))
	# layer 2: LSTM
	model.add(LSTM(output_dim=100, return_sequences=False))
	model.add(Dropout(0.2))
	# layer 3: dense
	# linear activation: a(x) = x
	model.add(Dense(output_dim=1, activation='linear'))
	# compile the model
	model.compile(loss="mse", optimizer="rmsprop")

	return model


def subtract_the_mean(matrix_vix):
	# shift all data by mean
	matrix_vix = np.array(matrix_vix)
	shifted_value = matrix_vix.mean()
	matrix_vix -= shifted_value
	print "Data  shape: ", matrix_vix.shape
	return matrix_vix, shifted_value


def predict(df, sequence_length, asset):

 	# convert the vector to a 2D matrix
	matrix_vix = convertSeriesToMatrix(df, sequence_length)

	matrix_vix, shifted_value = subtract_the_mean(matrix_vix)

	# split dataset: 90% for training and 10% for testing
	train_row = int(round(0.9 * matrix_vix.shape[0]))

	train_set = matrix_vix[:train_row, :]

	# shuffle the training set (but do not shuffle the test set)
	# to build a model that is not dependent on the data collection process
	np.random.shuffle(train_set)

	# the training set
	X_train = train_set[:, :-1]

	# the last column is the true value to compute the mean-squared-error loss
	y_train = train_set[:, -1]

	# the test set
	X_test = matrix_vix[train_row:, :-1]
	y_test = matrix_vix[train_row:, -1]

	# the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

	model = build_the_model()

	# train the model
	model.fit(X_train, y_train, batch_size=512, nb_epoch=50, validation_split=0.05, verbose=1)

	# # evaluate the result
	test_mse = model.evaluate(X_test, y_test, verbose=1)

	# print "Full value of test_mse is {}".format(test_mse)

	print '\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test))

	# get the predicted values
	predicted_values = model.predict(X_test)
	num_test_samples = len(predicted_values)
	predicted_values = np.reshape(predicted_values, (num_test_samples,1))

	plot_and_save(y_test, shifted_value, predicted_values, asset)


if __name__ == "__main__":

	# random seed
	np.random.seed(1234)

	sp = get_ordered_series('SP')

	vx = get_ordered_series("VIX")

	# predict(vx, 20, "VIX")

	predict(sp, 20, "S&P")