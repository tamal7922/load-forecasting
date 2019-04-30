from __future__ import print_function
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

NAME = f"ANN-{int(time.time())}"
# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix
    
# random seed
np.random.seed(1234)
#MAKING TRAINING AND TESTING DATA SEQUENCE   
# load raw data
df_raw = pd.read_csv('hourly_load_2016.csv', header=None)
df_raw_array = df_raw.values
# daily load
list_daily_load = [df_raw_array[i,:] for i in range(0, len(df_raw)) if i % 25 == 0]
# hourly load (23 loads for each day)
list_hourly_load = [df_raw_array[i,1]/100000 for i in range(0, len(df_raw)) if i % 25 != 0]
# the length of the sequnce for predicting the future value
sequence_length = 24

# convert the vector to a 2D matrix
matrix_load = convertSeriesToMatrix(list_hourly_load, sequence_length)

# shift all data by mean
matrix_load = np.array(matrix_load)
shifted_value = matrix_load.mean()
matrix_load -= shifted_value
print ("Data  shape: ", matrix_load.shape)

# split dataset: 90% for training and 10% for testing
train_row = int(round(0.9 * matrix_load.shape[0]))
train_set = matrix_load[:train_row, :]

# shuffle the training set (but do not shuffle the test set)
np.random.shuffle(train_set)
# the training set
X_train = train_set[:, :-1]
print('X train', X_train.shape)

# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1]
print('y train', y_train.shape)

# the test set
X_test = matrix_load[train_row:, :-1]
print('X test shape', X_test.shape)

y_test = matrix_load[train_row:, -1]
print('y test shape', y_test.shape)

#Build Model
model= Sequential()
model.add(Dense(35, input_dim= 23, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='linear'))

#Model compilation Parameters
model.compile(loss="mse", optimizer="rmsprop", metrics=["mape"])

#Tensorboard initia;isation and folder where to save.
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

#Model Training.
model.fit(X_train, y_train, batch_size=215, nb_epoch=50,
          validation_split=0.05, verbose=1,
          callbacks= [tensorboard])

#Model Evaluation
test_mse = model.evaluate(X_test, y_test, verbose=1)
print ('\nANN MSE on the test data set is %.3f over %d test samples.' % (test_mse[0], len(y_test)))
print ('\nANN MAPE on the test data set is %.3f over %d test samples.' % (test_mse[1], len(y_test)))

# get the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))


# plot the results
fig = plt.figure()
plt.plot(y_test + shifted_value, label = 'actual value')
plt.plot(predicted_values + shifted_value, label = 'predicted value')
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e5)')
plt.title('ANN Load forecasting')
plt.legend(loc='upper right', fontsize= 'x-small')

plt.show()
fig.savefig(f'{NAME}.png', bbox_inches='tight')
fig.savefig(f'{NAME}.png')

#tensorboard --logdir=logs/

