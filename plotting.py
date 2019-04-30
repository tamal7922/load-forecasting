import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint



SEQ_LEN = 24  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "unix_data"
EPOCHS = 30 # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def preprocess_df(df):
   
    #print('preprocess df shape',df.shape)
    #print('P R E \n',df.head())
    
    val_u=df['load'].mean()
    val_s=df['load'].std()
    
    df['load'] = preprocessing.scale(df['load'].values)  # stdarisation
    df.dropna(inplace=True)  # cleanup again... jic.

    #print('post process df shape',df.shape)
    #print('P O S T \n',df.head())
    
    df['future'] = df['load'].shift(-FUTURE_PERIOD_PREDICT) # future added in the recieved df
    df.dropna(inplace=True) # cleanup for missing values . loss of last 3 data
    #print('after future add, df shape', df.shape)
    #print('24th ele search:',df.head(24))

    
    seq_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the future
        if len(prev_days) == SEQ_LEN:  # make sure we have 24 sequences!
            seq_data.append([np.array(prev_days), i[-1]])

    #seq_data has 2 columns. seq and future. seq in itself is an array of 24 seq.
    #the last data of every seq is the current load data, i.e. 24th data        
    #i=0;
    #for seq, future in seq_data:
    #    i=i+1
    #    print(i, seq[-1], future)
    #    if i==6:
    #       print(i)
    #       break
            
    random.shuffle(seq_data)  # shuffle for good measure
    #print('seq data shape', len(seq_data), len(seq_data[0])) # loss of 1st 23 data from df as back seq should be of 23
    
    incr = []  # list that will store our buy sequences and targets
    decr = []  # list that will store our sell sequences and targets

    #print('ex seq data',seq_data[0])

    for seq, future in seq_data:  # iterate over the sequential data
        if future < seq[-1]:  # future cmp to current value
            decr.append([seq, future])  # append entire 24 length seq and future
        elif future >= seq[-1]:  # otherwise if the target is a 1...
            incr.append([seq, future])  # it's a buy!

    random.shuffle(decr)  # shuffle the dec
    random.shuffle(incr)  # shuffle the incr

    lower = min(len(incr), len(decr))  # what's the shorter length?
    #print(f"length incr:{len(incr)}, lengtgh decr: {len(decr)}")

    incr = incr[:lower]  # make sure both lists are only up to the shortest length.
    decr = decr[:lower]  # make sure both lists are only up to the shortest length.

    seq_data = incr+decr  # add them together
    random.shuffle(seq_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    #print('incr + decr seq shape ', len(seq_data), len(seq_data[0]))
       
    X = []
    y = []
    
    
    for seq, future in seq_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(future)  # y is the targets/labels (buys vs sell/notbuy)
    X=np.array(X)
    y=np.array(y)
    
    print('X array', X.shape)
    print('y shape', y.shape)
    return X, y, val_u, val_s  # return X and y...and make X a numpy array!


main_df = pd.DataFrame() # begin empty

ratios = ["unix_data"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration

    ratio = ratio.split('.csv')[0]  # split away the ticker from the file-name
    print(ratio)
    dataset = f'load/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=["time", "load"])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    #df.rename(columns={"load": f"{ratio}_load"}, inplace=True)
    print('LOADED SAMPLE DATA .....')
    print(df.head())
    
    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[["load"]]  # ignore the other columns besides price and volume

main_df=df

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??


#print(main_df.head())
# here, split away some slice of the future data from the main main_df.
times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]
print(last_5pct)

test_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

print('Creating Training Dataset....' )
train_x, train_y ,train_u, train_s= preprocess_df(main_df)
print('Creating Validation Dataset...')
test_x, test_y, test_u, test_s = preprocess_df(test_main_df)

#MODEL 
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Dense(1, activation='linear')) #node = number pr pred columns



# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer='RMSProp',
    #metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")


# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    #validation_data=(validation_x, validation_y),
    callbacks=[tensorboard]
    )
    
# Score model
score = model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', score)
#print('Test score:', score[1])
# Save model
model.save(f"models/{NAME}.h5")

#get predicted values
predicted = model.predict(test_x) # to be improved with reference
print('shape',predicted.shape)
num_sample=len(predicted)
predicted=np.reshape(predicted, (num_sample, 1))

#plot the results
fig=plt.figure()
plt.plot((test_y*test_s)+test_u)
plt.plot((predicted*test_s) + test_u)
plt.xlabel('hour')
plt.ylabel('Electricity load')
plt.show()
fig.savefig(f'plot_{NAME}.png', bbbox_inches= 'tight')

#------------------------To be put on terminal -------------
#tensorboard --logdir=logs/   #to see on tensorboard
#virtualenv --system-site-packages -p python3 ./venv
#source ./venv/bin/activate  #activate virtual env
#deactivate




