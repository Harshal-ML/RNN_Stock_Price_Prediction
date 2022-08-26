# Data-Preprocessing
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset_train = pd.read_csv("F:\\Machine learning\\Deep learning\\Part 3 - Recurrent Neural Networks\\Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler 
# MinMaxScaler is used for Data Normalisation
sc = MinMaxScaler(feature_range = (0,1))
train_set = sc.fit_transform(training_set)

# Creating Datastructure with 60 time stamp & 1 output
X_train =[] #it will contain previous 60 stockprices based on which next prediction will be made
y_train = [] # it will contain 61st day stock price with which model will calculate loss/cost function
for i in range(60, 1258):
    X_train.append(train_set[i-60:i,0])
    y_train.append(train_set[i,0])
X_train , y_train = np.array(X_train), np.array(y_train)
#this line of code converts lists to array

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
#tuple inside reshape denotes A 3D tensor with shape [batch, timesteps, feature]

# Builing the RNN
#Importing Keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Initializing RNN
rnn = Sequential()

# Adding 1st LSTM layer and some drop-out regularisation
rnn.add(LSTM(units = 50,return_sequences = True, input_shape= (X_train.shape[1],1) ))
# return_sequences = True only if we are about to add more LSTM layers in the RNN
rnn.add(Dropout(0.2))
#drop-out is used to avoid overfitting and to reduce complexity of running many models 
# dropout causes output from some nodes of a layer to be dropped so that network works with input from limited number of nodes

# Adding 2nd LSTM layer and drop-out regularisation
rnn.add(LSTM(units = 50, return_sequences=True))
rnn.add(Dropout(0.2))

# 3rd LSTM layer with drop-out regularisation
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(0.2))

# 4th LSTM layer with dropout regularisation
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(0.2))

# Adding the output layer
rnn.add(Dense(units = 1, activation = "linear"))
# activation = linear because we are doing regression
# because the dense class layer is always connected to all the neurons of previous layers

# compiling the RNN
rnn.compile(optimizer = "adam", loss = "mean_squared_error")
#adam optimizer usses stochastic gradient descent and performs relevent update of the weights

# Fitting RNN on to training set
rnn.fit(X_train , y_train , epochs= 100, batch_size = 32)

# MAking Predictions and Visualizing
# Getting Acttual Stock price values
test_dataset = pd.read_csv("F:\\Machine learning\\Deep learning\\Part 3 - Recurrent Neural Networks\\Google_Stock_Price_Test.csv")
real_stock_price = test_dataset.iloc[:,1:2]

# Getting predicted Stock price values

# we need to concatenate dataframes created while reading using pandas library
# doing this because we can not scale the values of test set because data leakage can take place
total_dataset=pd.concat((dataset_train["Open"], test_dataset["Open"]),axis =0)
# Horizontal concatenation axis = 1
# Vertical concatenation axis = 0
inputs = total_dataset.iloc[len(dataset_train)-60 :].values
# to predict a perticular day price we require previous 60 day prices
inputs = inputs.reshape(-1,1) #reshape because we did not use iloc function
inputs = sc.transform(inputs) # only transform will use training set value of Max and Min to scale test set
# now create X_test and Y_test out of these inputs in the correct shape as expected by RNN
X_test =[]
# doing only prediction so no y_test not required
for n in range (0, len(inputs)):
    X_test.append(inputs[n:n+60,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
predicted_stock_price = rnn.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#visualizing