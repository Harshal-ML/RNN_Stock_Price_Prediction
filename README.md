# RNN_Stock_Price_Prediction
Dataset was obtained from Kaggle link = (https://www.kaggle.com/datasets/medharawat/google-stock-price?select=Google_Stock_Price_Train.csv)
Single independent variable of opening price of the stock was used.
The opening price of 5 years was divided on 2-months basis, i.e 60 timesteps and 1 output. So 60days price is stored as short term memory.
After data pre-processing. Sequential class of Tesorflow library and LSTM, Dense classes of layers module was used to create an recurrent neural network.
Drop-out rate of 20% was used to make the model more robust.
Four layers of LSTM cells with 50 units each was used to create the network.
Adam optimizer was used with mean squared error as loss function.
Results are also attached.
