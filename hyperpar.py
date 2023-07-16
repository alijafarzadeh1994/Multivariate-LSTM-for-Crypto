from pandas import read_csv
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# load data
dataset = read_csv('dataset.csv', index_col=0)
dataset = dataset.iloc[:370,:]
result = [[0 for x in range(7)] for y in range(15)]
rmse1 = []
df = dataset
#def swap_cols(arr, frm, to):
    #arr[:,[frm, to]] = arr[:,[to, frm]]
    
    
    
    
import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from itertools import product

# define the hyperparameters to search
num_layers = [1, 2, 3, 4]
num_nodes = [16, 32, 64, 128, 256, 512]
dropout_rates = [0.0, 0.1, 0.2]
optimizers = ['adam', 'rmsprop', 'SGD']
best_rmse = float('inf')
best_model_summary = ''

for num_layer in num_layers:
    for node_combination in product(num_nodes, repeat=num_layer):
        for dropout_rate, optimizer in product(dropout_rates, optimizers):
            print(f"Training model with {num_layer} layer(s), nodes: {node_combination}, dropout rate: {dropout_rate}, optimizer: {optimizer}")

            # convert series to supervised learning
            def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
                n_vars = 1 if type(data) is list else data.shape[1]
                df = DataFrame(data)
                cols, names = list(), list()
                # input sequence (t-n, ... t-1)
                for i in range(n_in, 0, -1):
                    cols.append(df.shift(i))
                    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
                # forecast sequence (t, t+1, ... t+n)
                for i in range(0, n_out):
                    cols.append(df.shift(-i))
                    if i == 0:
                        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                    else:
                        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
                # put it all together
                agg = concat(cols, axis=1)
                agg.columns = names
                # drop rows with NaN values
                if dropnan:
                    agg.dropna(inplace=True)
                return agg

            # load dataset
            # Replace 'dataset' with the actual data you want to use

            values = dataset.values
            values = pd.DataFrame(values)
            cols = list(values)
            cols[num_layer - 1], cols[0] = cols[0], cols[num_layer - 1]
            values = values.iloc[:, cols]
            values = np.array(values)

            # ensure all data is float
            values = values.astype('float32')

            # normalize features
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)

            # frame as supervised learning
            reframed = series_to_supervised(scaled, 1, 1)

            # drop columns we don't want to predict
            reframed.drop(reframed.columns[list(range(41, 80))], axis=1, inplace=True)

            # split into train and test sets
            values = reframed.values
            n_train_hours = 280
            train = values[:n_train_hours, :]
            test = values[n_train_hours:, :]

            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]

            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

            # design network
            model = Sequential()
            model.add(LSTM(node_combination[0], input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))

            for node in node_combination[1:]:
                model.add(LSTM(node, return_sequences=True))
                model.add(Dropout(dropout_rate))

            model.add(LSTM(node_combination[-1], return_sequences=False))
            model.add(Dense(40))
            model.add(Dense(1))

            model.compile(loss='mse', optimizer=optimizer)

            # fit network
            history = model.fit(train_X, train_y, epochs=10, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

            # make a prediction
            yhat = model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

            # invert scaling for forecast
            inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:, 0]

            # invert scaling for actual
            test_y = test_y.reshape((len(test_y), 1))
            inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:, 0]

            # calculate RMSE
            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
            print('Test RMSE: %.3f' % rmse)

            # check if this model has the lowest RMSE so far
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_summary = model.summary()




