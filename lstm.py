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
    
    
    





def adf_test(series, title=''):
    """
    Returning an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC') # .dropna() handles differenced data
    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)
    for key, val in result[4].items():
        out[f'critical value ({key})']=val
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


print(adf_test(dataset['BTC-USD']))
#print(adf_test(dataset['XRP-USD']))






dataC = df.iloc[:,:15]
test_obs = 50
train = dataC[:-test_obs]
test = dataC[-test_obs:]
#search for p
for i in range(1, 11):
    model = VAR(train)
    results = model.fit(i)
    print('Order =', i)
    print('AIC:', results.aic)
    print('BIC:', results.bic)
    print('FPE:', results.fpe)
    print('Hannan-Quinn:', results.hqic)
    print('Schwarz/BIC:', results.bic)
    print()
    
    
    



for c in range(1,16):

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
  # load datase
  values = dataset.values
  values = pd.DataFrame(values)
  cols = list(values)
  cols[c-1], cols[0] = cols[0], cols[c-1]
  values = values.iloc[:,cols]
  values = np.array(values)
  # integer encode direction
  # ensure all data is float
  values = values.astype('float32')
  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)
  # frame as supervised learning
  reframed = series_to_supervised(scaled, 1, 1)
  print(np.shape(reframed))
  # drop columns we don't want to predict
  reframed.drop(reframed.columns[list(range(41,80))], axis=1, inplace=True)
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
  model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
  model.add(LSTM(32, return_sequences=True))
  model.add(LSTM(32, return_sequences=False))
    
  print(train_X.shape[1])
  print(train_X.shape[2])
  model.add(Dense(40))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='rmsprop')
  # fit network
  history = model.fit(train_X, train_y, epochs=5, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
  # plot history
  column_names = list(dataset.columns)
  pyplot.plot(history.history['loss'], label='train')
  pyplot.plot(history.history['val_loss'], label='test')
  pyplot.title('Training and Testing Loss for Column: ' + column_names[c-1])
  pyplot.legend()
  pyplot.savefig('fig0.png', dpi=400)
  pyplot.show()

  # make a prediction
  yhat = model.predict(test_X)
  test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
  # invert scaling for forecast
  inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,0]
  # invert scaling for actual
  test_y = test_y.reshape((len(test_y), 1))
  inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,0]
  # calculate RMSE
  rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
  print('Test RMSE: %.3f' % rmse)
  
  import matplotlib.pyplot as plt
  

  plt.plot(inv_y[:82], label='True')
  plt.plot(inv_yhat[:82],  label='Predict')
  plt.plot(range(82,89), inv_yhat[82:89], label="forecast 7")
  plt.title('True and Predicted Values for Column: ' + column_names[c-1])
  plt.legend()
  plt.savefig('fig%d.png'%c, dpi=400)
  plt.show()

  result[c-1]=list(inv_yhat)
  import numpy as np
  rmse1 = np.append(rmse1, rmse)
  
  
  
  
  

