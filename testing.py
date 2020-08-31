import json

from flask import Flask
from jinja2.runtime import to_string
from tensorflow.python.keras.models import load_model

app = Flask(__name__)
import time
import os

from tensorflow.python.keras.saving.saved_model_experimental import sequential
import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras.layers import LSTM
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import deque
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

N_STEPS = 80
# Lookup step, 1 is the next day
LOOKUP_STEP = 50
np.random.seed(314)
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 500

@app.route('/cryptopal/<string:coin_id>',methods=['GET', 'POST'])
def index(coin_id):

    coin_id= coin_id+"-USD"
    model_name = f"2020-08-26_{coin_id}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"
    #ticker_data_filename = os.path.join("data", f"{coin_id}_{date_now}.csv")

    def plot_graph(model, data):
        y_test = data["y_test"]
        X_test = data["X_test"]
        y_pred = model.predict(X_test)
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
        plt.plot(y_test[-200:], c='b')
        plt.plot(y_pred[-200:], c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.show()

    def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
                  test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
        """
        Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
        Params:
            ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
            n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
            scale (bool): whether to scale prices from 0 to 1, default is True
            shuffle (bool): whether to shuffle the data, default is True
            lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
            feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
        """
        # see if ticker is already a loaded stock from yahoo finance
        if isinstance(ticker, str):
            # load it from yahoo_fin library
            df = si.get_data(ticker)
        elif isinstance(ticker, pd.DataFrame):
            # already loaded, use it directly
            df = ticker
        else:
            raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

        # this will contain all the elements we want to return from this function
        result = {}
        # we will also return the original dataframe itself
        result['df'] = df.copy()

        # make sure that the passed feature_columns exist in the dataframe
        for col in feature_columns:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."

        if scale:
            column_scaler = {}
            # scale the data (prices) from 0 to 1
            for column in feature_columns:
                scaler = preprocessing.MinMaxScaler()
                df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler

            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler

        # add the target column (label) by shifting by `lookup_step`
        df['future'] = df['adjclose'].shift(-lookup_step)

        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[feature_columns].tail(lookup_step))

        # drop NaNs
        df.dropna(inplace=True)

        sequence_data = []
        sequences = deque(maxlen=n_steps)

        for entry, target in zip(df[feature_columns].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])

        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
        # this last_sequence will be used to predict in future dates that are not available in the dataset
        last_sequence = list(sequences) + list(last_sequence)
        # shift the last sequence by -1
        last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
        # add to result
        result['last_sequence'] = last_sequence

        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # reshape X to fit the neural network
        X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

        # split the dataset
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)
        # return the result
        return result

    data = load_data(coin_id, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                     feature_columns=FEATURE_COLUMNS, shuffle=False)

    def get_accuracy(model, data):
        print("we are at get accuracy")
        y_test = data["y_test"]
        X_test = data["X_test"]
        y_pred = model.predict(X_test)
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
        y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP],
                          y_pred[LOOKUP_STEP:]))
        y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP],
                          y_test[LOOKUP_STEP:]))
        return accuracy_score(y_test, y_pred)

    def predict(model, data, classification=False):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][:N_STEPS]
        # retrieve the column scalers
        column_scaler = data["column_scaler"]
        # reshape the last sequence
        last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
        return predicted_price

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path+'/results', model_name) + ".h5"

    name ='BTC'

    path = '%s/results/2020-08-26_%s-USD-huber_loss-adam-LSTM-seq-80-step-60-layers-3-units-256.h5'%(dir_path, name)
    print(path)
    model = load_model(path)
    model.summary()
    model.load_weights(path)
    # evaluate the model
    mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    print("Mean Absolute Error:", mean_absolute_error)

    # predict the future price
    future_price = predict(model, data)
    print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    print("Accuracy Score:", get_accuracy(model, data))
    #plot_graph(model, data)

    dict = {"name":str(coin_id),"FuturePrice":str(future_price), "Accuracy":str(get_accuracy(model,data))}
    json_dump = json.dumps(dict)
    return json_dump

if __name__ == "__main__":
    app.run(debug=True)