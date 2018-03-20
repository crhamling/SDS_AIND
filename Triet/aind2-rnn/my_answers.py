import numpy as np
from string import ascii_lowercase

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


#  fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    last_window_index = len(series) - window_size
    for i in range(0, last_window_index):
        window_end = i + window_size
        X.append(series[i:window_end])
        y.append(series[window_end])
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

#  build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    #layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model.add(LSTM(5, input_shape=(window_size, 1)))
    #layer 2 uses a fully connected module with one unit
    model.add(Dense(1))
    return model


###  return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    alphabet = list(ascii_lowercase)
    
    allowed_chars = set(punctuation + alphabet + [' '])
    atypical_chars = set(text) - allowed_chars

    for char in atypical_chars:
        text = text.replace(char,' ')
        
    return text

### fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    i = 0
    last_window_index = len(text) - window_size
    while i < last_window_index:
        window_end = i + window_size
        inputs.append(text[i:window_end])
        outputs.append(text[window_end])
        i += step_size

    return inputs,outputs

# build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    #layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    #layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    model.add(Dense(num_chars))
    #layer 3 should be a softmax activation ( since we are solving a multiclass classification)
    model.add(Activation('softmax'))
    return model
