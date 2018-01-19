import numpy as np
from string import ascii_lowercase

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
#
# and window-size into a set of input/output pairs for use with our RNN model
# https://stackoverflow.com/questions/36937869/sliding-window-over-pandas-dataframe
#
# Implement the function called window_transform_series in my_answers.py so that it runs 
# a sliding window along the input series and creates associated input/output pairs. 
# Note that this function should input a) the series and b) the window length, and return 
# the input/output subsequences. Make sure to format returned input/output as generally 
# shown in table above (where window_size = 5), and make sure your returned input is a 
# numpy array.
# Inspiration / R&D:
# https://www-m15.ma.tum.de/foswiki/pub/M15/Allgemeines/SummerSchool2016/perea_lect1.pdf
# https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator
# 
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Create a simple while loop that iterated through the series and 
    # creates one array of arrays (of window_size) that contains the 
    # base for the time series and another array that contains the series itself.
    i = 0
    while i < len(series) - window_size:
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
        i += 1    
    
    # reshape each 
    #
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
#
# This is a rather simple stacked (sequential) RNN that has an LSTM layer of 5 Memory
# Cells. The input dimmension will be 1 feature with 5 time steps, and the output
# will be a vector of 1 value which will be interpreted to be the next time step.
# FYI - The loss metric is included in the compile phase. 
#
# Use Keras to quickly build a two hidden layer RNN of the following specifications
# layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
# layer 2 uses a fully connected module with one unit
# the 'mean_squared_error' loss should be used (remember: we are performing regression here)
#
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
#
# This is a quick and dirty function to leverage the pythonic methods of questionable 
# character removal via list and sets (for fast lookups). Effectively, we define a list 
# of characters we want to keep, a list of the alphabet, and put the entire text into a 
# python "set" where we can do fast operations. Then we remove the unwanted characters
# from the set by removing everything but what we intend on keeping via the replace function
# on a character by character basis.
#
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    # alphabet = list(ascii_lowercase)
    alphabet = list(ascii_lowercase)
    allowed_chars = set(punctuation + alphabet + [' '])

    all_chars = set(text)
    chars_to_remove = all_chars - allowed_chars

    for char_to_remove in chars_to_remove:
        text = text.replace(char_to_remove,' ')

    return text

    
    
### TODO: fill out the function below that transforms the 
# input text and window-size into a set of input/output pairs for use with our RNN model
#
# The core of this function is exactly the core of it's doppelganger above except it does
# not require the extra step of reshaping the arrays to a form the NN requires.
#
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # Create a simple while loop that iterated through the series and 
    # creates one array of arrays (of window_size) that contains the 
    # base for the time series and another array that contains the series itself.
    i = 0
    while i < len(text) - window_size:
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size
        
    return inputs,outputs

    
    
# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
#
#
# Build a 3 layer RNN model of the following specification
#
# layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
# layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
# layer 3 (integrated) should be a softmax activation ( since we are solving a multiclass classification)
# Use the categorical_crossentropy loss - performed at compile step.
#
#
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model

    
  
    
    
    
    
    
    
    
    
    