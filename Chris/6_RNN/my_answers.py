import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from string import ascii_lowercase

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    for i in range(len(series)-window_size):
        X.append(series[i:i+window_size])
    
    y = series[window_size:]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(4,input_shape = (window_size,1)))
    model.add(Dense(1))
    return model


def cleaned_text(text):
    punc = ['!', ',', '.', ':', ';', '?',' ']
    ints = ['0','1','2','3','4','5','6','7','8','9']
    allowed_chars = punc + list(ascii_lowercase) + ints
    remove = set(text) - set(allowed_chars)
    
    for char in remove:
        text = text.replace(char,'')
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    i = 0
    while i < len(text) - window_size:
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(250, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model
