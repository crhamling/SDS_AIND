# Stock Udacity Imports
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt 
from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True  

# Stock Udacity Imports (more)
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  

# Selected Imports
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



# On a Windows Machine these come in handy for file systems and pathing issues.
#
import os, sys
sys.path.insert(0, os.path.abspath(".."))



# Utility Functions:

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
    
# define function to load train, test, and validation datasets
#
def load_dataset(path):
    data = load_files(path)
    # print("PATH: ", path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets    
    

# load train, test, and validation datasets
#
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# pre-process the data for Keras (long download)
#
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# load list of dog names
#
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# There are 133 different breeds of Dogs to be classified.
# >>> len(dog_names)
# 133



# print statistics about the dataset
#
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

# There are 133 total dog categories.
# There are 8351 total dog images.
# There are 6680 training dog images.
# There are 835 validation dog images.
# There are 836 test dog images.  
    
    
# Shapes of various Tensors
#
print(train_tensors.shape)
print(valid_tensors.shape)
print(test_tensors.shape)    
    
# (6680, 224, 224, 3)
# (835, 224, 224, 3)
# (836, 224, 224, 3)
    
    
    
# CONSTANTS

# DogBreed is a set of 6680 images 224x224 pixels on 3 channels
# There are 133 different breeds of dogs to be classified.
# 
IMG_CHANNELS = 3
IMG_ROWS = 224
IMG_COLS = 224

#constant
BATCH_SIZE = 128
NB_EPOCH = 11
NB_CLASSES = 133
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()   
    
    
### CNN - Network Architecture
### TODO: Define your architecture.

model = Sequential()

# kernel_size is an integer or tuple/list of two integers, specifying the width
# and height of the 2D convolution window (can be a single integer to specify the same value
# for all spatial dimensions)
# Note: I'm choosing 21 because it provides aproximately the same ratio as the CIFAR10 use 
# case of 10.6

model.add(Conv2D(113, kernel_size=5, padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
# Layer 2   
model.add(Conv2D(40, kernel_size=5, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
# Layer 3 
model.add(Conv2D(20, kernel_size=5, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 4
model.add(Flatten())
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary() 
    
    
    
# Compile the Model
#
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    


### TODO: specify the number of epochs that you would like to use to train the model.
# NB_EPOCH = 40, per the constants section.
epochs = NB_EPOCH

# Per CIFAR10 usecase
# model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
#    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, 
#    verbose=VERBOSE)


# TRAIN (Fit) the model
#

### Do NOT modify the code BELOW this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)  
    
    
### Do NOT modify the code ABOVE this line. 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









