# Stock Udacity Imports
from sklearn.datasets import load_files      
from keras import applications, optimizers
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


### TODO: Obtain bottleneck features from another pre-trained CNN.
# Takes a while to Load
#  https://alexisbcook.github.io/2017/using-transfer-learning-to-classify-images-with-keras/

### VGG16
#
# bottleneck_features_VGG16 = np.load('bottleneck_features/DogVGG16Data.npz')
# train_VGG16 = bottleneck_features_VGG16['train']
# valid_VGG16 = bottleneck_features_VGG16['valid']
# test_VGG16 = bottleneck_features_VGG16['test']

### Xception
#
bottleneck_features_Xception = np.load('bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features_Xception['train']
valid_Xception = bottleneck_features_Xception['valid']
test_Xception = bottleneck_features_Xception['test']

### Resnet50
#
# bottleneck_features_Resnet50 = np.load('bottleneck_features/DogResnet50Data.npz')
# train_Resnet50 = bottleneck_features_Resnet50['train']
# valid_Resnet50 = bottleneck_features_Resnet50['valid']
# test_Resnet50 = bottleneck_features_Resnet50['test']

### VGG19
#
# bottleneck_features_VGG19 = np.load('bottleneck_features/DogVGG19Data.npz')
# train_VGG19 = bottleneck_features_VGG19['train']
# valid_VGG19 = bottleneck_features_VGG19['valid']
# test_VGG19 = bottleneck_features_VGG19['test']

### InceptionV
#
# bottleneck_features_InceptionV3 = np.load('bottleneck_features/DogInceptionV3Data.npz')
# train_InceptionV3 = bottleneck_features_InceptionV3['train']
# valid_InceptionV3 = bottleneck_features_InceptionV3['valid']
# test_InceptionV3 = bottleneck_features_InceptionV3['test']



### Model Architecture
# The model uses the the pre-trained VGG-16 model as a fixed feature extractor, 
# where the last convolutional output of VGG-16 is fed as input to our model. 
# We only add a global average pooling layer and a fully connected layer, 
# where the latter contains one node for each dog category and is equipped with a softmax.
    
    
### CNN - Network Architecture
### TODO: Define your architecture.

# VGG16_model = Sequential()
# VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
# VGG16_model.add(Dense(133, activation='softmax'))
# VGG16_model.summary()
    
    
Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(Dense(133, activation='softmax'))
Xception_model.summary()    
    
    
# VGG19_model = Sequential()
# VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
# VGG19_model.add(Dense(133, activation='softmax'))
# VGG19_model.summary()      
    
    
# InceptionV3_model = Sequential()
# InceptionV3_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
# InceptionV3_model.add(Dense(133, activation='softmax'))
# InceptionV3_model.summary()     
    

# Resnet50_model = Sequential()
# Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
# Resnet50_model.add(Dense(133, activation='softmax'))
# Resnet50_model.summary()     
    
    
# Compile the Model
#
# model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
# VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    
# VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
# InceptionV3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
# Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
    

### TODO: specify the number of epochs that you would like to use to train the model.
# NB_EPOCH = 40, per the constants section.
epochs = 11

# Per CIFAR10 usecase
# model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
#    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, 
#    verbose=VERBOSE)


# TRAIN (Fit) the model
#

### Do NOT modify the code BELOW this line.

# CheckPointer
# checkpointer_VGG16 = ModelCheckpoint(filepath='saved_models/weights.best.from_VGG16_transfer_learning.hdf5', verbose=1, save_best_only=True)
checkpointer_Xception = ModelCheckpoint(filepath='saved_models/weights.best.from_Xception_transfer_learning.hdf5', verbose=1, save_best_only=True)
# checkpointer_VGG19 = ModelCheckpoint(filepath='saved_models/weights.best.from_VGG19_transfer_learning.hdf5', verbose=1, save_best_only=True)
# checkpointer_InceptionV3 = ModelCheckpoint(filepath='saved_models/weights.best.from_InceptionV3_transfer_learning.hdf5', verbose=1, save_best_only=True)
# checkpointer_Resnet50 = ModelCheckpoint(filepath='saved_models/weights.best.from_Resnet50_transfer_learning.hdf5', verbose=1, save_best_only=True)

          
# VGG16_model.fit(train_VGG16, train_targets, 
#           validation_data=(valid_VGG16, valid_targets),
#           epochs=20, batch_size=20, callbacks=[checkpointer_VGG16], verbose=1)          
    
Xception_model.fit(train_Xception, train_targets, 
       validation_data=(valid_Xception, valid_targets),
       epochs=20, batch_size=20, callbacks=[checkpointer_Xception], verbose=1)    
    
# VGG19_model.fit(train_VGG19, train_targets, 
#        validation_data=(valid_VGG19, valid_targets),
#        epochs=20, batch_size=20, callbacks=[checkpointer_VGG19], verbose=1)  
    
    
# InceptionV3_model.fit(train_InceptionV3, train_targets, 
#        validation_data=(valid_InceptionV3, valid_targets),
#        epochs=20, batch_size=20, callbacks=[checkpointer_InceptionV3], verbose=1)      
    
    
# Resnet50_model.fit(train_Resnet50, train_targets, 
#           validation_data=(valid_Resnet50, valid_targets),
#           epochs=20, batch_size=20, callbacks=[checkpointer_Resnet50], verbose=1)     
  
  
    
### Do NOT modify the code ABOVE this line. 
    
    
    
# Load The Models:
#
# VGG16_model.load_weights('saved_models/weights.best.from_VGG16_transfer_learning.hdf5')
Xception_model.load_weights('saved_models/weights.best.from_Xception_transfer_learning.hdf5') 
# VGG19_model.load_weights('saved_models/weights.best.from_VGG19_transfer_learning.hdf5') 
# InceptionV3_model.load_weights('saved_models/weights.best.from_InceptionV3_transfer_learning.hdf5') 
# Resnet50_model.load_weights('saved_models/weights.best.from_Resnet50_transfer_learning.hdf5') 
    
    
### Test The model
#

# VGG16
#
# VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]
# VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]
# InceptionV3_predictions = [np.argmax(InceptionV3_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]
# Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]


# report test accuracy

# VGG16
#
# test_accuracy_VGG16 = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
# print('Test accuracy - VGG16: %.4f%%' % test_accuracy_VGG16)
    
    
# Xception
#
test_accuracy_Xception = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
print('Test accuracy - Xception: %.4f%%' % test_accuracy_Xception)


# VGG19 
#
# test_accuracy_VGG19 = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
# print('Test accuracy - VGG19: %.4f%%' % test_accuracy_VGG19)

    
# InceptionV3 
#
# test_accuracy_InceptionV3 = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
# print('Test accuracy - InceptionV3: %.4f%%' % test_accuracy_InceptionV3)
    
    
# Resnet50 
#
# test_accuracy_Resnet50 = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
# print('Test accuracy - Resnet50: %.4f%%' % test_accuracy_Resnet50)    
    
    
    
# Results: VGG16 - (+3.)
# Test accuracy - VGG16: 43.3014%

# Results: Xception - (+44.)
# Test accuracy - Xception: 84.0909%
    
# Results: Resnet50 - (+39.)
# Test accuracy - Resnet50: 79.6651%    
    
# Results: VGG19 - (+5.)    
# Test accuracy - VGG19: 45.2153%   
    
# Results: InceptionV3 - (+41.)   
# Test accuracy - InceptionV3: 81.3397%
    
    
    
    
    
    
    
    
    
    
    
    
    
    









