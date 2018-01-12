from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

import matplotlib.pyplot as plt

#from quiver_engine import server
# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
# 
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

#constant
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()


#load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
 
# convert to categorical
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES) 

# float and normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# network
#
# Our net will learn 32 convolutional filters, each of which with a 3 x 3 size. The output
# dimension is the same one of the input shape, so it will be 32 x 32 and activation is ReLU,
# which is a simple way of introducing non-linearity. After that we have a max-pooling
# operation with pool size 2 x 2 and a dropout at 25%:
#    
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
# The next stage in the deep pipeline is a dense network with 512 units and ReLU activation
# followed by a dropout at 50% and by a softmax layer with 10 classes as output, one for each
# category:  
# 
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()


# Compile Model
#
#optim = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=OPTIM,	metrics=['accuracy'])

# train (fit) Model
#
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
	epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, 
	verbose=VERBOSE)
 
print('Testing...')
score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

#server.launch(model)


#save model
model_json = model.to_json()
open('cifar10_architecture.json', 'w').write(model_json)
model.save_weights('cifar10_weights.h5', overwrite=True)


# list all data in history
print(history.history.keys())

# summarize history for accuracy
#plt.plot(mo)
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




# ... model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
# _________________________________________________________________
# activation_1 (Activation)    (None, 32, 32, 32)        0         
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 16, 16, 32)        0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 8192)              0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 512)               4194816   
# _________________________________________________________________
# activation_2 (Activation)    (None, 512)               0         
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 512)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                5130      
# _________________________________________________________________
# activation_3 (Activation)    (None, 10)                0         
# =================================================================
# Total params: 4,200,842
# Trainable params: 4,200,842
# Non-trainable params: 0


# Our network reaches a test accuracy of 66.4% with 20 iterations.

# 
# Testing...
# 10000/10000 [==============================] - 3s 311us/step
# 
# Test score: 1.05169322224
# Test accuracy: 0.6792
# dict_keys(['val_loss', 'acc', 'val_acc', 'loss'])
# 





