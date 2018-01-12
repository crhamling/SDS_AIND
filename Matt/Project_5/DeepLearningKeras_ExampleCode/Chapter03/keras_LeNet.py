# import the necessary packages
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(1671)  # for reproducibility



# ConvNet Description
#
#


#define the ConvNet 
#
class LeNet:
	@staticmethod
	def build(input_shape, classes):
	    
	    # Architecture: Sequential - different predefined models are stacked together 
	    # in a linear pipeline of layers similar to a stack or a queue
		model = Sequential() 
		
		# We have a first convolutional stage with ReLU activations followed by a max-pooling. Our
        # net will learn 20 convolutional filters, each one of which has a size of 5 x 5. The output
        # dimension is the same one of the input shape, so it will be 28 x 28. Note that since the
        # Convolution2D is the first stage of our pipeline, we are also required to define its
        # input_shape. The max-pooling operation implements a sliding window that slides over
        # the layer and takes the maximum of each region with a step of two pixels vertically and
        # horizontally
		
		
		# CONV => RELU => POOL
		#
		# * 20 : - filters is the number of convolution kernels to use 
		#   (for example , the dimensionality of the output)
		# * kernel_size : width and height of the 2D convolution window = 5
		# * padding='same' means that we have an output that is the same size as the input, 
		#   for which the area around the input is padded with zeros.
		# * input_shape: - The model needs to know what input shape it should expect.
		# * The max-pooling operation implements a sliding window that slides over the layer 
		#   and takes the maximum of each region with a step of two pixels vertically and horizontally
		# Activation is RELU - a simple way of introducing non-linearity
        #
		model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
		# Then a second convolutional stage with ReLU activations follows, again by a max-pooling.
        # In this case, we increase the number of convolutional filters learned to 50 from the previous
        # 20. Increasing the number of filters in deeper layers is a common technique used in deep
        # learning:
		
		# CONV => RELU => POOL
		# 
		model.add(Conv2D(50, kernel_size=5, padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
		# Then we have a pretty standard flattening and a dense network of 500 neurons, followed by
        # a softmax classifier with 10 classes:
		
		# Flatten => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
 
		# a softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model

# Now we need some additional code for training the network, but this is very similar to
# what we have already described in Chapter 1, Neural Network Foundations. This time, we
# also show the code for printing the loss:

# network and training
#
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.2

IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
NB_CLASSES = 10  # number of outputs = number of digits
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)

# data: shuffled and split between train and test sets
#
(X_train, y_train), (X_test, y_test) = mnist.load_data()
K.set_image_dim_ordering("th")

# consider them as float and normalize
# 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 
X_test /= 255  

# we need a 60K x [1 x 28 x 28] shape as input to the CONVNET
# 
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# initialize the optimizer and model
# 
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

# Train (fit) the Model
#
history = model.fit(X_train, y_train, 
		batch_size=BATCH_SIZE, epochs=NB_EPOCH, 
		verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Score Model
#
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())

# PLOT : summarize history for accuracy
#
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# PLOT : summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# 
# RESULTS 
# ...
# ...
# ...
# Epoch 20/20
# 48000/48000 [==============================] - 145s 3ms/step - loss: 0.0025 - acc: 0.9993 - val_loss: 0.0369 - val_acc: 0.9924
# 10000/10000 [==============================] - 14s 1ms/step
# 
# Test score: 0.0310026965442
# Test accuracy: 0.9924
# dict_keys(['acc', 'val_loss', 'val_acc', 'loss'])
# 

