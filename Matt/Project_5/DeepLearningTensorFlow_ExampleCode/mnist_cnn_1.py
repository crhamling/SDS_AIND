import tensorflow as tf
import numpy as np
#import mnist_data 

# Set the following parameters, that indicate the number of samples to consider respectively
# for the training phase (128) and then the test phase (256):
#    
batch_size = 128
test_size = 256

# We define the following parameter, the value is 28 because a MNIST image is 28 pixels in height and width:
#
img_size = 28

# Regarding the number of classes, the value 10 means that we'll have one class for each of 10 digits (0-9):
#
num_classes = 10

# We collect the mnist data which will be copied into the data folder "MNIST_data"
#
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# We build the datasets for training (trX, trY) and testing the network (teX, teY):
#
trX, trY, teX, teY = mnist.train.images,\
                     mnist.train.labels, \
                     mnist.test.images, \
                     mnist.test.labels


# BEFORE:
# >>> trX.shape
# (55000, 784)
#
# >>> trX.shape
# (55000, 784)


# The trX and teX image sets must be reshaped according the input shape:
#
trX = trX.reshape(-1, img_size, img_size, 1)  # 28x28x1 input img
teX = teX.reshape(-1, img_size, img_size, 1)  # 28x28x1 input img

# AFTER:
# >>> trX.shape
# (55000, 28, 28, 1) => 55,000 images of dimmension 28 x 28 x 1
# 
# >>> trY.shape
# (55000, 10)  => 55,000 images of dimmension 10


# A placeholder variable, X, is defined for the *input images*. The data type for this tensor is set
# to float32 and the shape is set to [None, img_size, img_size, 1], where None
# means that the tensor may hold an *arbitrary* number of images:
# 55,000 images of dimmension 28 x 28 x 1
#
X = tf.placeholder("float", [None, img_size, img_size, 1])


# We set another placeholder variable, Y, for the *true labels* (0-9) associated with the images
# that were input data in the placeholder variable X.
# The shape of this placeholder variable is [None, num_classes] which means it may hold
# an *arbitrary* number of labels and each label is a vector of the length num_classes which is
# 10 in this case:
# 55,000 images of dimmension 10
#
Y = tf.placeholder("float", [None, num_classes])


# The init_weights FUNCTION builds new variables in the given shape and initializes the
# network's weights with random values as defined below (and actually used subsequently below that):
# USAGE:
# Create a variable.
# w = tf.Variable(<initial-value>, name=<optional-name>)
#
# random_normal(
#    shape,
#    mean=0.0,
#    stddev=1.0,
#    dtype=tf.float32,
#    seed=None,
#    name=None
# )
#
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# Each neuron of the FIRST convolutional layer is convoluted to a small subset of the input
# tensor, with a dimension of 3x3x1, while the value 32 is just the number of feature maps we
# are considering for this first layer. The weight w is then defined:
#    
w = init_weights([3, 3, 1, 32])       # 3x3x1 conv => 32 outputs


# The number of inputs is then increased of 32, which means that each neuron of the second
# convolutional layer is convoluted to 3x3x32 neurons of the first convolution layer. The w2
# weight is below:
# The value 64 represents the number of obtained output features.
# 
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv => 64 outputs


# The third convolutional layer is convoluted to 3x3x64 neurons of the previous layer, while
# 128 are the resulting features:
#    
w3 = init_weights([3, 3, 64, 128])    # 3x3x64 conv => 128 outputs


# The fourth layer is fully-connected. It receives 128x4x4 inputs, while the output is equal to
# 625:
#
w4 = init_weights([128 * 4 * 4, 625]) # FC 128x4x4 inputs => 625 outputs


# The output layer receives 625 inputs, while the output is the number of classes (10: 0-9):
#    
w_o = init_weights([625, num_classes])         # FC 625 inputs, 10 outputs (labels)


# These are the dropout parameters for the convolution and fully-connected layers:
# Note that these initializations are NOT actually done at this point; they are merely being
# defined in the TensorFlow graph:
#    
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


# It's time to define the network model. As we did for the network's weight definition, it will
# be a function.
# It receives as input, the X tensor, the weights tensors, and the dropout parameters for
# convolution and fully-connected layers:
#    
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

    # The tf.nn.conv2d() function executes the TensorFlow operation for the convolution.
    # Note that the strides are set to 1 in all dimensions.
    # Indeed, the first and last stride must always be 1, because the first is for the image number
    # and the last is for the input channel. The padding parameter is set to 'SAME' which means
    # the input image is padded with zeroes so that the size of the output is the same:
    #
    conv1 = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')

    # Then we pass the "conv1 layer" to a "relu layer". It calculates the max(x, 0) function for each
    # input pixel x, adding some non-linearity to the formula and allows us to learn more
    # complicated functions:
    #
    conv1_a = tf.nn.relu(conv1)
    
    # The resulting layer is then pooled by the tf.nn.max_pool operator.
    # It is a 2x2 max-pooling, which means that we are considering 2x2 windows and select the
    # largest value in each window. Then we move two pixels to the next window.
    #
    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1] ,strides=[1, 2, 2, 1], padding='SAME')
    
    # We try to reduce the overfitting, via the tf.nn.dropout() function, passing the conv1
    # layer and the p_keep_conv probability value:
    # 
    conv1 = tf.nn.dropout(conv1, p_keep_conv)


    # As you can see, the next two convolutional layers, conv2, conv3, are defined in the same
    # way as conv1:
    #
    conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2_a = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)
    #
    #
    conv3=tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv3)


    # Two fully-connected layers are added to the network. The input of the first FC_layer is the
    # convolution layer from the previous convolution:
    #
    FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    FC_layer = tf.reshape(FC_layer, [-1, w4.get_shape().as_list()[0]])    
    
    # A dropout function is again used to reduce the overfitting:
    # 
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)


    # The output layer receives the input as FC_layer and the w4 weight tensor. A relu and a
    # dropout operator are respectively applied:
    #
    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

    # The result variable is a vector of length 10 for determining which one of the 10 classes for
    # the input image belongs to:
    #
    result = tf.matmul(output_layer, w_o)
    return result

# The cross-entropy is the performance measure we used in this classifier. The cross-entropy
# is a continuous function that is always positive and is equal to zero, if the predicted output
# exactly matches the desired output. The goal of this optimization is therefore to minimize
# the cross-entropy so it gets as close to zero as possible by changing the variables of the
# network layers.
# 
# TensorFlow has a built-in function for calculating the cross-entropy. Note that the function
# calculates the softmax internally so we must use the output of py_x directly:
#
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)


# Now that we have defined the cross-entropy for each classified image, we have a measure
# of how well the model performs on each image individually. But using the cross-entropy to
# guide the optimization of the networks's variables we need a single scalar value, so we
# simply take the average of the cross-entropy for all the classified images:
#    
cost = tf.reduce_mean(Y_)


# To minimize the evaluated cost, we must define an optimizer. In this case, we adopt the
# implemented RMSPropOptimizer function which is an advanced form of gradient descent.
# The RMSPropOptimizer function also divides the learning rate by an exponentially
# decaying average of squared gradients. Hinton suggests setting the decay parameter to 0.9,
# while a good default value for the learning rate is 0.001:
#
optimizer  = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)


# Finally, we define predict_op that is the index with the largest value across dimensions
# from the output of the mode:
#    
predict_op = tf.argmax(py_x, 1)


# Note that optimization is not performed at this point. Nothing is calculated at all; we'll just
# add the optimizer object to the TensorFlow graph for later execution.


# Now we can proceed to implement a TensorFlow session:

with tf.Session() as sess:
    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    
    for i in range(100):
        # We get a batch of training examples, the training_batch tensor now holds a subset of
        # images and corresponding labels:
        #
        training_batch = \
                       zip(range(0, len(trX), \
                                 batch_size),
                             range(batch_size, \
                                   len(trX)+1, \
                                   batch_size))
        # Put the batch into feed_dict with the proper names for placeholder variables in the graph.
        # We run the optimizer using this batch of training data, TensorFlow assigns the variables in
        # a feed to the placeholder variables and then runs the optimizer:
        #
        for start, end in training_batch:
            sess.run(optimizer, feed_dict={X: trX[start:end],\
                                          Y: trY[start:end],\
                                          p_keep_conv: 0.8,\
                                          p_keep_hidden: 0.5})

        # At the same time, we get a shuffled batch of test samples:
        test_indices = np.arange(len(teX))# Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        # For each iteration, we display the accuracy evaluated on the batch set:
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==\
                         sess.run\
                         (predict_op,\
                          feed_dict={X: teX[test_indices],\
                                     Y: teY[test_indices], \
                                     p_keep_conv: 1.0,\
                                     p_keep_hidden: 1.0})))


"""
RUN RESULTS:

Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Successfully extracted to train-images-idx3-ubyte.mnist 9912422 bytes.
Loading ata/train-images-idx3-ubyte.mnist
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Successfully extracted to train-labels-idx1-ubyte.mnist 28881 bytes.
Loading ata/train-labels-idx1-ubyte.mnist
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Successfully extracted to t10k-images-idx3-ubyte.mnist 1648877 bytes.
Loading ata/t10k-images-idx3-ubyte.mnist
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Successfully extracted to t10k-labels-idx1-ubyte.mnist 4542 bytes.
Loading ata/t10k-labels-idx1-ubyte.mnist
(0, 0.95703125)
(1, 0.98046875)
(2, 0.9921875)
(3, 0.99609375)
(4, 0.99609375)
(5, 0.98828125)
(6, 0.99609375)
(7, 0.99609375)
(8, 0.98828125)
(9, 0.98046875)
(10, 0.99609375)
(11, 1.0)
(12, 0.9921875)
(13, 0.98046875)
(14, 0.98828125)
(15, 0.9921875)
(16, 0.9921875)
(17, 0.9921875)
(18, 0.9921875)
(19, 1.0)
(20, 0.98828125)
(21, 0.99609375)
(22, 0.98828125)
(23, 1.0)
(24, 0.9921875)
(25, 0.99609375)
(26, 0.99609375)
(27, 0.98828125)
(28, 0.98828125)
(29, 0.9921875)
(30, 0.99609375)
(31, 0.9921875)
(32, 0.99609375)
(33, 1.0)
(34, 0.99609375)
(35, 1.0)
(36, 0.9921875)
(37, 1.0)
(38, 0.99609375)
(39, 0.99609375)
(40, 0.99609375)
(41, 0.9921875)
(42, 0.98828125)
(43, 0.9921875)
(44, 0.9921875)
(45, 0.9921875)
(46, 0.9921875)
(47, 0.98828125)
(48, 0.99609375)
(49, 0.99609375)
(50, 1.0)
(51, 0.98046875)
(52, 0.99609375)
(53, 0.98828125)
(54, 0.99609375)
(55, 0.9921875)
(56, 0.99609375)
(57, 0.9921875)
(58, 0.98828125)
(59, 0.99609375)
(60, 0.99609375)
(61, 0.98828125)
(62, 1.0)
(63, 0.98828125)
(64, 0.98828125)
(65, 0.98828125)
(66, 1.0)
(67, 0.99609375)
(68, 1.0)
(69, 1.0)
(70, 0.9921875)
(71, 0.99609375)
(72, 0.984375)
(73, 0.9921875)
(74, 0.98828125)
(75, 0.99609375)
(76, 1.0)
(77, 0.9921875)
(78, 0.984375)
(79, 1.0)
(80, 0.9921875)
(81, 0.9921875)
(82, 0.99609375)
(83, 1.0)
(84, 0.98828125)
(85, 0.98828125)
(86, 0.99609375)
(87, 1.0)
(88, 0.99609375)

# MRV Run
#
...
...
...
90 1.0
91 0.99609375
92 0.99609375
93 0.98828125
94 1.0
95 0.99609375
96 1.0
97 0.9921875
98 0.99609375
99 0.9921875


"""
