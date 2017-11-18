# -*- coding: utf-8 -*-
# -*- built with Python 3.6 -*-
"""
cnn_build.py

Builds a CNN described as a computational graph in Tensorflow for the purpose of
classifying handwritten mathematical symbols. Training set and test set image matrices
must be in the form of pickled binary numpy arrays with flattened image vectors
in each row. The classes must by in the form of pickled binary numpy arrays
of the classes in one hot vector form. The TensorFlow graph is saved inside a directory
named "{1}_saved_model/cnn_graph" where {1} is the first argument to the function.

HOW TO RUN:
    
    python cnn_build.py {1} {2} {3} {4} [{5}]
    
WHERE:
    {1} = the name of the training data image matrix
    {2} = the name of the training data classes matrix
    {3} = the name of the test data image matrix (NA if not available)
    {4} = the name of the test data classes matrix (NA if not available)
    {5} = optional number of iterations to take. otherwise goes into interactive mode.

NOTE: this code is adapted from 'https://github.com/Hvass-Labs/TensorFlow-Tutorials'

@author: Brody Kutt (bjk4704)
"""


import tensorflow as tf  # built with version 0.12.1
import numpy as np
import time
from datetime import timedelta
import sys
import pickle
import random
import os


class input_data:
    """
    Class to help ease the use of the input data with optimizer.
    """
    def __init__(self, train_mat, train_cls, test_mat, test_cls):
        self.train_mat = train_mat  # Training image matrix
        self.train_cls = train_cls  # Training image true classes one hot encoded
        self.test_mat = test_mat  # Testing image matrix
        self.test_cls = test_cls  # Testing image true classes one hot encoded
        self.num_train_samp = len(train_mat)  # Calculate this once
        self.num_test_samp = len(test_mat)  # Calculate this once
        self.i = 0  # Index into batch sampling
        
    def scramble(self):
        """
        Scrambles the training and testing row orders.
        """
        new_order = random.sample(range(self.num_train_samp), self.num_train_samp)
        self.train_mat = self.train_mat[new_order, :]
        self.train_cls = self.train_cls[new_order, :]
        
        new_order = random.sample(range(self.num_test_samp), self.num_test_samp)
        self.test_mat = self.test_mat[new_order, :]
        self.test_cls = self.test_cls[new_order, :]
        
    def next_batch(self, train_batch_size):
        """
        Returns a batch of training data of the given size and updates index.
        """
        if(self.i + train_batch_size >= self.num_train_samp):
            x_batch = self.train_mat[self.i:self.num_train_samp-1, :]
            y_true_batch = self.train_cls[self.i:self.num_train_samp-1, :]
            
            diff = (self.i + train_batch_size) - self.num_train_samp
            x_batch = np.vstack((x_batch, self.train_mat[0:diff, :]))
            y_true_batch = np.vstack((y_true_batch, self.train_cls[0:diff, :]))
            self.i = diff
        else:
            x_batch = self.train_mat[self.i:self.i+train_batch_size, :]
            y_true_batch = self.train_cls[self.i:self.i+train_batch_size, :]
            self.i += train_batch_size
            
        return x_batch, y_true_batch
    
## MODEL HYPERPARAMETERS & PROGRAM CONSTANTS ##

# Convolutional Layer 1.
filter_size1 = 10         # Convolution filters are 10 x 10 pixels
num_filters1 = 16         # There are 16 of these filters

# Convolutional Layer 2.
filter_size2 = 8          # Convolution filters are 8 x 8 pixels
num_filters2 = 18         # There are 18 of these filters

# Convolutional Layer 3.
filter_size3 = 6          # Convolution filters are 6 x 6 pixels
num_filters3 = 24         # There are 24 of these filters

# Number of neurons in fully-connected layer 1
fc1_size = 162

# Number of neurons in fully-connected layer 2
fc2_size = 136

# Size of each image
img_size = 42

# Images are stored in one-dimensional arrays of this length
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale
num_channels = 1

# Use batch training
train_batch_size = 700

# Split the test-set into smaller batches of this size to limit RAM usage
test_batch_size = 700

# Learning rate for the ADAM optimizer
LEARNING_RATE = 0.0002

# L2 penalty parameter
LAMBDA = 0.0001


def new_weights(shape):
    """
    Returns new TensorFlow weights in the given shape with normally distributed random initializations
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    """
    Returns new TensorFlow biases with the given length with initialization 0.05
    """
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              # The previous layer
                   num_input_channels, # Num. channels in prev. layer
                   filter_size,        # Width and height of each filter
                   num_filters,        # Number of filters
                   use_pooling=True):  # Use 2x2 max-pooling
    """
    Helper function for creating a new convolutional layer.
    """

    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights (filters) with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases   # Adding biases to the results of the convolution.

    if use_pooling:  # Use pooling to down-sample the image resolution?
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')  # This is 2x2 max-pooling

    layer = tf.nn.relu(layer)  # Rectified Linear Unit (ReLU).
    
    return layer, weights


def flatten_layer(layer):
    """
    Reduces a 4d tensor to a 2d tensor to be able to be fed into a fully connected layer.
    """

    layer_shape = layer.get_shape()  # Shape of the input layer

    num_features = layer_shape[1:4].num_elements()  # img_height * img_width * num_channels
    
    # Reshape the layer to [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now: [num_images, img_height * img_width * num_channels]
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    """
    Helper function for creating a new fully connected layer.
    """

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)        

    return layer


def optimize(session,           # TensorFlow session
             optimizer,         # Tensorflow optimization method object
             data,              # The raw data class instance
             x,                 # Placeholder for image data
             y_true,            # Placeholder for true classes
             accuracy,          # Tensorflow object for calculating iteration accuracy
             train_batch_size,  # Number of training examples per batch
             total_iterations,  # Total number of iterations that have been done so far
             num_iterations):   # Number of iterations to add onto existing model
    """
    Function for performing training iterations. Built to be used incrementally
    as needed with repeated calls to train further in real time.
    """
    start_time = time.time()  # record starting data
    for i in range(total_iterations, total_iterations + num_iterations):

        # Get a batch of training examples
        x_batch, y_true_batch = data.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 1 == 0: # Print status every 100 iterations
            # Calculate the accuracy on the testing set
            acc = session.run(accuracy, feed_dict={x: data.test_mat, y_true: data.test_cls})
            msg = "Optimization Iteration: {0:>6}...Testing Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))

    total_iterations += num_iterations  # Update the total number of iterations performed
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def print_test_accuracy(session,         # TensorFlow session
                        data,            # The raw data class instance
                        x,               # Place holder for image data
                        y_true,          # Placeholder for true classes for the images
                        y_pred_cls):     # Placeholder for our model's predicted classes
    """
    Computes and prints the test set accuracy.
    """
    num_test = len(data.test_mat)   # Number of images in the test set
    cls_pred = np.zeros(shape=num_test, dtype=np.int)  # Allocate an array for the predicted classes
    
    i = 0  # The starting index for the next batch is denoted i
    while i < num_test:
        j = min(i + test_batch_size, num_test)  # End index for next batch
        images = data.test_mat[i:j, :]
        labels = data.test_cls[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}
        
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)  # Calculate the predicted class
        i = j  # Update indices for next batch

    cls_true = np.argmax(data.test_cls, axis=1)  # True class-numbers of the test-set
    correct = (cls_true == cls_pred)  # Create a boolean array whether each image is correctly classified
    correct_sum = correct.sum()  # Calculate the number of correctly classified images
    acc = float(correct_sum) / num_test  # Classification accuracy is (# correct/total)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    

def build_cnn(data):
    """
    Main execution function. Data parameter contains the raw data class instance.
    """
    num_classes = len(data.train_cls[0,:])  # determine this from data
    
    ## PLACEHOLDERS ##
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])  # redefine as 4d tensor
    
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')  # as one hot encoded vector
    y_true_cls = tf.argmax(y_true, dimension=1)  # back into class #'s
    
    ## FIRST CONVOLUTIONAL LAYER ##
    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                       num_input_channels=num_channels,
                       filter_size=filter_size1,
                       num_filters=num_filters1,
                       use_pooling=True)
        
    ## SECOND CONVOLUTIONAL LAYER ##
    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       num_input_channels=num_filters1,
                       filter_size=filter_size2,
                       num_filters=num_filters2,
                       use_pooling=True)
        
    ## THIRD CONVOLUTIONAL LAYER ##
    layer_conv3, weights_conv3 = \
        new_conv_layer(input=layer_conv2,
                       num_input_channels=num_filters2,
                       filter_size=filter_size3,
                       num_filters=num_filters3,
                       use_pooling=False)
        
    ## FLATTENED LAYER ##
    layer_flat, num_features = flatten_layer(layer_conv2)
    
    ## FIRST FULLY CONNECTED LAYER ##
    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc1_size,
                             use_relu=True)
    
    ## SECOND FULLY CONNECTED LAYER ##
    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc1_size,
                             num_outputs=fc2_size,
                             use_relu=True)
    
    ## THIRD FULLY CONNECTED LAYER ##
    layer_fc3 = new_fc_layer(input=layer_fc2,
                             num_inputs=fc2_size,
                             num_outputs=num_classes,
                             use_relu=False)
    
    ## SOFTMAX ON OUTPUT LAYER TO GET PROB. DISTRIBUTION ##
    y_pred = tf.nn.softmax(layer_fc3)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    # L2 REGULARIZATION
    l2 = LAMBDA * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        
    ## DEFINE CROSS ENTROPY LOSS ##
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,
                                                            labels=y_true)
    cost = tf.reduce_mean(cross_entropy + l2)  # Average loss across all images with regulatization
    
    ## DEFINE OPTIMIZATION METHOD ##
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    
    ## SOME PERFORMANCE MEASURES ##
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ## CREATING THE TENSORFLOW SESSION ##
    session = tf.Session()
    session.run(tf.global_variables_initializer())  # initialize variables
    
    ## BEGIN TRAINING ITERATIONS ##
    if(len(sys.argv) == 5):  # Interactive mode
        total_iterations = 0
        while(True):
            print('How many iterations of training?')
            num_iters = int(input())
            optimize(session, optimizer, data, x, y_true, accuracy, train_batch_size, total_iterations, num_iters)
            print_test_accuracy(session, data, x, y_true, y_pred_cls)
            total_iterations += num_iters
            print('Total Number of iterations so far is: ' + str(total_iterations) + '. Keep Going? Y or N')
            if(input() == "Y"):
                continue
            else:
                print('Do you want to save to disk? Y or N')
                if(input() == 'Y'):
                    print('Saving Model to Disk...')
                    saver = tf.train.Saver()
                    save_dir = sys.argv[1] + '_saved_model/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, 'cnn_graph')
                    tf.add_to_collection('vars', x)
                    tf.add_to_collection('vars', y_true)
                    tf.add_to_collection('vars', y_pred_cls)
                    saver.save(sess=session, save_path=save_path)
                break
    else:
        num_iters = int(sys.argv[5])
        optimize(session, optimizer, data, x, y_true, accuracy, train_batch_size, 0, num_iters)
        print_test_accuracy(session, data, x, y_true, y_pred_cls)
        print('Saving Model to Disk...')
        saver = tf.train.Saver()
        save_dir = sys.argv[1] + '_saved_model/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'cnn_graph')
        tf.add_to_collection('vars', x)
        tf.add_to_collection('vars', y_true)
        tf.add_to_collection('vars', y_pred_cls)
        saver.save(sess=session, save_path=save_path)
    session.close()  # finish and clean up
        
        
if __name__ == '__main__':
    if(len(sys.argv) == 5 or len(sys.argv) == 6):
        try:
            train_mat = pickle.load(open('../binary_data/' + sys.argv[1], "rb" ))
            train_cls = pickle.load(open('../binary_data/' + sys.argv[2], "rb" ))
            if(sys.argv[3] == 'NA'):
                test_mat = train_mat[0:2,:]  # dummy value
            else:
                test_mat = pickle.load(open('../binary_data/' + sys.argv[3], "rb" ))
            if(sys.argv[4] == 'NA'):
                test_cls = train_cls[0:2,:]  # dummy value
            else:
                test_cls = pickle.load(open('../binary_data/' + sys.argv[4], "rb" ))
        except FileNotFoundError:
            print("Wrong file or file path provided.")
            sys.exit(1)
        data = input_data(train_mat, train_cls, test_mat, test_cls)
        data.scramble()
        tf.reset_default_graph()  # start with clean graph
        build_cnn(data)
        
    else:
        print('Wrong number of arguments.')
        print("""
        HOW TO RUN:
    
            python cnn_build.py {1} {2} {3} {4} [{5}]
    
        WHERE:
            
            {1} = the name of the training data image matrix
            {2} = the name of the training data classes matrix
            {3} = the name of the test data image matrix (NA if not available)
            {4} = the name of the test data classes matrix (NA if not available)
            {5} = optional number of iterations to take. otherwise goes into interactive mode.
            """)