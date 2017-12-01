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

'''
********** after editing ********** Zhiyuan Li, 2017.11.27
How to run
    python cnn_build.py NA NA NA NA [iter_num]

    e.g.
    python cnn_build.py NA NA NA NA 5000
'''

import tensorflow as tf  # built with version 0.12.1
import numpy as np
import time
from datetime import timedelta
import sys
import pickle
import random
import os

from utils import *
from ops import *
    
## MODEL HYPERPARAMETERS & PROGRAM CONSTANTS ##

#******** image size *********
# Size of each image
img_size = 32
# Number of colour channels for the images: 1 channel for gray-scale
num_channels = 3
# Images are stored in one-dimensional arrays of this length
img_size_flat = img_size * img_size * num_channels
# Tuple with height and width of images used to reshape arrays
img_shape = (img_size, img_size)

#********** learning *********** 
# Use batch training
train_batch_size = 128
# Split the test-set into smaller batches of this size to limit RAM usage
test_batch_size = 128
# Learning rate for the ADAM optimizer
LEARNING_RATE = 0.0002
# L2 penalty parameter
LAMBDA = 0.0001

#************ CNN *****************
# Convolutional Layer 1.
filter_size1 = 7         # Convolution filters are 10 x 10 pixels
num_filters1 = 64         # There are 16 of these filters
# Convolutional Layer 2.
filter_size2 = 5        # Convolution filters are 8 x 8 pixels
num_filters2 = 96         # There are 18 of these filters
# Convolutional Layer 3.
filter_size3 = 3          # Convolution filters are 6 x 6 pixels
num_filters3 = 128         # There are 24 of these filters
# Number of neurons in fully-connected layer 1
fc1_size = 384
# Number of neurons in fully-connected layer 2
fc2_size = 192

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

        if i % 100 == 0: # Print status every 100 iterations
            # Calculate the accuracy on the testing set
            acc = session.run(accuracy, feed_dict={x: data.test_mat, y_true: data.test_cls})
            msg = "Iteration: {0:>6}   Testing Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))

    total_iterations += num_iterations  # Update the total number of iterations performed
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

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

    layer_conv1, weights_conv1 = \
        conv2d(x_image,
        	[filter_size1,filter_size1,num_channels,num_filters1],
        	'conv1',
        	bias=0.0)

    layer_conv1=pool_layer(layer_conv1,
        size=3,
        stride=2,
        name='pool1')

    layer_conv1=lrn_layer(layer_conv1,name='norm1')

    layer_conv2, weights_conv2 = \
        conv2d(layer_conv1,
        	[filter_size2,filter_size2,num_filters1,num_filters2],
        	'conv2',
        	bias=0.1)

    layer_conv2=pool_layer(layer_conv2,
        size=3,
        stride=2,
        name='pool2')

    layer_conv2=lrn_layer(layer_conv2,name='norm2')

    layer_conv3, weights_conv3 = \
        conv2d(layer_conv2,
        	[filter_size3,filter_size3,num_filters2,num_filters3],
        	'conv3',
        	bias=0.1)
    
    layer_conv3=pool_layer(layer_conv3,
        size=3,
        stride=2,
        name='pool3')
    

    layer_conv3=lrn_layer(layer_conv3,name='norm3')

    '''
    dim = np.prod(layer_conv2.get_shape().as_list()[1:])
    layer_flat = tf.reshape(layer_conv2, shape=[-1, dim], name='flat')
    layer_fc1=linear(layer_flat,[dim,fc1_size],'fc1',bias=True)
    layer_fc2=linear(layer_fc1,[fc1_size,fc2_size],'fc2',bias=True)
    layer_fc3=linear(layer_fc2,[fc2_size,num_classes],'fc3',relu=False, bias=True)
    '''

    layer_fc1=fc_layer(layer_conv3,
    	neurons=fc1_size,
    	decay=True,
    	name='fc1',
    	bias=0.1)

    h_fc1_drop = tf.nn.dropout(layer_fc1, 0.5)

    layer_fc2=fc_layer(layer_fc1,
    	neurons=fc2_size,
    	decay=True,
    	name='fc2',
    	bias=0.1)

    #h_fc2_drop = tf.nn.dropout(layer_fc2, 0.5)

    layer_fc3=fc_layer(layer_fc2,
    	neurons=10,
    	decay=False,
    	relu=False,
    	name='fc3')

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
            print_test_accuracy(session, data, x, y_true, y_pred_cls,test_batch_size)
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
        optimize(session, optimizer, data, x, y_true, accuracy,train_batch_size, 0, num_iters)
        print_test_accuracy(session, data, x, y_true, y_pred_cls,test_batch_size)
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
    print len(sys.argv)
    if(len(sys.argv) == 5 or len(sys.argv) == 6):
        try:
            #train_mat = pickle.load(open('../binary_data/' + sys.argv[1], "rb" ))
            #train_cls = pickle.load(open('../binary_data/' + sys.argv[2], "rb" ))
            m1,c1=read_cifar_batch_data_lab('./cifar/data_batch_1')
            m2,c2=read_cifar_batch_data_lab('./cifar/data_batch_2')
            m3,c3=read_cifar_batch_data_lab('./cifar/data_batch_3')
            m4,c4=read_cifar_batch_data_lab('./cifar/data_batch_4')
            m5,c5=read_cifar_batch_data_lab('./cifar/data_batch_5')
            train_mat=np.vstack((m1,m2,m3,m4,m5))
            train_cls=np.vstack((c1,c2,c3,c4,c5))
            print train_mat.shape
            print train_cls.shape
            if(sys.argv[3] == 'NA'):
                #test_mat = train_mat[0:2,:]  # dummy value
                test_mat,_=read_cifar_batch_data_lab('./cifar/test_batch')
            else:
                #test_mat = pickle.load(open('../binary_data/' + sys.argv[3], "rb" ))
                test_mat,_=read_cifar_batch_data_lab(sys.argv[3])
            if(sys.argv[4] == 'NA'):
                #test_cls = train_cls[0:2,:]  # dummy value
                _,test_cls=read_cifar_batch_data_lab('./cifar/test_batch')
            else:
                #test_cls = pickle.load(open('../binary_data/' + sys.argv[4], "rb" ))
                _,test_cls=read_cifar_batch_data_lab(sys.argv[3])
        except ValueError:
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