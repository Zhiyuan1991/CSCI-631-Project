# -*- coding: utf-8 -*-
# -*- built with Python 3.6 -*-
"""
cnn_predict.py

Loads in a CNN saved as a TensorFlow .meta object and then performs a prediction
on all the samples provided in the output of ink2pickle.py. Outputs a .csv file 
containing the predictions in the expected format for CROHME in the current 
working directory.

HOW TO RUN:
    
    python cnn_predict.py {1} {2} {3} [{4}]
    
WHERE:
    
    {1} = the name of the TensorFlow .meta graph ex. "saved_model/cnn_graph"
    {2} = the name of the test image matrix binary pickle file ex. "real_test_img"
    {3} = the name of the .csv file containing file names ex. "real_test.csv" (NA if not availables)
    {4} = optional parameter to specify if junk is included as a class.
          Defaul is no. Type 'Y' here to include it.

@author: Brody Kutt (bjk4704)
"""

import tensorflow as tf  # built with version 0.12.1
import numpy as np
import sys
import pickle
import csv


def get_filenames():
    """
    Returns a list of filenames taken from the csv file provided.
    """
    filenames = []
    with open('../csv_data/' + sys.argv[3], newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            filenames.append(row[0])
    return filenames


def predict(session,         # TensorFlow session
            data,            # The raw data in a numpy array
            x,               # Place holder for image data
            y_true,          # Placeholder for true classes for the images
            y_pred_cls):     # Placeholder for our model's predicted classes
    """
    Computes predictions for the images in data and outputs a .csv with the
    results.
    """
    # Split the data-set in batches of this size to limit RAM usage
    test_batch_size = 256

    num_test = len(data)   # Number of images in the test set
    cls_pred = np.zeros(shape=num_test, dtype=np.int)  # Allocate an array for the predicted classes
    
    i = 0  # The starting index for the next batch is denoted i
    while i < num_test:
        j = min(i + test_batch_size, num_test)  # End index for next batch
        images = data[i:j, :]
        feed_dict = {x: images}  # Create a feed-dict with these images and labels
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)  # Calculate the predicted class
        i = j  # Update indices for next batch
    
    ## CHANGE NUMBERED CLASSES BACK TO SYMBOLS ##
    if(len(sys.argv) == 5):
        SYMB_DICT = pickle.load(open('../binary_data/SYMB_DICT_W_JUNK', 'rb'))
    else:
        SYMB_DICT = pickle.load(open('../binary_data/SYMB_DICT', 'rb'))
    symb_pred = []
    for i in range(len(cls_pred)):
        symb_pred.append(list(SYMB_DICT.keys())[list(SYMB_DICT.values()).index(cls_pred[i])])

    ## WRITE NEW CSV FILE WITH RESULTS ##
    if(sys.argv[3] != 'NA'):
        filenames = get_filenames()
        with open('cnn_' + sys.argv[2] + '_output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
            for i in range(len(data)):
                writer.writerow([filenames[i], symb_pred[i]])
    else:
        with open('cnn_' + sys.argv[2] + '_output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
            for i in range(len(data)):
                writer.writerow([symb_pred[i]])
    

if __name__ == '__main__':
    if(len(sys.argv) == 4 or len(sys.argv) == 5):
        
        ## GET DATA FROM CSV FILE ##
        data = pickle.load(open('../binary_data/' + sys.argv[2], 'rb'))
        
        # Create a clean graph and import the MetaGraphDef nodes
        tf.reset_default_graph()
        with tf.Session() as sess:
          # Import the previously exported meta graph and variables
          saver = tf.train.import_meta_graph(sys.argv[1] + '.meta')
          saver.restore(sess, sys.argv[1])
          all_vars = tf.get_collection('vars')
          x = all_vars[0]
          y_true = all_vars[1]
          y_pred_cls = all_vars[2]
          predict(sess, data, x, y_true, y_pred_cls)
          
    else:
        print('Wrong number of arguments.')
        print("""
        HOW TO RUN:
    
            python cnn_predict.py {1} {2} {3} [{4}]
    
        WHERE:
            
            {1} = the name of the TensorFlow .meta graph ex. "saved_model/cnn_graph"
            {2} = the name of the test image matrix binary pickle file ex. "real_test_img"
            {3} = the name of the .csv file containing file names ex. "real_test.csv" (NA if not available)
            {4} = optional parameter to specify if junk is included as a class.
                  Defaul is no. Type 'Y' here to include it.
            """)