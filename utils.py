import cPickle
import numpy as np
import random
from skimage import color

def read_cifar_batch_data(filename):
	with open(filename, 'rb') as fo:
		dict = cPickle.load(fo)
	data=dict['data']
	labels=dict['labels']

	#reset image structure
	data=data.reshape([-1, 3, 32, 32])
	data=data.transpose([0, 2, 3, 1])
	data=data.reshape([-1,3*32*32])

	l=len(labels)
	class_num=10
	one_hot=np.zeros((l,class_num))
	one_hot[np.arange(l),labels]=1
	return data,one_hot

def read_cifar_batch_data_lab(filename):
	with open(filename, 'rb') as fo:
		dict = cPickle.load(fo)
	data=dict['data']
	labels=dict['labels']

	#reset image structure
	data=data.reshape([-1, 3, 32, 32])
	data=data.transpose([0, 2, 3, 1])
	data=color.rgb2lab(data)
	data=data.reshape([-1,3*32*32])

	l=len(labels)
	class_num=10
	one_hot=np.zeros((l,class_num))
	one_hot[np.arange(l),labels]=1
	return data,one_hot

def print_test_accuracy(session,         # TensorFlow session
                        data,            # The raw data class instance
                        x,               # Place holder for image data
                        y_true,          # Placeholder for true classes for the images
                        y_pred_cls,
                        test_batch_size):     # Placeholder for our model's predicted classes
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