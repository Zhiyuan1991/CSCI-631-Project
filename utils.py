import cPickle
import numpy as np
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
	class_num=100
	one_hot=np.zeros((l,class_num))
	one_hot[np.arange(l),labels]=1
	return data,one_hot

def read_cifar_batch_data_lab(filename):
	with open(filename, 'rb') as fo:
		dict = cPickle.load(fo)
	data=dict['data']
	labels=dict['labels']

	#reset image structure
	data=data.reshape([32, 32, 3])
	data=color.rgb2lab(data)
	data=data.reshape([-1, 3, 32, 32])
	data=data.transpose([0, 2, 3, 1])
	data=data.reshape([-1,3*32*32])

	l=len(labels)
	class_num=100
	one_hot=np.zeros((l,class_num))
	one_hot[np.arange(l),labels]=1
	return data,one_hot
