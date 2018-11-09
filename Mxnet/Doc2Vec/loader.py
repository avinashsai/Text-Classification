from __future__ import print_function
import os
import re
import numpy as np 
from sklearn.utils import shuffle

def readfiles():
	train_data = []
	train_labels = np.zeros(25000)

	path = '/home/avinashsai/Documents/Datasets/IMDB'
	
	with open(path+'/train_pos.txt','r',encoding='latin1') as f:
		for line in f:
			train_data.append(line[:-1])

	with open(path+'/train_neg.txt','r',encoding='latin1') as f:
		for line in f:
			train_data.append(line[:-1])

	train_labels[0:12500] = 1
	train_data,train_labels = shuffle(train_data,train_labels)

	test_data = []
	test_labels = np.zeros(25000)

	with open(path+'/test_pos.txt','r',encoding='latin1') as f:
		for line in f:
			test_data.append(line[:-1])

	with open(path+'/test_neg.txt','r',encoding='latin1') as f:
		for line in f:
			test_data.append(line[:-1])

	test_labels[0:12500] = 1
	test_data,test_labels = shuffle(test_data,test_labels)


	return train_data,train_labels,test_data,test_labels


