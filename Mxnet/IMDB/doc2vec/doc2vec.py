from __future__ import print_function
import os
import re
import numpy as np 
import sklearn
from sklearn.utils import shuffle
import gensim
from gensim.models import doc2vec 
from gensim.models.doc2vec import Doc2Vec,TaggedDocument


def tag_sentence(sentence,label):
	return TaggedDocument(sentence,label)

def create_tags(train_data,train_labels,test_data,test_labels):
	tagged_data = []
	for i in range(25000):
		if(train_labels[i]==1):
			tag = ['train_pos_'+str(i)]
		else:
			tag = ['train_neg_'+str(i)]
		tagged_data.append(tag_sentence(train_data[i].split(),tag))

	for i in range(25000):
		if(test_labels[i]==1):
			tag = ['test_pos_'+str(i)]
		else:
			tag = ['test_neg_'+str(i)]
		tagged_data.append(tag_sentence(test_data[i].split(),tag))

	return tagged_data

def create_doc2vec_model(windowsize,modeltype):

	d2vmodel = Doc2Vec(min_count=1,sample=1e-4, negative=5, workers=7,vector_size=100,
		window=windowsize,dm=modeltype)

	return d2vmodel

def train_doc2vec_model(model,num_epochs,data):

	model.build_vocab(data)

	for epoch in range(num_epochs):
		print(epoch,end=" ")
		data = sklearn.utils.shuffle(data)
		model.train(data,total_examples=model.corpus_count,epochs=model.epochs)

	return model,data 

def generate_vectors(model,data,train_l,test_l):
	train_vectors = np.zeros((train_l,100))
	test_vectors =np.zeros((test_l,100))

	train_labels = np.zeros(train_l)
	test_labels = np.zeros(test_l)

	train_index = 0
	test_index = 0
	
	for i in range(50000):
		tag = (data[i].tags)[0]
		ind,lab,_ = tag.split("_")
		if(ind=="train"):
			train_vectors[train_index] = model[tag]
			if(lab=="pos"):
				train_labels[train_index] = 1

			train_index+=1

		else:
			test_vectors[test_index] = model[tag]
			if(lab=="pos"):
				test_labels[test_index] = 1

			test_index+=1

	return train_vectors,train_labels,test_vectors,test_labels
