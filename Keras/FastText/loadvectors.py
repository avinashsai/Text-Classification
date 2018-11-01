import os
import re
import numpy as np 

def sentence_vector(sent_len,sentence,model,embed_dim):
	sent_vector = np.zeros((sent_len,embed_dim))
	words = sentence.split()
	count = 1
	for i in range(len(words)):
		if(words[i] in model.wv.vocab):
			vector = model.wv[words[i]]
			if(vector is not None):
				sent_vector[count,:] = vector 
				count+=1
		if(count==sent_len):
			return sent_vector

	return sent_vector

def load_imdb_vectors(Xtrain,Xtest,embed_dim,sent_len,model):

	train_vectors = np.zeros((len(Xtrain),sent_len,embed_dim))

	for i in range(len(Xtrain)):
		train_vectors[i,:,:] = sentence_vector(sent_len,Xtrain[i],model,embed_dim)

	test_vectors = np.zeros((len(Xtest),sent_len,embed_dim))

	for i in range(len(Xtest)):
		test_vectors[i,:,:] = sentence_vector(sent_len,Xtest[i],model,embed_dim)

	return train_vectors,test_vectors

def load_rotten_vectors(Xtrain,Xtest,embed_dim,sent_len,model):

	train_vectors = np.zeros((len(Xtrain),sent_len,embed_dim))

	for i in range(len(Xtrain)):
		train_vectors[i,:,:] = sentence_vector(sent_len,Xtrain[i],model,embed_dim)

	test_vectors = np.zeros((len(Xtest),sent_len,embed_dim))

	for i in range(len(Xtest)):
		test_vectors[i,:,:] = sentence_vector(sent_len,Xtest[i],model,embed_dim)

	return train_vectors,test_vectors