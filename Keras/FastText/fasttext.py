import sys
import os
import numpy as np 

import gensim
from gensim.models import FastText


class fasttext():
	def __init__(self,windowsize,vectorsize,epochs,mincount):
		self.windowsize = windowsize
		self.vectorsize = vectorsize
		self.epochs = epochs
		self.mincount = mincount

	def define_model(self):
		model = FastText(min_count=self.mincount,window=self.windowsize,
			size=self.vectorsize)
		return model

	def imdb_fasttext_model(self,imdb_corpus,imdb_train,imdb_test):
		imdb_model = self.define_model()
		imdb_model.build_vocab(imdb_corpus)

		total_corpus = imdb_train+imdb_test
		imdb_model.train(total_corpus,total_examples=imdb_model.corpus_count,epochs=self.epochs)

		return imdb_model

	def rotten_fasttext_model(self,rotten_corpus,rotten_train,rotten_test):
		rotten_model = self.define_model()
		rotten_model.build_vocab(rotten_corpus)

		total_corpus = rotten_train+rotten_test
		rotten_model.train(total_corpus,total_examples=rotten_model.corpus_count,epochs=self.epochs)

		return rotten_model