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


	def imdb_fasttext_model(self,imdb_train):
		words = []
		for sentence in imdb_train:
			words+=sentence.split()

		imdb_model = FastText(words,min_count=self.mincount,size=self.vectorsize,
			window=self.windowsize,iter=self.epochs)

		return imdb_model

	def rotten_fasttext_model(self,rotten_train):
		words = []
		for sentence in rotten_train:
			words+=sentence.split()

		rotten_model = FastText(words,min_count=self.mincount,size=self.vectorsize,
			window=self.windowsize,iter=self.epochs)

		return rotten_model