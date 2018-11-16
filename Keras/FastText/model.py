import sys
import os
import numpy as np 

import tensorflow as tf 
import keras
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras import metrics
from keras.utils import to_categorical

class Model():
	def __init__(self,batchsize,optim,epochs,hiddensize):
		self.batchsize = batchsize
		self.hiddensize = hiddensize
		self.optim = optim
		self.epochs = epochs


	def imdb_model(self,sentencelen,embeddim):
		model = Sequential()
		model.add(LSTM(self.hiddensize,input_shape=(sentencelen,embeddim)))
		model.add(Dense(1,activation='sigmoid'))

		model.compile(optimizer=self.optim,loss='binary_crossentropy',metrics=['accuracy'])

		return model

	def rotten_model(self,sentencelen,embeddim):
		model = Sequential()
		model.add(LSTM(self.hiddensize,input_shape=(sentencelen,embeddim)))
		model.add(Dense(5,activation='softmax'))

		model.compile(optimizer=self.optim,loss='categorical_crossentropy',metrics=['accuracy'])

		return model

	def train_model(self,model,Xtrain,ytrain):
		if(np.amax(ytrain)>1):
			ytrain = to_categorical(ytrain)
		model.fit(Xtrain,ytrain,batch_size=self.batchsize,epochs=self.epochs)
		return model

	def compute_accuracy(self,model,Xtest,ytest):
		if(np.amax(ytest)>1):
			ytest = to_categorical(ytest)
		acc = model.evaluate(Xtest,ytest)[1]
		return acc*100

