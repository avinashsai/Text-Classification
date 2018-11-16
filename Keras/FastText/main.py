from __future__ import print_function
import sys
import os
import argparse
import numpy as np 

from preprocess import *
from fasttext import *
from loadvectors import *
from model import *


def main():

	parser = argparse.ArgumentParser(description='Enter hyperparameters')

	parser.add_argument('-fe','--FastTextEpochs',type=int,help='Fast Text Epoch Size',default=15)
	parser.add_argument('-w','--WindowSize',type=int,help='Window Size',default=10)
	parser.add_argument('-v','--VectorSize',type=int,help='Vector Size',default=100)
	parser.add_argument('-c','--mincount',type=int,help='Min Count of Words',default=1)
	parser.add_argument('-b','--batchsize',type=int,help='Batch Size',default=32)
	parser.add_argument('-e','--NeuralNetEpochs',type=int,help='Training Epochs',default=10)
	parser.add_argument('-o','--optimizer',type=str,help='Optimizer',default='adam')
	parser.add_argument('-s','--hiddensize',type=int,help='Hidden Size',default=4)
	
	args = vars(parser.parse_args())

	Fast_Text_Epochs = args['FastTextEpochs']
	Window_Size = args['WindowSize']
	Vector_Size = args['VectorSize']
	Min_Count = args['mincount']
	batchsize = args['batchsize']
	Epochs = args['NeuralNetEpochs']
	optim = args['optimizer']
	hidden_size = args['hiddensize']


	imdb_path = '/home/avinashsai/Documents/Datasets/IMDB/'
	rotten_path = '../../Datasets/Rotten Tomatoes/'
	testsize = 0.2

	imdb_len = 25
	rt_len = 10

	data = load_data(imdb_path,rotten_path,testsize)

	imdb_Xtrain,imdb_Xtest,imdb_ytrain,imdb_ytest,imdb_corpus = data.load_imdb_data()
	rotten_Xtrain,rotten_Xtest,rotten_ytrain,rotten_ytest,rotten_corpus = data.load_rotten_tomatoes_data()

	print("IMDB Data")
	print(imdb_Xtrain[0:2])
	print(imdb_Xtest[0:2])

	print("Rotten Tomatoes Data")
	print(rotten_Xtrain[0:2])
	print(rotten_Xtest[0:2]) 

	ft = fasttext(Window_Size,Vector_Size,Min_Count,Fast_Text_Epochs)

	imdb_ft_model = ft.imdb_fasttext_model(imdb_Xtrain)
	rotten_ft_model = ft.rotten_fasttext_model(rotten_Xtrain)

	imdb_train,imdb_test = load_imdb_vectors(imdb_Xtrain,imdb_Xtest,Vector_Size,imdb_len,imdb_ft_model)
	rotten_train,rotten_test = load_rotten_vectors(rotten_Xtrain,rotten_Xtest,Vector_Size,rt_len,rotten_ft_model)


	net = Model(batchsize,optim,Epochs,hidden_size)

	imdb_net = net.imdb_model(imdb_len,Vector_Size)
	
	imdb_net = net.train_model(imdb_net,imdb_train,imdb_ytrain)
	imdb_acc = net.compute_accuracy(imdb_net,imdb_test,imdb_ytest)
	print("IMDB Accuracy is {}".format(imdb_acc))


	rotten_net = net.rotten_model(rt_len,Vector_Size)

	rotten_net = net.train_model(rotten_net,rotten_train,rotten_ytrain)
	rotten_acc = net.compute_accuracy(rotten_net,rotten_test,rotten_ytest)
	print("Rotten Accuracy is {}".format(rotten_acc))


if __name__ == '__main__':
	main()