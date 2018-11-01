import re

import numpy as np 
import pandas as pd 

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



class load_data():
	def __init__(self,imdb_path,rotten_path,testsize):
		self.stopword = stopwords.words('english')
		self.imdb_path = imdb_path
		self.rotten_path = rotten_path
		self.testsize = testsize

	def preprocess(self,text):
		text = re.sub(r"it\'s","it is",str(text))
		text = re.sub(r"i\'d","i would",str(text))
		text = re.sub(r"don\'t","do not",str(text))
		text = re.sub(r"he\'s","he is",str(text))
		text = re.sub(r"there\'s","there is",str(text))
		text = re.sub(r"that\'s","that is",str(text))
		text = re.sub(r"can\'t", "can not", text)
		text = re.sub(r"cannot", "can not ", text)
		text = re.sub(r"what\'s", "what is", text)
		text = re.sub(r"What\'s", "what is", text)
		text = re.sub(r"\'ve ", " have ", text)
		text = re.sub(r"n\'t", " not ", text)
		text = re.sub(r"i\'m", "i am ", text)
		text = re.sub(r"I\'m", "i am ", text)
		text = re.sub(r"\'re", " are ", text)
		text = re.sub(r"\'d", " would ", text)
		text = re.sub(r"\'ll", " will ", text)
		text = re.sub(r"\'s"," is",text)
		text = re.sub(r"[^a-zA-Z0-9]"," ",str(text))
		words = text.split()

		return " ".join(word.lower() for word in words if word.lower() not in self.stopword)


	def load_imdb_data(self):
		Xtrain = []
		Xtest = []
		words = []
		with open(self.imdb_path+'train_pos.txt','r',encoding='latin1') as f:
			for each_line in f.readlines():
				Xtrain.append(each_line[:-1])
				words+=each_line[:-1].split()

		with open(self.imdb_path+'train_neg.txt','r',encoding='latin1') as f:
			for each_line in f.readlines():
				Xtrain.append(each_line[:-1])
				words+=each_line[:-1].split()

		with open(self.imdb_path+'test_pos.txt','r',encoding='latin1') as f:
			for each_line in f.readlines():
				Xtest.append(each_line[:-1])
				words+=each_line[:-1].split()

		with open(self.imdb_path+'test_neg.txt','r',encoding='latin1') as f:
			for each_line in f.readlines():
				Xtest.append(each_line[:-1])
				words+=each_line[:-1].split()

		ytrain = np.zeros(len(Xtrain))
		ytrain[0:12500] = 1

		ytest = np.zeros(len(Xtest))
		ytest[0:12500] = 1

		Xtrain,ytrain = shuffle(Xtrain,ytrain)
		Xtest,ytest = shuffle(Xtest,ytest)

		assert(len(Xtrain)==len(ytrain))
		assert(len(Xtest)==len(ytest))

		return Xtrain,Xtest,ytrain,ytest,words

	def load_rotten_tomatoes_data(self):
		rotten_data = pd.read_csv(self.rotten_path+'data.tsv',sep='	')
		rotten_data.drop(['PhraseId','SentenceId'],inplace=True,axis=1)

		labels = rotten_data['Sentiment'].as_matrix()

		corpus = []
		words = []

		for i in range(len(rotten_data)):
			corpus.append(self.preprocess(rotten_data["Phrase"][i]))
			words+=corpus[i].split()

		Xtrain,Xtest,ytrain,ytest = train_test_split(corpus,labels,test_size=self.testsize,random_state=42)

		assert(len(Xtrain)==len(ytrain))
		assert(len(Xtest)==len(ytest))

		return Xtrain,Xtest,ytrain,ytest,words