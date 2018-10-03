import sys
import os
import re
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.utils import shuffle

import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import LSTM,Dense,Dropout
from keras.models import Model,Sequential
from keras.layers import Input,Bidirectional
from keras.layers import Embedding

from keras import metrics


def read_data():
  
  train_data = []
  with open('train_pos.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      train_data.append(line[:-1])
  with open('train_neg.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      train_data.append(line[:-1])
      
  test_data = []
  with open('test_pos.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      test_data.append(line[:-1])
  with open('test_neg.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      test_data.append(line[:-1])
      
  
  train_labels = np.zeros(25000)
  train_labels[0:12500] = 1
  
  test_labels = np.zeros(25000)
  test_labels[0:12500] = 1
  
  train_data,train_labels = shuffle(train_data,train_labels)
  test_data,test_labels = shuffle(test_data,test_labels)
  
  return train_data,test_data,train_labels,test_labels

train_data,test_data,train_labels,test_labels = read_data()

train_length = len(train_data)
test_length = len(test_data)

assert train_length==train_labels.shape[0]
assert test_length==test_labels.shape[0]

maxwords = 20000

max_sentence_length = 60
embeddingdim = 100
hiddensize = 50

def generate_indices(train,test,max_num_words,max_sent_len):
  
  tokenizer = Tokenizer(num_words=maxwords)
  tokenizer.fit_on_texts(train)
  
  train_tokens = tokenizer.texts_to_sequences(train)
  train_indices = pad_sequences(train_tokens,max_sent_len)
  
  test_tokens = tokenizer.texts_to_sequences(test)
  test_indices = pad_sequences(test_tokens,max_sent_len)
  
  return train_indices,test_indices

train_indices,test_indices = generate_indices(train_data,test_data,maxwords,max_sentence_length)

def model(max_sent_len,max_num_words,embedding_dim,n_hidden):
  indices = Input(shape=(max_sent_len,))
  vectors = Embedding(input_dim=max_num_words,output_dim=embedding_dim,trainable=False,input_length=max_sent_len)(indices)
  lstm = LSTM(n_hidden,return_sequences=False)
  lstm_out = lstm(vectors)
  probs = Dense(1,activation='sigmoid')(lstm_out)
  
  lstm_model = Model(inputs=indices,output=probs)
  
  return lstm_model

lstm_model = model(max_sentence_length,maxwords,embeddingdim,hiddensize)

lr_rate = 0.001
batchsize = 50
numepochs = 50

lstm_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

lstm_model.fit(train_indices,train_labels,validation_data=(test_indices,test_labels),batch_size=batchsize,epochs=numepochs)

scores = lstm_model.evaluate(test_indices,test_labels)

