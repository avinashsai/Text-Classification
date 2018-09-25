from __future__ import print_function
import re
import os
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

import gensim
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec,TaggedDocument

from sklearn.utils import shuffle

import tensorflow as tf
import keras
from keras.layers import Conv1D,Flatten,Dropout,Dense,BatchNormalization
from keras import metrics
from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,LSTM,Bidirectional,Activation
from keras.models import Model

path = '../../../Datasets/IMDB'

train_data = []
train_labels = np.zeros(25000)
train_labels[0:12500] = 1

with open(path+'/train_pos.txt','r',encoding='latin1') as f:
  for line in f:
    train_data.append(line[:-1])

with open(path+'/train_neg.txt','r',encoding='latin1') as f:
  for line in f:
    train_data.append(line[:-1])

print(len(train_data))

test_data = []
test_labels = np.zeros(25000)
test_labels[0:12500] = 1

with open(path+'/test_pos.txt','r',encoding='latin1') as f:
  for line in f:
    test_data.append(line[:-1])

with open(path+'/test_neg.txt','r',encoding='latin1') as f:
  for line in f:
    test_data.append(line[:-1])

print(len(test_data))

def tag_sentence(sentence,tag):
  return TaggedDocument(sentence,tag)

data = []

train_length = len(train_data)
test_length = len(test_data)

for i in range(25000):
  tag = ["train_"+str(i)]
  data.append(tag_sentence(train_data[i].split(),tag))

for i in range(25000):
  tag = ["test_"+str(i)]
  data.append(tag_sentence(test_data[i].split(),tag))

doc2vec_model = Doc2Vec(min_count=1, window=10, vector_size=100, sample=1e-4, negative=5, workers=7)

doc2vec_model.build_vocab(data)

def make_shuffle(sentences):
    return shuffle(sentences)

for epoch in range(20):
  print(epoch,end=" ")
  data = make_shuffle(data)
  doc2vec_model.train(data,total_examples=doc2vec_model.corpus_count,epochs=doc2vec_model.epochs)
  

train_vectors = np.zeros((train_length,100))

for i in range(train_length):
  train_vectors[i] = doc2vec_model.docvecs["train_"+str(i)]

test_vectors = np.zeros((test_length,100))

for i in range(test_length):
  test_vectors[i] = doc2vec_model.docvecs["test_"+str(i)]


train_vecs = train_vectors.reshape((train_length,10,10))

def conv1d_model():
  model = Sequential()
  model.add(Conv1D(10,3,activation='relu',input_shape=(10,10)))
  model.add(Dropout(0.5))
  model.add(Conv1D(20,3,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(40,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(8,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1,activation='sigmoid'))
  return model

cnn1d_model = conv1d_model()

optim = keras.optimizers.Adam(lr=0.0001)

cnn1d_model.compile(optimizer=optim,loss='binary_crossentropy',metrics=['accuracy'])

cnn1d_model.fit(train_vecs,train_labels,batch_size=50,epochs=100)

test_vecs = test_vectors.reshape((test_length,10,10))

acc = cnn1d_model.evaluate(test_vecs,test_labels)[0]

print(acc)

