import torch
import torch.nn as nn
from torch import autograd

import torch.nn.functional as F
import torch.optim as optim

import torchtext

import collections
from collections import Counter
import re
import os
import sys
import numpy as np

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopword = stopwords.words('english')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(text): 
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
  text = re.sub(r"[^a-zA-Z]"," ",str(text))
  sents = text.split()
  return " ".join(word.lower() for word in sents if word.lower() not in stopword  and len(word)>1)

def load_files(path):
  
  train_corpus = []
  
  with open(path+'train_pos.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      train_corpus.append(preprocess(line[:-1]))
  
  with open(path+'train_neg.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      train_corpus.append(preprocess(line[:-1]))
      
      
  test_corpus = []
  
  with open(path+'test_pos.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      test_corpus.append(preprocess(line[:-1]))
  
  with open(path+'test_neg.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      test_corpus.append(preprocess(line[:-1]))
      
  return train_corpus,test_corpus

path = ""
train_corpus,test_corpus = load_files(path)

train_length = len(train_corpus)
test_length = len(test_corpus)

numclasses = 1

train_labels = torch.zeros(train_length)
train_labels[0:12500] = 1

test_labels = torch.zeros(test_length)
test_labels[0:12500] = 1

words = []
for sentence in train_corpus:
  words+=sentence.split()

for sentence in test_corpus:
  words+=sentence.split()

maxwords = 20000

maxsentencelength = 40

counts = Counter(words)

vocabulary_ = torchtext.vocab.Vocab(counts,max_size=maxwords,min_freq=2)

mapping = vocabulary_.stoi

train_indices = torch.zeros((train_length,maxsentencelength))

def generate_indices(sentence):
  indices = torch.zeros(maxsentencelength)
  toks = sentence.split()
  count = 0
  for word in toks:
    if word in mapping:
      indices[count] =  mapping[word]
      count+=1
    if(count>=maxsentencelength):
      return indices
  return indices

for i in range(train_length):
  train_indices[i] = generate_indices(train_corpus[i])

test_indices = torch.zeros((test_length,maxsentencelength))

for i in range(test_length):
  test_indices[i] = generate_indices(test_corpus[i])

train_indices = train_indices.long()
test_indices = test_indices.long()

train_array = torch.utils.data.TensorDataset(train_indices,train_labels)
train_loaders = torch.utils.data.DataLoader(train_array,batch_size=batchsize)

test_array = torch.utils.data.TensorDataset(test_indices,test_labels)
test_loaders = torch.utils.data.DataLoader(test_array,batch_size=batchsize)

class lstm(nn.Module):
  def __init__(self,hiddensize,batchsize,embeddim,sentlen,numclasses,maxwords):
    super(lstm,self).__init__()
    self.hiddensize = hiddensize
    self.batchsize = batchsize
    self.embeddim = embeddim
    self.sentencelength = sentlen
    self.outsize = numclasses
    self.maxwords = maxwords
    
    self.embed = nn.Embedding(self.maxwords+1,self.embeddim)
    self.lstm = nn.LSTM(self.embeddim,self.hiddensize,batch_first=True)
    self.dense = nn.Linear(self.hiddensize,self.outsize)
    self.act = nn.Sigmoid()
    
  def forward(self,x):
    h0 = torch.zeros(1, x.size(0), self.hiddensize)
    c0 = torch.zeros(1, x.size(0), self.hiddensize)
    
    out = self.embed(x)
    out,_ = self.lstm(out,(h0,c0))
    out = self.dense(out[:, -1, :])
    out = self.act(out)
    
    return out

hiddenlayers = 50
batchsize = 50

learning_rate = 0.0001

embedding_dim = 100

model = lstm(hiddenlayers,batchsize,embedding_dim,maxsentencelength,numclasses,maxwords)

inp = train_indices[:batchsize,:]
out = model(inp)
print(out.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

numepochs = 1

def evaluate_acc(net,loader):
  
  correct = 0
  
  for batch_idx,(X,y) in enumerate(loader):
    
    #X,y = X.to(device),y.to(device)
    outputs = net(X)
    y = y.double()
    y = y.view(y.size(0))
    yp = (outputs>=0.5).double()
    yp = yp.view(yp.size(0))
    correct+=(torch.sum(y==yp).item())
  
  return float((100*correct)/25000)

for epoch in range(numepochs):
  for batch_idx,(Xtrain,ytrain) in enumerate(train_loaders):
    
    #Xtrain,ytrain = Xtrain.to(device),ytrain.to(device)
    ytrain = ytrain.long()
    
    ypred = model(Xtrain)
    ytrain = ytrain.view(-1,1).float()
    loss = F.binary_cross_entropy(ypred,ytrain)
    
    optimizer.zero_grad()
   
    loss.backward()
    optimizer.step()
    
  train_acc = evaluate_acc(model,train_loaders)
  print("Epoch {} loss {} Train Accuracy {}".format(epoch,loss,train_acc))
  
  
test_acc = evaluate_acc(model,test_loaders)
print("Test Accuracy {}".format(test_acc))

