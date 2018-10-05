import mxnet as mx
from mxnet import gluon,autograd,nd

import gluonnlp as glp

def load_files(path):
  
  train_corpus = []
  test_corpus = []
  
  with open(path+'train_pos.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      train_corpus.append(line[:-1])
      
  with open(path+'train_neg.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      train_corpus.append(line[:-1])
      
  with open(path+'test_pos.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      test_corpus.append(line[:-1])
      
  with open(path+'test_neg.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      test_corpus.append(line[:-1])
      
  return train_corpus,test_corpus

path = ''
train_corpus,test_corpus = load_files(path)

train_length = len(train_corpus)
test_length = len(test_corpus)

train_labels = mx.nd.zeros(train_length)
train_labels[0:12500] = 1

test_labels = mx.nd.zeros(test_length)
test_labels[0:12500] = 1

maxwords = 20000

words = []
for sentence in train_corpus:
  words+=sentence.split()
  
for sentence in test_corpus:
  words+=sentence.split()

counter = glp.data.count_tokens(words)

maxsentencelength = 50
embeddingdim = 100

def get_indices(words):
  indices = mx.nd.zeros(maxsentencelength)
  count = 0
  for word in words:
    if(word in counter):
      indices[count] = counter[word]
      count+=1
    if(count>=maxsentencelength):
      return indices
  return indices

train_indices = mx.nd.zeros((train_length,maxsentencelength))

for i in range(train_length):
  words = train_corpus[i].split()
  train_indices[i] = get_indices(words)

test_indices = mx.nd.zeros((test_length,maxsentencelength))

for i in range(test_length):
  words = test_corpus[i].split()
  test_indices[i] = get_indices(words)

batchsize = 50

lr = 0.00001

train_arraydataset = gluon.data.ArrayDataset(train_indices,train_labels)
train_loader = gluon.data.DataLoader(train_arraydataset,batch_size=batchsize)

test_arraydataset = gluon.data.ArrayDataset(test_indices,test_labels)
test_loader = gluon.data.DataLoader(test_arraydataset,batch_size=batchsize)

model = gluon.nn.Sequential()

with model.name_scope():
  model.add(gluon.nn.Embedding(maxwords,embeddingdim))
  model.add(gluon.rnn.LSTM(32))
  model.add(gluon.nn.Dense(32))
  model.add(gluon.nn.Dense(1))
  model.add(gluon.nn.Activation('sigmoid'))

model.collect_params().initialize(mx.init.Xavier(magnitude=2))

binary_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss()

optimizer = 'rmsprop'

batches = int(train_length/batchsize)

numepochs = 2

optim = gluon.Trainer(model.collect_params(), optimizer, {'learning_rate': 0.001})

def evaluate_accuracy(net,X,y):
  acc = mx.metric.Accuracy()
  pred = net(X)
  ypred = (pred>=0.5)
  ypred = ypred.reshape(len(ypred))
  acc.update(ypred,y)
  return acc.get()[1]

for epoch in range(numepochs):
  total_loss = mx.nd.zeros(batches)
  for batch_idx,(Xtrain,ytrain) in enumerate(train_loader):
    
    with autograd.record():
      output = model(Xtrain)
      loss = binary_cross_entropy(output,ytrain)
    
    loss.backward()
    optim.step(batchsize)
    curr_loss = nd.mean(loss).asscalar()
    total_loss[batch_idx] = curr_loss
      
      
  actual_loss = nd.mean(total_loss).asscalar()
  train_acc = evaluate_accuracy(model,train_indices,train_labels)
  
  print("Epoch {} Loss {} Train Accuracy {}".format(epoch,actual_loss,train_acc))
  

test_acc = evaluate_accuracy(model,test_indices,test_labels)
print("Epoch {} Loss {} Test Accuracy {}".format(epoch,actual_loss,test_acc))

