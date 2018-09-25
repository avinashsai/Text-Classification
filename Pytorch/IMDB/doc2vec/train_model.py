import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_optimizer(optimizer,net,learningrate):
	if(optimizer=="adam"):
		return torch.optim.Adam(net.parameters(),lr=learningrate)
	elif(optimizer=="sgd"):
		return torch.optim.SGD(net.parameters(),lr=learningrate)
	elif(optimizer=='adadelta'):
		return torch.optim.Adadelta(net.parameters(),lr=learningrate)
	elif(optimizer=='adagrad'):
		return torch.optim.Adagrad(net.parameters(),lr=learningrate)
	elif(optimizer=="rmsprop"):
		return torch.optim.RMSprop(net.parameters(),lr=learningrate)
	else:
		raise ValueError("Please Enter optimizers (adam,sgd,adadelta,adagrad,rmsprop)")

def calculate_accuracy(net,X,y):
	pred = net(X)
	ypred = (pred>=0.5).double()
	y = y.double()
	correct = 0
	for i in range(len(ypred)):
		if(ypred[i]==y[i]):
			correct+=1
	return correct


def train_pytorch_model(net,Xtrain,ytrain,batchsize,numepochs,criteria,learningrate):
	numbatches = (int(Xtrain.shape[0])/batchsize)

	optimizer = get_optimizer(criteria,net,learningrate)
	
	for epoch in range(numepochs):
		for i in range(numbatches):
			X = Xtrain[i*(batchsize):(i+1)*batchsize,:,:,:]
			y = ytrain[i*(batchsize):(i+1)*batchsize]
			y = y.view(-1,1)
			output = net(X)

			loss = F.binary_cross_entropy(output,y)

			loss.backward()
			optimizer.step()

		train_acc = calculate_accuracy(net,Xtrain,ytrain)

		print("Epoch {} Loss {} Train Accuracy {}".format(epoch,loss.item(),((train_acc/25000)*100)))

	return net

