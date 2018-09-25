import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()

		self.conv1 = nn.Sequential(nn.Conv2d(1,32,3,padding=6),
			                       nn.ReLU(),
			                       nn.BatchNorm2d(32),
			                       nn.MaxPool2d(2),
			                       nn.Dropout2d(0.2))
		self.conv2 = nn.Sequential(nn.Conv2d(32,16,3,padding=6),
			                       nn.ReLU(),
			                       nn.BatchNorm2d(16),
			                       nn.MaxPool2d(2),
			                       nn.Dropout2d(0.2))
		self.conv3 = nn.Sequential(nn.Conv2d(16,6,3,padding=6),
			                       nn.ReLU(),
			                       nn.BatchNorm2d(6),
			                       nn.MaxPool2d(2),
			                       nn.Dropout2d(0.2))
		self.dense1 = nn.Sequential(nn.Linear(600,100),
			                        nn.ReLU(),
			                        nn.Dropout2d(0.5))
		self.dense2 = nn.Sequential(nn.Linear(100,30),
			                        nn.ReLU())
		self.dense3 = nn.Sequential(nn.Linear(30,8),
			                        nn.ReLU())
		self.dense4 = nn.Sequential(nn.Linear(8,1),
			                        nn.Sigmoid())


	def forward(self,x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = out.view(out.size(0),-1)
		out = self.dense1(out)
		out = self.dense2(out)
		out = self.dense3(out)
		out = self.dense4(out)

		return out

def define_pytorch_model(arch):
	model = CNN()
	model = model.to(arch)
	return model