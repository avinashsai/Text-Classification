import mxnet as mx 
from mxnet import nd,gluon,autograd


def define_gluon_model(arch):
	if(arch=="gpu"):
		ctx = mx.gpu()
	else:
		ctx = mx.cpu()

	net = gluon.nn.Sequential()

	with net.name_scope():
		net.add(gluon.nn.Conv2D(channels=32,kernel_size=3,padding=6))
		net.add(gluon.nn.BatchNorm())
		net.add(gluon.nn.Activation('relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=(2,2)))
		net.add(gluon.nn.Dropout(0.2))
		net.add(gluon.nn.Conv2D(channels=16,kernel_size=3,padding=6))
		net.add(gluon.nn.BatchNorm())
		net.add(gluon.nn.Activation('relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=(2,2)))
		net.add(gluon.nn.Dropout(0.2))
		net.add(gluon.nn.Conv2D(channels=6,kernel_size=3,padding=6))
		net.add(gluon.nn.BatchNorm())
		net.add(gluon.nn.Activation('relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=(2,2)))
		net.add(gluon.nn.Dropout(0.2))
		net.add(gluon.nn.Flatten())
		net.add(gluon.nn.Dense(100,activation='relu'))
		net.add(gluon.nn.Dropout(0.5))
		net.add(gluon.nn.Dense(30,activation='relu'))
		net.add(gluon.nn.Dense(8,activation='relu'))
		net.add(gluon.nn.Dense(1,activation='sigmoid'))

	return net,ctx


def calculate_gluon_accuracy(net,X,y):
	if(X.ndim!=4):	
		X = X.reshape((int(X.shape[0]),10,10,1))
	acc = mx.metric.Accuracy()
	pred = net(X)
	ypred = (pred>=0.5)
	ypred = ypred.reshape(len(ypred))
	acc.update(ypred,y)
	return acc.get()[1]

def train_gluon_model(net,Xtrain,ytrain,context,lr,batchsize,numepochs,optimizer):

	net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=context)

	binary_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss()

	optim = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': lr})

	train_l = len(Xtrain)

	batches = int(train_l/batchsize)

	Xtrain = Xtrain.reshape((train_l,10,10,1))

	for epoch in range(numepochs):
		total_loss = mx.nd.zeros(batches)
		for i in range(0,batches):
			X = Xtrain[i*batchsize:(i+1)*batchsize,:,:]
			y = ytrain[i*batchsize:(i+1)*batchsize]

			with autograd.record():
				output = net(X)
				loss = binary_cross_entropy(output,y)

			loss.backward()
			optim.step(batchsize)

			curr_loss = nd.mean(loss).asscalar()
			total_loss[i] = curr_loss
			#print(curr_loss,end=" ")

		actual_loss = nd.mean(total_loss).asscalar()
		train_acc = calculate_gluon_accuracy(net,Xtrain,ytrain)

		print("Epoch {} Loss {} Train Accuracy {}".format(epoch,actual_loss,train_acc))


	return net

