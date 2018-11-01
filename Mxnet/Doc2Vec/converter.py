import mxnet as mx 
from mxnet import nd,gluon,autograd


def convert_to_nd(X,y,ctx):

	X_mx = mx.nd.array(X,ctx)
	y_mx = mx.nd.array(y,ctx)

	return X_mx,y_mx
