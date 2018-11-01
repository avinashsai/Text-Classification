import torch


def convert_to_tensor(X,y,ctx):

	X_tensor = torch.from_numpy(X)
	y_tensor = torch.from_numpy(y)

	X_tensor = X_tensor.float()
	y_tensor = y_tensor.float()

	X_tensor = X_tensor.reshape(int(X_tensor.shape[0]),1,10,10).to(ctx)
	y_tensor = y_tensor.to(ctx)

	
	return X_tensor,y_tensor
