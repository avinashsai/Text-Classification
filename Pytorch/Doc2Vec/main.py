from __future__ import print_function
import os
import re
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import argparse 


import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from loader import *
from doc2vec import *

from converter import *

from pytorch_model import *

from train_model import *

def main():
    
    train_data,train_labels,test_data,test_labels = readfiles()

    train_length = len(train_data)
    test_length = len(test_data)

    parser = argparse.ArgumentParser(description='Enter hyperparameters')
    
    parser.add_argument('-dm','--model',type=int,help='skip gram or Bag of Words',required=True)
    parser.add_argument('-de','--doc2vecepochs',type=int,help='Doc2vec Model Number of epochs',required=True)
    parser.add_argument('-w','--windowsize',help='Window Size',required=True)

    parser.add_argument('-b','--batchsize',type=int,help='Batch Size',required=True)
    parser.add_argument('-e','--epochs',type=int,help='Model Epochs',required=True)
    parser.add_argument('-l','--learning_rate',type=float,help='Learning rate',required=True)
    parser.add_argument('-o','--optimizer',help='optimizer',required=True)
    parser.add_argument('-c','--cpu_gpu',help='Enter cpu for cpu else gpu for gpu',required=True)
    

    args = vars(parser.parse_args())
    if((25000%(args['batchsize']))!=0):
        raise ValueError("Please Specify Batch Size that divides 25000")
    
    
    tagged_data = create_tags(train_data,train_labels,test_data,test_labels)

    doc2vec_model = create_doc2vec_model(args['windowsize'],args['model'])

    doc2vec_model,tagged_data = train_doc2vec_model(doc2vec_model,args['doc2vecepochs'],tagged_data)

    train_vectors,train_labels,test_vectors,test_labels = generate_vectors(doc2vec_model,tagged_data,train_length,test_length)
    
    
    model = define_pytorch_model(args['cpu_gpu'])
    train_vectors_tensor,train_labels_tensor = convert_to_tensor(train_vectors,train_labels,args['cpu_gpu'])
    test_vectors_tensor,test_labels_tensor = convert_to_tensor(test_vectors,test_labels,args['cpu_gpu'])

    model = train_pytorch_model(model,train_vectors_tensor,train_labels_tensor,args['batchsize'],args['epochs'],args['optimizer'],args['learning_rate'])
    test_accuracy = calculate_accuracy(model,test_vectors_tensor,test_labels_tensor)

    print("Test Accuracy :".format((test_accuracy/25000)*100))
    

if __name__ == '__main__':
	main()
