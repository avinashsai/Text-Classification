from __future__ import print_function
import os
import re
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import argparse 

from loader import *
from doc2vec import *

from model_gluon import *


from converter import *

def main():
    train_data,train_labels,test_data,test_labels = readfiles()

    train_length = len(train_data)
    test_length = len(test_data)

    parser = argparse.ArgumentParser(description='Enter hyperparameters')
    
    parser.add_argument('-dm','--model',help='skip gram or Bag of Words',required=True)
    parser.add_argument('-de','--doc2vecepochs',help='Doc2vec Model Number of epochs',required=True)
    parser.add_argument('-w','--windowsize',help='Window Size',required=True)

    parser.add_argument('-b','--batchsize',help='Batch Size',required=True)
    parser.add_argument('-e','--epochs',help='Model Epochs',required=True)
    parser.add_argument('-l','--learning_rate',help='Learning rate',required=True)
    parser.add_argument('-o','--optimizer',help='optimizer',required=True)
    parser.add_argument('-c','--cpu_gpu',help='Enter cpu for cpu else gpu for gpu',required=True)
    

    args = vars(parser.parse_args())
    
    
    tagged_data = create_tags(train_data,train_labels,test_data,test_labels)

    doc2vec_model = create_doc2vec_model(int(args['windowsize']),int(args['model']))

    doc2vec_model,tagged_data = train_doc2vec_model(doc2vec_model,int(args['doc2vecepochs']),tagged_data)

    train_vectors,train_labels,test_vectors,test_labels = generate_vectors(doc2vec_model,tagged_data,train_length,test_length)
    
    model,ctx = define_gluon_model(args['cpu_gpu'])

   
    train_vectors_mx,train_labels_mx = convert_to_nd(train_vectors,train_labels,ctx)
    test_vectors_mx,test_labels_mx = convert_to_nd(test_vectors,test_labels,ctx)

    batchsize = int(args['batchsize'])
    num_epochs = int(args['epochs'])
    learningrate = float(args['learning_rate'])

    model = train_gluon_model(model,train_vectors_mx,train_labels_mx,ctx,learningrate,batchsize,num_epochs,args['optimizer'])
    test_accuracy = calculate_gluon_accuracy(model,test_vectors_mx,test_labels_mx)
    print("Test Accuracy {}".format(test_accuracy))
   

if __name__ == '__main__':
	main()