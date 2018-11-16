
# Getting started
This is the implementation of FastText using gensim.FastText is an extension to Word2Vec proposed by Facebook research.

# Implementation
Fasttext is trained on IMDB and Rotten Tomatoes datasets separately and then sentiment classification is performed on both tha datasets using LSTM network

# Usage
1. Clone this repository

```
git clone https://github.com/avinashsai/Text-Classification.git
```

2. Run the files using

```
cd Keras/FastText

python main.py --fe (Number of epochs FastText model is to be trained)
               --w (Window Size of FastText model)
               --v (Vector Size)
               --c (Min Count of words)
               --b (Batch Size of LSTM)
               --e (Number of Epochs LSTM to be trained)
               --o (Optimizer)
               --s (Number of hidden LSTM cells)
