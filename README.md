# Chinese Blog Sentiment Classification

Recurrent neural network models which classify the emotion expressed in an input text of Chinese text.

## Directory

* /preprocessing
  * Provides code to map the (dataset raw xml) -> (Python object abstractions) -> (word embedding arrays)
* /notebooks
  * /preprocessing_routines.ipynb - Creates a tokenizer and word embedding matrix to match the vocabulary found in the dataset and then caches them to Python pickles..
  * /visualization.ipynb - A notebook to graph distribution of sentiment labels and randomly sample and read individual blog posts.
* /modeling
  * Implementation of models utilizing Tensorflow Keras LSTM neural networks.

## Prerequisites

The training code was developed on a single instance Ubuntu 18.04.1 machine using an Nvidia GeForce RTX 2070 8GB GPU

## Download Datasets

This project contains code to process the Ren-CECps dataset. This can be obtained by contacting the publisher through this link: http://a1-www.is.tokushima-u.ac.jp/member/ren/Ren-CECps1.0/Ren-CECps1.0.html

This project also makes use of the Tencent AILab word embeddings for preprocessing text before feeding into models for training. Find the word embeddings here: https://ai.tencent.com/ailab/nlp/en/embedding.html
