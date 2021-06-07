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
* /deployment
  * `linode_stackscript.sh` - Automates deployment of a remote Tensorflow training server on Linode.
  * `setup_tf_server.py` - Copy raw and preprocessed data to a remote training server.

## Prerequisites

The training code was developed on a single instance Ubuntu 18.04.1 machine using an Nvidia GeForce RTX 2070 8GB GPU

