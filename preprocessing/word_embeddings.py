"""
Convert Tencent AllLab Chinese Embeddings from .txt to .pkl of Embeddings and Tokenizer

Reference: https://www.programmersought.com/article/41874653456/
"""
from os.path import join
import pickle as pkl
from gensim.models import KeyedVectors
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from settings.settings import (
    EMBEDDING_DATA_ROOT,
    RAW_WORD_EMBEDDING_PATH,
)


def get_embeddings_matrix(tokenizer, embedding_input_path):
    """
    Retrieve the word embeddings from 'Tencent_AILab_ChineseEmbedding.txt' for a given vocabulary
    pre-compiled in a Keras tokenizer object
    
    Input
    -----
    embedding_input_path - str: a complete filepath for 'Tencent_AILab_ChineseEmbedding.txt'
    tokenizer - Tokenizer: A Keras tokizer object pre-compiled with a target vocabulary
    
    References
    ----------
    https://keras.io/examples/nlp/pretrained_word_embeddings/#load-pretrained-word-embeddings
    https://www.dlology.com/blog/tutorial-chinese-sentiment-analysis-with-hotel-review-data/
    """
    number_of_tokens = len(tokenizer.word_index)
    embedding_length = 200

    # Make matrix num_tokens + 1 since 0 is a reserved index in Tokenizer
    embedding_matrix = np.zeros((number_of_tokens+1, embedding_length))
    found_words = set()

    with open(embedding_input_path, "r", encoding="utf-8") as file_object:
        # First line contains num of lines in document and embedding length (200)
        first_line = next(file_object)
        num_lines = int(first_line.split(" ")[0])

        # Populated embedding_matrix
        for i in tqdm(range(num_lines)):
            line = next(file_object)
            embedding = line.rstrip("\n").split(" ")
            word = embedding[0]
            if word in tokenizer.word_index:
                vector = [ float(coordinate) for coordinate in embedding[1:] ]
                token_index = tokenizer.word_index[word]
                embedding_matrix[token_index] = vector

                found_words.add(word)

    did_not_find = (tokenizer.word_index.keys() - found_words)
    print("Found %s words" % len(found_words))
    print("Did not find %s words: %s\n" % (len(did_not_find), did_not_find))
    return embedding_matrix

def to_pkl(content, output_path):
    with open(output_path, "wb") as f:
        pkl.dump(content, f)
    print("Created file: %s" % output_path)

def load_pkl(input_path):
    with open(input_path, "rb") as f:
        loaded_obj = pkl.load(f)

def save_word_embeddings_and_tokenizer(tokenizer, embedding_input_path, embedding_output_path, tokenizer_output_path):
    embedding_matrix = get_embeddings_matrix(tokenizer, embedding_input_path)
    to_pkl(tokenizer, tokenizer_output_path)
    to_pkl(embedding_matrix, embedding_output_path)
