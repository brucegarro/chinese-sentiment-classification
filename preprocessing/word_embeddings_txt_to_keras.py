"""
Convert Tencent AllLab Chinese Embeddings from .txt to .pkl of Embeddings and Tokenizer

Reference: https://www.programmersought.com/article/41874653456/
"""
from os.path import join
import pickle as pk
from gensim.models import KeyedVectors
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


from settings.settings import (
    EMBEDDING_DATA_ROOT,
    RAW_WORD_EMBEDDING_PATH,
)

 # Load pkl file
def load_pkl(input_path):
    with open(input_path, "rb") as f:
        loaded_obj = pk.load(f)


# Write to pkl file
def to_pkl(content, output_path):
    with open(output_path, "wb") as f:
        pk.dump(content, f)


# def load_tencent_word_embedding(embedding_input_path):
#     n = 0
#     with open("tencent.txt", "a", encoding="utf-8", errors="ignore") as w_f:
#         with open("Tencent_AILab_ChineseEmbedding.txt", "r", encoding="utf-8", errors="ignore")as f:
#                          for i in tqdm(range(8824330)): # It seems that the word vector ranges downloaded in different periods are not the same
#                 data = f.readline()
#                 a = data.split()
#                 if i == 0:
#                                          w_f.write("8748463 200\n") # The number of lines written may also be different
#                 if len(a) == 201:
#                     if not a[0].isdigit():
#                         n = n + 1
#                         w_f.write(data)
#          print(n) # output the cleaned range
#     model = KeyedVectors.load_word2vec_format("tencent.txt", binary=False, unicode_errors="ignore")
#     print("successfully load tencent word embedding!")

def save_charembedding(embedding_input_path, embedding_output_path, tokenizer_output_path):
    flag, keras_embedding, words = 0, [], []
 
    with open(embedding_input_path,"r", encoding="utf-8") as file:
        for line in file:
            flag += 1
            if flag >= 3:
                vectorlist = line.split() # Split a line, divided into vocabulary and word vector
                if len(vectorlist[0]) == 1: # Word:"\u4e00" <= vectorlist[0] <="\u9fff"
                    vector = list(map(lambda x:float(x),vectorlist[1:])) # Process the word vector
                    vec = np.array(vector) # Convert list to array
                    keras_embedding.append(vec)
                    words.append(vectorlist[0])
        
        res = np.array(keras_embedding)
        to_pkl(res, embedding_output_path) # Save Tencent word vector
 
        # Create tokenizer Tokenzier object
        tokenizer = Tokenizer()
 
        # fit_on_texts method
        tokenizer.fit_on_texts(words)
        to_pkl(tokenizer, tokenizer_output_path) # Save Tencent word tokenizer


if __name__ == "__main__":
    # Create convert text embeddings to Keras objects and save
    embedding_input_path = RAW_WORD_EMBEDDING_PATH
    embedding_output_path = join(EMBEDDING_DATA_ROOT, "tencent_keras_embedding.pkl")
    tokenizer_output_path = join(EMBEDDING_DATA_ROOT, "tencent_keras_tokenizer.pkl")

    save_charembedding = save_charembedding(
        embedding_input_path=embedding_input_path, 
        embedding_output_path=embedding_output_path,
        tokenizer_output_path=tokenizer_output_path,
    )

    query = "Come on, Wuhan. Come on, China."
    text = "".join(list(query))
    tokenizer = load_pkl(tokenizer_output_path)
    seq = tokenizer.texts_to_sequences([text])
    print(query, seq)
