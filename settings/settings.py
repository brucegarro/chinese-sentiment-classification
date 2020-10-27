from os.path import join

from settings.local import *


EMBEDDING_MATRIX_PATH = join(EMBEDDING_DATA_ROOT, "embedding_matrix_for_vocabulary.pkl")
KERAS_TOKENIZER_PATH = join(EMBEDDING_DATA_ROOT, "keras_tokenizer_for_vocabulary.pkl")
