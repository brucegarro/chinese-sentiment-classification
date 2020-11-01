from os.path import join

from settings.local import *


DATA_PATH = join(ALL_REPOS_ROOT, "Ren_CECps-Dictionary/data/CEC_emotionCoprus")
EMBEDDING_DATA_ROOT = join(ALL_REPOS_ROOT, "chinese-sentiment-classification-data", "embedding")
RAW_WORD_EMBEDDING_PATH = join(EMBEDDING_DATA_ROOT, "Tencent_AILab_ChineseEmbedding.txt")

EMBEDDING_MATRIX_PATH = join(EMBEDDING_DATA_ROOT, "embedding_matrix_for_vocabulary.pkl")
KERAS_TOKENIZER_PATH = join(EMBEDDING_DATA_ROOT, "keras_tokenizer_for_vocabulary.pkl")

SAVED_MODELS_PATH = join(REPO_ROOT, "saved_models")
