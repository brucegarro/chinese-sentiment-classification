import os
from os.path import join


if os.getenv("REPO_PATH"):
	ALL_REPOS_ROOT = os.getenv("REPO_PATH")
else:
	from settings.local import ALL_REPOS_ROOT

REPO_ROOT = join(ALL_REPOS_ROOT, "chinese-sentiment-classification")

REN_CEC_PATH = os.getenv("REN_CEC_PATH", default=join(ALL_REPOS_ROOT, "Ren_CECps-Dictionary"))
DATA_PATH = join(REN_CEC_PATH, "data/CEC_emotionCoprus")


EMBEDDING_DATA_ROOT = os.getenv("EMBEDDING_DATA_ROOT",
	default=join(ALL_REPOS_ROOT, "chinese-sentiment-classification-data", "embedding"))
RAW_WORD_EMBEDDING_PATH = join(EMBEDDING_DATA_ROOT, "Tencent_AILab_ChineseEmbedding.txt")

EMBEDDING_MATRIX_NAME = "embedding_matrix_for_vocabulary.pkl"
KERAS_TOKENIZER_NAME = "keras_tokenizer_for_vocabulary.pkl"
EMBEDDING_MATRIX_PATH = join(EMBEDDING_DATA_ROOT, EMBEDDING_MATRIX_NAME)
KERAS_TOKENIZER_PATH = join(EMBEDDING_DATA_ROOT, KERAS_TOKENIZER_NAME)

SAVED_MODELS_PATH = join(REPO_ROOT, "saved_models")
LOGS_PATH = join(REPO_ROOT, "logs")
