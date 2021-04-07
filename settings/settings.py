import os
from os.path import join


if os.getenv("REPO_PATH"):
    ALL_REPOS_ROOT = os.getenv("REPO_PATH")
else:
    from settings.local import ALL_REPOS_ROOT

REPO_ROOT = join(ALL_REPOS_ROOT, "chinese-sentiment-classification")

REN_CEC_PATH = os.getenv("REN_CEC_PATH", default=join(ALL_REPOS_ROOT, "Ren_CECps-Dictionary"))
DATA_PATH = join(REN_CEC_PATH, "data/CEC_emotionCoprus")

# Define paths to cached embeddings
EMBEDDING_DATA_ROOT = os.getenv("EMBEDDING_DATA_ROOT",
    default=join(ALL_REPOS_ROOT, "chinese-sentiment-classification-data", "embedding"))
RAW_WORD_EMBEDDING_PATH = join(EMBEDDING_DATA_ROOT, "Tencent_AILab_ChineseEmbedding.txt")

class TFProjectSettings(object):
    EMBEDDING_MATRIX_NAME = "embedding_matrix_for_vocabulary.pkl"
    TOKENIZER_NAME = "keras_tokenizer_for_vocabulary.pkl"
    EMBEDDING_MATRIX_PATH = join(EMBEDDING_DATA_ROOT, EMBEDDING_MATRIX_NAME)
    TOKENIZER_PATH = join(EMBEDDING_DATA_ROOT, TOKENIZER_NAME)

class PTProjectSettings(object):
    EMBEDDING_MATRIX_NAME = "embedding_matrix_for_vocabulary.pkl"
    TOKENIZER_NAME = "pt_tokenizer_for_vocabulary.pkl"
    EMBEDDING_MATRIX_PATH = join(EMBEDDING_DATA_ROOT, "pt_project", EMBEDDING_MATRIX_NAME)
    TOKENIZER_PATH = join(EMBEDDING_DATA_ROOT, "pt_project", TOKENIZER_NAME)

# Set PROJECT_TYPE to "pytorch" or "tensorflow"
PROJECT_TYPE = "pytorch"

def get_project_settings(project_type):
    if project_type == "tensorflow":
        return TFProjectSettings
    elif project_type == "pytorch":
        return PTProjectSettings
    else:
        raise Exception("Set PROJECT_TYPE to 'tensorflow' or 'pytorch'")

PROJECT_SETTINGS = get_project_settings(PROJECT_TYPE)

SAVED_MODELS_PATH = join(REPO_ROOT, "saved_models")
LOGS_PATH = join(REPO_ROOT, "logs")
