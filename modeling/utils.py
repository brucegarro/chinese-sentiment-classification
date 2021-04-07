from tensorflow.compat.v1 import GPUOptions
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant

from preprocessing.word_embeddings import load_pkl
from settings.settings import PROJECT_SETTINGS


def get_gpu_configurations():
    GLOBAL_GPU_USAGE_LIMIT = 0.75
    gpu_options = GPUOptions(per_process_gpu_memory_fraction=GLOBAL_GPU_USAGE_LIMIT)
    return gpu_options

def get_tokenizer():
    tokenizer = load_pkl(PROJECT_SETTINGS.TOKENIZER_PATH)
    return tokenizer

def get_embedding_layer():
    embedding_matrix = load_pkl(PROJECT_SETTINGS.EMBEDDING_MATRIX_PATH)

    num_tokens, embedding_dim = embedding_matrix.shape

    embedding_layer = Embedding(
        input_dim=num_tokens,
        output_dim=embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
    )
    return embedding_layer, num_tokens, embedding_dim
