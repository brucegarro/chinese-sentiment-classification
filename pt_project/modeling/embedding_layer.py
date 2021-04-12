import torch
import torch.nn as nn

from settings.settings import PTProjectSettings
from modeling.utils import get_embedding_matrix


def get_embedding_layer():
    embedding_matrix = get_embedding_matrix(path=PTProjectSettings.EMBEDDING_MATRIX_PATH)
    num_embeddings, embedding_dim = embedding_matrix.shape
    embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
    
    # Load pre-loaded Tencent embeddings
    # embedding_layer.load_state_dict({'weight': embedding_matrix})
    embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))

    # Disable embedding retraining
    embedding_layer.weight.requires_grad = False

    return embedding_layer, num_embeddings, embedding_dim
