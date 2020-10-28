from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.initializers import Constant

from preprocessing.word_embeddings import load_pkl
from preprocessing.document_manager import DocumentManager
from preprocessing.dataset_manager import DatasetManager
from settings.settings import EMBEDDING_MATRIX_PATH, KERAS_TOKENIZER_PATH


if __name__ == "__main__":
    # Create Embeddings from pre-compiled embeddings matrix
    embedding_matrix = load_pkl(EMBEDDING_MATRIX_PATH)
    tokenizer = load_pkl(KERAS_TOKENIZER_PATH)

    # Load the dataset
    dataset_manager = DatasetManager()
    train_x, train_y = dataset_manager.create_dataset_from_documents()

    # Create Model
    num_tokens = len(tokenizer.word_index)
    embedding_dim = embedding_matrix[0].size

    embedding_layer = Embedding(
        input_dim=num_tokens,
        output_dim=embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
    )

    model = Sequential([

    ])
    model.fit(train_X, train_Y, validation_data=(valid_X, valid_Y)