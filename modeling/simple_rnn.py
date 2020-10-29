import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import SGD

from preprocessing.word_embeddings import load_pkl
from preprocessing.document_manager import DocumentManager
from preprocessing.dataset_manager import DatasetManager
from settings.settings import EMBEDDING_MATRIX_PATH, KERAS_TOKENIZER_PATH

def get_gpu_configurations():
    GLOBAL_GPU_USAGE_LIMIT = 0.75
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage_limit)
    return gpu_options

if __name__ == "__main__":
    # Load Embeddings from pre-compiled embeddings matrix
    embedding_matrix = load_pkl(EMBEDDING_MATRIX_PATH)
    tokenizer = load_pkl(KERAS_TOKENIZER_PATH)

    # Load the dataset
    doc_manager = DocumentManager()
    doc_manager.cache_documents()

    dataset_manager = DatasetManager(tokenizer)
    dataset = dataset_manager.get_dataset_from_documents(doc_manager)

    train_sequences = dataset["train_sequences"]
    train_labels = dataset["train_labels"]
    valid_sequences = dataset["valid_sequences"]
    valid_labels = dataset["valid_labels"]

    # Create Model
    num_tokens = len(tokenizer.word_index) + 1 # ~50,000
    embedding_dim = embedding_matrix[0].size # 200
    num_labels = train_labels.shape[1] # 8

    # Hyperparameters
    dropout_rate = 0.0
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 50

    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Instantiate Model Layers
    embedding_layer = Embedding(
        input_dim=num_tokens,
        output_dim=embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=False,
    )

    bidirectional_lstm = Bidirectional(LSTM(embedding_dim))
    fully_connected_layer = Dense(units=embedding_dim) # TODO: use ReLu as activation
    classification_layer = Dense(
        units=num_labels,
        activation="softmax",
    )

    model = Sequential([
        embedding_layer,
        bidirectional_lstm,
        fully_connected_layer,
        classification_layer,
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    # Fit Model Layers
    # TODO: Limit gpu usage
    # gpu_options = get_gpu_configurations()

    model.fit(
        train_sequences,
        train_labels,
        validation_data=(valid_sequences, valid_labels),
        batch_size=batch_size,
        epochs=training_epochs,
    )
