from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from preprocessing.document_manager import DocumentManager
from preprocessing.dataset_manager import DatasetManager
from preprocessing.enums import EmotionTag
from modeling.utils import get_tokenizer, get_embedding_layer
from modeling.train import train_model


def simple_rnn_model(hyperparameters):
    num_labels = len(EmotionTag)
    embedding_layer, num_tokens, embedding_dim = get_embedding_layer()

    bidirectional_lstm = Bidirectional(LSTM(embedding_dim))
    # bidirectional_lstm = Bidirectional(LSTM(embedding_dim, return_sequences=True))
    # second_bidirectional_lstm = Bidirectional(LSTM(100))
    fully_connected_layer = Dense(units=embedding_dim, activation="relu")
    classification_layer = Dense(
        units=num_labels,
        activation="sigmoid",
    )

    model = Sequential([
        embedding_layer,
        bidirectional_lstm,
        # second_bidirectional_lstm,
        fully_connected_layer,
        Dropout(hyperparameters["dropout_rate"]),
        classification_layer,
    ])

    # Fit Model
    optimizer = Adam(hyperparameters["learning_rate"])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    return model

def train_simple_rnn():
    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load the dataset
    doc_manager = DocumentManager()
    doc_manager.cache_documents()

    dataset_manager = DatasetManager(tokenizer)
    dataset = dataset_manager.get_dataset_from_documents(doc_manager)

    # Hyperparameters
    hyperparameters = {
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "training_epochs": 25,
        "batch_size": 64,
    }

    model = simple_rnn_model(hyperparameters)

    train_model(model, dataset, hyperparameters)

    return model


if __name__ == "__main__":
    trained_model = train_simple_rnn()