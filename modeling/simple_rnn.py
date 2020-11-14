from os.path import join
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import get_logger
get_logger().setLevel('ERROR')

from preprocessing.document_manager import DocumentManager
from preprocessing.dataset_manager import DatasetManager
from preprocessing.enums import EmotionTag
from modeling.utils import get_tokenizer, get_embedding_layer
from modeling.train import train_model, get_model_checkpoint
from modeling.metrics import (
    rounded_to_tenth_categorical_accuracy,
    rounded_equal,
    rounded_mean_absolute_error,
)
from settings.settings import SAVED_MODELS_PATH


SIMPLE_RNN_PATH = join(SAVED_MODELS_PATH, "simple_rnn")

def simple_rnn_model(hyperparameters):
    num_labels = len(EmotionTag)
    embedding_layer, num_tokens, embedding_dim = get_embedding_layer()

    bidirectional_lstm = Bidirectional(LSTM(embedding_dim))
    # bidirectional_lstm = Bidirectional(LSTM(embedding_dim, return_sequences=True))
    # second_bidirectional_lstm = Bidirectional(LSTM(100))
    fully_connected_layer = Dense(units=embedding_dim, activation="relu")
    classification_layer = Dense(
        units=num_labels,
        activation="linear",
        # activation="categorical_crossentropy",
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
        # loss="categorical_crossentropy",
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=([
            rounded_equal,
            "accuracy",
            # rounded_to_tenth_categorical_accuracy,
            # rounded_mean_absolute_error,
        ],),
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
        # "learning_rate": 0.001,
        "learning_rate": 0.005,
        # "training_epochs": 25,
        "training_epochs": 25,
        "batch_size": 64,
    }

    model = simple_rnn_model(hyperparameters)
    callbacks = [
        get_model_checkpoint(output_filepath=SIMPLE_RNN_PATH),
    ]

    train_model(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        callbacks=callbacks,
    )

    return model


if __name__ == "__main__":
    trained_model = train_simple_rnn()