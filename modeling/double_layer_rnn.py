from os.path import join
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

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


DOUBLE_LAYER_RNN_PATH = join(SAVED_MODELS_PATH, "double_layer_rnn")

def double_layer_rnn_model(hyperparameters):
    num_labels = len(EmotionTag)
    embedding_layer, num_tokens, embedding_dim = get_embedding_layer()

    bidirectional_lstm = Bidirectional(LSTM(embedding_dim, return_sequences=True))
    second_bidirectional_lstm = Bidirectional(LSTM(embedding_dim))
    fully_connected_layer = Dense(units=embedding_dim, activation="relu")
    classification_layer = Dense(
        units=num_labels,
        # activation="sigmoid",
        activation="linear",
    )

    # Reference for how dropout is being selected for this model.
    # https://becominghuman.ai/learning-note-dropout-in-recurrent-networks-part-1-57a9c19a2307
    #   Gal and Ghahramani [6] also propose a new way to regularize word embedding, in addition
    #   to apply dropout on inputs. They suggest dropout be used on word type, instead of individual words.
    #   That is, randomly setting rows of the embedding matrix to zero.
    model = Sequential([
        embedding_layer,
        Dropout(0.05),
        bidirectional_lstm,
        second_bidirectional_lstm,
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
        metrics=[
            rounded_equal,
            "accuracy",
        ],
    )

    return model

def train_double_layer_rnn(use_previous=False):
    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load the dataset
    doc_manager = DocumentManager()
    doc_manager.cache_documents()

    dataset_manager = DatasetManager(tokenizer)
    dataset = dataset_manager.get_dataset_from_documents(doc_manager)

    # Hyperparameters
    hyperparameters = {
        "dropout_rate": 0.1,
        "learning_rate": 0.005,
        "training_epochs": 25,
        "batch_size": 64,
    }

    if use_previous:
        model = load_model(DOUBLE_LAYER_RNN_PATH, custom_objects={"rounded_equal": rounded_equal})
    else:
        model = double_layer_rnn_model(hyperparameters)

    callbacks = [
        get_model_checkpoint(output_filepath=DOUBLE_LAYER_RNN_PATH),
    ]

    train_model(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        callbacks=callbacks,
    )

    return model


if __name__ == "__main__":
    use_previous = False
    trained_model = train_double_layer_rnn(use_previous=use_previous)
