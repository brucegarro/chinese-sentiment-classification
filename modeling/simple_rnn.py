import itertools
from os.path import join
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import get_logger
from tensorboard.plugins.hparams import api as hp
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
from settings.settings import SAVED_MODELS_PATH, LOGS_PATH


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

    input_dropout_layer = Dropout(hyperparameters["dropout"]["input_dropout"])
    lstm_output_dropout_layer = Dropout(hyperparameters["dropout"]["lstm_output_dropout"])
    fully_connected_dropout_layer = Dropout(hyperparameters["dropout"]["fully_connected_dropout"])

    model = Sequential([
        embedding_layer,
        input_dropout_layer,
        bidirectional_lstm,
        lstm_output_dropout_layer,
        fully_connected_layer,
        fully_connected_dropout_layer,
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


def train_simple_rnn(use_previous=True):
    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load the dataset
    doc_manager = DocumentManager()
    doc_manager.cache_documents()

    dataset_manager = DatasetManager(tokenizer)
    dataset = dataset_manager.get_dataset_from_documents(doc_manager)

    # Define hyperparameters to tune
    dropout = [
        hp.HParam("input_dropout", hp.Discrete([0.0, 0.05])),
        hp.HParam("recurrent_dropout", hp.Discrete([0.0, 0.1])),
        hp.HParam("lstm_output_dropout", hp.Discrete([0.0, 0.1])),
        hp.HParam("fully_connected_dropout", hp.Discrete([0.0, 0.1])),
    ]
    dropout_permutations = ([
        {
            dropout[i]: tup[i] for i in range(len(dropout))
        } for tup in list(itertools.product(*[ hparam.domain.values for hparam in dropout])) 
    ])
    metrics = [
        hp.Metric("rounded_equal", display_name="Rounded Equal"),
        hp.Metric("accuracy", display_name="Accuracy"),
    ]

    parameter_tuning_log_path = join(LOGS_PATH, "simple_rnn_dropout")

    for dropout_params in dropout_permutations:
        tag = "_".join([ str(param) for param in dropout_params.values() ])
        hyperparameters = {
            "dropout": { hparam.name: val for hparam, val in dropout_params.items() },
            # "learning_rate": 0.001,
            "learning_rate": 0.005,
            # "training_epochs": 1,
            "training_epochs": 10,
            "batch_size": 64,
        }

        save_path = "_".join([SIMPLE_RNN_PATH, tag])
        run_log_path = join(parameter_tuning_log_path, tag, "validation")

        with tf.summary.create_file_writer(run_log_path).as_default():
            hp.hparams_config(
                hparams=dropout,
                metrics=metrics,
            )

            if use_previous:
                try:
                    model = load_model(save_path, custom_objects={"rounded_equal": rounded_equal})
                except OSError:
                    model = simple_rnn_model(hyperparameters)
            else:
                model = simple_rnn_model(hyperparameters)
            callbacks = [
                get_model_checkpoint(output_filepath=save_path), # Save best model found
                tf.keras.callbacks.TensorBoard(log_dir=run_log_path),
                hp.KerasCallback(run_log_path, dropout_params), # Log parameter tuning results
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