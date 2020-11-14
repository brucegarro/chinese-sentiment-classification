from os.path import join
import tensorflow as tf
from tensorflow.compat.v1 import Session, ConfigProto
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy, mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint

from modeling.utils import get_gpu_configurations
from settings.settings import SAVED_MODELS_PATH


def get_model_checkpoint(output_filepath=None):
    if not output_filepath:
        output_filepath = join(SAVED_MODELS_PATH, "tmp_model")

    return ModelCheckpoint(
        filepath=output_filepath,
        save_freq="epoch",
        save_weights_only=False,
        save_best_only=True,
        monitor="val_rounded_equal",
        mode="max",
        verbose=1,
    )

def train_model(model, dataset, hyperparameters, callbacks=()):
    train_sequences = dataset["train_sequences"]
    train_labels = dataset["train_labels"]
    valid_sequences = dataset["valid_sequences"]
    valid_labels = dataset["valid_labels"]

    # Train Model

    # TODO: Limit GPU memory growth
    # with Session(config=ConfigProto(gpu_options=get_gpu_configurations())) as sess:

    model.fit(
        train_sequences,
        train_labels,
        validation_data=(valid_sequences, valid_labels),
        batch_size=hyperparameters["batch_size"],
        epochs=hyperparameters["training_epochs"],
        callbacks=callbacks,
        verbose=1,
    )
