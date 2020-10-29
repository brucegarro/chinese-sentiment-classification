from tensorflow.compat.v1 import Session, ConfigProto

from modeling.utils import get_gpu_configurations


def train_model(model, dataset, hyperparameters):
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
    )
