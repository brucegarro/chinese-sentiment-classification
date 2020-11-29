from os.path import join
from fabric import task, Connection
from patchwork.transfers import rsync

from settings.settings import (
    REN_CEC_PATH,
    EMBEDDING_MATRIX_NAME,
    KERAS_TOKENIZER_NAME,
    EMBEDDING_MATRIX_PATH,
    KERAS_TOKENIZER_PATH,
)


def transfer_embedding_data(connection):
    # Add Embedding Data Folder
    EMBEDDING_REMOTE_PATH = join(connection.config.run.env["REPO_PATH"], "chinese-sentiment-classification-data/embedding")
    if connection.run("test -d %s" % EMBEDDING_REMOTE_PATH, warn=True).failed:
        connection.run("mkdir -p %s" % EMBEDDING_REMOTE_PATH)
        print("Created: %s\n" % EMBEDDING_REMOTE_PATH)

    # Transfer embedding matrix and tokenizer
    EMBEDDING_MATRIX_REMOTE_PATH = join(EMBEDDING_REMOTE_PATH, EMBEDDING_MATRIX_NAME)
    KERAS_TOKENIZER_REMOTE_PATH = join(EMBEDDING_REMOTE_PATH, KERAS_TOKENIZER_NAME)
    for local_filepath, remote_filepath in (
        (EMBEDDING_MATRIX_PATH, EMBEDDING_MATRIX_REMOTE_PATH),
        (KERAS_TOKENIZER_PATH, KERAS_TOKENIZER_REMOTE_PATH),
    ):
        if connection.run("test -f %s" % remote_filepath, warn=True).failed:
            rsync(connection, local_filepath, EMBEDDING_REMOTE_PATH)
            print("Created: %s\n" % remote_filepath)

def setup_directories(connection):
    # Create Repo Path
    if connection.run("test -d $REPO_PATH", warn=True).failed:
        connection.run("mkdir $REPO_PATH")
        print("Created: %s" % connection.config.run.env["REPO_PATH"])

    # Put Ren CEC Repo
    REN_CEC_REMOTE_PATH = join(connection.config.run.env["REPO_PATH"], "Ren_CECps-Dictionary")
    if connection.run("test -d %s" % REN_CEC_REMOTE_PATH, warn=True).failed:
        connection.run("mkdir -p %s" % REN_CEC_REMOTE_PATH)
        print("Created: %s\n" % REN_CEC_REMOTE_PATH)

        rsync(connection, REN_CEC_PATH, connection.config.run.env["REPO_PATH"])
        print("Put %s to %s\n" % (REN_CEC_PATH, REN_CEC_REMOTE_PATH))

    transfer_embedding_data(connection)


@task()
def setup_tf_server(connection):
    """
    Usage
    -----
    fab setup-tf-server -H user@host -i ~/.ssh/{key} --prompt-for-login-password
    """
    print("Connected to %s@%s" % (connection.user, connection.host))

    connection.inline_ssh_env = True
    connection.config.run.env["REPO_PATH"] = "~/repos"

    setup_directories(connection)
