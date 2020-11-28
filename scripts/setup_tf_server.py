from fabric import task, Connection


def setup_directories(connection):
    # Create Repo Path
    if connection.run("test -d $REPO_PATH", warn=True).failed:
        connection.run("mkdir $REPO_PATH")
        print("Created: %s" % connection.config.run.env["REPO_PATH"])


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
