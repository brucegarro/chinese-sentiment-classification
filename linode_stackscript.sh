#!/bin/bash
# Modification of Sentdex Linode deployment scripts
# https://pythonprogramming.net/cloud-gpu-compare-and-setup-linode-rtx-6000/

# Update packages
apt-get update


# Show Nvidia hardware available
lspci | grep -i nvidia

# Non-interactive mode, use default answers
export DEBIAN_FRONTEND=noninteractive

# Workaround for libc6 bug - asking about service restart in non-interactive mode
# https://bugs.launchpad.net/ubuntu/+source/eglibc/+bug/935681
echo 'libc6 libraries/restart-without-asking boolean true' | debconf-set-selections

# Install Pip & Virtualenv
curl https://bootstrap.pypa.io/get-pip.py | sudo python3.6
apt-get -y install virtualenv

# Add Nvidia repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
apt-get update

# Install drivers, CUDA and cuDNN
apt-get -y install --no-install-recommends nvidia-driver-440
apt-get -y install --no-install-recommends cuda-10-0 libcudnn7=\*+cuda10.0 libcudnn7-dev=\*+cuda10.0
apt-get -y install --no-install-recommends cuda-10-1 libcudnn7=\*+cuda10.1 libcudnn7-dev=\*+cuda10.1
apt-get -y install --no-install-recommends libnvinfer5=5.\*+cuda10.0 libnvinfer-dev=5.\*+cuda10.0
apt-get -y install --no-install-recommends libnvinfer5=5.\*+cuda10.1 libnvinfer-dev=5.\*+cuda10.1

# Clone project git repository
mkdir ~/repos
(cd ~/repos && git clone https://github.com/brucegarro/chinese-sentiment-classification.git)

# Create and activate a virtualenv
mkdir ~/.virtualenvs
(cd ~/.virtualenvs && virtualenv -p python3 csc)
source ~/.virtualenvs/csc/bin/activate

# Install Python libraries
pip install -r ~/repos/chinese-sentiment-classification/requirements.txt

# Setup some environment variables used in the project
echo 'export PYTHONPATH="${PYTHONPATH}:/root/repos/chinese-sentiment-classification"' >> ~/.bashrc
echo 'export REPO_PATH="/root/repos"' >> ~/.bashrc

reboot
