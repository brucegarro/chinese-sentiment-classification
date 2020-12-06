#!/bin/bash

# Set this project ID manually
export PROJECT_ID="chinese-sentiment-clf"
export TPU_NAME="chinese-sentiment-classification"
export CSC_BUCKET_NAME="chinese-sentiment-clf-bucket"
export BUCKET_ADDRESS="gs://$CSC_BUCKET_NAME"

gcloud config set project ${PROJECT_ID}

# Create Service Account for Cloud TPU (if necessary)
gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID

# Create a bucket for this project (if neccessary)
gsutil mb -p $PROJECT_ID -c standard -l us-central -b on $BUCKET_ADDRESS

# Create a Cloud TPU instances in Google Cloud
ctpu up --project=${PROJECT_ID} \
 --zone=us-central1-b \
 --tf-version=2.3.1 \
 --name=chinese-sentiment-classification

# SSH into the instance (if necessary)
gcloud compute ssh chinese-sentiment-classification --zone=us-central1-b

 # Setup Environment variables after SSH
export TPU_NAME="chinese-sentiment-classification"
export CSC_BUCKET_NAME="chinese-sentiment-clf-bucket"
export BUCKET_ADDRESS="gs://$CSC_BUCKET_NAME"

mkdir ~/repos

# Download data from Google Cloud Storage buckets and then symlink them to expected filesystem locations
mkdir ~/buckets
(gsutil -m cp -R $BUCKET_ADDRESS ~/buckets)

ln -s ~/buckets/$CSC_BUCKET_NAME/repos/Ren_CECps-Dictionary/ ~/repos/Ren_CECps-Dictionary
ln -s ~/buckets/$CSC_BUCKET_NAME/chinese-sentiment-classification-data/ ~/repos/chinese-sentiment-classification-data

# Clone project git repository
(cd ~/repos && git clone https://github.com/brucegarro/chinese-sentiment-classification.git)

 # Install Python libraries
pip3 install -r ~/repos/chinese-sentiment-classification/requirements.txt

# Setup some environment variables used in the project
export PYTHONPATH="${PYTHONPATH}:$HOME/repos/chinese-sentiment-classification"
export REPO_PATH="$HOME/repos/"

# Run a test model
cd ~/repos/chinese-sentiment-classification/
python3 modeling/simple_rnn_tpu.py False False

# How to Cleanup
exit

# Delete the TPU instance
ctpu delete --project=${PROJECT_ID} \
  --name=$TPU_NAME \
  --zone=us-central1-b

# Check what TPUs are running
ctpu status --project=${PROJECT_ID}} \
  --name=$TPU_NAME \
  --zone=us-central1-b
