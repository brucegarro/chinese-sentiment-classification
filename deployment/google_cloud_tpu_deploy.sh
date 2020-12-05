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

# SSH into the instance
gcloud compute ssh chinese-sentiment-classification --zone=us-central1-b

 # Setup Environment variables after SSH
export TPU_NAME="chinese-sentiment-classification"
export CSC_BUCKET_NAME="chinese-sentiment-clf-bucket"
export BUCKET_ADDRESS="gs://$CSC_BUCKET_NAME"

mkdir ~/repos

# TODO: Download the whole bucket to ~/buckets, then down REN_CEC_PATH and chinese-sentiment-classification-data

# Clone project git repository
(cd ~/repos && git clone https://github.com/brucegarro/chinese-sentiment-classification.git)

 # Install Python libraries
pip3 install -r ~/repos/chinese-sentiment-classification/requirements.txt

# Setup some environment variables used in the project
export PYTHONPATH="${PYTHONPATH}:$HOME/repos/chinese-sentiment-classification"
export REN_CEC_PATH="$BUCKET_ADDRESS/repos/Ren_CECps-Dictionary/"
export EMBEDDING_DATA_ROOT="$BUCKET_ADDRESS/chinese-sentiment-classification-data/embedding/"
export REPO_PATH="$HOME/repos/chinese-sentiment-classification/"


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
