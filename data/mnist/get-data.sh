#!/bin/bash

# Script to download MNIST dataset files from alternate mirror
# Original source: http://yann.lecun.com/exdb/mnist/

# Create directory if it doesn't exist
mkdir -p "$(dirname "$0")"
cd "$(dirname "$0")" || exit 1

BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

# Download training data
wget -N "${BASE_URL}/train-images-idx3-ubyte.gz"
wget -N "${BASE_URL}/train-labels-idx1-ubyte.gz"

# Download test data
wget -N "${BASE_URL}/t10k-images-idx3-ubyte.gz"
wget -N "${BASE_URL}/t10k-labels-idx1-ubyte.gz"

# Unpack all files
gunzip -f train-images-idx3-ubyte.gz
gunzip -f train-labels-idx1-ubyte.gz
gunzip -f t10k-images-idx3-ubyte.gz
gunzip -f t10k-labels-idx1-ubyte.gz

echo "MNIST data downloaded and extracted successfully"