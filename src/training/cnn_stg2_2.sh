#!/bin/bash

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting execution at $(date)"

echo "Running script1..."

python cnn_stg2_2.py > cnn-stg2-2-log-$TIMESTAMP.txt 2>&1


echo "All scripts finished at $(date)"
