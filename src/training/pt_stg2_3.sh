#!/bin/bash

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting execution at $(date)"

echo "Running script1..."

python pre-trained-stg2-3.py > pt-stg2-3-log-$TIMESTAMP.txt 2>&1


echo "All scripts finished at $(date)"
