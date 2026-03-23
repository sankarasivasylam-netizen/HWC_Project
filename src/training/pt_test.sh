#!/bin/bash

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting execution at $(date)"

echo "Running script1..."

python pre-trained-1.py > pre-run-1-log-$TIMESTAMP.txt 2>&1


echo "All scripts finished at $(date)"
