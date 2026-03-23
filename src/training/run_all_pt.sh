#!/bin/bash

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting execution at $(date)"

echo "Running script1..."

python pre-trained-1.py > pre-run-1-log-$TIMESTAMP.txt 2>&1

echo "Running script2..."
python pre-trained-2.py > pre-run-2-log-$TIMESTAMP.txt 2>&1

echo "Running script3..."
python pre-trained-3.py > pre-run-3-log-$TIMESTAMP.txt 2>&1

echo "Running script4..."
python pre-trained-4.py > pre-run-4-log-$TIMESTAMP.txt 2>&1

echo "Running script5..."
python pre-trained-5.py > pre-run-5-log-$TIMESTAMP.txt 2>&1

echo "Running script6..."
python pre-trained-6.py > pre-run-6-log-$TIMESTAMP.txt 2>&1

echo "Running script7..."
python pre-trained-7.py > pre-run-7-log-$TIMESTAMP.txt 2>&1

echo "Running script8..."
python pre-trained-8.py > pre-run-8-log-$TIMESTAMP.txt 2>&1

echo "All scripts finished at $(date)"
