#!/bin/bash
set -e

# 1. Install Dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 2. Process Dataset
echo "Processing dataset..."
python process_dataset.py --base_dir . --zip_file test-coco-upload.zip

# 3. Train Model
echo "Starting training..."
# Run for a few epochs to verify, user can increase later
python train_model.py --data_dir alopecia_dataset/augmented_dataset --output_dir models --epochs 5

echo "Pipeline completed successfully!"
