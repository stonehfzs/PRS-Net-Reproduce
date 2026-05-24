#!/bin/bash

# ============================================================
# PRS-Net Batch Visualization Script
# ============================================================

# Configuration
# Path to the processed dataset root (containing categorized folders)
PROC_DATA_DIR="ModelNet10_processed"
# Path where the output HTML files will be stored
OUTPUT_DIR="visual_results"
# Path to the trained model weights
MODEL_PATH="prsnet_model.pth"

# 1. Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "PRS-Net Batch Visualization"
echo "Searching for test files in: $PROC_DATA_DIR"
echo "============================================================"

# 2. Find categories and process first 10 test files for each
echo "Searching for test files (max 10 per category)..."

# Get all unique category directories that contain a test folder
CATEGORIES=$(find "$PROC_DATA_DIR" -type d -path "*/test" | sed "s|/test$||" | sort -u)

for cat_dir in $CATEGORIES; do
    category=$(basename "$cat_dir")
    echo "------------------------------------------------------------"
    echo "Category: $category"
    
    # Get first 10 .pt files in the test directory
    CAT_TEST_FILES=$(find "$cat_dir/test" -maxdepth 1 -name "*.pt" | sort | head -n 10)
    
    if [ -z "$CAT_TEST_FILES" ]; then
        echo "No .pt files found in $category/test"
        continue
    fi
    
    for data_file in $CAT_TEST_FILES; do
        filename=$(basename "$data_file" .pt)
        out_path="$OUTPUT_DIR/${category}_${filename}.html"
        
        echo "Processing: $filename"
        # Run visualization (outputs to visual_result.html by default)
        python visualize.py --data "$data_file" --model "$MODEL_PATH"
        # Move the result to our target directory with unique name
        mv visual_result.html "$out_path"
    done
done

echo "============================================================"
echo "Batch visualization completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "============================================================"
