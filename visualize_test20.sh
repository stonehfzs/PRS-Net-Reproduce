#!/bin/bash
set -euo pipefail

PROC_DATA_DIR="${1:-ModelNet40_processed}"
MODEL_PATH="${2:-prsnet_model.pth}"
OUTPUT_DIR="${3:-visual_results_test20}"
MAX_PER_CATEGORY=20

mkdir -p "$OUTPUT_DIR"

find "$PROC_DATA_DIR" -type d -path "*/test" | sort | while read -r test_dir; do
    category_dir="$(dirname "$test_dir")"
    category="$(basename "$category_dir")"
    category_output_dir="$OUTPUT_DIR/$category"

    mkdir -p "$category_output_dir"

    echo "Category: $category"

    find "$test_dir" -maxdepth 1 -type f -name "*.pt" | sort | head -n "$MAX_PER_CATEGORY" | while read -r data_file; do
        filename="$(basename "$data_file" .pt)"
        output_html="$category_output_dir/${filename}.html"

        echo "  Visualizing: $filename"
        python visualize.py --data "$data_file" --model "$MODEL_PATH" --output "$output_html"
    done
done

echo "Done. Results saved to: $OUTPUT_DIR"
