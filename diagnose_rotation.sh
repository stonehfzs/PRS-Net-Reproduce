#!/bin/bash
set -euo pipefail

PROC_DATA_DIR="${1:-ModelNet40_processed}"
MODEL_PATH="${2:-prsnet_model.pth}"
OUTPUT_DIR="${3:-rotation_diagnostics}"
MAX_FILES="${4:-20}"

CATEGORIES=(
    vase
    bottle
    cup
    lamp
    flower_pot
    plant
    stool
)

mkdir -p "$OUTPUT_DIR"

for CATEGORY in "${CATEGORIES[@]}"; do
    TEST_DIR="$PROC_DATA_DIR/$CATEGORY/test"
    CATEGORY_OUTPUT_DIR="$OUTPUT_DIR/$CATEGORY"

    if [ ! -d "$TEST_DIR" ]; then
        echo "Skip missing category: $CATEGORY"
        continue
    fi

    mkdir -p "$CATEGORY_OUTPUT_DIR"

    echo "Category: $CATEGORY"
    echo "Test directory: $TEST_DIR"
    echo "Output directory: $CATEGORY_OUTPUT_DIR"

    find "$TEST_DIR" -maxdepth 1 -type f -name "*.pt" | sort | head -n "$MAX_FILES" | while read -r data_file; do
        filename="$(basename "$data_file" .pt)"
        output_html="$CATEGORY_OUTPUT_DIR/${filename}.html"

        echo "  Visualizing: $filename"
        python visualize.py --data "$data_file" --model "$MODEL_PATH" --output "$output_html"
    done
done

echo "Done. Results saved to: $OUTPUT_DIR"
