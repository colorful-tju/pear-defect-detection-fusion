#!/bin/bash

# ============================================================================
# Prepare Priors Script
#
# This script generates likelihood.npy and topology_mask.npy for the entire
# dataset using Project A (Image Segmentation).
#
# Usage:
#   bash scripts/prepare_priors.sh [--splits train,val,test] [--limit N]
# ============================================================================

set -e  # Exit on error

# Default values
SPLITS="train,val,test"
LIMIT=""
OVERWRITE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --splits)
            SPLITS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Project paths
PROJECT_A_ROOT="/Users/renxd/code/Image Segmentation"
FUSION_ROOT="/Users/renxd/code/pear-defect-detection-fusion"

# Dataset configuration
DATA_YAML="/home/robot/yolo/datasets/PearSurfaceDefects/li_data.yaml"

# Model checkpoints (from Project A)
SEG_CKPT="${PROJECT_A_ROOT}/outputs/checkpoints/unet_best.pt"
UQ_CKPT="${PROJECT_A_ROOT}/outputs/checkpoints/uq_best.pt"

# Config file
CONFIG="${PROJECT_A_ROOT}/configs/pear_topology.yaml"  # Use auto device selection

# Output directory
OUTPUT_DIR="${FUSION_ROOT}/outputs/priors"

# Check if Project A exists
if [ ! -d "$PROJECT_A_ROOT" ]; then
    echo "Error: Project A not found at $PROJECT_A_ROOT"
    exit 1
fi

# Check if checkpoints exist
if [ ! -f "$SEG_CKPT" ]; then
    echo "Error: Segmentation checkpoint not found at $SEG_CKPT"
    echo "Please train the segmentation model first using Project A."
    exit 1
fi

if [ ! -f "$UQ_CKPT" ]; then
    echo "Error: UQ checkpoint not found at $UQ_CKPT"
    echo "Please train the uncertainty model first using Project A."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================================"
echo "Preparing Priors for Pear Defect Detection Fusion"
echo "============================================================================"
echo "Project A Root: $PROJECT_A_ROOT"
echo "Fusion Root: $FUSION_ROOT"
echo "Data YAML: $DATA_YAML"
echo "Splits: $SPLITS"
echo "Output Directory: $OUTPUT_DIR"
echo "============================================================================"

# Change to Project A directory
cd "$PROJECT_A_ROOT"

# Activate conda environment (if needed)
# conda activate pear-topo

# Run inference
echo "Running pear-topo infer-dataset..."
pear-topo infer-dataset \
    --config "$CONFIG" \
    --data-yaml "$DATA_YAML" \
    --seg-ckpt "$SEG_CKPT" \
    --uq-ckpt "$UQ_CKPT" \
    --splits $(echo $SPLITS | tr ',' ' ') \
    --out "$OUTPUT_DIR" \
    $LIMIT \
    $OVERWRITE

echo "============================================================================"
echo "Priors generation completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================================================"

# Verify output structure
echo "Verifying output structure..."
for split in $(echo $SPLITS | tr ',' ' '); do
    MANIFEST="${OUTPUT_DIR}/manifests/${split}.json"
    if [ -f "$MANIFEST" ]; then
        NUM_IMAGES=$(jq '. | length' "$MANIFEST")
        echo "  - $split: $NUM_IMAGES images"
    else
        echo "  - $split: manifest not found (may be empty)"
    fi
done

echo "============================================================================"
echo "Done! You can now run fusion inference."
echo "============================================================================"
