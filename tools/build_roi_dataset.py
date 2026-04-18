#!/usr/bin/env python3
"""
Build ROI Dataset Script

Constructs YOLO-compatible ROI patch dataset for training local detector.

Usage:
    python tools/build_roi_dataset.py --config configs/e1_config.yaml
    python tools/build_roi_dataset.py --config configs/e1_config.yaml --splits train val
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion.roi_proposal.roi_dataset_builder import ROIDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description='Build ROI dataset for local detector training')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val'],
        help='Dataset splits to process (default: train val)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override output directory if specified
    if args.output_dir:
        config['roi_dataset']['output_dir'] = args.output_dir

    # Initialize dataset builder
    print("\nInitializing ROI dataset builder...")
    builder = ROIDatasetBuilder(config)

    # Build dataset
    print(f"\nBuilding ROI dataset for splits: {args.splits}")
    builder.build_dataset(
        original_dataset_root=config['dataset']['root'],
        priors_root=config['priors']['root_dir'],
        output_dir=config['roi_dataset']['output_dir'],
        splits=args.splits
    )

    print("\n" + "="*60)
    print("ROI dataset construction completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
