#!/usr/bin/env python3
"""
E1 Standalone Training Pipeline

This is a standalone version that doesn't require external projects.
It assumes you have already prepared:
1. Priors data (likelihood.npy, topology_mask.npy) in outputs/priors/
2. Pre-trained global detector model
3. YOLO dataset

Usage:
    python tools/train_e1_standalone.py --config configs/e1_config_standalone.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion.roi_proposal.roi_dataset_builder import ROIDatasetBuilder


def check_priors_exist(priors_root: str, splits: list = ['train', 'val']) -> bool:
    """Check if priors have been generated."""
    priors_root = Path(priors_root)

    if not priors_root.exists():
        return False

    # Check manifests
    for split in splits:
        manifest_path = priors_root / 'manifests' / f'{split}.json'
        if not manifest_path.exists():
            return False

    return True


def build_roi_dataset(config: dict):
    """Build ROI training dataset."""
    print("\n" + "="*60)
    print("Building ROI Training Dataset")
    print("="*60)

    builder = ROIDatasetBuilder(config)

    builder.build_dataset(
        original_dataset_root=config['dataset']['root'],
        priors_root=config['priors']['root_dir'],
        output_dir=config['roi_dataset']['output_dir'],
        splits=['train', 'val']
    )


def train_local_detector(config: dict):
    """Train local detector on ROI patches."""
    print("\n" + "="*60)
    print("Training Local Detector")
    print("="*60)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not found. Please install: pip install ultralytics"
        )

    # Load pretrained model
    pretrained = config['training'].get('pretrained', 'yolo26s.pt')
    print(f"\nLoading pretrained model: {pretrained}")
    model = YOLO(pretrained)

    # Training parameters
    training_config = config['training']
    roi_dataset_dir = config['roi_dataset']['output_dir']
    data_yaml = Path(roi_dataset_dir) / 'data.yaml'

    if not data_yaml.exists():
        raise FileNotFoundError(f"ROI dataset config not found: {data_yaml}")

    print(f"\nTraining configuration:")
    print(f"  - Dataset: {data_yaml}")
    print(f"  - Epochs: {training_config['epochs']}")
    print(f"  - Image size: {training_config['imgsz']}")
    print(f"  - Batch size: {training_config['batch']}")
    print(f"  - Device: {training_config['device']}")

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=training_config['epochs'],
        imgsz=training_config['imgsz'],
        batch=training_config['batch'],
        device=training_config['device'],
        project=training_config['output_dir'],
        name='local_detector',
        amp=training_config.get('amp', True),
        cache=training_config.get('cache', True),
        workers=training_config.get('workers', 8)
    )

    print("\nLocal detector training completed!")
    print(f"Best checkpoint: {training_config['output_dir']}/local_detector/weights/best.pt")

    return results


def main():
    parser = argparse.ArgumentParser(description='E1 standalone training pipeline')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to E1 configuration file'
    )
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Skip ROI dataset construction (use existing dataset)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip local detector training'
    )

    args = parser.parse_args()

    # Load configuration
    print("="*60)
    print("E1 Standalone Training Pipeline")
    print("="*60)
    print(f"\nLoading configuration from: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nExperiment: {config.get('experiment_name', 'e1_global_local')}")
    print(f"Dataset: {config['dataset']['root']}")
    print(f"Priors: {config['priors']['root_dir']}")
    print(f"ROI dataset: {config['roi_dataset']['output_dir']}")
    print(f"Output: {config['training']['output_dir']}")

    # Check priors exist
    if not check_priors_exist(config['priors']['root_dir']):
        print("\n" + "="*60)
        print("ERROR: Priors not found!")
        print("="*60)
        print("\nYou need to generate priors first.")
        print("\nOption 1: Use Project A to generate priors")
        print("  cd /path/to/Image_Segmentation")
        print("  conda activate pear-topo")
        print("  pear-topo infer-dataset \\")
        print("    --config configs/pear_topology_4090.yaml \\")
        print("    --data-yaml /path/to/data.yaml \\")
        print("    --seg-ckpt outputs/checkpoints/unet_best.pt \\")
        print("    --uq-ckpt outputs/checkpoints/uq_best.pt \\")
        print("    --splits train val test \\")
        print("    --out /path/to/fusion/outputs/priors")
        print("\nOption 2: Copy pre-generated priors")
        print("  cp -r /path/to/existing/priors outputs/")
        print("\nThen run this script again.")
        return 1

    print("\n✓ Priors found")

    # Build ROI dataset
    if not args.skip_dataset:
        build_roi_dataset(config)
    else:
        print("\nSkipping ROI dataset construction (--skip-dataset)")

    # Train local detector
    if not args.skip_training:
        train_local_detector(config)
    else:
        print("\nSkipping local detector training (--skip-training)")

    print("\n" + "="*60)
    print("E1 Training Pipeline Completed Successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run inference: python scripts/infer_e1_fusion.py --config configs/e1_config_standalone.yaml")
    print("2. Evaluate results: Compare with baseline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
