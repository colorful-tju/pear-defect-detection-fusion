#!/usr/bin/env python3
"""
E1 Unified Training Pipeline

Complete training pipeline for E1 experiment:
1. Generate priors using Project A (if not exists)
2. Build ROI training dataset
3. Train local detector

Usage:
    python tools/train_e1_pipeline.py --config configs/e1_config.yaml
    python tools/train_e1_pipeline.py --config configs/e1_config.yaml --skip-priors
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion.roi_proposal.roi_dataset_builder import ROIDatasetBuilder


def priors_exist(priors_root: str, splits: list = ['train', 'val']) -> bool:
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


def generate_priors_with_project_a(config: dict):
    """
    Generate priors using Project A (Image Segmentation).

    Uses subprocess with conda environment to call pear-topo CLI.
    """
    print("\n" + "="*60)
    print("Step 1: Generating Priors with Project A")
    print("="*60)

    project_a_root = config['project_a_root']
    project_a_config = config.get('project_a_config', 'pear_topology.yaml')
    priors_root = config['priors']['root_dir']
    data_yaml = config['dataset']['data_yaml']

    # Build command
    cmd = [
        'conda', 'run', '-n', 'pear-topo',
        'pear-topo', 'infer-dataset',
        '--config', f"{project_a_root}/configs/{project_a_config}",
        '--data-yaml', data_yaml,
        '--seg-ckpt', f"{project_a_root}/outputs/checkpoints/unet_best.pt",
        '--uq-ckpt', f"{project_a_root}/outputs/checkpoints/uq_best.pt",
        '--splits', 'train', 'val', 'test',
        '--out', priors_root
    ]

    print(f"\nRunning command:")
    print(" ".join(cmd))

    # Execute
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\nError output:")
        print(result.stderr)
        raise RuntimeError(f"Project A inference failed with code {result.returncode}")

    print("\nPriors generated successfully!")
    print(f"Output directory: {priors_root}")


def build_roi_dataset(config: dict):
    """Build ROI training dataset."""
    print("\n" + "="*60)
    print("Step 2: Building ROI Training Dataset")
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
    print("Step 3: Training Local Detector")
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
        amp=True,
        cache=True,
        workers=8
    )

    print("\nLocal detector training completed!")
    print(f"Best checkpoint: {training_config['output_dir']}/local_detector/weights/best.pt")

    return results


def main():
    parser = argparse.ArgumentParser(description='E1 unified training pipeline')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to E1 configuration file'
    )
    parser.add_argument(
        '--skip-priors',
        action='store_true',
        help='Skip priors generation (use existing priors)'
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
    print("E1 Unified Training Pipeline")
    print("="*60)
    print(f"\nLoading configuration from: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nExperiment: {config.get('experiment_name', 'e1_global_local')}")
    print(f"Dataset: {config['dataset']['root']}")
    print(f"Priors: {config['priors']['root_dir']}")
    print(f"ROI dataset: {config['roi_dataset']['output_dir']}")
    print(f"Output: {config['training']['output_dir']}")

    # Step 1: Generate priors
    if not args.skip_priors:
        if priors_exist(config['priors']['root_dir']):
            print("\nPriors already exist, skipping generation.")
            print("Use --skip-priors to skip this check.")
        else:
            generate_priors_with_project_a(config)
    else:
        print("\nSkipping priors generation (--skip-priors)")

    # Step 2: Build ROI dataset
    if not args.skip_dataset:
        build_roi_dataset(config)
    else:
        print("\nSkipping ROI dataset construction (--skip-dataset)")

    # Step 3: Train local detector
    if not args.skip_training:
        train_local_detector(config)
    else:
        print("\nSkipping local detector training (--skip-training)")

    print("\n" + "="*60)
    print("E1 Training Pipeline Completed Successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run inference: python scripts/infer_e1_fusion.py --config configs/e1_config.yaml")
    print("2. Evaluate results: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
