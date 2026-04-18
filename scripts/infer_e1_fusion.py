#!/usr/bin/env python3
"""
E1 Fusion Inference Script

Runs E1 inference pipeline: global detector + local detector + fusion.

Usage:
    python scripts/infer_e1_fusion.py --config configs/e1_config.yaml --source test_images/
    python scripts/infer_e1_fusion.py --config configs/e1_config.yaml --source image.jpg --visualize
"""

import argparse
import yaml
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion.roi_proposal.roi_infer import ROIInferencer
from fusion.roi_proposal.roi_fusion import FusionEngine


def load_models(config: dict):
    """Load global and local YOLO models."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not found. Please install: pip install ultralytics")

    print("Loading models...")
    global_model = YOLO(config['models']['global_detector'])
    local_model = YOLO(config['models']['local_detector'])

    print(f"  - Global detector: {config['models']['global_detector']}")
    print(f"  - Local detector: {config['models']['local_detector']}")

    return global_model, local_model


def run_global_detection(model, image_path: str, config: dict):
    """Run global detector on full image."""
    inference_config = config['inference']['global']

    results = model.predict(
        source=image_path,
        conf=inference_config['conf'],
        iou=inference_config['iou'],
        verbose=False
    )

    # Extract detections
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'conf': float(conf),
                'cls': int(cls),
                'source': 'global'
            })

    return detections


def visualize_detections(image_path: str, detections: list, output_path: str, class_names: dict = None):
    """Visualize detections on image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Cannot load image {image_path}")
        return

    # Color map for sources
    colors = {
        'global': (0, 255, 0),    # Green
        'local': (255, 0, 0),     # Blue
        'merged': (0, 255, 255)   # Yellow
    }

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        source = det.get('source', 'merged')
        color = colors.get(source, (255, 255, 255))

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        cls = det['cls']
        conf = det['conf']
        if class_names and cls in class_names:
            label = f"{class_names[cls]} {conf:.2f}"
        else:
            label = f"cls{cls} {conf:.2f}"

        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # Save
    cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser(description='E1 fusion inference')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to E1 configuration file'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Image file or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/e1_detections',
        help='Output directory'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save visualization images'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split (for loading priors)'
    )

    args = parser.parse_args()

    # Load configuration
    print("="*60)
    print("E1 Fusion Inference")
    print("="*60)
    print(f"\nLoading configuration from: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

    # Load models
    global_model, local_model = load_models(config)

    # Initialize components
    print("\nInitializing components...")
    roi_inferencer = ROIInferencer(local_model, config)
    fusion_engine = FusionEngine(config)

    # Get image list
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    else:
        raise ValueError(f"Invalid source: {args.source}")

    print(f"\nProcessing {len(image_paths)} images...")

    # Load class names
    data_yaml_path = config['dataset']['data_yaml']
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', {})

    # Process each image
    all_results = {}

    for image_path in tqdm(image_paths, desc="Running E1 inference"):
        image_name = image_path.stem

        # 1. Global detection
        global_dets = run_global_detection(global_model, str(image_path), config)

        # 2. Local detection (ROI-based)
        local_dets = roi_inferencer.infer_single_image(
            str(image_path),
            split=args.split,
            verbose=False
        )

        # 3. Fusion
        merged_dets = fusion_engine.merge(global_dets, local_dets)

        # Save results
        all_results[str(image_path)] = {
            'global_detections': global_dets,
            'local_detections': local_dets,
            'merged_detections': merged_dets,
            'num_global': len(global_dets),
            'num_local': len(local_dets),
            'num_merged': len(merged_dets)
        }

        # Visualize if requested
        if args.visualize:
            vis_path = vis_dir / f"{image_name}_e1.jpg"
            visualize_detections(str(image_path), merged_dets, str(vis_path), class_names)

    # Save results to JSON
    results_path = output_dir / 'e1_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("E1 Inference Completed!")
    print("="*60)

    total_global = sum(r['num_global'] for r in all_results.values())
    total_local = sum(r['num_local'] for r in all_results.values())
    total_merged = sum(r['num_merged'] for r in all_results.values())

    print(f"\nProcessed {len(image_paths)} images")
    print(f"Total detections:")
    print(f"  - Global: {total_global}")
    print(f"  - Local: {total_local}")
    print(f"  - Merged: {total_merged}")
    print(f"\nResults saved to: {results_path}")

    if args.visualize:
        print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
