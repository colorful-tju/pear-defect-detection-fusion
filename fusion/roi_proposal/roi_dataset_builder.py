"""
ROI Dataset Builder Module

Constructs YOLO-compatible training dataset from ROI patches.

Key features:
- Extracts ROI patches for all training images
- Assigns labels based on GT box overlap
- Maps GT boxes to patch coordinate system
- Handles hard negative sampling
- Outputs YOLO-format dataset structure
"""

import cv2
import numpy as np
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fusion.roi_proposal.roi_generator import ROIGenerator
from fusion.roi_proposal.roi_mapper import (
    original_to_patch,
    compute_iou,
    boxes_overlap,
    validate_coordinates
)


class ROIDatasetBuilder:
    """
    Builds YOLO-compatible ROI patch dataset for training local detector.
    """

    def __init__(self, config: Dict):
        """
        Initialize ROI dataset builder.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.roi_generator = ROIGenerator(config)

        # Label assignment parameters
        label_config = config.get('roi_dataset', {}).get('label_assignment', {})
        self.iou_threshold = label_config.get('iou_threshold', 0.3)
        self.overlap_ratio = label_config.get('overlap_ratio', 0.5)

        # Hard negative sampling parameters
        hard_neg_config = config.get('roi_dataset', {}).get('hard_negative', {})
        self.hard_negative_enabled = hard_neg_config.get('enabled', True)
        self.hard_negative_likelihood_threshold = hard_neg_config.get('likelihood_threshold', 0.5)
        self.hard_negative_max_ratio = hard_neg_config.get('max_ratio', 0.3)

        print(f"ROIDatasetBuilder initialized:")
        print(f"  - IoU threshold: {self.iou_threshold}")
        print(f"  - Overlap ratio: {self.overlap_ratio}")
        print(f"  - Hard negatives: {self.hard_negative_enabled}")

    def load_yolo_labels(self, label_path: str, image_shape: Tuple[int, int]) -> List[Tuple]:
        """
        Load YOLO format labels and convert to xyxy format.

        Args:
            label_path: Path to YOLO label file (.txt)
            image_shape: (H, W) of the image

        Returns:
            List of (x1, y1, x2, y2, class_id) tuples
        """
        if not Path(label_path).exists():
            return []

        h, w = image_shape
        boxes = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Convert normalized xywh to pixel xyxy
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h

                boxes.append((x1, y1, x2, y2, class_id))

        return boxes

    def is_positive_roi(
        self,
        roi_coords: Tuple[int, int, int, int],
        gt_boxes: List[Tuple]
    ) -> bool:
        """
        Determine if ROI is positive based on GT box overlap.

        Positive criteria (any of):
        1. ROI contains GT box center
        2. IoU(ROI, GT) > threshold
        3. GT box area overlap > overlap_ratio

        Args:
            roi_coords: (x1, y1, x2, y2) ROI coordinates
            gt_boxes: List of (x1, y1, x2, y2, class_id) GT boxes

        Returns:
            True if positive, False otherwise
        """
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords

        for gt_box in gt_boxes:
            x1, y1, x2, y2, _ = gt_box

            # Criterion 1: GT box center in ROI
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                return True

            # Criterion 2: IoU > threshold
            iou = compute_iou(roi_coords, (x1, y1, x2, y2))
            if iou > self.iou_threshold:
                return True

            # Criterion 3: Overlap ratio
            if boxes_overlap(roi_coords, (x1, y1, x2, y2)):
                # Compute overlap area
                x1_i = max(roi_x1, x1)
                y1_i = max(roi_y1, y1)
                x2_i = min(roi_x2, x2)
                y2_i = min(roi_y2, y2)
                overlap_area = (x2_i - x1_i) * (y2_i - y1_i)

                # GT box area
                gt_area = (x2 - x1) * (y2 - y1)

                if overlap_area / gt_area > self.overlap_ratio:
                    return True

        return False

    def is_hard_negative(
        self,
        roi_coords: Tuple[int, int, int, int],
        gt_boxes: List[Tuple],
        likelihood_map: np.ndarray
    ) -> bool:
        """
        Check if ROI is a hard negative (high likelihood but no GT).

        Args:
            roi_coords: (x1, y1, x2, y2) ROI coordinates
            gt_boxes: List of GT boxes
            likelihood_map: Likelihood map [H, W]

        Returns:
            True if hard negative, False otherwise
        """
        if not self.hard_negative_enabled:
            return False

        # Check no GT overlap
        for gt_box in gt_boxes:
            iou = compute_iou(roi_coords, gt_box[:4])
            if iou > 0.1:  # Small overlap threshold
                return False

        # Check high likelihood score
        x1, y1, x2, y2 = roi_coords
        roi_likelihood = likelihood_map[y1:y2, x1:x2]
        mean_likelihood = np.mean(roi_likelihood)

        return mean_likelihood > self.hard_negative_likelihood_threshold

    def map_gt_to_patch(
        self,
        roi_coords: Tuple[int, int, int, int],
        gt_boxes: List[Tuple],
        patch_size: Tuple[int, int]
    ) -> List[Tuple]:
        """
        Map GT boxes from original image to patch coordinates.

        Args:
            roi_coords: (x1, y1, x2, y2) ROI position in original image
            gt_boxes: List of (x1, y1, x2, y2, class_id) in original coords
            patch_size: (H, W) of the patch

        Returns:
            List of (cx, cy, w, h, class_id) in normalized patch coords (YOLO format)
        """
        mapped_boxes = []

        for gt_box in gt_boxes:
            x1, y1, x2, y2, cls = gt_box

            # Check if GT box overlaps with ROI
            if not boxes_overlap((x1, y1, x2, y2), roi_coords):
                continue

            # Transform to patch coordinates (normalized)
            try:
                cx, cy, w, h = original_to_patch(
                    (x1, y1, x2, y2),
                    roi_coords,
                    patch_size,
                    normalize=True
                )

                # Skip if box is too small after transformation
                if w < 0.01 or h < 0.01:  # Less than 1% of patch size
                    continue

                # Clip to [0, 1]
                cx = max(0, min(cx, 1))
                cy = max(0, min(cy, 1))
                w = max(0, min(w, 1))
                h = max(0, min(h, 1))

                mapped_boxes.append((cx, cy, w, h, cls))

            except Exception as e:
                print(f"Warning: Failed to map GT box {gt_box}: {e}")
                continue

        return mapped_boxes

    def build_dataset(
        self,
        original_dataset_root: str,
        priors_root: str,
        output_dir: str,
        splits: List[str] = ['train', 'val']
    ):
        """
        Build complete ROI dataset for all splits.

        Args:
            original_dataset_root: Root of original YOLO dataset
            priors_root: Root of priors directory
            output_dir: Output directory for ROI dataset
            splits: List of splits to process
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output structure
        for split in splits:
            (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # Load original dataset config
        dataset_root = Path(original_dataset_root)
        data_yaml_path = self.config['dataset']['data_yaml']

        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Mapping: patch_id -> metadata
        mapping = {}

        # Statistics
        stats = {split: {'positive': 0, 'hard_negative': 0, 'total': 0} for split in splits}

        # Process each split
        for split in splits:
            print(f"\nProcessing {split} split...")

            # Get image list
            images_dir = dataset_root / 'images' / split
            labels_dir = dataset_root / 'labels' / split

            if not images_dir.exists():
                print(f"Warning: {images_dir} not found, skipping...")
                continue

            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

            # Process each image
            for image_path in tqdm(image_files, desc=f"Building {split} ROI dataset"):
                image_name = image_path.stem
                label_path = labels_dir / f"{image_name}.txt"

                # Load image to get shape
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Warning: Cannot load {image_path}")
                    continue

                image_shape = image.shape[:2]  # (H, W)

                # Load GT labels
                gt_boxes = self.load_yolo_labels(str(label_path), image_shape)

                # Generate ROI patches
                try:
                    patches = self.roi_generator.generate_rois(
                        str(image_path),
                        split=split
                    )
                except Exception as e:
                    print(f"Warning: Failed to generate ROIs for {image_path}: {e}")
                    continue

                # Load likelihood for hard negative sampling
                if self.hard_negative_enabled:
                    try:
                        priors = self.roi_generator.priors_loader.load_priors(
                            str(image_path),
                            split=split,
                            resize_to_original=True
                        )
                        likelihood_map = priors['likelihood']
                    except:
                        likelihood_map = None
                else:
                    likelihood_map = None

                # Process each ROI patch
                for patch_info in patches:
                    roi_coords = patch_info['roi_coords']
                    patch = patch_info['patch']
                    patch_size = patch_info['patch_size']
                    roi_id = patch_info['roi_id']

                    # Check if positive or hard negative
                    is_positive = self.is_positive_roi(roi_coords, gt_boxes)
                    is_hard_neg = False

                    if not is_positive and likelihood_map is not None:
                        is_hard_neg = self.is_hard_negative(roi_coords, gt_boxes, likelihood_map)

                    # Skip if neither positive nor hard negative
                    if not is_positive and not is_hard_neg:
                        continue

                    # Generate patch filename
                    patch_name = f"{image_name}_roi_{roi_id}"
                    patch_image_path = output_dir / 'images' / split / f"{patch_name}.jpg"
                    patch_label_path = output_dir / 'labels' / split / f"{patch_name}.txt"

                    # Save patch image
                    patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(patch_image_path), patch_bgr)

                    # Map GT boxes to patch coordinates
                    if is_positive:
                        mapped_boxes = self.map_gt_to_patch(roi_coords, gt_boxes, patch_size)

                        # Save labels
                        with open(patch_label_path, 'w') as f:
                            for cx, cy, w, h, cls in mapped_boxes:
                                f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                        stats[split]['positive'] += 1
                    else:
                        # Hard negative: create empty label file
                        patch_label_path.touch()
                        stats[split]['hard_negative'] += 1

                    stats[split]['total'] += 1

                    # Save mapping (convert NumPy types to Python native types)
                    mapping[patch_name] = {
                        'original_image': str(image_path),
                        'roi_coords': [int(x) for x in roi_coords],  # Convert to Python int
                        'patch_size': [int(x) for x in patch_size],  # Convert to Python int
                        'split': split,
                        'is_positive': bool(is_positive),  # Convert to Python bool
                        'is_hard_negative': bool(is_hard_neg)  # Convert to Python bool
                    }

        # Save mapping
        mapping_path = output_dir / 'mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        # Create data.yaml for YOLO
        roi_data_yaml = {
            'path': str(output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': data_config.get('nc', 3),
            'names': data_config.get('names', {})
        }

        data_yaml_path = output_dir / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml.dump(roi_data_yaml, f)

        # Print statistics
        print("\n" + "="*60)
        print("ROI Dataset Construction Complete!")
        print("="*60)
        for split in splits:
            print(f"\n{split.upper()} Split:")
            print(f"  - Positive patches: {stats[split]['positive']}")
            print(f"  - Hard negative patches: {stats[split]['hard_negative']}")
            print(f"  - Total patches: {stats[split]['total']}")

        print(f"\nDataset saved to: {output_dir}")
        print(f"Mapping saved to: {mapping_path}")
        print(f"Data config saved to: {data_yaml_path}")


if __name__ == "__main__":
    print("ROI Dataset Builder module loaded successfully!")
