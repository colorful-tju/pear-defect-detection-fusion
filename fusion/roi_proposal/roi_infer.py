"""
ROI Inference Module

Performs YOLO inference on ROI patches and maps results back to original coordinates.

Key features:
- Loads local detector model
- Generates ROI patches for test images
- Runs YOLO inference on each patch
- Maps detections back to original image coordinates
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fusion.roi_proposal.roi_generator import ROIGenerator
from fusion.roi_proposal.roi_mapper import patch_to_original


class ROIInferencer:
    """
    Performs ROI-based inference using local detector.
    """

    def __init__(self, local_model, config: Dict):
        """
        Initialize ROI inferencer.

        Args:
            local_model: YOLO model for local detection
            config: Configuration dictionary
        """
        self.local_model = local_model
        self.config = config
        self.roi_generator = ROIGenerator(config)

        # Inference parameters
        inference_config = config.get('inference', {}).get('local', {})
        self.conf_threshold = inference_config.get('conf', 0.20)
        self.iou_threshold = inference_config.get('iou', 0.45)

        print(f"ROIInferencer initialized:")
        print(f"  - Confidence threshold: {self.conf_threshold}")
        print(f"  - IoU threshold: {self.iou_threshold}")

    def infer_single_image(
        self,
        image_path: str,
        split: str = 'test',
        verbose: bool = False
    ) -> List[Dict]:
        """
        Run ROI-based inference on a single image.

        Args:
            image_path: Path to the image
            split: Dataset split
            verbose: Whether to print verbose output

        Returns:
            List of detections, each containing:
                - 'bbox': [x1, y1, x2, y2] in original image coords
                - 'conf': confidence score
                - 'cls': class ID
                - 'source': 'local'
                - 'roi_id': ROI ID
        """
        # Generate ROI patches
        patches = self.roi_generator.generate_rois(image_path, split=split)

        if len(patches) == 0:
            if verbose:
                print(f"No ROIs generated for {image_path}")
            return []

        # Run inference on each patch
        all_detections = []

        for patch_info in patches:
            patch = patch_info['patch']
            roi_coords = patch_info['roi_coords']
            patch_size = patch_info['patch_size']
            roi_id = patch_info['roi_id']

            # Convert RGB to BGR for YOLO
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

            # Run YOLO inference
            results = self.local_model.predict(
                source=patch_bgr,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            # Extract detections
            if len(results) == 0:
                continue

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Process each detection
            for box in result.boxes:
                # Get box coordinates in patch
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Map to original image coordinates
                orig_x1, orig_y1, orig_x2, orig_y2 = patch_to_original(
                    (x1, y1, x2, y2),
                    roi_coords,
                    patch_size=None,
                    normalized=False
                )

                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                all_detections.append({
                    'bbox': [orig_x1, orig_y1, orig_x2, orig_y2],
                    'conf': conf,
                    'cls': cls,
                    'source': 'local',
                    'roi_id': roi_id
                })

        if verbose:
            print(f"ROI inference: {len(patches)} patches, {len(all_detections)} detections")

        return all_detections

    def infer_batch(
        self,
        image_paths: List[str],
        split: str = 'test',
        verbose: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Run ROI-based inference on a batch of images.

        Args:
            image_paths: List of image paths
            split: Dataset split
            verbose: Whether to print progress

        Returns:
            Dictionary mapping image_path -> list of detections
        """
        results = {}

        for image_path in image_paths:
            if verbose:
                print(f"Processing: {image_path}")

            detections = self.infer_single_image(image_path, split=split, verbose=verbose)
            results[image_path] = detections

        return results


if __name__ == "__main__":
    print("ROI Inferencer module loaded successfully!")
