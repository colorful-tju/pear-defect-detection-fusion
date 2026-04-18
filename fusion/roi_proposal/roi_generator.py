"""
ROI Generator Module

Generates ROI patches from topology masks for training and inference.

Key features:
- Loads topology masks and metadata using PriorsLoader
- Applies morphological post-processing
- Extracts ROIs using ROIProposer
- Crops RGB patches from original images
- Saves patches with coordinate mapping metadata
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.priors_loader import PriorsLoader
from src.roi_proposal import ROIProposer, ROI


class ROIGenerator:
    """
    Generates ROI patches from topology masks.

    This class combines PriorsLoader and ROIProposer to extract ROI patches
    from original images based on topology masks from Project A.
    """

    def __init__(self, config: Dict):
        """
        Initialize ROI generator.

        Args:
            config: Configuration dictionary with keys:
                - priors.root_dir: Path to priors directory
                - roi.min_area: Minimum ROI area
                - roi.max_area: Maximum ROI area
                - roi.expansion_ratio: ROI expansion ratio
                - roi.morphology.kernel_size: Morphological kernel size
                - roi.morphology.operation: Morphological operation (closing/opening)
        """
        # Initialize priors loader
        priors_root = config.get('priors', {}).get('root_dir', 'outputs/priors')
        self.priors_loader = PriorsLoader(priors_root, use_cache=True)

        # Initialize ROI proposer
        roi_config = config.get('roi', {})
        self.roi_proposer = ROIProposer(
            min_area=roi_config.get('min_area', 100),
            max_area=roi_config.get('max_area', 50000),
            expansion_ratio=roi_config.get('expansion_ratio', 0.2)
        )

        # Morphological processing parameters
        morph_config = roi_config.get('morphology', {})
        self.morph_kernel_size = morph_config.get('kernel_size', 5)
        self.morph_operation = morph_config.get('operation', 'closing')

        print(f"ROIGenerator initialized:")
        print(f"  - Priors root: {priors_root}")
        print(f"  - Min area: {self.roi_proposer.min_area}")
        print(f"  - Max area: {self.roi_proposer.max_area}")
        print(f"  - Expansion ratio: {self.roi_proposer.expansion_ratio}")
        print(f"  - Morphology: {self.morph_operation} (kernel={self.morph_kernel_size})")

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological post-processing to topology mask.

        Args:
            mask: Binary mask [H, W] with values 0 or 1

        Returns:
            Processed mask
        """
        # Ensure uint8 type
        mask = mask.astype(np.uint8)

        # Create morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )

        # Apply morphological operation
        if self.morph_operation == 'closing':
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif self.morph_operation == 'opening':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        else:
            raise ValueError(f"Unknown morphology operation: {self.morph_operation}")

        return mask

    def generate_rois(
        self,
        image_path: str,
        split: str = 'test',
        apply_morphology: bool = True
    ) -> List[Dict]:
        """
        Generate ROI patches for a given image.

        Args:
            image_path: Path to the original image
            split: Dataset split (train/val/test)
            apply_morphology: Whether to apply morphological post-processing

        Returns:
            List of dictionaries, each containing:
                - 'patch': RGB patch [H, W, 3]
                - 'roi_coords': (x1, y1, x2, y2) in original image
                - 'roi_obj': ROI object with additional metadata
                - 'patch_size': (H, W) of the patch
        """
        # 1. Load priors (automatically handles size alignment)
        priors = self.priors_loader.load_priors(
            image_path,
            split=split,
            resize_to_original=True  # Critical: resize to original image size
        )

        topology_mask = priors['topology_mask']  # Now in original size
        metadata = priors['metadata']

        # 2. Apply morphological post-processing
        if apply_morphology:
            topology_mask = self._apply_morphology(topology_mask)

        # 3. Extract ROIs using ROIProposer
        rois = self.roi_proposer.extract_rois(
            topology_mask,
            expand=True  # Apply expansion
        )

        if len(rois) == 0:
            print(f"Warning: No ROIs extracted for {image_path}")
            return []

        # 4. Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        # Convert BGR to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 5. Crop patches from original image
        patches = []
        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = roi.to_xyxy()

            # Ensure coordinates are within image bounds
            h, w = original_image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Skip if invalid
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop patch from ORIGINAL image (not mask)
            patch = original_image[y1:y2, x1:x2]

            patches.append({
                'patch': patch,
                'roi_coords': (x1, y1, x2, y2),
                'roi_obj': roi,
                'patch_size': (patch.shape[0], patch.shape[1]),  # (H, W)
                'roi_id': i
            })

        return patches

    def save_patches(
        self,
        patches: List[Dict],
        output_dir: str,
        image_name: str,
        save_format: str = 'jpg'
    ) -> List[str]:
        """
        Save ROI patches to disk.

        Args:
            patches: List of patch dictionaries from generate_rois()
            output_dir: Output directory
            image_name: Base name for the image (without extension)
            save_format: Image format (jpg/png)

        Returns:
            List of saved patch file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for patch_info in patches:
            roi_id = patch_info['roi_id']
            patch = patch_info['patch']

            # Generate filename
            patch_filename = f"{image_name}_roi_{roi_id}.{save_format}"
            patch_path = output_dir / patch_filename

            # Convert RGB to BGR for OpenCV
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

            # Save patch
            cv2.imwrite(str(patch_path), patch_bgr)
            saved_paths.append(str(patch_path))

        return saved_paths

    def visualize_rois(
        self,
        image_path: str,
        patches: List[Dict],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize ROIs on the original image.

        Args:
            image_path: Path to original image
            patches: List of patch dictionaries from generate_rois()
            output_path: Optional path to save visualization

        Returns:
            Visualization image [H, W, 3]
        """
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        # Draw ROI boxes
        for patch_info in patches:
            x1, y1, x2, y2 = patch_info['roi_coords']
            roi_id = patch_info['roi_id']

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw ROI ID
            cv2.putText(
                image,
                f"ROI {roi_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            # Draw centroid
            roi_obj = patch_info['roi_obj']
            cx, cy = int(roi_obj.centroid[0]), int(roi_obj.centroid[1])
            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, image)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    # Test ROI generation
    import yaml

    print("Testing ROI Generator...")

    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "fusion_config.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Using default config...")
        config = {
            'priors': {'root_dir': 'outputs/priors'},
            'roi': {
                'min_area': 100,
                'max_area': 50000,
                'expansion_ratio': 0.2,
                'morphology': {
                    'kernel_size': 5,
                    'operation': 'closing'
                }
            }
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Initialize generator
    generator = ROIGenerator(config)

    # Test on a sample image (if priors exist)
    test_image = "/home/robot/yolo/datasets/PearSurfaceDefects/images/val/cam0_1_557_pear_2.jpg"

    if Path(test_image).exists():
        print(f"\nTesting on: {test_image}")

        try:
            patches = generator.generate_rois(test_image, split='val')
            print(f"Generated {len(patches)} ROI patches")

            for i, patch_info in enumerate(patches):
                print(f"  Patch {i}:")
                print(f"    - ROI coords: {patch_info['roi_coords']}")
                print(f"    - Patch size: {patch_info['patch_size']}")
                print(f"    - Area: {patch_info['roi_obj'].area}")

        except Exception as e:
            print(f"Error: {e}")
            print("Make sure priors have been generated first.")
    else:
        print(f"Test image not found: {test_image}")
        print("Skipping test.")

    print("\nROI Generator module loaded successfully!")
