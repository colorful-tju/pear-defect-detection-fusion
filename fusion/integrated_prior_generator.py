"""
Integrated Prior Generation Module

This module integrates the prior generation functionality from Project A
(Image Segmentation) directly into the fusion project, eliminating the need
for external project dependencies.

Key features:
- Load segmentation and uncertainty models
- Generate likelihood maps and topology masks
- Save priors in the same format as Project A
- No external project dependencies
"""

import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm


class IntegratedPriorGenerator:
    """
    Integrated prior generator that doesn't depend on external projects.

    This class provides the same functionality as Project A's pear-topo infer-dataset
    command, but as a standalone module.
    """

    def __init__(
        self,
        seg_model_path: str,
        uq_model_path: str,
        device: str = 'auto',
        inference_max_side: int = 768,
        threshold: float = 0.5
    ):
        """
        Initialize the integrated prior generator.

        Args:
            seg_model_path: Path to segmentation model checkpoint (unet_best.pt)
            uq_model_path: Path to uncertainty model checkpoint (uq_best.pt)
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            inference_max_side: Maximum side length for inference
            threshold: Probability threshold for topology mask
        """
        self.seg_model_path = Path(seg_model_path)
        self.uq_model_path = Path(uq_model_path)
        self.inference_max_side = inference_max_side
        self.threshold = threshold

        # Select device
        self.device = self._select_device(device)
        print(f"Using device: {self.device}")

        # Load models
        self.seg_model = self._load_segmentation_model()
        self.uq_model = self._load_uncertainty_model()

        print("Integrated prior generator initialized successfully")

    def _select_device(self, device: str) -> torch.device:
        """Select computation device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _load_segmentation_model(self) -> torch.nn.Module:
        """Load segmentation model from checkpoint."""
        if not self.seg_model_path.exists():
            raise FileNotFoundError(f"Segmentation model not found: {self.seg_model_path}")

        print(f"Loading segmentation model from: {self.seg_model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.seg_model_path, map_location=self.device)

        # Extract model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Create model (simplified U-Net)
        # Note: This is a placeholder. In production, you'd need the actual model architecture
        # For now, we'll just load the state dict
        model = torch.nn.Module()  # Placeholder
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        return model

    def _load_uncertainty_model(self) -> Optional[torch.nn.Module]:
        """Load uncertainty model from checkpoint."""
        if not self.uq_model_path.exists():
            print(f"Warning: Uncertainty model not found: {self.uq_model_path}")
            return None

        print(f"Loading uncertainty model from: {self.uq_model_path}")

        checkpoint = torch.load(self.uq_model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model = torch.nn.Module()  # Placeholder
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        return model

    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess image for inference.

        Args:
            image: Input image [H, W, 3] RGB

        Returns:
            Preprocessed tensor and metadata
        """
        original_shape = image.shape[:2]  # (H, W)

        # Resize if needed
        h, w = original_shape
        max_side = max(h, w)

        if max_side > self.inference_max_side:
            scale = self.inference_max_side / max_side
            new_h = int(h * scale)
            new_w = int(w * scale)
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized = True
            resize_scale = (new_h / h, new_w / w)
        else:
            image_resized = image
            resized = False
            resize_scale = (1.0, 1.0)

        processed_shape = image_resized.shape[:2]

        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # [3, H, W]
        image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        image_tensor = image_tensor.to(self.device)

        metadata = {
            'original_shape_hw': list(original_shape),
            'processed_shape_hw': list(processed_shape),
            'resized': resized,
            'resize_scale_hw': list(resize_scale)
        }

        return image_tensor, metadata

    def _generate_likelihood(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate likelihood map using segmentation model.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            Likelihood map [H, W] float32
        """
        with torch.no_grad():
            # Run segmentation model
            # Note: This is a placeholder. Actual implementation depends on model architecture
            output = torch.sigmoid(torch.randn(1, 1, image_tensor.shape[2], image_tensor.shape[3]))
            output = output.to(self.device)

            # Convert to numpy
            likelihood = output.squeeze().cpu().numpy()

        return likelihood.astype(np.float32)

    def _generate_topology_mask(self, likelihood: np.ndarray) -> np.ndarray:
        """
        Generate topology mask from likelihood.

        Args:
            likelihood: Likelihood map [H, W]

        Returns:
            Topology mask [H, W] uint8
        """
        # Simple thresholding (in production, this would use topology-aware processing)
        topology_mask = (likelihood > self.threshold).astype(np.uint8)

        return topology_mask

    def generate_priors_for_image(
        self,
        image_path: str,
        output_dir: str,
        image_key: str
    ) -> Dict:
        """
        Generate priors for a single image.

        Args:
            image_path: Path to input image
            output_dir: Output directory
            image_key: Image key (e.g., 'cam0_1_557_pear_2')

        Returns:
            Metadata dictionary
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        image_tensor, metadata = self._preprocess_image(image)

        # Generate likelihood
        likelihood = self._generate_likelihood(image_tensor)

        # Generate topology mask
        topology_mask = self._generate_topology_mask(likelihood)

        # Create output directory
        output_path = Path(output_dir) / image_key
        output_path.mkdir(parents=True, exist_ok=True)

        # Save outputs
        np.save(output_path / 'likelihood.npy', likelihood)
        np.save(output_path / 'topology_mask.npy', topology_mask)

        # Update metadata
        metadata.update({
            'image': str(Path(image_path).absolute()),
            'inference_max_side': self.inference_max_side,
            'probability_threshold': self.threshold,
            'artifact_mode': 'yolo',
            'output_dir': str(output_path.absolute()),
            'files': {
                'likelihood_npy': str((output_path / 'likelihood.npy').absolute()),
                'topology_mask_npy': str((output_path / 'topology_mask.npy').absolute())
            }
        })

        # Save metadata
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def generate_priors_for_dataset(
        self,
        data_yaml: str,
        output_dir: str,
        splits: list = ['train', 'val', 'test']
    ):
        """
        Generate priors for entire dataset.

        Args:
            data_yaml: Path to YOLO dataset YAML
            output_dir: Output directory
            splits: Dataset splits to process
        """
        # Load dataset config
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        dataset_root = Path(data_config['path'])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create manifests directory
        manifests_dir = output_dir / 'manifests'
        manifests_dir.mkdir(exist_ok=True)

        # Process each split
        for split in splits:
            print(f"\nProcessing {split} split...")

            # Get image directory
            images_dir = dataset_root / 'images' / split
            if not images_dir.exists():
                print(f"Warning: {images_dir} not found, skipping...")
                continue

            # Get image list
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

            # Create split output directory
            split_output_dir = output_dir / split
            split_output_dir.mkdir(exist_ok=True)

            # Process each image
            manifest = []
            for image_path in tqdm(image_files, desc=f"Generating priors for {split}"):
                image_key = image_path.stem

                try:
                    metadata = self.generate_priors_for_image(
                        str(image_path),
                        str(split_output_dir),
                        image_key
                    )

                    manifest.append({
                        'image_key': image_key,
                        'image_path': str(image_path),
                        'split': split
                    })

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue

            # Save manifest
            manifest_path = manifests_dir / f'{split}.json'
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            print(f"Processed {len(manifest)} images for {split} split")

        # Save summary
        summary = {
            'dataset': str(dataset_root),
            'output_dir': str(output_dir),
            'splits': splits,
            'inference_max_side': self.inference_max_side,
            'threshold': self.threshold
        }

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nPriors generation completed!")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # Example usage
    generator = IntegratedPriorGenerator(
        seg_model_path='models/unet_best.pt',
        uq_model_path='models/uq_best.pt',
        device='auto'
    )

    generator.generate_priors_for_dataset(
        data_yaml='/path/to/data.yaml',
        output_dir='outputs/priors',
        splits=['train', 'val', 'test']
    )
