"""
ROI Proposal Module for E1 Fusion Detection

This module provides ROI-based detection components for the E1 experiment:
- ROI generation from topology masks
- ROI dataset construction for training
- Coordinate transformations
- ROI-based inference
- Detection fusion
"""

__version__ = "0.1.0"

from .roi_generator import ROIGenerator
from .roi_mapper import (
    original_to_patch,
    patch_to_original,
    validate_coordinates,
    compute_iou,
    boxes_overlap
)

__all__ = [
    "ROIGenerator",
    "original_to_patch",
    "patch_to_original",
    "validate_coordinates",
    "compute_iou",
    "boxes_overlap",
]
