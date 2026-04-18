"""
Package initialization for pear defect detection fusion.
"""

__version__ = "0.1.0"
__author__ = "Pear Defect Detection Team"

from .priors_loader import PriorsLoader, load_priors_manifest
from .roi_proposal import ROI, ROIProposer

__all__ = [
    "PriorsLoader",
    "load_priors_manifest",
    "ROI",
    "ROIProposer",
]
