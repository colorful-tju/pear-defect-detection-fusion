"""
ROI Mapper Module

Provides coordinate transformation utilities for mapping between
original image coordinates and ROI patch coordinates.

Key functions:
- original_to_patch: Map bounding box from original image to patch coords
- patch_to_original: Map bounding box from patch to original image coords
- validate_coordinates: Ensure coordinates are valid
- compute_iou: Calculate IoU between two boxes
- boxes_overlap: Check if two boxes overlap
"""

import numpy as np
from typing import Tuple, List


def original_to_patch(
    bbox: Tuple[float, float, float, float],
    roi_coords: Tuple[int, int, int, int],
    patch_size: Tuple[int, int],
    normalize: bool = True
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box from original image coordinates to patch coordinates.

    Args:
        bbox: (x1, y1, x2, y2) in original image coordinates
        roi_coords: (roi_x1, roi_y1, roi_x2, roi_y2) ROI position in original image
        patch_size: (patch_h, patch_w) actual patch dimensions
        normalize: If True, return normalized coordinates [0, 1]; else pixel coordinates

    Returns:
        Transformed bbox in patch coordinates
        - If normalize=True: (cx, cy, w, h) normalized to [0, 1] (YOLO format)
        - If normalize=False: (x1, y1, x2, y2) in pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    patch_h, patch_w = patch_size

    # Transform to patch coordinates
    patch_x1 = x1 - roi_x1
    patch_y1 = y1 - roi_y1
    patch_x2 = x2 - roi_x1
    patch_y2 = y2 - roi_y1

    # Clip to patch boundaries
    patch_x1 = max(0, min(patch_x1, patch_w))
    patch_y1 = max(0, min(patch_y1, patch_h))
    patch_x2 = max(0, min(patch_x2, patch_w))
    patch_y2 = max(0, min(patch_y2, patch_h))

    if normalize:
        # Convert to YOLO format (normalized center + size)
        center_x = (patch_x1 + patch_x2) / 2 / patch_w
        center_y = (patch_y1 + patch_y2) / 2 / patch_h
        width = (patch_x2 - patch_x1) / patch_w
        height = (patch_y2 - patch_y1) / patch_h
        return (center_x, center_y, width, height)
    else:
        return (patch_x1, patch_y1, patch_x2, patch_y2)


def patch_to_original(
    bbox: Tuple[float, float, float, float],
    roi_coords: Tuple[int, int, int, int],
    patch_size: Tuple[int, int] = None,
    normalized: bool = False
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box from patch coordinates to original image coordinates.

    Args:
        bbox: Bounding box in patch coordinates
            - If normalized=True: (cx, cy, w, h) normalized [0, 1]
            - If normalized=False: (x1, y1, x2, y2) in pixel coordinates
        roi_coords: (roi_x1, roi_y1, roi_x2, roi_y2) ROI position in original image
        patch_size: (patch_h, patch_w) required if normalized=True
        normalized: Whether input bbox is normalized

    Returns:
        (x1, y1, x2, y2) in original image coordinates
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords

    if normalized:
        if patch_size is None:
            raise ValueError("patch_size required when normalized=True")

        patch_h, patch_w = patch_size
        cx, cy, w, h = bbox

        # Convert from normalized to pixel coordinates
        patch_x1 = (cx - w / 2) * patch_w
        patch_y1 = (cy - h / 2) * patch_h
        patch_x2 = (cx + w / 2) * patch_w
        patch_y2 = (cy + h / 2) * patch_h
    else:
        patch_x1, patch_y1, patch_x2, patch_y2 = bbox

    # Transform to original image coordinates
    orig_x1 = roi_x1 + patch_x1
    orig_y1 = roi_y1 + patch_y1
    orig_x2 = roi_x1 + patch_x2
    orig_y2 = roi_y1 + patch_y2

    return (orig_x1, orig_y1, orig_x2, orig_y2)


def validate_coordinates(
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int],
    min_size: int = 5
) -> bool:
    """
    Validate that bounding box coordinates are valid.

    Args:
        bbox: (x1, y1, x2, y2) bounding box
        image_shape: (H, W) image dimensions
        min_size: Minimum box width/height in pixels

    Returns:
        True if valid, False otherwise
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape

    # Check bounds
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False

    # Check ordering
    if x2 <= x1 or y2 <= y1:
        return False

    # Check minimum size
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return False

    return True


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1: (x1, y1, x2, y2) first box
        box2: (x1, y1, x2, y2) second box

    Returns:
        IoU value in [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def boxes_overlap(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> bool:
    """
    Check if two boxes overlap.

    Args:
        box1: (x1, y1, x2, y2) first box
        box2: (x1, y1, x2, y2) second box

    Returns:
        True if boxes overlap, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Check if boxes are separated
    if x2_1 <= x1_2 or x2_2 <= x1_1:
        return False
    if y2_1 <= y1_2 or y2_2 <= y1_1:
        return False

    return True


def clip_bbox_to_image(
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Clip bounding box to image boundaries.

    Args:
        bbox: (x1, y1, x2, y2) bounding box
        image_shape: (H, W) image dimensions

    Returns:
        Clipped (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape

    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    return (x1, y1, x2, y2)


def convert_yolo_to_xyxy(
    yolo_bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Convert YOLO format (normalized cx, cy, w, h) to xyxy format.

    Args:
        yolo_bbox: (cx, cy, w, h) normalized to [0, 1]
        image_shape: (H, W) image dimensions

    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    cx, cy, w, h = yolo_bbox
    img_h, img_w = image_shape

    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h

    return (x1, y1, x2, y2)


def convert_xyxy_to_yolo(
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Convert xyxy format to YOLO format (normalized cx, cy, w, h).

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates
        image_shape: (H, W) image dimensions

    Returns:
        (cx, cy, w, h) normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = image_shape

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    return (cx, cy, w, h)


if __name__ == "__main__":
    # Test coordinate transformations
    print("Testing coordinate transformations...")

    # Test case 1: original to patch
    bbox_orig = (150, 100, 250, 200)  # x1, y1, x2, y2 in original image
    roi_coords = (100, 50, 300, 250)  # ROI position
    patch_size = (200, 200)  # patch dimensions

    bbox_patch_norm = original_to_patch(bbox_orig, roi_coords, patch_size, normalize=True)
    print(f"Original bbox: {bbox_orig}")
    print(f"ROI coords: {roi_coords}")
    print(f"Patch size: {patch_size}")
    print(f"Patch bbox (normalized): {bbox_patch_norm}")

    # Test case 2: patch to original
    bbox_orig_recovered = patch_to_original(
        bbox_patch_norm, roi_coords, patch_size, normalized=True
    )
    print(f"Recovered original bbox: {bbox_orig_recovered}")

    # Test case 3: IoU computation
    box1 = (0, 0, 100, 100)
    box2 = (50, 50, 150, 150)
    iou = compute_iou(box1, box2)
    print(f"\nIoU between {box1} and {box2}: {iou:.3f}")

    # Test case 4: Overlap check
    overlap = boxes_overlap(box1, box2)
    print(f"Boxes overlap: {overlap}")

    print("\nAll tests passed!")
