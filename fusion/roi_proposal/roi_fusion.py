"""
ROI Fusion Module

Merges global and local detections with intelligent fusion strategies.

Key features:
- Priority-based NMS (local > global when overlapping)
- Confidence-based filtering
- Optional score reweighting using likelihood maps
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fusion.roi_proposal.roi_mapper import compute_iou


class FusionEngine:
    """
    Merges detections from global and local detectors.
    """

    def __init__(self, config: Dict):
        """
        Initialize fusion engine.

        Args:
            config: Configuration dictionary with keys:
                - fusion.method: Fusion method (priority_nms/confidence_nms)
                - fusion.iou_threshold: IoU threshold for NMS
                - fusion.priority: Priority source (local/global/confidence)
        """
        fusion_config = config.get('fusion', {})
        self.method = fusion_config.get('method', 'priority_nms')
        self.iou_threshold = fusion_config.get('iou_threshold', 0.5)
        self.priority = fusion_config.get('priority', 'local')

        print(f"FusionEngine initialized:")
        print(f"  - Method: {self.method}")
        print(f"  - IoU threshold: {self.iou_threshold}")
        print(f"  - Priority: {self.priority}")

    def merge(
        self,
        global_dets: List[Dict],
        local_dets: List[Dict],
        likelihood_map: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Merge global and local detections.

        Args:
            global_dets: List of global detections
            local_dets: List of local detections
            likelihood_map: Optional likelihood map for score reweighting

        Returns:
            List of merged detections
        """
        if self.method == 'priority_nms':
            return self._priority_nms(global_dets, local_dets)
        elif self.method == 'confidence_nms':
            return self._confidence_nms(global_dets, local_dets)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

    def _priority_nms(
        self,
        global_dets: List[Dict],
        local_dets: List[Dict]
    ) -> List[Dict]:
        """
        Merge detections with priority-based NMS.

        When detections overlap (IoU > threshold):
        - If priority='local': keep local detection
        - If priority='global': keep global detection
        - If priority='confidence': keep higher confidence detection

        Args:
            global_dets: Global detections
            local_dets: Local detections

        Returns:
            Merged detections
        """
        # Tag detections with source
        all_dets = []

        for det in global_dets:
            det_copy = det.copy()
            if 'source' not in det_copy:
                det_copy['source'] = 'global'
            all_dets.append(det_copy)

        for det in local_dets:
            det_copy = det.copy()
            if 'source' not in det_copy:
                det_copy['source'] = 'local'
            all_dets.append(det_copy)

        if len(all_dets) == 0:
            return []

        # Sort by confidence (descending)
        all_dets.sort(key=lambda x: x['conf'], reverse=True)

        # Priority-based NMS
        keep = []

        while len(all_dets) > 0:
            current = all_dets.pop(0)
            keep.append(current)

            # Remove overlapping boxes based on priority
            remaining = []

            for det in all_dets:
                iou = compute_iou(
                    tuple(current['bbox']),
                    tuple(det['bbox'])
                )

                if iou > self.iou_threshold:
                    # Overlapping: apply priority rule
                    if self.priority == 'local':
                        # Keep local over global
                        if det['source'] == 'local' and current['source'] == 'global':
                            # Replace current with local detection
                            keep[-1] = det
                        # else: keep current, discard det
                    elif self.priority == 'global':
                        # Keep global over local
                        if det['source'] == 'global' and current['source'] == 'local':
                            keep[-1] = det
                    elif self.priority == 'confidence':
                        # Keep higher confidence (already sorted, so keep current)
                        pass
                    # Discard det (overlapping with lower priority)
                else:
                    # Not overlapping: keep det
                    remaining.append(det)

            all_dets = remaining

        return keep

    def _confidence_nms(
        self,
        global_dets: List[Dict],
        local_dets: List[Dict]
    ) -> List[Dict]:
        """
        Merge detections with standard confidence-based NMS.

        Args:
            global_dets: Global detections
            local_dets: Local detections

        Returns:
            Merged detections
        """
        # Combine all detections
        all_dets = []

        for det in global_dets:
            det_copy = det.copy()
            if 'source' not in det_copy:
                det_copy['source'] = 'global'
            all_dets.append(det_copy)

        for det in local_dets:
            det_copy = det.copy()
            if 'source' not in det_copy:
                det_copy['source'] = 'local'
            all_dets.append(det_copy)

        if len(all_dets) == 0:
            return []

        # Sort by confidence (descending)
        all_dets.sort(key=lambda x: x['conf'], reverse=True)

        # Standard NMS
        keep = []

        while len(all_dets) > 0:
            current = all_dets.pop(0)
            keep.append(current)

            # Remove overlapping boxes
            remaining = []

            for det in all_dets:
                iou = compute_iou(
                    tuple(current['bbox']),
                    tuple(det['bbox'])
                )

                if iou <= self.iou_threshold:
                    remaining.append(det)
                # else: discard (overlapping with higher confidence)

            all_dets = remaining

        return keep

    def reweight_scores(
        self,
        detections: List[Dict],
        likelihood_map: np.ndarray,
        alpha: float = 0.5,
        beta: float = 0.5,
        aggregation: str = 'mean'
    ) -> List[Dict]:
        """
        Reweight detection scores using likelihood map.

        Formula: new_score = original_score * (alpha + beta * likelihood_score)

        Args:
            detections: List of detections
            likelihood_map: Likelihood map [H, W]
            alpha: Base weight
            beta: Likelihood weight
            aggregation: Likelihood aggregation method (mean/max/weighted_mean)

        Returns:
            Detections with reweighted scores
        """
        reweighted_dets = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            # Ensure integer coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Clip to image bounds
            h, w = likelihood_map.shape
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Extract likelihood within box
            if x2 > x1 and y2 > y1:
                box_likelihood = likelihood_map[y1:y2, x1:x2]

                # Aggregate likelihood
                if aggregation == 'mean':
                    likelihood_score = np.mean(box_likelihood)
                elif aggregation == 'max':
                    likelihood_score = np.max(box_likelihood)
                elif aggregation == 'weighted_mean':
                    # Weight by distance from center
                    cy, cx = box_likelihood.shape[0] // 2, box_likelihood.shape[1] // 2
                    y_coords, x_coords = np.ogrid[:box_likelihood.shape[0], :box_likelihood.shape[1]]
                    distances = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
                    weights = 1 / (distances + 1)
                    likelihood_score = np.average(box_likelihood, weights=weights)
                else:
                    likelihood_score = np.mean(box_likelihood)
            else:
                likelihood_score = 0.0

            # Reweight score
            original_score = det['conf']
            new_score = original_score * (alpha + beta * likelihood_score)
            new_score = min(1.0, max(0.0, new_score))  # Clip to [0, 1]

            # Create reweighted detection
            det_copy = det.copy()
            det_copy['conf'] = new_score
            det_copy['original_conf'] = original_score
            det_copy['likelihood_score'] = likelihood_score

            reweighted_dets.append(det_copy)

        return reweighted_dets


if __name__ == "__main__":
    # Test fusion
    print("Testing FusionEngine...")

    config = {
        'fusion': {
            'method': 'priority_nms',
            'iou_threshold': 0.5,
            'priority': 'local'
        }
    }

    engine = FusionEngine(config)

    # Test detections
    global_dets = [
        {'bbox': [100, 100, 200, 200], 'conf': 0.8, 'cls': 1},
        {'bbox': [300, 300, 400, 400], 'conf': 0.7, 'cls': 1},
    ]

    local_dets = [
        {'bbox': [105, 105, 205, 205], 'conf': 0.6, 'cls': 1},  # Overlaps with first global
        {'bbox': [500, 500, 600, 600], 'conf': 0.9, 'cls': 2},
    ]

    merged = engine.merge(global_dets, local_dets)

    print(f"\nGlobal detections: {len(global_dets)}")
    print(f"Local detections: {len(local_dets)}")
    print(f"Merged detections: {len(merged)}")

    for i, det in enumerate(merged):
        print(f"  Detection {i}: source={det['source']}, conf={det['conf']:.2f}, bbox={det['bbox']}")

    print("\nFusionEngine module loaded successfully!")
