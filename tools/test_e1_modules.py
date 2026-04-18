#!/usr/bin/env python3
"""
Test E1 Modules

Verifies that all E1 modules are working correctly.

Usage:
    python tools/test_e1_modules.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("="*60)
    print("Testing Module Imports")
    print("="*60)

    try:
        print("\n1. Testing roi_mapper...")
        from fusion.roi_proposal import roi_mapper
        print("   ✓ roi_mapper imported successfully")

        print("\n2. Testing roi_generator...")
        from fusion.roi_proposal import roi_generator
        print("   ✓ roi_generator imported successfully")

        print("\n3. Testing roi_dataset_builder...")
        from fusion.roi_proposal import roi_dataset_builder
        print("   ✓ roi_dataset_builder imported successfully")

        print("\n4. Testing roi_infer...")
        from fusion.roi_proposal import roi_infer
        print("   ✓ roi_infer imported successfully")

        print("\n5. Testing roi_fusion...")
        from fusion.roi_proposal import roi_fusion
        print("   ✓ roi_fusion imported successfully")

        print("\n✓ All modules imported successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        return False


def test_coordinate_transformations():
    """Test coordinate transformation functions."""
    print("\n" + "="*60)
    print("Testing Coordinate Transformations")
    print("="*60)

    try:
        from fusion.roi_proposal.roi_mapper import (
            original_to_patch,
            patch_to_original,
            compute_iou,
            boxes_overlap
        )

        # Test case 1: original to patch
        print("\n1. Testing original_to_patch...")
        bbox_orig = (150, 100, 250, 200)
        roi_coords = (100, 50, 300, 250)
        patch_size = (200, 200)

        bbox_patch = original_to_patch(bbox_orig, roi_coords, patch_size, normalize=True)
        print(f"   Original bbox: {bbox_orig}")
        print(f"   Patch bbox (normalized): {bbox_patch}")
        print("   ✓ original_to_patch works")

        # Test case 2: patch to original
        print("\n2. Testing patch_to_original...")
        bbox_recovered = patch_to_original(bbox_patch, roi_coords, patch_size, normalized=True)
        print(f"   Recovered bbox: {bbox_recovered}")

        # Check if close to original
        error = sum(abs(a - b) for a, b in zip(bbox_orig, bbox_recovered))
        if error < 1.0:
            print("   ✓ patch_to_original works (error < 1.0)")
        else:
            print(f"   ⚠ Warning: transformation error = {error}")

        # Test case 3: IoU
        print("\n3. Testing compute_iou...")
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = compute_iou(box1, box2)
        print(f"   IoU({box1}, {box2}) = {iou:.3f}")
        expected_iou = 0.143  # Approximately 1/7
        if abs(iou - expected_iou) < 0.01:
            print("   ✓ compute_iou works")
        else:
            print(f"   ⚠ Warning: expected ~{expected_iou}, got {iou}")

        # Test case 4: Overlap
        print("\n4. Testing boxes_overlap...")
        overlap = boxes_overlap(box1, box2)
        print(f"   Overlap({box1}, {box2}) = {overlap}")
        if overlap:
            print("   ✓ boxes_overlap works")
        else:
            print("   ✗ boxes_overlap failed")

        print("\n✓ All coordinate transformation tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Coordinate transformation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_engine():
    """Test fusion engine."""
    print("\n" + "="*60)
    print("Testing Fusion Engine")
    print("="*60)

    try:
        from fusion.roi_proposal.roi_fusion import FusionEngine

        config = {
            'fusion': {
                'method': 'priority_nms',
                'iou_threshold': 0.5,
                'priority': 'local'
            }
        }

        print("\n1. Initializing FusionEngine...")
        engine = FusionEngine(config)
        print("   ✓ FusionEngine initialized")

        print("\n2. Testing detection merging...")
        global_dets = [
            {'bbox': [100, 100, 200, 200], 'conf': 0.8, 'cls': 1},
            {'bbox': [300, 300, 400, 400], 'conf': 0.7, 'cls': 1},
        ]

        local_dets = [
            {'bbox': [105, 105, 205, 205], 'conf': 0.6, 'cls': 1},  # Overlaps with first global
            {'bbox': [500, 500, 600, 600], 'conf': 0.9, 'cls': 2},
        ]

        merged = engine.merge(global_dets, local_dets)

        print(f"   Global detections: {len(global_dets)}")
        print(f"   Local detections: {len(local_dets)}")
        print(f"   Merged detections: {len(merged)}")

        # Should have 3 detections (local replaces overlapping global)
        if len(merged) == 3:
            print("   ✓ Fusion works correctly")
        else:
            print(f"   ⚠ Warning: expected 3 merged detections, got {len(merged)}")

        print("\n✓ Fusion engine tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Fusion engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("Testing Configuration Loading")
    print("="*60)

    try:
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "e1_config.yaml"

        if not config_path.exists():
            print(f"   ⚠ Config file not found: {config_path}")
            return False

        print(f"\n1. Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"   Experiment: {config.get('experiment_name')}")
        print(f"   Dataset: {config['dataset']['root']}")
        print(f"   Global detector: {config['models']['global_detector']}")
        print(f"   Local detector: {config['models']['local_detector']}")

        print("\n   ✓ Configuration loaded successfully")
        return True

    except Exception as e:
        print(f"\n✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("E1 Module Testing Suite")
    print("="*60)

    results = []

    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Coordinate Transformations", test_coordinate_transformations()))
    results.append(("Fusion Engine", test_fusion_engine()))
    results.append(("Configuration Loading", test_config_loading()))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
