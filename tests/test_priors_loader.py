"""
Unit tests for priors_loader module.
"""

import pytest
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from priors_loader import PriorsLoader, load_priors_manifest


@pytest.fixture
def temp_priors_dir():
    """Create a temporary priors directory structure."""
    temp_dir = tempfile.mkdtemp()
    priors_root = Path(temp_dir) / "priors"

    # Create directory structure
    val_dir = priors_root / "val" / "test_image"
    val_dir.mkdir(parents=True)

    # Create dummy likelihood.npy
    likelihood = np.random.rand(512, 640).astype(np.float32)
    np.save(val_dir / "likelihood.npy", likelihood)

    # Create dummy topology_mask.npy
    topology_mask = np.random.randint(0, 2, (512, 640), dtype=np.uint8)
    np.save(val_dir / "topology_mask.npy", topology_mask)

    # Create metadata.json
    metadata = {
        "image": "/path/to/test_image.jpg",
        "split": "val",
        "original_shape_hw": [1024, 1280],
        "processed_shape_hw": [512, 640],
        "resized": True,
        "resize_scale_hw": [0.5, 0.5]
    }
    with open(val_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create manifest
    manifests_dir = priors_root / "manifests"
    manifests_dir.mkdir(parents=True)
    manifest = [
        {
            "image_key": "test_image",
            "image_path": "/path/to/test_image.jpg",
            "split": "val"
        }
    ]
    with open(manifests_dir / "val.json", "w") as f:
        json.dump(manifest, f)

    yield priors_root

    # Cleanup
    shutil.rmtree(temp_dir)


def test_priors_loader_init(temp_priors_dir):
    """Test PriorsLoader initialization."""
    loader = PriorsLoader(str(temp_priors_dir))
    assert loader.priors_root == temp_priors_dir
    assert loader.use_cache is True


def test_get_image_key():
    """Test image key extraction."""
    loader = PriorsLoader("dummy_path", use_cache=False)

    # Test with full path
    image_path = "/path/to/images/val/test_image.jpg"
    key = loader.get_image_key(image_path, "val")
    assert key == "test_image"

    # Test with relative path
    image_path = "test_image.jpg"
    key = loader.get_image_key(image_path, "val")
    assert key == "test_image"


def test_load_priors_no_resize(temp_priors_dir):
    """Test loading priors without resizing."""
    loader = PriorsLoader(str(temp_priors_dir))

    priors = loader.load_priors(
        "test_image.jpg",
        split="val",
        resize_to_original=False
    )

    assert "likelihood" in priors
    assert "topology_mask" in priors
    assert "metadata" in priors

    # Check shapes (should be processed shape)
    assert priors["likelihood"].shape == (512, 640)
    assert priors["topology_mask"].shape == (512, 640)


def test_load_priors_with_resize(temp_priors_dir):
    """Test loading priors with resizing to original."""
    loader = PriorsLoader(str(temp_priors_dir))

    priors = loader.load_priors(
        "test_image.jpg",
        split="val",
        resize_to_original=True
    )

    # Check shapes (should be original shape)
    assert priors["likelihood"].shape == (1024, 1280)
    assert priors["topology_mask"].shape == (1024, 1280)


def test_cache_functionality(temp_priors_dir):
    """Test caching mechanism."""
    loader = PriorsLoader(str(temp_priors_dir), use_cache=True)

    # First load
    priors1 = loader.load_priors("test_image.jpg", split="val")
    assert loader.get_cache_size() == 1

    # Second load (should use cache)
    priors2 = loader.load_priors("test_image.jpg", split="val")
    assert loader.get_cache_size() == 1

    # Check that cached result is returned
    assert np.array_equal(priors1["likelihood"], priors2["likelihood"])

    # Clear cache
    loader.clear_cache()
    assert loader.get_cache_size() == 0


def test_load_priors_manifest(temp_priors_dir):
    """Test loading manifest file."""
    manifest = load_priors_manifest(str(temp_priors_dir), split="val")

    assert isinstance(manifest, list)
    assert len(manifest) == 1
    assert manifest[0]["image_key"] == "test_image"


def test_file_not_found():
    """Test error handling for missing files."""
    with pytest.raises(FileNotFoundError):
        loader = PriorsLoader("/nonexistent/path")

    # Create a valid loader but try to load nonexistent image
    with tempfile.TemporaryDirectory() as temp_dir:
        priors_root = Path(temp_dir) / "priors"
        priors_root.mkdir()

        loader = PriorsLoader(str(priors_root))

        with pytest.raises(FileNotFoundError):
            loader.load_priors("nonexistent_image.jpg", split="val")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
