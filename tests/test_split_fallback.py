from pathlib import Path

import pytest
from PIL import Image

from training.data import _resolve_split_dir, load_image_split


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(path)


def _make(root: Path, split: str, n: int = 2) -> None:
    for cls in ("REAL", "FAKE"):
        for i in range(n):
            _png(root / split / cls / f"{i}.png")


def test_val_falls_back_to_test(tmp_path):
    _make(tmp_path, "train", 3)
    _make(tmp_path, "test", 2)
    vx, vy = load_image_split(tmp_path, "val")
    assert len(vx) == 4
    assert all(Path(p).parent.parent.name == "test" for p in vx)
    assert set(vy) == {"authentic", "ai_generated"}
    tx, _ = load_image_split(tmp_path, "train")
    assert len(tx) == 6


@pytest.mark.parametrize("name", ["valid", "validation"])
def test_val_aliases(tmp_path, name):
    _make(tmp_path, name)
    _make(tmp_path, "test")
    assert _resolve_split_dir(tmp_path, "jaw", "val") == tmp_path / name


def test_real_val_preferred_over_test(tmp_path):
    _make(tmp_path, "val")
    _make(tmp_path, "test")
    assert _resolve_split_dir(tmp_path, "jaw", "val") == tmp_path / "val"


def test_nested_cifake_test(tmp_path):
    _make(tmp_path / "cifake", "test")
    assert _resolve_split_dir(tmp_path, "jaw", "val") == tmp_path / "cifake" / "test"


def test_train_does_not_fall_back(tmp_path):
    _make(tmp_path, "test")
    assert _resolve_split_dir(tmp_path, "jaw", "train") is None


def test_jaw_loader_includes_test_images(tmp_path):
    from util.loaders.jaw import JAWDatasetLoader

    root = tmp_path / "cifake"
    _make(root, "train", 3)
    _make(root, "test", 2)
    ds = JAWDatasetLoader().load_cifake(root)
    paths = [str(p) for p in ds.image_paths]
    assert len(paths) == 10
    assert any("test" in Path(p).parts for p in paths)
