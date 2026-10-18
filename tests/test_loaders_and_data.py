"""
Unit tests for enhanced data loaders, label normalization,
auto-partitioning, and sample dataset generation.
"""

from pathlib import Path
from PIL import Image

from training.data import load_image_split, load_text_split, load_splits
from util.datasets.dataset_config import DatasetConfig
from util.loaders.cart import CARTDatasetLoader, normalize_text_label
from util.loaders.jaw import JAWDatasetLoader, normalize_image_label
from util.manager import SafeStepDataManager


def test_label_normalization():
    # JAW image labels
    assert normalize_image_label("real") == "authentic"
    assert normalize_image_label("REAL") == "authentic"
    assert normalize_image_label("nature") == "authentic"
    assert normalize_image_label("authentic") == "authentic"
    assert normalize_image_label("fake") == "ai_generated"
    assert normalize_image_label("FAKE") == "ai_generated"
    assert normalize_image_label("ai") == "ai_generated"
    assert normalize_image_label("synthetic") == "ai_generated"
    assert normalize_image_label("deepfake") == "ai_generated"

    # CART text labels
    assert normalize_text_label("human") == "human"
    assert normalize_text_label("real") == "human"
    assert normalize_text_label("0") == "human"
    assert normalize_text_label("chatgpt") == "ai_generated"
    assert normalize_text_label("ai_generated") == "ai_generated"
    assert normalize_text_label("fake") == "ai_generated"
    assert normalize_text_label("1") == "ai_generated"


def test_jaw_sample_dataset_generation(tmp_path: Path):
    target = JAWDatasetLoader.create_sample_dataset(tmp_path / "sample_jaw", samples_per_class=10)
    assert (target / "train" / "authentic").exists()
    assert (target / "train" / "ai_generated").exists()
    assert (target / "val" / "authentic").exists()
    assert (target / "val" / "ai_generated").exists()

    train_auth = list((target / "train" / "authentic").glob("*.png"))
    val_auth = list((target / "val" / "authentic").glob("*.png"))
    assert len(train_auth) == 10
    assert len(val_auth) >= 2


def test_cart_sample_dataset_generation(tmp_path: Path):
    target = CARTDatasetLoader.create_sample_dataset(tmp_path / "sample_cart", samples_per_class=20)
    assert (target / "train" / "texts.txt").exists()
    assert (target / "train" / "labels.txt").exists()
    assert (target / "val" / "texts.txt").exists()
    assert (target / "val" / "labels.txt").exists()

    tr_texts = (target / "train" / "texts.txt").read_text().splitlines()
    assert len(tr_texts) == 40  # 20 human + 20 ai


def test_load_image_split_with_cifake_structure(tmp_path: Path):
    # Simulate Kaggle CIFAKE: train/REAL and train/FAKE
    train_dir = tmp_path / "train"
    real_dir = train_dir / "REAL"
    fake_dir = train_dir / "FAKE"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    img = Image.new("RGB", (32, 32), color="red")
    img.save(real_dir / "0001.jpg")
    img.save(real_dir / "0002.jpg")
    img.save(fake_dir / "0001.jpg")

    paths, labels = load_image_split(tmp_path, "train")
    assert len(paths) == 3
    assert labels.count("authentic") == 2
    assert labels.count("ai_generated") == 1


def test_load_image_split_unpartitioned_auto_split(tmp_path: Path):
    # Flat directory without train/val folders: real/ and fake/
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    img = Image.new("RGB", (32, 32), color="blue")
    for i in range(10):
        img.save(real_dir / f"r_{i}.png")
        img.save(fake_dir / f"f_{i}.png")

    train_p, train_l = load_image_split(tmp_path, "train")
    val_p, val_l = load_image_split(tmp_path, "val")

    assert len(train_p) == 16
    assert len(val_p) == 4
    # No path overlap
    assert set(train_p).isdisjoint(set(val_p))


def test_safestep_data_manager_prepare_sample_data(tmp_path: Path):
    out = SafeStepDataManager.prepare_sample_data(model="all", target_dir=tmp_path / "data")
    assert "jaw" in out
    assert "cart" in out
    assert "phishing" in out

    # Verify JAW can be loaded by load_splits
    jaw_tr_x, jaw_tr_y, jaw_va_x, jaw_va_y = load_splits("jaw", tmp_path / "data" / "jaw")
    assert len(jaw_tr_x) > 0
    assert len(jaw_va_x) > 0

    # Verify CART can be loaded by load_splits
    cart_tr_x, cart_tr_y, cart_va_x, cart_va_y = load_splits("cart", tmp_path / "data" / "cart")
    assert len(cart_tr_x) > 0
    assert len(cart_va_x) > 0


def test_load_text_split_no_data_without_demo_returns_empty(tmp_path):
    for split in ("train", "val"):
        texts, labels = load_text_split(tmp_path, split)
        assert texts == [] and labels == []
        texts, labels = load_text_split(tmp_path / "missing", split, allow_demo=False)
        assert texts == [] and labels == []


def test_load_text_split_no_data_with_demo_uses_placeholders(tmp_path):
    texts, labels = load_text_split(tmp_path, "train", allow_demo=True)
    assert len(texts) == len(labels) > 0
    assert set(labels) <= {"human", "ai_generated"}
