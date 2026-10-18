"""--resume with a bad path must fail fast instead of silently training from scratch."""
from pathlib import Path

import pytest

import train


@pytest.mark.parametrize("fn", [train.train_cart, train.train_jaw, train.train_phishing])
def test_missing_resume_dir_returns_false(fn, tmp_path):
    ok = fn(
        data_dir=tmp_path / "no_data",
        checkpoint_dir=tmp_path / "ckpt",
        device="cpu",
        workers=1,
        estimate_only=False,
        resume=tmp_path / "does_not_exist",
        verbose=False,
        config_overrides={},
        runs_dir=tmp_path / "runs",
    )
    assert ok is False


def test_resume_dir_missing_file_returns_false(tmp_path):
    (tmp_path / "neural_model.pt").write_bytes(b"")  # fusion_model.json absent
    assert train._check_resume(tmp_path, train.CART_RESUME_FILES, "CART") is False


def test_resume_ok_and_none(tmp_path):
    (tmp_path / "jaw_model.pt").write_bytes(b"")
    assert train._check_resume(tmp_path, train.JAW_RESUME_FILES, "JAW") is True
    assert train._check_resume(None, train.JAW_RESUME_FILES, "JAW") is True


def test_all_bad_resume_returns_false(tmp_path):
    ok = train.train_all_parallel(
        data_dir=tmp_path, checkpoint_dir=tmp_path, device="cpu", workers=1,
        estimate_only=False, resume_cart=tmp_path / "cart", resume_jaw=tmp_path / "jaw",
        verbose=False, runs_dir=tmp_path / "runs",
    )
    assert ok is False
