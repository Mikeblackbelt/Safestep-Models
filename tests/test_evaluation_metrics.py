"""
Unit tests for Core Evaluation Metrics Engine (training/evaluation.py).
"""

import json
import math
import os
import tempfile
import numpy as np
import pytest

from training.evaluation import (
    compute_evaluation_metrics,
    export_roc_curve,
    render_ascii_roc_curve,
)


def test_binary_continuous_probabilities():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_evaluation_metrics(y_true, y_pred, pos_label=1)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
    assert len(metrics["roc_curve"]["fpr"]) > 0
    assert len(metrics["roc_curve"]["tpr"]) > 0
    assert len(metrics["roc_curve"]["thresholds"]) > 0


def test_binary_discrete_fallback_trapezoidal():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])  # discrete {0, 1}
    metrics = compute_evaluation_metrics(y_true, y_pred, pos_label=1)

    # TPR = 2/3, FPR = 1/2
    # Sensitivity = 2/3, Specificity = 1/2
    # Balanced Accuracy = (2/3 + 1/2) / 2 = 7/12 = 0.5833333333333333
    expected_auc = 7.0 / 12.0
    assert metrics["auroc"] == pytest.approx(expected_auc)
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert metrics["accuracy"] == pytest.approx(3.0 / 5.0)


def test_single_class_edge_case_protection():
    y_true_all_pos = [1, 1, 1, 1]
    y_pred_probs = [0.9, 0.8, 0.7, 0.6]

    with pytest.warns(UserWarning, match="Only one class present in y_true"):
        metrics = compute_evaluation_metrics(y_true_all_pos, y_pred_probs, pos_label=1)

    assert math.isnan(metrics["auroc"])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_curve"]["fpr"] == []
    assert metrics["roc_curve"]["tpr"] == []
    assert metrics["roc_curve"]["thresholds"] == []

    y_true_all_neg = [0, 0, 0]
    y_pred_disc = [0, 0, 0]
    with pytest.warns(UserWarning, match="Only one class present in y_true"):
        metrics_neg = compute_evaluation_metrics(y_true_all_neg, y_pred_disc, pos_label=1)
    assert math.isnan(metrics_neg["auroc"])


def test_input_validation_errors():
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_evaluation_metrics([], [])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_evaluation_metrics([0, 1], [0.5])


def test_multi_class_ovr():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred_2d = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ]
    metrics = compute_evaluation_metrics(y_true, y_pred_2d, multi_class="ovr")
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert len(metrics["roc_curve"]["fpr"]) > 0


def test_multi_class_ovo():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred_2d = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ]
    metrics = compute_evaluation_metrics(y_true, y_pred_2d, multi_class="ovo")
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0


def test_multi_class_discrete_labels():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred_disc = [0, 1, 2, 0, 2, 2]
    metrics = compute_evaluation_metrics(y_true, y_pred_disc, multi_class="ovr")
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_export_roc_curve_json_and_csv():
    roc_data = {
        "fpr": [0.0, 0.1, 0.5, 1.0],
        "tpr": [0.0, 0.7, 0.9, 1.0],
        "thresholds": [1.0, 0.8, 0.4, 0.0],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "subdir", "roc.json")
        csv_path = os.path.join(tmpdir, "subdir", "roc.csv")

        out_json = export_roc_curve(roc_data, json_path, format="json")
        assert os.path.exists(out_json)
        with open(out_json) as f:
            data = json.load(f)
            assert data["fpr"] == [0.0, 0.1, 0.5, 1.0]

        out_csv = export_roc_curve(roc_data, csv_path, format="csv")
        assert os.path.exists(out_csv)
        with open(out_csv) as f:
            lines = [line.strip() for line in f.readlines()]
            assert lines[0] == "fpr,tpr,threshold"
            assert lines[1] == "0.0,0.0,1.0"

        with pytest.raises(ValueError, match="Unsupported format"):
            export_roc_curve(roc_data, os.path.join(tmpdir, "roc.txt"), format="txt")


def test_render_ascii_roc_curve():
    roc_data = {
        "fpr": [0.0, 0.1, 0.3, 1.0],
        "tpr": [0.0, 0.6, 0.85, 1.0],
        "thresholds": [1.0, 0.8, 0.5, 0.0],
    }
    rendered = render_ascii_roc_curve(roc_data, width=30, height=10)
    assert "TPR ^" in rendered
    assert "1.0 |" in rendered
    assert "0.0 |" in rendered
    assert "*" in rendered
    assert "." in rendered

    empty_rendered = render_ascii_roc_curve({"fpr": [], "tpr": []})
    assert "unavailable" in empty_rendered

    non_finite = render_ascii_roc_curve({"fpr": [float("nan")], "tpr": [float("nan")]})
    assert "unavailable" in non_finite


def test_phishing_evaluate_and_predict_proba():
    from phishing.config import PhishingConfig
    from phishing.train import PhishingTrainer

    trainer = PhishingTrainer(PhishingConfig(use_warhammer=False))
    train_texts = ["URGENT: confirm password at http://evil.com", "Team meeting at 2pm on Thursday"] * 10
    train_labels = ["phishing", "legitimate"] * 10
    val_texts = ["Your account is suspended: click http://phish.net", "Quarterly financial review slides"] * 5
    val_labels = ["phishing", "legitimate"] * 5

    trainer.fit(train_texts, train_labels, val_texts, val_labels)

    # Test trainer.evaluate
    metrics = trainer.evaluate(val_texts, val_labels)
    assert "auroc" in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert "accuracy" in metrics
    assert "roc_curve" in metrics
    assert len(metrics["roc_curve"]["fpr"]) > 0

    # Test detector.predict_proba
    probs = trainer.detector.predict_proba(val_texts)
    assert probs.shape == (len(val_texts), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Test detector.evaluate
    det_metrics = trainer.detector.evaluate(val_texts, val_labels)
    assert "auroc" in det_metrics
    assert det_metrics["accuracy"] == metrics["accuracy"]


def test_cart_evaluate_and_predict_proba():
    from cart.config import CARTConfig
    from cart.detector import CARTDetector
    from cart.train import CARTTrainer

    config = CARTConfig()
    detector = CARTDetector(config=config, device="cpu")
    texts = [
        "Natural human writing with typical sentence length variance and simple style.",
        "Delve into the comprehensive overview of pivotal artificial intelligence mechanisms.",
    ]
    labels = ["human", "ai_generated"]

    # Test detector.predict_proba
    probs = detector.predict_proba(texts)
    assert probs.shape == (2, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    # Test detector.evaluate
    metrics = detector.evaluate(texts, labels)
    assert "accuracy" in metrics
    assert "auroc" in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0

    # Test trainer.evaluate
    trainer = CARTTrainer(config=config, device="cpu")
    trainer_metrics = trainer.evaluate(texts, labels)
    assert "accuracy" in trainer_metrics
    assert "auroc" in trainer_metrics


def test_jaw_evaluate_and_predict_proba(tmp_path):
    from jaw.config import JAWConfig
    from jaw.detector import JAWDetector
    from jaw.train import JAWTrainer
    from tests.generate_test_assets import generate_test_assets

    auth_path, syn_path = generate_test_assets(tmp_path)
    paths = [auth_path, syn_path]
    labels = ["authentic", "ai_generated"]

    config = JAWConfig()
    detector = JAWDetector(config=config, device="cpu")

    # Test detector.predict_proba
    probs = detector.predict_proba(paths)
    assert probs.shape == (2, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    # Test detector.evaluate
    metrics = detector.evaluate(paths, labels)
    assert "accuracy" in metrics
    assert "auroc" in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0

    # Test trainer.evaluate
    trainer = JAWTrainer(config=config, device="cpu")
    trainer_metrics = trainer.evaluate(paths, labels)
    assert "accuracy" in trainer_metrics
    assert "auroc" in trainer_metrics

