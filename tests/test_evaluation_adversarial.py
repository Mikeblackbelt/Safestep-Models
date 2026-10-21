"""
Adversarial stress tests and analytical oracles for Core Evaluation Metrics Engine (training/evaluation.py).
Constructed by challenger_m1_1 to stress-test edge cases, boundary conditions,
and verify mathematical equivalence against analytical oracles.
"""

import math
import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, auc

from training.evaluation import (
    compute_evaluation_metrics,
    export_roc_curve,
    render_ascii_roc_curve,
)


# =============================================================================
# 1. Stress Tests: Identical Scores & Ties
# =============================================================================

def test_adversarial_identical_scores_all_zeros():
    """Identical scores of 0.0 for binary labels should yield AUROC = 0.5."""
    y_true = [0, 1, 0, 1]
    y_pred = [0.0, 0.0, 0.0, 0.0]
    metrics = compute_evaluation_metrics(y_true, y_pred)
    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_adversarial_identical_scores_all_ones():
    """Identical scores of 1.0 for binary labels should yield AUROC = 0.5."""
    y_true = [0, 1, 0, 1]
    y_pred = [1.0, 1.0, 1.0, 1.0]
    metrics = compute_evaluation_metrics(y_true, y_pred)
    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_adversarial_identical_scores_constant_float():
    """Constant float scores of 0.50 or any other float should yield AUROC = 0.5."""
    y_true = [0, 1, 0, 1]
    for c in [0.5, 0.25, 0.75, 0.99]:
        metrics = compute_evaluation_metrics(y_true, [c, c, c, c])
        assert metrics["auroc"] == pytest.approx(0.5)


# =============================================================================
# 2. Stress Tests: Inverted Probabilities (Adversarial Classifiers)
# =============================================================================

def test_adversarial_inverted_probabilities_continuous():
    """Completely inverted continuous probabilities must yield AUROC = 0.0, Accuracy = 0.0."""
    y_true = [0, 0, 1, 1]
    y_pred = [0.95, 0.85, 0.15, 0.05]
    metrics = compute_evaluation_metrics(y_true, y_pred)
    assert metrics["auroc"] == pytest.approx(0.0)
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


def test_adversarial_inverted_probabilities_discrete():
    """Completely inverted discrete labels must yield AUROC = 0.0, Accuracy = 0.0."""
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    metrics = compute_evaluation_metrics(y_true, y_pred)
    assert metrics["auroc"] == pytest.approx(0.0)
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


# =============================================================================
# 3. Stress Tests: Empty & Single-Class Arrays
# =============================================================================

def test_adversarial_empty_arrays():
    """Empty arrays must raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_evaluation_metrics([], [])


def test_adversarial_single_sample_array():
    """A dataset of size N=1 must trigger single-class protection without crash."""
    with pytest.warns(UserWarning, match="Only one class present in y_true"):
        metrics = compute_evaluation_metrics([1], [0.8])
    assert math.isnan(metrics["auroc"])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_curve"]["fpr"] == []


def test_adversarial_single_class_multiclass_label():
    """A dataset containing only one class ID (e.g. class 2) should return NaN AUROC."""
    with pytest.warns(UserWarning, match="Only one class present in y_true"):
        metrics = compute_evaluation_metrics(
            [2, 2, 2],
            [[0.1, 0.1, 0.8], [0.2, 0.1, 0.7], [0.1, 0.2, 0.7]]
        )
    assert math.isnan(metrics["auroc"])
    assert metrics["accuracy"] == 1.0


# =============================================================================
# 4. Stress Tests: Extreme Sizes & Class Imbalance
# =============================================================================

def test_adversarial_extreme_scale_and_severe_imbalance():
    """Tests 10,000 samples with 1:9999 class imbalance."""
    y_true = np.array([1] + [0] * 9999)
    scores = np.linspace(0, 0.5, 10000)
    scores[0] = 0.99  # positive sample has highest score

    metrics = compute_evaluation_metrics(y_true, scores)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert not math.isnan(metrics["auroc"])


# =============================================================================
# 5. Mathematical Oracles: Mann-Whitney U & Balanced Accuracy Equivalence
# =============================================================================

def test_adversarial_oracle_mann_whitney_u_equivalence():
    """
    AUROC must mathematically equal the normalized Wilcoxon-Mann-Whitney U statistic:
    U = sum_{i in Pos, j in Neg} [I(s_i > s_j) + 0.5 * I(s_i == s_j)] / (N_pos * N_neg).
    """
    def mann_whitney_u_auroc(y_true, scores, pos_label=1):
        y_true_arr = np.asarray(y_true)
        scores_arr = np.asarray(scores)
        pos = scores_arr[y_true_arr == pos_label]
        neg = scores_arr[y_true_arr != pos_label]
        wins = 0.0
        for p in pos:
            for n in neg:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        return wins / (len(pos) * len(neg))

    np.random.seed(42)
    for _ in range(25):
        n = np.random.randint(15, 60)
        y_true = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
        if len(np.unique(y_true)) < 2:
            continue
        scores = np.random.uniform(0, 1, size=n)
        metrics = compute_evaluation_metrics(y_true, scores)
        oracle_auc = mann_whitney_u_auroc(y_true, scores)
        assert metrics["auroc"] == pytest.approx(oracle_auc, abs=1e-12)


def test_adversarial_oracle_balanced_accuracy_equivalence():
    """
    For discrete binary predictions, trapezoidal AUROC must strictly equal Balanced Accuracy:
    AUROC_discrete = (TPR + TNR) / 2 = Balanced Accuracy.
    """
    np.random.seed(123)
    for _ in range(25):
        n = np.random.randint(20, 80)
        y_true = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
        if len(np.unique(y_true)) < 2:
            continue
        y_pred = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
        metrics = compute_evaluation_metrics(y_true, y_pred)
        expected_bacc = balanced_accuracy_score(y_true, y_pred)
        assert metrics["auroc"] == pytest.approx(expected_bacc, abs=1e-12)


# =============================================================================
# 6. Stress Tests: Visualization Edge Cases
# =============================================================================

def test_adversarial_ascii_extreme_dimensions():
    """ASCII renderer should handle small 1x1 dimensions gracefully."""
    roc_data = {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0]}
    rendered = render_ascii_roc_curve(roc_data, width=1, height=1)
    assert "TPR ^" in rendered
    assert "*" in rendered


# =============================================================================
# 7. Documented Vulnerabilities / Bug Confirmation Tests (Marked XFAIL)
# =============================================================================

@pytest.mark.xfail(
    reason="Bug 1: Non-contiguous or 1-indexed multi-class discrete labels crash "
           "with ValueError: Number of classes in y_true not equal to the number of columns in 'y_score'"
)
def test_adversarial_multiclass_non_contiguous_labels_failure():
    """Exposes Bug 1: 1-indexed discrete labels [1, 2, 3] crash multi-class evaluation."""
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    # Current implementation raises ValueError due to one-hot dimension mismatch
    metrics = compute_evaluation_metrics(y_true, y_pred, multi_class="ovr")
    assert metrics["auroc"] == 1.0


@pytest.mark.xfail(
    reason="Bug 2: Multi-class prediction matrix (N, C) evaluated on a slice with only 2 classes "
           "crashes with ValueError: Target is multiclass but average='binary'"
)
def test_adversarial_multiclass_shape_with_two_classes_in_batch_failure():
    """Exposes Bug 2: 3-class probability matrix evaluated when batch only contains 2 classes crashes."""
    y_true = [0, 1, 0, 1]
    y_pred_3col = [
        [0.1, 0.1, 0.8],  # predicts class 2
        [0.1, 0.7, 0.2],
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
    ]
    # Current implementation takes binary branch and crashes in precision_recall_fscore_support
    metrics = compute_evaluation_metrics(y_true, y_pred_3col)
    assert 0.0 <= metrics["accuracy"] <= 1.0


@pytest.mark.xfail(
    reason="Bug 3: When pos_label=0 in binary classification, metrics['auroc'] is 0.0 while "
           "roc_curve coordinates have AUC = 1.0 (inverted AUROC / internal discrepancy)"
)
def test_adversarial_binary_pos_label_zero_inversion_discrepancy():
    """Exposes Bug 3: pos_label=0 causes internal contradiction between auroc and roc_curve."""
    y_true = [0, 0, 1, 1]
    # Col 0 is prob(class 0), col 1 is prob(class 1)
    y_pred_2d = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
    ])
    # Model is perfect for class 0
    metrics = compute_evaluation_metrics(y_true, y_pred_2d, pos_label=0)
    curve_auc = auc(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"])
    # Both auroc and curve_auc should be 1.0, and they should match
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(curve_auc)


@pytest.mark.xfail(
    reason="Bug 4: 2D prediction array of shape (N, 1) raises IndexError on prob_col extraction"
)
def test_adversarial_2d_single_column_index_error():
    """Exposes Bug 4: 2D single-column predictions (N, 1) crash with IndexError."""
    y_true = [0, 1, 0, 1]
    y_pred_1col = np.array([[0.2], [0.8], [0.3], [0.9]])
    metrics = compute_evaluation_metrics(y_true, y_pred_1col, pos_label=1)
    assert 0.0 <= metrics["auroc"] <= 1.0


@pytest.mark.xfail(
    reason="Bug 5: Multi-class AUROC and ROC curve calculation has no try/except protection against NaN/Inf"
)
def test_adversarial_multiclass_unhandled_nan_exception():
    """Exposes Bug 5: Multi-class probabilities with NaN crash with unhandled ValueError."""
    y_true = [0, 1, 2]
    y_pred_nan = [
        [0.8, 0.1, 0.1],
        [0.1, np.nan, 0.1],
        [0.1, 0.1, 0.8],
    ]
    # In binary, this logs warning and returns NaN; in multi-class it crashes with uncaught ValueError
    metrics = compute_evaluation_metrics(y_true, y_pred_nan, multi_class="ovr")
    assert math.isnan(metrics["auroc"])
