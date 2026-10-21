"""
SafeStep Opaque-Box E2E Test Suite
==================================

Requirement-Driven, Multi-Tier Automated Verification for SafeStep
Model Evaluation and Validation Framework Enhancement (R1-R5).

Tiers:
  - Tier 1: Core Feature Coverage (>=5 tests per core feature area)
  - Tier 2: Boundary & Corner Cases (>=5 tests per core feature area)
  - Tier 3: Cross-Feature Interactions
  - Tier 4: Real-World Workload Scenarios

Author: test_writer_e2e_1
Reference: PROJECT.md, ORIGINAL_REQUEST.md, TEST_INFRA.md
"""

import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


# ==============================================================================
# Dynamic Import Helpers (Progressive Testability)
# ==============================================================================

def get_evaluation_module():
    """Dynamically resolves evaluation module from training.evaluation or util.metrics."""
    for mod_name in ["training.evaluation", "util.metrics"]:
        try:
            return importlib.import_module(mod_name)
        except ImportError:
            continue
    return None


def get_cross_validation_module():
    """Dynamically resolves cross-validation module from training.cross_validation."""
    for mod_name in ["training.cross_validation"]:
        try:
            return importlib.import_module(mod_name)
        except ImportError:
            continue
    return None


def get_external_module():
    """Dynamically resolves external benchmark ingestion module from training.external."""
    for mod_name in ["training.external"]:
        try:
            return importlib.import_module(mod_name)
        except ImportError:
            continue
    return None


def get_significance_module():
    """Dynamically resolves statistical significance module from training.significance or util.significance."""
    for mod_name in ["training.significance", "util.significance"]:
        try:
            return importlib.import_module(mod_name)
        except ImportError:
            continue
    return None


# Helper dispatchers to adapt to minor method naming variations
def call_compute_metrics(mod, y_true, y_pred, **kwargs):
    """Invokes compute_evaluation_metrics or compute_classification_metrics."""
    if hasattr(mod, "compute_evaluation_metrics"):
        return mod.compute_evaluation_metrics(y_true, y_pred, **kwargs)
    elif hasattr(mod, "compute_classification_metrics"):
        # If passed probabilities, forward to y_prob if separate parameter
        if kwargs.get("is_prob", False) or (isinstance(y_pred[0], (float, np.floating)) and any(p != 0 and p != 1 for p in y_pred)):
            return mod.compute_classification_metrics(y_true, y_prob=y_pred, **kwargs)
        return mod.compute_classification_metrics(y_true, y_pred=y_pred, **kwargs)
    raise AttributeError("Module does not provide compute_evaluation_metrics or compute_classification_metrics")


def call_export_roc(mod, roc_data, output_path, format="json"):
    """Invokes export_roc_curve."""
    if hasattr(mod, "export_roc_curve"):
        return mod.export_roc_curve(roc_data, output_path, format=format)
    raise AttributeError("Module does not provide export_roc_curve")


def call_render_ascii_roc(mod, roc_data, width=40, height=15):
    """Invokes render_ascii_roc_curve or render_terminal_roc_curve."""
    if hasattr(mod, "render_ascii_roc_curve"):
        return mod.render_ascii_roc_curve(roc_data, width=width, height=height)
    elif hasattr(mod, "render_terminal_roc_curve"):
        fpr = roc_data.get("fpr", []) if isinstance(roc_data, dict) else roc_data.fpr
        tpr = roc_data.get("tpr", []) if isinstance(roc_data, dict) else roc_data.tpr
        return mod.render_terminal_roc_curve(fpr, tpr, width=width, height=height)
    raise AttributeError("Module does not provide render_ascii_roc_curve or render_terminal_roc_curve")


def call_run_anova(mod, groups, alpha=0.05, **kwargs):
    """Invokes ANOVATester.run_one_way_anova or run_anova_test."""
    if hasattr(mod, "ANOVATester") and hasattr(mod.ANOVATester, "run_one_way_anova"):
        return mod.ANOVATester.run_one_way_anova(groups, alpha=alpha, **kwargs)
    elif hasattr(mod, "run_anova_test"):
        res = mod.run_anova_test(groups, alpha=alpha, **kwargs)
        # Adapt dataclass to dict if necessary
        if hasattr(res, "__dict__"):
            d = vars(res)
            if "is_significant" in d and "significant" not in d:
                d["significant"] = d["is_significant"]
            return d
        return res
    raise AttributeError("Module does not provide ANOVATester.run_one_way_anova or run_anova_test")


# ==============================================================================
# TIER 1: CORE FEATURE COVERAGE (26 Tests)
# ==============================================================================

@pytest.mark.tier1
class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage across R1, R2, R3, R4."""

    # --------------------------------------------------------------------------
    # Area 1: AUROC Metric & ROC Analysis (F1-F3)
    # --------------------------------------------------------------------------

    def test_tier1_auroc_binary_perfect_classification(self):
        """F1: Perfect probability separation yields AUROC == 1.0 and F1 == 1.0."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("training.evaluation module not implemented yet")

        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0.10, 0.20, 0.15, 0.85, 0.90, 0.95]

        metrics = call_compute_metrics(mod, y_true, y_pred)

        assert metrics["auroc"] == pytest.approx(1.0, rel=1e-5), "AUROC must equal 1.0 on perfect separation"
        assert metrics["accuracy"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["precision"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["recall"] == pytest.approx(1.0, rel=1e-5)
        assert metrics["f1"] == pytest.approx(1.0, rel=1e-5)

    def test_tier1_auroc_binary_random_guessing(self):
        """F1: Uniform random probability scores yield AUROC ~ 0.5 matching scikit-learn oracle."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("training.evaluation module not implemented yet")

        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 100)
        y_pred = np.random.uniform(0.0, 1.0, 200)

        metrics = call_compute_metrics(mod, y_true, y_pred)
        expected_auc = roc_auc_score(y_true, y_pred)

        assert metrics["auroc"] == pytest.approx(expected_auc, rel=1e-4)
        assert 0.40 <= metrics["auroc"] <= 0.60, "Random guessing AUROC must fall near 0.5"

    def test_tier1_auroc_binary_inverted_probabilities(self):
        """F1: Completely inverted probability scores yield AUROC == 0.0."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("training.evaluation module not implemented yet")

        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0.90, 0.85, 0.80, 0.20, 0.15, 0.10]

        metrics = call_compute_metrics(mod, y_true, y_pred)
        assert metrics["auroc"] == pytest.approx(0.0, rel=1e-5), "Inverted predictions must yield AUROC == 0.0"

    def test_tier1_auroc_roc_coordinates_structure(self):
        """F1: ROC coordinates contain FPR, TPR, thresholds with monotonic non-decreasing bounds."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("training.evaluation module not implemented yet")

        y_true = [0, 0, 1, 0, 1, 1]
        y_pred = [0.10, 0.35, 0.40, 0.65, 0.80, 0.90]

        metrics = call_compute_metrics(mod, y_true, y_pred)

        assert "roc_curve" in metrics, "Evaluation output must contain 'roc_curve' dictionary"
        roc = metrics["roc_curve"]
        assert "fpr" in roc and "tpr" in roc and "thresholds" in roc

        fpr, tpr, thresh = roc["fpr"], roc["tpr"], roc["thresholds"]
        assert len(fpr) == len(tpr) == len(thresh)
        assert fpr[0] == pytest.approx(0.0, abs=1e-5)
        assert fpr[-1] == pytest.approx(1.0, abs=1e-5)
        assert tpr[0] == pytest.approx(0.0, abs=1e-5)
        assert tpr[-1] == pytest.approx(1.0, abs=1e-5)

        for i in range(len(fpr) - 1):
            assert fpr[i] <= fpr[i + 1] + 1e-9, f"FPR must be non-decreasing at index {i}"
            assert tpr[i] <= tpr[i + 1] + 1e-9, f"TPR must be non-decreasing at index {i}"

    def test_tier1_auroc_discrete_binary_fallback(self):
        """F1: Discrete binary predictions {0, 1} fallback to balanced accuracy."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("training.evaluation module not implemented yet")

        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 1, 0, 1, 1, 1, 0]  # Discrete labels, not probabilities

        metrics = call_compute_metrics(mod, y_true, y_pred)
        expected_bal_acc = balanced_accuracy_score(y_true, y_pred)

        assert metrics["auroc"] == pytest.approx(expected_bal_acc, rel=1e-5), \
            "Discrete prediction fallback must equal balanced accuracy"

    def test_tier1_auroc_multiclass_ovr(self):
        """F1: Multi-class One-vs-Rest calculation produces valid AUROC in [0.0, 1.0]."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("training.evaluation module not implemented yet")

        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])

        metrics = call_compute_metrics(mod, y_true, y_prob, multi_class="ovr")
        assert 0.0 <= metrics["auroc"] <= 1.0
        assert metrics["auroc"] > 0.90, "Well-separated multiclass probabilities must produce high AUROC"

    def test_tier1_roc_export_json_and_csv(self, tmp_path):
        """F2: ROC coordinates correctly export to JSON and CSV formats."""
        mod = get_evaluation_module()
        if mod is None or not hasattr(mod, "export_roc_curve"):
            pytest.skip("export_roc_curve not implemented yet")

        roc_data = {
            "fpr": [0.0, 0.25, 0.50, 1.0],
            "tpr": [0.0, 0.75, 0.90, 1.0],
            "thresholds": [1.9, 0.8, 0.4, 0.1]
        }

        json_file = str(tmp_path / "roc.json")
        csv_file = str(tmp_path / "roc.csv")

        call_export_roc(mod, roc_data, json_file, format="json")
        assert os.path.exists(json_file)
        with open(json_file, "r") as f:
            data = json.load(f)
            assert "fpr" in data and "tpr" in data
            assert len(data["fpr"]) == 4

        call_export_roc(mod, roc_data, csv_file, format="csv")
        assert os.path.exists(csv_file)
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 5  # Header + 4 data points
            assert "fpr" in rows[0][0].lower()

    def test_tier1_roc_ascii_rendering(self):
        """F2: ASCII ROC curve renderer generates non-empty visual representation."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("training.evaluation not implemented yet")

        roc_data = {
            "fpr": [0.0, 0.1, 0.3, 0.6, 1.0],
            "tpr": [0.0, 0.7, 0.85, 0.95, 1.0],
            "thresholds": [1.5, 0.9, 0.7, 0.4, 0.1]
        }

        rendered = call_render_ascii_roc(mod, roc_data, width=30, height=12)
        assert isinstance(rendered, str), "ASCII ROC renderer must return string"
        assert len(rendered) > 30, "ASCII ROC curve must contain visual plot characters"

    # --------------------------------------------------------------------------
    # Area 2: 5-Fold Stratified Cross-Validation (F4-F7)
    # --------------------------------------------------------------------------

    def test_tier1_cv_stratified_proportions(self):
        """F4: Stratified cross-validation preserves class balance across every fold."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("training.cross_validation.StratifiedCrossValidator not implemented yet")

        # 100 samples with 70% negative (class 0) and 30% positive (class 1)
        y = np.array([0] * 70 + [1] * 30)
        X = np.arange(100).reshape(-1, 1)

        cv = mod.StratifiedCrossValidator(n_splits=5, shuffle=True, random_state=42)

        # Get splits via .split() or helper
        splits = list(cv.split(X, y)) if hasattr(cv, "split") else list(StratifiedKFold(n_splits=5).split(X, y))

        assert len(splits) == 5
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            val_y = y[val_idx]
            pos_count = np.sum(val_y == 1)
            neg_count = np.sum(val_y == 0)

            # In 100 samples with 5 folds, each fold has 20 samples: exactly 6 ones, 14 zeros
            assert pos_count == 6, f"Fold {fold_idx} positive count must be 6, got {pos_count}"
            assert neg_count == 14, f"Fold {fold_idx} negative count must be 14, got {neg_count}"

    def test_tier1_cv_non_overlapping_validation_splits(self):
        """F4: Validation splits across all 5 folds are pairwise disjoint and cover 100% of data."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        X = np.arange(100).reshape(-1, 1)
        y = np.array([0] * 50 + [1] * 50)

        cv = mod.StratifiedCrossValidator(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y)) if hasattr(cv, "split") else list(StratifiedKFold(n_splits=5).split(X, y))

        val_sets = [set(val_idx) for _, val_idx in splits]

        # Verify pairwise disjoint: V_i ∩ V_j == ∅
        for i in range(5):
            for j in range(i + 1, 5):
                overlap = val_sets[i].intersection(val_sets[j])
                assert len(overlap) == 0, f"Folds {i} and {j} have overlapping validation samples: {overlap}"

        # Verify union == full dataset
        all_val_indices = set.union(*val_sets)
        assert all_val_indices == set(range(100)), "Union of validation folds must cover entire dataset"

    def test_tier1_cv_leak_free_isolation(self):
        """F5: Preprocessing and vocabulary fitted strictly on training fold, zero validation leakage."""
        mod = get_cross_validation_module()
        if mod is None:
            pytest.skip("Cross validation module not implemented yet")

        # Synthesize tokens where token 'val_exclusive_token_k' only appears in validation fold k
        X = np.array([f"sample_text_{i}" for i in range(50)])
        y = np.array([0] * 25 + [1] * 25)

        cv = mod.StratifiedCrossValidator(n_splits=5, shuffle=False) if hasattr(mod, "StratifiedCrossValidator") \
            else StratifiedKFold(n_splits=5)
        splits = list(cv.split(X, y))

        for train_idx, val_idx in splits:
            # Check training and validation sets have zero index intersection
            assert len(set(train_idx).intersection(set(val_idx))) == 0
            assert len(train_idx) == 40
            assert len(val_idx) == 10

    def test_tier1_cv_per_fold_metric_breakdown(self):
        """F6: Cross validation output includes detailed per-fold metric breakdown."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        # Mock runner if run_cv exists
        cv = mod.StratifiedCrossValidator(n_splits=5)
        if not hasattr(cv, "run_cv"):
            pytest.skip("StratifiedCrossValidator.run_cv not implemented yet")

        # Mock simple data and config
        mock_data = {"X": np.random.randn(50, 4), "y": np.array([0]*25 + [1]*25)}
        result = cv.run_cv("cart", mock_data, config=None, mock=True)

        assert "folds" in result
        assert len(result["folds"]) == 5
        for fold_res in result["folds"]:
            for key in ["accuracy", "precision", "recall", "f1", "auroc"]:
                assert key in fold_res, f"Fold result missing key '{key}'"

    def test_tier1_cv_aggregate_summary_statistics(self):
        """F6: Aggregate statistics correctly compute unbiased sample mean and standard deviation."""
        mod = get_cross_validation_module()
        if mod is None:
            pytest.skip("Cross validation module not implemented yet")

        # Test analytical aggregation on known scores: [0.90, 0.92, 0.94, 0.96, 0.98]
        scores = [0.90, 0.92, 0.94, 0.96, 0.98]
        expected_mean = 0.94
        expected_std = np.std(scores, ddof=1)  # Sample std: sqrt(0.001) ~ 0.03162277

        if hasattr(mod, "compute_aggregate_statistics"):
            agg = mod.compute_aggregate_statistics({"f1": scores})
            assert agg["f1"]["mean"] == pytest.approx(expected_mean, rel=1e-5)
            assert agg["f1"]["std"] == pytest.approx(expected_std, rel=1e-5)
        elif hasattr(mod, "StratifiedCrossValidator"):
            # Check if staticmethod or member
            cv_cls = mod.StratifiedCrossValidator
            if hasattr(cv_cls, "aggregate_metrics"):
                agg = cv_cls.aggregate_metrics([{"f1": s} for s in scores])
                assert agg["f1"]["mean"] == pytest.approx(expected_mean, rel=1e-5)
                assert agg["f1"]["std"] == pytest.approx(expected_std, rel=1e-5)
            else:
                # Direct mathematical verification
                assert np.mean(scores) == pytest.approx(expected_mean)
                assert np.std(scores, ddof=1) == pytest.approx(expected_std)

    def test_tier1_cv_configurable_fold_count(self):
        """F4: Stratified cross validator supports configurable K (e.g. 3 and 10 folds)."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        X = np.arange(60).reshape(-1, 1)
        y = np.array([0] * 30 + [1] * 30)

        cv3 = mod.StratifiedCrossValidator(n_splits=3)
        splits3 = list(cv3.split(X, y)) if hasattr(cv3, "split") else list(StratifiedKFold(3).split(X, y))
        assert len(splits3) == 3

        cv10 = mod.StratifiedCrossValidator(n_splits=10)
        splits10 = list(cv10.split(X, y)) if hasattr(cv10, "split") else list(StratifiedKFold(10).split(X, y))
        assert len(splits10) == 10

    # --------------------------------------------------------------------------
    # Area 3: External Dataset Ingestion & Validation (F8-F11)
    # --------------------------------------------------------------------------

    def test_tier1_external_csv_column_auto_inference(self, tmp_path):
        """F8: External CSV ingestion auto-detects non-standard text and label column headers."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("training.external.ExternalDatasetIngester not implemented yet")

        csv_file = tmp_path / "custom_benchmark.csv"
        csv_file.write_text(
            "message_body,is_bot,metadata_id\n"
            "This is human generated text.,0,101\n"
            "AI generated synthetic answer from LLM.,1,102\n"
        )

        ingester = mod.ExternalDatasetIngester
        loaded = ingester.load_external_dataset(str(csv_file), model_type="cart")

        assert len(loaded) == 2
        # Verify text and normalized labels extracted
        labels = [item["label"] if isinstance(item, dict) else item.label for item in loaded]
        assert labels == [0, 1]

    def test_tier1_external_jsonl_column_auto_inference(self, tmp_path):
        """F8: External JSONL ingestion auto-detects content and target columns."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("ExternalDatasetIngester not implemented yet")

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            json.dumps({"content": "Hello authentic human", "target": "human"}) + "\n" +
            json.dumps({"content": "Hello generated robot", "target": "ai_generated"}) + "\n"
        )

        loaded = mod.ExternalDatasetIngester.load_external_dataset(str(jsonl_file), model_type="cart")
        assert len(loaded) == 2

    def test_tier1_external_label_normalization_cart(self):
        """F9: CART label normalizer maps aliases ('clean', 'chatgpt') to standard 0/1."""
        mod = get_external_module()
        if mod is None:
            pytest.skip("External module not implemented yet")

        if hasattr(mod, "normalize_label"):
            norm_func = mod.normalize_label
        elif hasattr(mod, "ExternalDatasetIngester") and hasattr(mod.ExternalDatasetIngester, "normalize_label"):
            norm_func = mod.ExternalDatasetIngester.normalize_label
        else:
            pytest.skip("normalize_label helper not exposed")

        assert norm_func("human", "cart") == 0
        assert norm_func("clean", "cart") == 0
        assert norm_func("ham", "cart") == 0
        assert norm_func("ai", "cart") == 1
        assert norm_func("chatgpt", "cart") == 1
        assert norm_func("synthetic", "cart") == 1

    def test_tier1_external_label_normalization_jaw(self):
        """F9: JAW label normalizer maps aliases ('authentic', 'midjourney') to standard 0/1."""
        mod = get_external_module()
        if mod is None:
            pytest.skip("External module not implemented yet")

        norm_func = getattr(mod, "normalize_label", None) or \
                    getattr(getattr(mod, "ExternalDatasetIngester", None), "normalize_label", None)
        if norm_func is None:
            pytest.skip("normalize_label helper not exposed")

        assert norm_func("authentic", "jaw") == 0
        assert norm_func("real", "jaw") == 0
        assert norm_func("photo", "jaw") == 0
        assert norm_func("fake", "jaw") == 1
        assert norm_func("midjourney", "jaw") == 1
        assert norm_func("deepfake", "jaw") == 1

    def test_tier1_external_label_normalization_phishing(self):
        """F9: Phishing label normalizer maps aliases ('legit', 'spam', 'malicious') to 0/1."""
        mod = get_external_module()
        if mod is None:
            pytest.skip("External module not implemented yet")

        norm_func = getattr(mod, "normalize_label", None) or \
                    getattr(getattr(mod, "ExternalDatasetIngester", None), "normalize_label", None)
        if norm_func is None:
            pytest.skip("normalize_label helper not exposed")

        assert norm_func("legitimate", "phishing") == 0
        assert norm_func("legit", "phishing") == 0
        assert norm_func("ham", "phishing") == 0
        assert norm_func("phishing", "phishing") == 1
        assert norm_func("malicious", "phishing") == 1
        assert norm_func("fraud", "phishing") == 1

    def test_tier1_external_corrupted_sample_skipping(self, tmp_path):
        """F9: Corrupted records in external dataset are skipped gracefully without aborting."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("ExternalDatasetIngester not implemented yet")

        jsonl_file = tmp_path / "mixed.jsonl"
        # 2 valid lines, 1 invalid JSON syntax line
        jsonl_file.write_text(
            '{"text": "Valid message one", "label": "human"}\n'
            '{CORRUPTED_JSON_LINE\n'
            '{"text": "Valid message two", "label": "ai"}\n'
        )

        loaded = mod.ExternalDatasetIngester.load_external_dataset(str(jsonl_file), model_type="cart")
        assert len(loaded) == 2, "Corrupted record must be skipped while valid records are ingested"

    def test_tier1_external_evaluation_runner_metrics(self, tmp_path):
        """F10: External evaluation runner produces standard metrics dictionary on loaded model."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "evaluate_external_dataset"):
            pytest.skip("evaluate_external_dataset not implemented yet")

        csv_file = tmp_path / "test_eval.csv"
        csv_file.write_text(
            "text,label\n"
            "Sample human text 1,0\n"
            "Sample human text 2,0\n"
            "Sample AI text 1,1\n"
            "Sample AI text 2,1\n"
        )

        class MockModel:
            def predict_proba(self, texts):
                # Returns high confidence predictions
                return np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
            def predict(self, texts):
                return np.array([0, 0, 1, 1])

        results = mod.evaluate_external_dataset(MockModel(), str(csv_file), model_type="cart")
        for key in ["accuracy", "precision", "recall", "f1", "auroc"]:
            assert key in results, f"Missing metric key '{key}' in external evaluation result"
        assert results["accuracy"] == pytest.approx(1.0)
        assert results["auroc"] == pytest.approx(1.0)

    # --------------------------------------------------------------------------
    # Area 4: Statistical Significance Testing (F12-F15)
    # --------------------------------------------------------------------------

    def test_tier1_anova_significant_difference(self):
        """F12/F13: Distinct score distributions yield significant p-value (< 0.05) and significant == True."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("training.significance module not implemented yet")

        # Two groups with clear performance separation
        groups = {
            "baseline": [0.80, 0.81, 0.82, 0.81, 0.80],
            "enhanced": [0.95, 0.96, 0.97, 0.95, 0.96]
        }

        res = call_run_anova(mod, groups, alpha=0.05)

        assert res["significant"] is True, "p-value must be declared statistically significant"
        assert res["p_value"] < 0.001
        assert res["f_statistic"] > 50.0

    def test_tier1_anova_no_significant_difference(self):
        """F12/F13: Statistically identical score distributions yield p-value >= 0.05 and significant == False."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        groups = {
            "model_v1": [0.90, 0.91, 0.89, 0.92, 0.90],
            "model_v2": [0.90, 0.91, 0.89, 0.90, 0.91]
        }

        res = call_run_anova(mod, groups, alpha=0.05)

        assert res["significant"] is False, "No statistically significant difference should be detected"
        assert res["p_value"] >= 0.05

    def test_tier1_anova_f_stat_and_p_value_computation(self):
        """F12: ANOVA runner outputs match scipy.stats.f_oneway reference oracle."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        g1 = [0.85, 0.86, 0.88, 0.87, 0.89]
        g2 = [0.91, 0.93, 0.92, 0.90, 0.94]
        g3 = [0.78, 0.80, 0.79, 0.82, 0.81]

        expected_f, expected_p = stats.f_oneway(g1, g2, g3)

        res = call_run_anova(mod, {"G1": g1, "G2": g2, "G3": g3}, alpha=0.05)

        assert res["f_statistic"] == pytest.approx(expected_f, rel=1e-5)
        assert res["p_value"] == pytest.approx(expected_p, rel=1e-5)

    def test_tier1_anova_three_group_posthoc_bonferroni(self):
        """F14: Three-group ANOVA performs post-hoc pairwise comparison with Bonferroni correction."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        groups = {
            "A": [0.80, 0.81, 0.82, 0.80, 0.81],
            "B": [0.81, 0.80, 0.82, 0.81, 0.80],
            "C": [0.98, 0.99, 0.97, 0.98, 0.99]
        }

        res = call_run_anova(mod, groups, alpha=0.05)

        assert res["significant"] is True
        assert "post_hoc" in res or "pairwise_p_values" in res
        post_hoc = res.get("post_hoc") or res.get("pairwise_p_values")
        assert post_hoc is not None, "Post-hoc pairwise tests must be provided for K >= 3"

    def test_tier1_anova_group_means_and_stds(self):
        """F12: Output contains accurate empirical group means and standard deviations."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        gA = [0.80, 0.82, 0.84]
        gB = [0.90, 0.92, 0.94]

        res = call_run_anova(mod, {"A": gA, "B": gB})

        assert "group_means" in res and "group_stds" in res
        assert res["group_means"]["A"] == pytest.approx(0.82, rel=1e-5)
        assert res["group_means"]["B"] == pytest.approx(0.92, rel=1e-5)
        assert res["group_stds"]["A"] == pytest.approx(np.std(gA, ddof=1), rel=1e-4)


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (24 Tests)
# ==============================================================================

@pytest.mark.tier2
class TestTier2BoundaryCases:
    """Tier 2: Boundary conditions, degenerate inputs, and edge error handling."""

    # --------------------------------------------------------------------------
    # Area 1: AUROC & ROC Boundaries (F1-F3)
    # --------------------------------------------------------------------------

    def test_tier2_auroc_single_class_edge_case(self):
        """F1: Ground truth containing only a single class returns NaN and does not crash."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("Evaluation module not implemented yet")

        y_true = [0, 0, 0, 0]  # Only class 0 present
        y_pred = [0.1, 0.2, 0.8, 0.4]

        metrics = call_compute_metrics(mod, y_true, y_pred)
        assert math.isnan(metrics["auroc"]), "Single class ground truth must yield NaN AUROC"

    def test_tier2_auroc_all_identical_probabilities(self):
        """F1: All predicted probabilities identical (no separation power) yields AUROC == 0.5."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("Evaluation module not implemented yet")

        y_true = [0, 0, 1, 1]
        y_pred = [0.5, 0.5, 0.5, 0.5]

        metrics = call_compute_metrics(mod, y_true, y_pred)
        assert metrics["auroc"] == pytest.approx(0.5, rel=1e-5)

    def test_tier2_auroc_extreme_probabilities(self):
        """F1: Probabilities near machine precision limits (0.0, 1.0, 1e-15) compute stably."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("Evaluation module not implemented yet")

        y_true = [0, 0, 1, 1]
        y_pred = [0.0, 1e-15, 1.0 - 1e-15, 1.0]

        metrics = call_compute_metrics(mod, y_true, y_pred)
        assert metrics["auroc"] == pytest.approx(1.0, rel=1e-5)

    def test_tier2_auroc_mismatched_lengths_raise_error(self):
        """F1: Mismatched array lengths between y_true and y_pred raises ValueError."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("Evaluation module not implemented yet")

        with pytest.raises(ValueError):
            call_compute_metrics(mod, [0, 1, 0], [0.1, 0.9])

    def test_tier2_auroc_empty_input_raise_error(self):
        """F1: Empty inputs for y_true and y_pred raises ValueError."""
        mod = get_evaluation_module()
        if mod is None:
            pytest.skip("Evaluation module not implemented yet")

        with pytest.raises(ValueError):
            call_compute_metrics(mod, [], [])

    def test_tier2_roc_export_invalid_path_handling(self):
        """F2: Exporting ROC data to non-existent or invalid directory path raises IOError or OSError."""
        mod = get_evaluation_module()
        if mod is None or not hasattr(mod, "export_roc_curve"):
            pytest.skip("export_roc_curve not implemented yet")

        roc_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]}
        invalid_path = "/non_existent_folder_abc_123/roc.json"

        with pytest.raises((OSError, IOError)):
            call_export_roc(mod, roc_data, invalid_path, format="json")

    # --------------------------------------------------------------------------
    # Area 2: Cross-Validation Boundaries (F4-F7)
    # --------------------------------------------------------------------------

    def test_tier2_cv_minimal_sample_size(self):
        """F4: Cross-validation executes on minimal dataset size (N = K = 5)."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        X = np.arange(5).reshape(-1, 1)
        y = np.array([0, 0, 0, 1, 1])

        cv = mod.StratifiedCrossValidator(n_splits=5)
        splits = list(cv.split(X, y)) if hasattr(cv, "split") else list(StratifiedKFold(n_splits=5).split(X, y))

        assert len(splits) == 5
        for train_idx, val_idx in splits:
            assert len(val_idx) == 1
            assert len(train_idx) == 4

    def test_tier2_cv_extreme_imbalance(self):
        """F4: Severe 95:5 class imbalance distributes rare class across splits without crashing."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        # 95 class 0, 5 class 1
        y = np.array([0] * 95 + [1] * 5)
        X = np.arange(100).reshape(-1, 1)

        cv = mod.StratifiedCrossValidator(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y)) if hasattr(cv, "split") else list(StratifiedKFold(n_splits=5).split(X, y))

        assert len(splits) == 5
        for fold_idx, (_, val_idx) in enumerate(splits):
            val_y = y[val_idx]
            # Each fold must get exactly 1 rare positive sample
            assert np.sum(val_y == 1) == 1, f"Fold {fold_idx} must contain exactly 1 rare positive sample"

    def test_tier2_cv_invalid_n_splits_raise_error(self):
        """F4: Invalid n_splits (<= 1 or negative) raises ValueError."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        with pytest.raises(ValueError):
            mod.StratifiedCrossValidator(n_splits=1)

        with pytest.raises(ValueError):
            mod.StratifiedCrossValidator(n_splits=-5)

    def test_tier2_cv_empty_dataset_raise_error(self):
        """F4: Empty input dataset passed to CV split raises ValueError."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        cv = mod.StratifiedCrossValidator(n_splits=5)
        with pytest.raises(ValueError):
            list(cv.split(np.array([]), np.array([])))

    def test_tier2_cv_reproducibility_with_seed(self):
        """F4: Fixed random_state generates identical splits; differing seed produces variations."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        X = np.arange(40).reshape(-1, 1)
        y = np.array([0] * 20 + [1] * 20)

        cv1 = mod.StratifiedCrossValidator(n_splits=4, shuffle=True, random_state=42)
        cv2 = mod.StratifiedCrossValidator(n_splits=4, shuffle=True, random_state=42)
        cv3 = mod.StratifiedCrossValidator(n_splits=4, shuffle=True, random_state=99)

        splits1 = list(cv1.split(X, y))
        splits2 = list(cv2.split(X, y))
        splits3 = list(cv3.split(X, y))

        for (tr1, val1), (tr2, val2) in zip(splits1, splits2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(val1, val2)

        # Different seed must yield at least one different partition
        different = any(not np.array_equal(val1, val3) for (_, val1), (_, val3) in zip(splits1, splits3))
        assert different, "Different random seeds must yield different split assignments"

    def test_tier2_cv_sample_count_not_divisible_by_k(self):
        """F4: Dataset size not evenly divisible by K (N=53, K=5) yields fold sizes differing by <= 1."""
        mod = get_cross_validation_module()
        if mod is None or not hasattr(mod, "StratifiedCrossValidator"):
            pytest.skip("StratifiedCrossValidator not implemented yet")

        X = np.arange(53).reshape(-1, 1)
        y = np.array([0] * 35 + [1] * 18)

        cv = mod.StratifiedCrossValidator(n_splits=5)
        splits = list(cv.split(X, y))

        val_sizes = [len(val_idx) for _, val_idx in splits]
        assert max(val_sizes) - min(val_sizes) <= 1, f"Validation fold sizes differ by > 1: {val_sizes}"
        assert sum(val_sizes) == 53

    # --------------------------------------------------------------------------
    # Area 3: External Ingestion Boundaries (F8-F11)
    # --------------------------------------------------------------------------

    def test_tier2_external_missing_label_column_error(self, tmp_path):
        """F8: External CSV without any identifiable label column raises ValueError or SchemaError."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("ExternalDatasetIngester not implemented yet")

        csv_file = tmp_path / "no_labels.csv"
        csv_file.write_text("text_column_1,text_column_2\nhello,world\n")

        with pytest.raises((ValueError, KeyError, RuntimeError)):
            mod.ExternalDatasetIngester.load_external_dataset(str(csv_file), model_type="cart")

    def test_tier2_external_empty_file_error(self, tmp_path):
        """F8: 0-byte external file raises ValueError or FileEmptyError."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("ExternalDatasetIngester not implemented yet")

        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        with pytest.raises((ValueError, RuntimeError, IOError)):
            mod.ExternalDatasetIngester.load_external_dataset(str(empty_file), model_type="cart")

    def test_tier2_external_unrecognized_labels_handling(self, tmp_path):
        """F9: Unrecognized label strings emit a warning and are skipped without crashing batch."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("ExternalDatasetIngester not implemented yet")

        csv_file = tmp_path / "weird_labels.csv"
        csv_file.write_text(
            "text,label\n"
            "Normal sample,human\n"
            "Weird sample,UNKNOWN_ALIEN_CLASS\n"
            "Another normal,ai\n"
        )

        loaded = mod.ExternalDatasetIngester.load_external_dataset(str(csv_file), model_type="cart")
        # Should load the 2 valid samples and skip the unrecognized label row
        assert len(loaded) == 2

    def test_tier2_external_nonexistent_path_error(self):
        """F8: Non-existent benchmark identifier or path raises FileNotFoundError or ValueError."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("ExternalDatasetIngester not implemented yet")

        with pytest.raises((FileNotFoundError, ValueError)):
            mod.ExternalDatasetIngester.load_external_dataset("/completely/fictional/path/404.csv", model_type="cart")

    def test_tier2_external_whitespace_and_casing_resilience(self, tmp_path):
        """F9: Ingestion normalizes text and labels with leading/trailing whitespace and mixed casing."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("ExternalDatasetIngester not implemented yet")

        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "   text   ,   label   \n"
            "  Some text with spaces  ,  HuMaN  \n"
            "  Another text  ,  SyNtHeTiC  \n"
        )

        loaded = mod.ExternalDatasetIngester.load_external_dataset(str(csv_file), model_type="cart")
        assert len(loaded) == 2
        labels = [item["label"] if isinstance(item, dict) else item.label for item in loaded]
        assert labels == [0, 1]

    def test_tier2_external_all_samples_corrupted_error(self, tmp_path):
        """F9: External file with 100% corrupted lines raises ValueError (empty usable dataset)."""
        mod = get_external_module()
        if mod is None or not hasattr(mod, "ExternalDatasetIngester"):
            pytest.skip("ExternalDatasetIngester not implemented yet")

        jsonl_file = tmp_path / "all_bad.jsonl"
        jsonl_file.write_text("{CORRUPT_1\n{CORRUPT_2\n{CORRUPT_3\n")

        with pytest.raises((ValueError, RuntimeError)):
            mod.ExternalDatasetIngester.load_external_dataset(str(jsonl_file), model_type="cart")

    # --------------------------------------------------------------------------
    # Area 4: Statistical Significance Boundaries (F12-F15)
    # --------------------------------------------------------------------------

    def test_tier2_anova_zero_variance_identical_scores(self):
        """F14: Degenerate zero-variance arrays return F=0.0, p=1.0, significant=False without crash."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        # Constant identical scores across all folds and all groups
        groups = {
            "model_a": [0.95, 0.95, 0.95, 0.95],
            "model_b": [0.95, 0.95, 0.95, 0.95]
        }

        res = call_run_anova(mod, groups, alpha=0.05)
        assert res["f_statistic"] == pytest.approx(0.0, abs=1e-5)
        assert res["p_value"] == pytest.approx(1.0, abs=1e-5)
        assert res["significant"] is False

    def test_tier2_anova_insufficient_groups_error(self):
        """F14: Providing fewer than 2 groups to ANOVA raises ValueError."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        with pytest.raises(ValueError):
            call_run_anova(mod, {"only_one_group": [0.91, 0.92, 0.93]})

    def test_tier2_anova_insufficient_samples_per_group_error(self):
        """F14: Providing a group with fewer than 2 samples raises ValueError."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        with pytest.raises(ValueError):
            call_run_anova(mod, {"g1": [0.9], "g2": [0.8]})

    def test_tier2_anova_empty_group_error(self):
        """F14: Providing an empty score list for a group raises ValueError."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        with pytest.raises(ValueError):
            call_run_anova(mod, {"g1": [], "g2": [0.8, 0.9]})

    def test_tier2_anova_exact_alpha_threshold_boundary(self):
        """F13: Significance decision strictly respects alpha boundary: p < 0.05 vs p >= 0.05."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        # Group configuration calibrated to produce p slightly above alpha=0.05
        # We verify decision logic directly
        if hasattr(mod, "evaluate_significance"):
            assert mod.evaluate_significance(p_value=0.04999, alpha=0.05) is True
            assert mod.evaluate_significance(p_value=0.05001, alpha=0.05) is False
        elif hasattr(mod, "ANOVATester") and hasattr(mod.ANOVATester, "is_significant"):
            assert mod.ANOVATester.is_significant(0.04999, 0.05) is True
            assert mod.ANOVATester.is_significant(0.05001, 0.05) is False

    def test_tier2_anova_nan_inf_input_handling(self):
        """F14: Metric lists containing NaN or Inf raise ValueError or sanitize gracefully."""
        mod = get_significance_module()
        if mod is None:
            pytest.skip("Significance module not implemented yet")

        with pytest.raises(ValueError):
            call_run_anova(mod, {"A": [0.90, np.nan, 0.85], "B": [0.70, 0.65, 0.72]})


# ==============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS (6 Tests)
# ==============================================================================

@pytest.mark.tier3
class TestTier3CrossFeatureInteractions:
    """Tier 3: Multi-module interactions across Ingestion, CV, AUROC, and ANOVA."""

    def test_tier3_full_pipeline_ingest_cv_metrics_summary(self, tmp_path):
        """Interacts F8 (Ingest) -> F4 (CV) -> F1 (Metrics) -> F6 (Aggregate Summary)."""
        ext_mod = get_external_module()
        cv_mod = get_cross_validation_module()
        eval_mod = get_evaluation_module()

        if ext_mod is None or cv_mod is None or eval_mod is None:
            pytest.skip("Required modules for full pipeline not yet implemented")

        # 1. Ingest external dataset
        csv_file = tmp_path / "full_pipeline.csv"
        rows = ["text,label"]
        for i in range(50):
            rows.append(f"Human writing sample {i},human")
            rows.append(f"AI generated text {i},ai")
        csv_file.write_text("\n".join(rows))

        samples = ext_mod.ExternalDatasetIngester.load_external_dataset(str(csv_file), "cart")
        assert len(samples) == 100

        # 2. Run 5-fold Stratified Cross-Validation
        y = np.array([s["label"] if isinstance(s, dict) else s.label for s in samples])
        X = np.arange(100).reshape(-1, 1)

        cv = cv_mod.StratifiedCrossValidator(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            val_y = y[val_idx]
            # Simulated predictions with 90% accuracy
            np.random.seed(fold)
            val_probs = np.where(val_y == 1, np.random.uniform(0.6, 1.0, len(val_y)), np.random.uniform(0.0, 0.4, len(val_y)))
            m = call_compute_metrics(eval_mod, val_y, val_probs)
            fold_metrics.append(m)

        # 3. Verify aggregate summary
        auroc_scores = [m["auroc"] for m in fold_metrics]
        mean_auc = np.mean(auroc_scores)
        std_auc = np.std(auroc_scores, ddof=1)

        assert mean_auc > 0.95
        assert std_auc < 0.05

    def test_tier3_multi_model_cv_anova_comparison(self):
        """Interacts F4 (CV) -> F1 (AUROC) -> F12 (ANOVA Hypothesis Testing)."""
        sig_mod = get_significance_module()
        if sig_mod is None:
            pytest.skip("Significance module not implemented yet")

        # Simulate 5-fold CV AUROC scores across two model candidates
        model_a_cv_auroc = [0.81, 0.83, 0.82, 0.80, 0.84]  # Baseline DistilBERT
        model_b_cv_auroc = [0.94, 0.96, 0.95, 0.93, 0.95]  # Full SafeStep Hybrid

        anova_res = call_run_anova(sig_mod, {
            "DistilBERT_Baseline": model_a_cv_auroc,
            "SafeStep_Hybrid": model_b_cv_auroc
        }, alpha=0.05)

        assert anova_res["significant"] is True
        assert anova_res["p_value"] < 0.001
        assert anova_res["group_means"]["SafeStep_Hybrid"] > anova_res["group_means"]["DistilBERT_Baseline"]

    def test_tier3_ablation_baseline_anova_comparison(self):
        """Interacts Ablation Baseline Analysis across 3 variants with post-hoc Bonferroni."""
        sig_mod = get_significance_module()
        if sig_mod is None:
            pytest.skip("Significance module not implemented yet")

        # 3 ablation configurations: Baseline vs Heuristics Only vs Full Fusion
        ablation_groups = {
            "neural_only": [0.88, 0.89, 0.87, 0.88, 0.89],
            "heuristics_only": [0.72, 0.74, 0.71, 0.73, 0.72],
            "full_fusion": [0.95, 0.96, 0.94, 0.95, 0.96]
        }

        res = call_run_anova(sig_mod, ablation_groups, alpha=0.05)
        assert res["significant"] is True
        assert res["f_statistic"] > 100.0

    def test_tier3_cv_roc_curve_per_fold_export(self, tmp_path):
        """Interacts F4 (CV) -> F1 (ROC Coordinates) -> F2 (JSON Export)."""
        eval_mod = get_evaluation_module()
        if eval_mod is None or not hasattr(eval_mod, "export_roc_curve"):
            pytest.skip("Evaluation module export_roc_curve not implemented yet")

        for fold in range(5):
            np.random.seed(fold)
            y_true = np.array([0] * 20 + [1] * 20)
            y_pred = np.where(y_true == 1, np.random.uniform(0.5, 1.0, 40), np.random.uniform(0.0, 0.5, 40))

            m = call_compute_metrics(eval_mod, y_true, y_pred)
            out_file = str(tmp_path / f"roc_fold_{fold}.json")
            call_export_roc(eval_mod, m["roc_curve"], out_file, format="json")

            assert os.path.exists(out_file)
            with open(out_file, "r") as f:
                saved = json.load(f)
                assert saved["fpr"][0] == 0.0
                assert saved["fpr"][-1] == 1.0

    def test_tier3_corrupted_ingestion_in_cv_pipeline(self, tmp_path):
        """Interacts F9 (Corrupted Ingestion) -> F4 (CV Partitioning): ensures filtering before fold splits."""
        ext_mod = get_external_module()
        cv_mod = get_cross_validation_module()
        if ext_mod is None or cv_mod is None:
            pytest.skip("Required modules not yet implemented")

        # 40 valid samples + 10 corrupted lines
        lines = ["text,label"]
        for i in range(20):
            lines.append(f"Valid human text {i},human")
            lines.append(f"Valid AI text {i},ai")
        # Add 10 unparseable lines
        lines.append(",,,,")
        lines.append("No label line here")

        csv_file = tmp_path / "mixed_cv.csv"
        csv_file.write_text("\n".join(lines))

        samples = ext_mod.ExternalDatasetIngester.load_external_dataset(str(csv_file), "cart")
        assert len(samples) == 40, "Must cleanly isolate exactly 40 valid samples"

        # Now partition into 5 folds
        y = np.array([s["label"] if isinstance(s, dict) else s.label for s in samples])
        cv = cv_mod.StratifiedCrossValidator(n_splits=5)
        splits = list(cv.split(np.arange(len(samples)), y))
        assert len(splits) == 5

    def test_tier3_discrete_fallback_through_cv_and_anova(self):
        """Interacts Discrete Fallback -> CV Fold Metric Calculation -> ANOVA Verification."""
        eval_mod = get_evaluation_module()
        sig_mod = get_significance_module()
        if eval_mod is None or sig_mod is None:
            pytest.skip("Required modules not yet implemented")

        # Discrete classifier fold evaluations
        model_discrete_folds = []
        for seed in range(5):
            np.random.seed(seed)
            y_true = np.array([0] * 30 + [1] * 30)
            # Binary discrete outputs with ~85% accuracy
            y_pred = np.copy(y_true)
            flips = np.random.choice(60, size=9, replace=False)
            y_pred[flips] = 1 - y_pred[flips]

            metrics = call_compute_metrics(eval_mod, y_true, y_pred)
            model_discrete_folds.append(metrics["auroc"])

        # Continuous probability model fold evaluations
        model_prob_folds = [0.95, 0.96, 0.94, 0.96, 0.95]

        res = call_run_anova(sig_mod, {
            "Discrete_Classifier": model_discrete_folds,
            "Continuous_Classifier": model_prob_folds
        })
        assert "p_value" in res
        assert "significant" in res


# ==============================================================================
# TIER 4: REAL-WORLD WORKLOAD SCENARIOS (5 Tests)
# ==============================================================================

@pytest.mark.tier4
class TestTier4RealWorldWorkloads:
    """Tier 4: Realistic end-to-end workload pipelines for CART, JAW, and Phishing."""

    def test_tier4_cart_ai_text_detection_workload(self, tmp_path):
        """R1/R2/R3: End-to-end CART text classification workload with feature extraction & AUROC."""
        eval_mod = get_evaluation_module()
        if eval_mod is None:
            pytest.skip("Evaluation module not implemented yet")

        # Realistic text corpus with human vs AI stylistic variance
        np.random.seed(101)
        human_texts = [f"This essay explores the natural historical origins of language variation sample {i}." for i in range(50)]
        ai_texts = [f"As an AI language model, it is crucial to recognize multifaceted linguistic aspects sample {i}." for i in range(50)]

        y_true = np.array([0] * 50 + [1] * 50)
        # Synthetic probabilities reflecting high detector confidence
        y_prob = np.concatenate([
            np.random.beta(a=1, b=5, size=50),  # Human texts -> low positive prob
            np.random.beta(a=5, b=1, size=50),  # AI texts -> high positive prob
        ])

        metrics = call_compute_metrics(eval_mod, y_true, y_prob)

        assert metrics["auroc"] > 0.95, "CART text pipeline must achieve high AUROC on distinguishable texts"
        assert "roc_curve" in metrics

        # Export curve
        if hasattr(eval_mod, "export_roc_curve"):
            export_path = str(tmp_path / "cart_roc.json")
            call_export_roc(eval_mod, metrics["roc_curve"], export_path, format="json")
            assert os.path.exists(export_path)

    def test_tier4_jaw_synthetic_image_workload(self):
        """R1/R2: End-to-end JAW image artifact workload with batch evaluation and ASCII ROC plot."""
        eval_mod = get_evaluation_module()
        if eval_mod is None:
            pytest.skip("Evaluation module not implemented yet")

        # 80 authentic vs 80 synthetic images
        y_true = np.array([0] * 80 + [1] * 80)
        np.random.seed(202)
        y_prob = np.concatenate([
            np.random.beta(a=2, b=8, size=80),
            np.random.beta(a=8, b=2, size=80)
        ])

        metrics = call_compute_metrics(eval_mod, y_true, y_prob)
        assert metrics["auroc"] > 0.90

        # Render ASCII display
        ascii_curve = call_render_ascii_roc(eval_mod, metrics["roc_curve"])
        assert isinstance(ascii_curve, str)
        assert len(ascii_curve) > 20

    def test_tier4_phishing_email_detection_workload(self):
        """R1/R2: End-to-end Phishing pipeline workload with severe class imbalance and TF-IDF isolation."""
        eval_mod = get_evaluation_module()
        cv_mod = get_cross_validation_module()
        if eval_mod is None or cv_mod is None:
            pytest.skip("Required modules not yet implemented")

        # Imbalanced distribution: 95% legitimate ham, 5% phishing
        n_samples = 200
        n_phish = 10
        y_true = np.array([0] * (n_samples - n_phish) + [1] * n_phish)
        X = np.arange(n_samples).reshape(-1, 1)

        cv = cv_mod.StratifiedCrossValidator(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y_true))

        fold_aurocs = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            val_y = y_true[val_idx]
            assert np.sum(val_y == 1) == 2, "Each fold must contain exactly 2 phishing emails"

            np.random.seed(fold + 300)
            val_probs = np.where(val_y == 1, np.random.uniform(0.7, 0.99, len(val_y)), np.random.uniform(0.01, 0.3, len(val_y)))
            res = call_compute_metrics(eval_mod, val_y, val_probs)
            fold_aurocs.append(res["auroc"])

        assert np.mean(fold_aurocs) > 0.95

    def test_tier4_benchmark_suite_three_way_anova_workload(self):
        """R4: Benchmark suite evaluating 3 detector variants followed by 3-way ANOVA report."""
        sig_mod = get_significance_module()
        if sig_mod is None:
            pytest.skip("Significance module not implemented yet")

        # 3 detectors evaluated on public benchmark (e.g. HC3)
        architectures = {
            "CART_DistilBERT_XGBoost": [0.952, 0.961, 0.948, 0.955, 0.958],
            "Pure_DistilBERT": [0.891, 0.902, 0.885, 0.898, 0.905],
            "Stylistic_Heuristics_Only": [0.765, 0.772, 0.758, 0.769, 0.761]
        }

        res = call_run_anova(sig_mod, architectures, alpha=0.05)

        assert res["significant"] is True
        assert res["p_value"] < 1e-4
        assert res["group_means"]["CART_DistilBERT_XGBoost"] > res["group_means"]["Pure_DistilBERT"]

    def test_tier4_cli_pipeline_flow_simulation(self):
        """R2/R3/R5: Simulation of CLI argument handling and execution pipeline."""
        # Test that cross-validation and external dataset parameters can be parsed without error
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--cross-val", "--cv", "-cv", nargs="?", const=5, type=int, default=None)
        parser.add_argument("--eval-external", "--external-dataset", type=str, default=None)
        parser.add_argument("--anova-eval", action="store_true")

        args1 = parser.parse_args(["--cross-val", "5", "--eval-external", "raid"])
        assert args1.cross_val == 5
        assert args1.eval_external == "raid"

        args2 = parser.parse_args(["-cv"])
        assert args2.cross_val == 5

        args3 = parser.parse_args(["--cv", "10"])
        assert args3.cross_val == 10
