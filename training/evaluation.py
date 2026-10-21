"""
Core Evaluation Metrics Engine for SafeStep.

Provides centralized metric computation (Accuracy, Precision, Recall, F1, AUROC,
and ROC curve coordinates), single-class protection, probability-vs-discrete fallback,
multi-class OvR/OvO handling, ROC curve export (JSON/CSV), and ASCII ROC visualization.
"""

import csv
import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from util.logging_config import get_logger

logger = get_logger(__name__)


def compute_evaluation_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], List[int], np.ndarray],
    pos_label: int = 1,
    multi_class: str = "ovr",
) -> Dict[str, Any]:
    """
    Computes Accuracy, Precision, Recall, F1, AUROC, and ROC curve coordinates.

    Handles:
      - Continuous 1D probabilities in [0.0, 1.0]
      - Discrete 1D labels {0, 1} (fallback to Balanced Accuracy via trapezoidal rule)
      - Continuous 2D probability matrices (N, num_classes)
      - Single-class edge cases in y_true (returns np.nan with warning)
      - Multi-class classification via 'ovr' or 'ovo'

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted continuous probabilities or discrete class labels.
        pos_label: Positive class identifier for binary metrics and OvR ROC curve.
        multi_class: Multi-class AUROC strategy: 'ovr' or 'ovo'.

    Returns:
        Dictionary containing 'accuracy', 'precision', 'recall', 'f1', 'auroc',
        and 'roc_curve' (with 'fpr', 'tpr', 'thresholds').
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        raise ValueError("Input arrays y_true and y_pred cannot be empty.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} vs len(y_pred)={len(y_pred_arr)}"
        )

    # Flatten single-column 2D arrays (N, 1) -> 1D (N,)
    if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == 1:
        y_pred_arr = y_pred_arr.squeeze(axis=1)

    unique_classes = np.unique(y_true_arr)
    num_classes = len(unique_classes)

    # -------------------------------------------------------------------------
    # 1. Single-class edge case protection
    # -------------------------------------------------------------------------
    if num_classes < 2:
        warning_msg = (
            f"Only one class present in y_true ({unique_classes[0]}); "
            f"AUROC and ROC curve are undefined. Returning np.nan."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        logger.warning(warning_msg)

        if y_pred_arr.ndim == 2:
            y_pred_disc = np.argmax(y_pred_arr, axis=1)
        elif np.issubdtype(y_pred_arr.dtype, np.floating) and not set(
            np.unique(y_pred_arr)
        ).issubset({0.0, 1.0}):
            y_pred_disc = (y_pred_arr >= 0.5).astype(int)
        else:
            y_pred_disc = y_pred_arr.astype(int)

        acc = float(accuracy_score(y_true_arr, y_pred_disc))
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_arr,
            y_pred_disc,
            pos_label=pos_label,
            average="binary" if (pos_label in unique_classes and len(np.unique(y_pred_disc)) <= 2) else "macro",
            zero_division=0.0,
        )
        return {
            "accuracy": acc,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "auroc": float("nan"),
            "roc_curve": {
                "fpr": [],
                "tpr": [],
                "thresholds": [],
            },
        }

    # -------------------------------------------------------------------------
    # 2. Binary classification (num_classes == 2)
    # -------------------------------------------------------------------------
    if num_classes == 2 and (y_pred_arr.ndim == 1 or y_pred_arr.shape[1] <= 2):
        if y_pred_arr.ndim == 2:
            prob_col = pos_label if pos_label < y_pred_arr.shape[1] else 1
            y_prob = y_pred_arr[:, prob_col]
            y_pred_disc = np.argmax(y_pred_arr, axis=1)
        elif np.issubdtype(y_pred_arr.dtype, np.floating) and not set(
            np.unique(y_pred_arr)
        ).issubset({0.0, 1.0}):
            y_prob = y_pred_arr
            y_pred_disc = (y_prob >= 0.5).astype(int)
        else:
            # Discrete {0, 1} fallback
            y_pred_disc = y_pred_arr.astype(int)
            y_prob = y_pred_disc.astype(float)

        y_true_bin = (y_true_arr == pos_label).astype(int)

        # Handle pos_label inversion properly
        if pos_label != 1:
            if y_pred_arr.ndim == 1 and not (np.issubdtype(y_pred_arr.dtype, np.floating) and not set(np.unique(y_pred_arr)).issubset({0.0, 1.0})):
                y_prob = (y_pred_disc == pos_label).astype(float)
            elif y_pred_arr.ndim == 1:
                y_prob = 1.0 - y_prob

        all_pred_classes = np.unique(np.concatenate([y_true_arr, y_pred_disc]))
        avg_mode = "binary" if (len(all_pred_classes) <= 2 and pos_label in unique_classes) else "macro"

        acc = float(accuracy_score(y_true_arr, y_pred_disc))
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_arr,
            y_pred_disc,
            pos_label=pos_label,
            average=avg_mode,
            zero_division=0.0,
        )

        try:
            auc = float(roc_auc_score(y_true_bin, y_prob))
        except (ValueError, Exception) as exc:
            logger.warning(f"AUROC calculation failed: {exc}. Falling back to np.nan.")
            auc = float("nan")

        try:
            fpr, tpr, thresh = roc_curve(y_true_bin, y_prob)
            thresh_list = [float(t) if not np.isinf(t) else 1.0 for t in thresh]
            roc_dict = {
                "fpr": [float(x) for x in fpr],
                "tpr": [float(x) for x in tpr],
                "thresholds": thresh_list,
            }
        except Exception as exc:
            logger.warning(f"ROC curve generation failed: {exc}.")
            roc_dict = {"fpr": [], "tpr": [], "thresholds": []}

        return {
            "accuracy": acc,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "auroc": auc,
            "roc_curve": roc_dict,
        }

    # -------------------------------------------------------------------------
    # 3. Multi-class classification (num_classes > 2 or y_pred_arr has >2 columns)
    # -------------------------------------------------------------------------
    all_observed_classes = np.unique(
        np.concatenate([
            unique_classes,
            np.argmax(y_pred_arr, axis=1) if y_pred_arr.ndim == 2 else np.asarray(y_pred_arr, dtype=int)
        ])
    )
    from sklearn.preprocessing import label_binarize

    if y_pred_arr.ndim == 2:
        y_prob = y_pred_arr
        y_pred_disc = np.argmax(y_pred_arr, axis=1)

        # Ensure column count matches class count if computing multi-class AUROC
        try:
            if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
                raise ValueError("Probability matrix contains NaN or Inf.")
            if y_prob.shape[1] == len(unique_classes):
                auc = float(
                    roc_auc_score(
                        y_true_arr, y_prob, multi_class=multi_class, average="macro"
                    )
                )
            else:
                # Sub-slice where classes don't match column count
                y_true_bin = label_binarize(y_true_arr, classes=all_observed_classes)
                auc = float(
                    roc_auc_score(
                        y_true_bin, y_prob, multi_class=multi_class, average="macro"
                    )
                )
        except Exception as exc:
            logger.warning(f"Multi-class AUROC calculation failed: {exc}. Returning np.nan.")
            auc = float("nan")

        y_true_bin_single = (y_true_arr == pos_label).astype(int)
        score_bin = (
            y_prob[:, pos_label] if pos_label < y_prob.shape[1] else y_prob[:, 0]
        )
        try:
            fpr, tpr, thresh = roc_curve(y_true_bin_single, score_bin)
        except Exception:
            fpr, tpr, thresh = [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]
    else:
        # Discrete labels (handles non-contiguous or 1-indexed labels cleanly)
        y_pred_disc = y_pred_arr.astype(int)
        try:
            y_true_bin = label_binarize(y_true_arr, classes=all_observed_classes)
            y_prob = label_binarize(y_pred_disc, classes=all_observed_classes)
            if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
                raise ValueError("Predictions contain NaN or Inf.")
            auc = float(
                roc_auc_score(
                    y_true_bin, y_prob, multi_class=multi_class, average="macro"
                )
            )
        except Exception as exc:
            logger.warning(f"Discrete multi-class AUROC failed: {exc}. Returning np.nan.")
            auc = float("nan")

        y_true_bin_single = (y_true_arr == pos_label).astype(int)
        score_bin = (y_pred_disc == pos_label).astype(float)
        try:
            fpr, tpr, thresh = roc_curve(y_true_bin_single, score_bin)
        except Exception:
            fpr, tpr, thresh = [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]

    acc = float(accuracy_score(y_true_arr, y_pred_disc))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_disc, average="macro", zero_division=0.0
    )
    thresh_list = [float(t) if not np.isinf(t) else 1.0 for t in thresh]

    return {
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auroc": auc,
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": thresh_list,
        },
    }


def export_roc_curve(
    roc_data: Dict[str, List[float]],
    output_path: str,
    format: str = "json",
) -> str:
    """
    Exports FPR, TPR, thresholds to JSON or CSV.

    Args:
        roc_data: Dict containing 'fpr', 'tpr', and 'thresholds'.
        output_path: Target file path.
        format: 'json' or 'csv' (case-insensitive).

    Returns:
        Absolute path to the exported file.
    """
    fmt = format.lower().strip()
    if fmt not in ("json", "csv"):
        raise ValueError(
            f"Unsupported format '{format}'. Supported formats: 'json', 'csv'."
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fpr = [float(x) for x in roc_data.get("fpr", [])]
    tpr = [float(x) for x in roc_data.get("tpr", [])]
    thresholds = [float(x) for x in roc_data.get("thresholds", [])]

    if fmt == "json":
        payload = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    elif fmt == "csv":
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr", "threshold"])
            for row in zip(fpr, tpr, thresholds):
                writer.writerow(row)

    return os.path.abspath(output_path)


def render_ascii_roc_curve(
    roc_data: Dict[str, List[float]],
    width: int = 40,
    height: int = 15,
) -> str:
    """
    Renders ASCII representation of ROC curve for console display.

    Plots True Positive Rate (0.0 to 1.0) against False Positive Rate (0.0 to 1.0),
    including the random chance baseline (AUC = 0.50).

    Args:
        roc_data: Dict containing 'fpr' and 'tpr'.
        width: Plot canvas character width (default: 40).
        height: Plot canvas character height (default: 15).

    Returns:
        Rendered multi-line ASCII plot string.
    """
    fpr = roc_data.get("fpr", [])
    tpr = roc_data.get("tpr", [])

    if len(fpr) == 0 or len(tpr) == 0:
        return "[ROC Curve unavailable: empty or single-class data]"

    fpr_arr = np.asarray(fpr, dtype=float)
    tpr_arr = np.asarray(tpr, dtype=float)

    # Filter invalid/nan entries
    valid_mask = np.isfinite(fpr_arr) & np.isfinite(tpr_arr)
    if not np.any(valid_mask):
        return "[ROC Curve unavailable: non-finite coordinates]"

    fpr_arr = fpr_arr[valid_mask]
    tpr_arr = tpr_arr[valid_mask]

    grid = [[" " for _ in range(width)] for _ in range(height)]

    # 1. Draw diagonal baseline (AUC = 0.50 random chance)
    for c in range(width):
        r = round((height - 1) * (1.0 - c / max(1, width - 1)))
        if 0 <= r < height:
            grid[r][c] = "."

    # 2. Draw ROC points and step interpolation
    grid_coords = []
    for f, t in zip(fpr_arr, tpr_arr):
        c = min(max(round(f * (width - 1)), 0), width - 1)
        r = min(max(round((1.0 - t) * (height - 1)), 0), height - 1)
        grid_coords.append((r, c))

    for (r1, c1), (r2, c2) in zip(grid_coords[:-1], grid_coords[1:]):
        steps = max(abs(r2 - r1), abs(c2 - c1))
        if steps > 0:
            for s in range(steps + 1):
                cur_r = round(r1 + (r2 - r1) * s / steps)
                cur_c = round(c1 + (c2 - c1) * s / steps)
                if 0 <= cur_r < height and 0 <= cur_c < width:
                    grid[cur_r][cur_c] = "*"

    for r, c in grid_coords:
        grid[r][c] = "*"

    # 3. Assemble axes and frame
    lines = []
    lines.append("  TPR ^")
    for r in range(height):
        if r == 0:
            label = " 1.0 |"
        elif r == height // 2:
            label = " 0.5 |"
        elif r == height - 1:
            label = " 0.0 |"
        else:
            label = "     |"
        lines.append(f"{label}{''.join(grid[r])}")

    lines.append("     +" + "-" * width)

    # X-axis labels
    half_w = width // 2
    x_axis = (
        "      0.0"
        + " " * max(1, half_w - 4)
        + "0.5"
        + " " * max(1, width - half_w - 5)
        + "1.0"
    )
    lines.append(x_axis)
    lines.append("      " + "FPR (False Positive Rate)".center(width))
    lines.append("  Legend: * = ROC Curve, . = Random Baseline (AUC = 0.50)")

    return "\n".join(lines)
