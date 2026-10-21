"""
Statistical Significance Testing Module for SafeStep.

Provides automated Analysis of Variance (One-Way ANOVA), p-value evaluation
at significance threshold alpha = 0.05, group summary statistics, and post-hoc
pairwise comparisons with Bonferroni correction for K >= 3 models or folds.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from util.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_significance(p_value: float, alpha: float = 0.05) -> bool:
    """
    Evaluates whether a p-value is strictly less than the significance threshold alpha.

    Args:
        p_value: Computed probability value.
        alpha: Significance threshold (default: 0.05).

    Returns:
        True if p_value < alpha, False otherwise.
    """
    return bool(p_value < alpha)


class ANOVATester:
    """
    One-Way Analysis of Variance (ANOVA) Testing Engine.

    Evaluates whether metric differences across cross-validation folds,
    architectures, or ablation baselines are statistically significant.
    """

    @classmethod
    def is_significant(cls, p_value: float, alpha: float = 0.05) -> bool:
        return evaluate_significance(p_value, alpha)

    @classmethod
    def run_one_way_anova(
        cls,
        groups: Dict[str, List[float]],
        alpha: float = 0.05,
        metric_name: str = "metric",
    ) -> Dict[str, Any]:
        """
        Runs One-Way ANOVA across group metric distributions.

        Args:
            groups: Mapping from group/model name to list of fold metric scores.
            alpha: Significance threshold (default: 0.05).
            metric_name: Name of evaluated metric (e.g. 'f1', 'auroc').

        Returns:
            Dictionary containing 'f_statistic', 'p_value', 'significant',
            'is_significant', 'alpha', 'group_means', 'group_stds', and 'post_hoc'.
        """
        if not isinstance(groups, dict) or len(groups) < 2:
            raise ValueError(
                f"ANOVA requires at least 2 groups for comparison, got {len(groups) if isinstance(groups, dict) else 0}"
            )

        group_arrays: Dict[str, np.ndarray] = {}
        all_values: List[float] = []

        for name, vals in groups.items():
            if not vals or len(vals) < 2:
                raise ValueError(
                    f"Group '{name}' must contain at least 2 samples, got {len(vals) if vals is not None else 0}"
                )
            arr = np.asarray(vals, dtype=float)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise ValueError(f"Group '{name}' contains NaN or infinite values.")
            group_arrays[name] = arr
            all_values.extend(arr.tolist())

        # Group means and sample standard deviations (ddof=1)
        group_means = {k: float(np.mean(v)) for k, v in group_arrays.items()}
        group_stds = {k: float(np.std(v, ddof=1)) for k, v in group_arrays.items()}

        # Edge case: zero variance across all samples (identical scores)
        if len(set(all_values)) <= 1 or np.allclose(all_values, all_values[0]):
            return {
                "f_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "is_significant": False,
                "alpha": float(alpha),
                "group_means": group_means,
                "group_stds": group_stds,
                "post_hoc": None,
                "pairwise_p_values": None,
                "metric_name": metric_name,
            }

        # Calculate F-statistic and p-value
        f_stat, p_val = stats.f_oneway(*group_arrays.values())
        f_stat = float(f_stat)
        p_val = float(p_val)
        is_sig = evaluate_significance(p_val, alpha)

        # Post-hoc pairwise testing if K >= 3
        post_hoc: Optional[Dict[str, Any]] = None
        group_names = list(groups.keys())
        k_groups = len(group_names)

        if k_groups >= 3:
            post_hoc = {}
            n_comparisons = (k_groups * (k_groups - 1)) // 2
            bonferroni_alpha = alpha / n_comparisons

            for i in range(k_groups):
                for j in range(i + 1, k_groups):
                    g1 = group_names[i]
                    g2 = group_names[j]
                    pair_key = f"{g1}_vs_{g2}"

                    t_stat, raw_p = stats.ttest_ind(
                        group_arrays[g1], group_arrays[g2], equal_var=False
                    )
                    adj_p = min(1.0, float(raw_p) * n_comparisons)
                    pair_sig = evaluate_significance(float(raw_p), bonferroni_alpha)

                    post_hoc[pair_key] = {
                        "group1": g1,
                        "group2": g2,
                        "t_statistic": float(t_stat),
                        "raw_p_value": float(raw_p),
                        "adjusted_p_value": adj_p,
                        "p_value": adj_p,
                        "significant": pair_sig,
                    }

        result = {
            "f_statistic": f_stat,
            "p_value": p_val,
            "significant": is_sig,
            "is_significant": is_sig,
            "alpha": float(alpha),
            "group_means": group_means,
            "group_stds": group_stds,
            "post_hoc": post_hoc,
            "pairwise_p_values": post_hoc,
            "metric_name": metric_name,
        }
        return result


def run_anova_test(
    groups: Dict[str, List[float]],
    alpha: float = 0.05,
    **kwargs,
) -> Dict[str, Any]:
    """Helper dispatcher for ANOVATester.run_one_way_anova."""
    return ANOVATester.run_one_way_anova(groups, alpha=alpha, **kwargs)


def format_anova_report(result: Dict[str, Any]) -> str:
    """Renders a human-readable text report of the ANOVA test outcome."""
    lines = [
        "===========================================================",
        "               ANOVA SIGNIFICANCE TEST REPORT              ",
        "===========================================================",
        f"Metric:            {result.get('metric_name', 'score')}",
        f"F-Statistic:       {result.get('f_statistic', 0.0):.4f}",
        f"p-Value:           {result.get('p_value', 1.0):.4e}",
        f"Alpha Threshold:   {result.get('alpha', 0.05)}",
        f"Significant Diff:  {'YES (p < alpha)' if result.get('significant') else 'NO (p >= alpha)'}",
        "-----------------------------------------------------------",
        "Group Statistics:",
    ]
    for grp, mean_val in result.get("group_means", {}).items():
        std_val = result.get("group_stds", {}).get(grp, 0.0)
        lines.append(f"  • {grp:25s}: Mean = {mean_val:.4f} ± {std_val:.4f}")

    post_hoc = result.get("post_hoc")
    if post_hoc:
        lines.append("-----------------------------------------------------------")
        lines.append("Post-Hoc Pairwise Comparisons (Bonferroni Corrected):")
        for pair, details in post_hoc.items():
            sig_str = "SIGNIFICANT" if details.get("significant") else "not significant"
            lines.append(
                f"  • {pair:30s}: Adj p = {details.get('adjusted_p_value', 1.0):.4e} ({sig_str})"
            )
    lines.append("===========================================================")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    # Demo CLI execution
    demo_groups = {
        "Baseline_DistilBERT": [0.81, 0.83, 0.82, 0.80, 0.84],
        "SafeStep_Hybrid": [0.94, 0.96, 0.95, 0.93, 0.95],
    }
    report = ANOVATester.run_one_way_anova(demo_groups)
    print(format_anova_report(report))
