# SafeStep Test Infrastructure & E2E Verification Plan

**Document Version:** 1.0.0  
**Status:** Active / Ready for Execution  
**Author:** test_writer_e2e_1  
**Target System:** SafeStep Model Evaluation and Validation Framework  
**Authoritative Specifications:** `/mnt/c/Users/mmati/Safestep-Models/.agents/teamwork/ORIGINAL_REQUEST.md`, `/mnt/c/Users/mmati/Safestep-Models/.agents/teamwork/orchestrator_1/PROJECT.md`

---

## 1. Test Philosophy & Design Principles

The SafeStep Model Evaluation and Validation test infrastructure is built upon five architectural testing principles:

1. **Opaque-Box Requirement-Driven Verification:**
   Tests validate functional contracts, observable outputs, mathematical invariants, and error behaviors as specified in `ORIGINAL_REQUEST.md` (R1–R5) and `PROJECT.md` interface contracts, without coupling to internal private methods.

2. **Progressive Testability & Deterministic Isolation:**
   Every test case is self-contained and independently executable. Modules are verified using pure dependencies (`numpy`, `scipy`, `scikit-learn`, `pytest`). If a milestone module is under active construction, tests gracefully detect implementation status while retaining strict contract assertions.

3. **No Facade Tests:**
   All assertions verify real computational logic, mathematical identities, non-overlapping fold partitions, or statistical distributions. Trivial tautologies (e.g. `assert True`) are strictly prohibited.

4. **Defensive Edge & Boundary Hardening:**
   Every feature area is subjected to adversarial corner cases: single-class degenerate distributions, zero-variance arrays, empty datasets, corrupt image/text assets, and boundary threshold tests ($p = 0.05$).

5. **Multi-Tier Layered Quality Architecture:**
   Tests are partitioned into 4 distinct verification tiers ensuring depth from unit contract coverage to integrated real-world workloads.

---

## 2. Feature Inventory Coverage Matrix

The following matrix maps the project Feature Inventory (`PROJECT.md` F1–F19) and requirements R1–R5 to the 4 testing tiers.

| Feature ID | Feature Name | Requirement | Primary Module Contract | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Cross-Feature) | Tier 4 (Workload) |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **F1** | Core AUROC & Metrics Engine | R1 | `training.evaluation.compute_evaluation_metrics` | 5 tests | 4 tests | 2 tests | 2 tests |
| **F2** | ROC Curve Export & Display | R1 | `training.evaluation.export_roc_curve`, `render_ascii_roc_curve` | 2 tests | 1 test | 1 test | 1 test |
| **F3** | Model Pipeline AUROC Integration | R1 | Model proba/prediction interface | 1 test | 1 test | 1 test | 1 test |
| **F4** | 5-Fold Stratified Split Generator | R2 | `training.cross_validation.StratifiedCrossValidator` | 2 tests | 3 tests | 1 test | 1 test |
| **F5** | Leak-Free Cross-Validation Runner | R2 | `StratifiedCrossValidator.run_cv` | 1 test | 1 test | 1 test | 1 test |
| **F6** | Aggregate CV Metrics & Reporting | R2 | Mean ± Std aggregate summary | 2 tests | 1 test | 1 test | 1 test |
| **F7** | Cross-Validation CLI Integration | R2 | `--cross-val 5` / `-cv 5` arguments | 1 test | 1 test | 1 test | 1 test |
| **F8** | External Dataset Schema Ingestion | R3 | `training.external.ExternalDatasetIngester` | 2 tests | 2 tests | 1 test | 1 test |
| **F9** | Benchmark Normalization & Robustness | R3 | Alias normalization & corrupted file skipping | 3 tests | 2 tests | 1 test | 1 test |
| **F10** | External Dataset Validation Runner | R3 | `training.external.evaluate_external_dataset` | 1 test | 1 test | 1 test | 1 test |
| **F11** | External Dataset CLI Integration | R3 | `--eval-external` / `--external-dataset` | 1 test | 1 test | 1 test | 1 test |
| **F12** | Automated ANOVA Significance Engine | R4 | `training.significance.ANOVATester` | 2 tests | 2 tests | 1 test | 1 test |
| **F13** | Significance Threshold & Decision | R4 | α = 0.05 decision logic (`significant: bool`) | 1 test | 1 test | 1 test | 1 test |
| **F14** | Multi-Group Post-Hoc & Edge Cases | R4 | Degenerate variance & Bonferroni pairwise | 2 tests | 2 tests | 1 test | 1 test |
| **F15** | Significance Testing CLI & Runner | R4 | ANOVA runner interface | 1 test | 1 test | 1 test | 1 test |
| **F16** | Beginner Onboarding Guide | R5 | `docs/onboarding.md` validation | - | - | - | 1 test |
| **F17** | CLI & Documentation Sync | R5 | CLI flag docs consistency | - | - | - | 1 test |
| **F18** | Opaque-Box E2E Test Suite | E2E | `tests/test_e2e_framework.py` | 26 tests | 24 tests | 6 tests | 5 tests |
| **F19** | 100% E2E Verification & Hardening | M_FINAL | Full test suite execution & zero regression | Pass | Pass | Pass | Pass |

---

## 3. Four-Tier Testing Methodology

### Tier 1: Core Feature Coverage ($\ge 5$ tests per feature area)
Tier 1 establishes that each individual capability functions according to its contract under standard conditions:
- **AUROC & Metrics (F1–F3):**
  - Binary AUROC computation on perfect separation ($1.0$), random guessing ($\approx 0.5$), and inverted probabilities ($0.0$).
  - ROC curve coordinate generation ($\text{FPR}, \text{TPR}, \text{thresholds}$) verifying monotonic non-decreasing progression.
  - Discrete binary prediction fallback to Balanced Accuracy when probabilities are unavailable.
  - Multi-class One-vs-Rest (OvR) macro-average AUROC computation.
  - ROC coordinate export to JSON and CSV formats.
  - High-resolution ASCII ROC curve console renderer.
- **5-Fold Stratified Cross-Validation (F4–F7):**
  - Stratified fold generation preserving class ratio invariants across all folds.
  - Pairwise disjoint validation splits guaranteeing zero test-set overlap.
  - Leak-free isolation verifying preprocessors and scalers are fitted strictly on training folds.
  - Per-fold metric breakdown reporting Accuracy, Precision, Recall, F1, and AUROC.
  - Aggregate statistics calculating unbiased sample mean $\mu$ and sample standard deviation $\sigma$.
  - Configurable split counts ($K=3, 5, 10$).
- **External Dataset Ingestion & Validation (F8–F11):**
  - Universal column auto-inference across CSV and JSONL formats (`text`, `body`, `label`, `is_bot`).
  - Label alias normalization for CART (`human` vs `ai_generated`), JAW (`authentic` vs `ai_generated`), and Phishing (`legitimate` vs `phishing`).
  - Corrupted record and unparseable image tolerance (graceful skip without aborting).
  - External validation runner producing complete metrics dictionary.
- **Statistical Significance Testing (F12–F15):**
  - One-Way ANOVA F-statistic and p-value calculation via `scipy.stats.f_oneway`.
  - Significance decision at $\alpha = 0.05$ threshold (`significant: True/False`).
  - Degenerate zero-variance protection returning $F=0.0, p=1.0$ without division-by-zero crash.
  - Post-hoc pairwise analysis with Bonferroni correction ($\alpha_{\text{adj}} = \alpha / \binom{K}{2}$).
  - Group empirical means and standard deviations reporting.

### Tier 2: Boundary & Corner Cases ($\ge 5$ tests per feature area)
Tier 2 stress-tests extreme conditions, edge inputs, and potential failure modes:
- **AUROC Boundary Cases:**
  - Single-class ground truth edge case: returns `np.nan` with a warning, never raises unhandled exception.
  - All identical prediction probabilities (AUROC $= 0.5$).
  - Extreme probability scores (floating point boundaries: $0.0, 1.0, 10^{-15}, 1 - 10^{-15}$).
  - Mismatched vector lengths and empty array handling raising descriptive `ValueError`.
  - JSON/CSV export to invalid file paths raising descriptive `IOError`.
- **Cross-Validation Boundary Cases:**
  - Minimal dataset size ($N=K=5$, 1 sample per validation fold).
  - Severe class imbalance ($95\%$ negative, $5\%$ positive) verifying balanced distribution.
  - Invalid fold parameters ($K \le 1$ or negative) raising descriptive `ValueError`.
  - Empty dataset handling raising descriptive `ValueError`.
  - Random seed reproducibility (fixed seed gives identical splits; differing seed produces variations).
  - Sample size not cleanly divisible by $K$ (e.g. $N=53, K=5$).
- **External Ingestion Boundary Cases:**
  - Missing label column in external dataset raising clear `ValueError` or `SchemaError`.
  - Empty input files (0-byte CSV/JSONL) raising descriptive error.
  - Unrecognized label strings logging warnings without terminating the batch.
  - Non-existent benchmark identifier or directory path raising clear error.
  - Whitespace and casing resilience (`"  HuMaN  \n"` $\to 0$).
  - 100% corrupted dataset error handling.
- **ANOVA Significance Boundary Cases:**
  - Constant identical scores across all groups ($SS_{\text{within}} = 0$) returning $F=0.0, p=1.0$.
  - Insufficient groups ($K < 2$) raising `ValueError`.
  - Insufficient samples per group ($n_i < 2$) raising `ValueError`.
  - Empty group lists raising descriptive `ValueError`.
  - Exact $\alpha$ threshold boundary test ($p = 0.04999 \implies$ True, $p = 0.05001 \implies$ False).
  - Sanitization of `NaN` or `Inf` metric scores.

### Tier 3: Cross-Feature Interactions
Tier 3 tests end-to-end multi-module coordination:
1. **Ingest $\to$ CV $\to$ AUROC Metric Summary:** Ingest external dataset, partition into 5 stratified folds, compute per-fold AUROC and classification metrics, and aggregate summary statistics.
2. **Multi-Model ANOVA Comparison:** Execute 5-fold CV across two candidate model configurations and pass fold AUROC distributions to ANOVA to statistically verify superiority.
3. **Ablation Baseline ANOVA Comparison:** Evaluate baseline vs feature-ablated detector across folds and identify statistically significant components using post-hoc Bonferroni.
4. **Cross-Validation ROC Coordinate Aggregation:** Compute and export ROC curve coordinates for each fold, verifying coordinate integrity and monotonicity across all splits.
5. **Corrupted Ingestion Fault Tolerance in CV:** External dataset with $20\%$ corrupted records is cleanly filtered during ingestion, allowing remaining valid data to complete 5-fold CV without error.
6. **Discrete Fallback Integration in CV and ANOVA:** Detectors providing only binary classifications trigger the trapezoidal fallback in CV, and resulting balanced accuracy distributions are successfully tested in ANOVA.

### Tier 4: Real-World Workload Scenarios
Tier 4 reproduces realistic deployment pipelines:
1. **CART AI Text Detection Workload:** Realistic synthetic human vs AI text corpus with stylistic features evaluated under 5-fold CV with AUROC reporting and ROC JSON export.
2. **JAW Synthetic Image Workload:** Multimodal image feature representations evaluated across batches with AUROC calculation and ASCII ROC curve display.
3. **Phishing Email Workload:** Severe class imbalance ($95\%$ ham, $5\%$ phishing) evaluated with leak-free TF-IDF training across 5 stratified folds.
4. **Benchmark Suite 3-Way ANOVA Workload:** Simulated external benchmark (e.g. HC3/CIFAKE) comparing three detector variants followed by 3-way ANOVA with post-hoc Bonferroni.
5. **CLI End-to-End Simulation Workload:** Simulation of command-line invocation flags (`--cross-val 5`, `--eval-external`) executing full pipeline and producing summary outputs.

---

## 4. Expected Output Derivation & Authoritative Reference Sources

| Component | Test Input | Authoritative Reference / Oracle | Verification Standard |
|---|---|---|---|
| **Binary AUROC** | True labels $y \in \{0, 1\}$, probs $s \in [0, 1]$ | `sklearn.metrics.roc_auc_score` | Equal within numerical tolerance $\pm 10^{-6}$ |
| **ROC Coordinates** | Scores and binary ground truth | `sklearn.metrics.roc_curve` | FPR, TPR monotonically increasing; start at $(0,0)$, end at $(1,1)$ |
| **Binary Fallback** | Discrete predictions $\hat{y} \in \{0, 1\}$ | `sklearn.metrics.balanced_accuracy_score` | Trapezoidal area $= \frac{\text{TPR} + (1-\text{FPR})}{2}$ |
| **Stratified CV** | $N$ samples, binary labels $y$ | `sklearn.model_selection.StratifiedKFold` | Fold proportions match within $\le 1/N$; pairwise intersection $\emptyset$ |
| **Aggregate Stats** | $K$ fold metric values $[M_1, \dots, M_K]$ | Analytical sample mean $\mu$, std $\sigma$ | $\mu = \frac{1}{K} \sum M_i, \sigma = \sqrt{\frac{\sum (M_i - \mu)^2}{K-1}}$ |
| **One-Way ANOVA** | Score groups $G_1, \dots, G_K$ | `scipy.stats.f_oneway(*groups)` | F-stat and p-value match scipy oracle; decision matches $p < 0.05$ |
| **Post-Hoc Test** | Pairwise groups $(G_i, G_j)$ | Bonferroni adjusted $\alpha / \binom{K}{2}$ | Pairwise t-test against adjusted threshold |
| **Schema Ingestion** | CSV / JSONL with variant column names | Priority header match list | Correctly extracts text and normalizes label aliases |

---

## 5. Test Suite Execution & Organization

The automated tests are located at:
`tests/test_e2e_framework.py`

### Test Runner Commands
```bash
# Run entire test suite
pytest tests/ -v

# Run only the E2E framework test suite
pytest tests/test_e2e_framework.py -v

# Run specific tier
pytest tests/test_e2e_framework.py -k "tier1" -v
pytest tests/test_e2e_framework.py -k "tier2" -v
pytest tests/test_e2e_framework.py -k "tier3" -v
pytest tests/test_e2e_framework.py -k "tier4" -v

# Run specific feature area
pytest tests/test_e2e_framework.py -k "auroc" -v
pytest tests/test_e2e_framework.py -k "cv" -v
pytest tests/test_e2e_framework.py -k "external" -v
pytest tests/test_e2e_framework.py -k "anova" -v
```

---

## 6. Summary of Test Quantities

| Test Category | Target Scope | Test Count |
|---|---|:---:|
| **Tier 1: Feature Coverage** | AUROC (8), CV (6), Ingestion (7), ANOVA (5) | **26** |
| **Tier 2: Boundary & Corner Cases** | AUROC (6), CV (6), Ingestion (6), ANOVA (6) | **24** |
| **Tier 3: Cross-Feature Interactions** | End-to-end multi-module pipelines | **6** |
| **Tier 4: Real-World Workload Scenarios** | Realistic CART, JAW, Phishing, Benchmark pipelines | **5** |
| **Total Test Suite** | Comprehensive Opaque-Box E2E Tests | **61** |
