# SafeStep: Newcomer's Guide to Training & Evaluation

Welcome to **SafeStep**! This guide walks you step-by-step through setting up, training models, running 5-fold cross-validation, computing AUROC metrics, evaluating on external benchmark datasets, and performing statistical significance testing (ANOVA).

---

## 📋 Table of Contents
1. [Prerequisites & Quick Setup](#1-prerequisites--quick-setup)
2. [SafeStep Models Overview](#2-safestep-models-overview)
3. [Training Models from Scratch](#3-training-models-from-scratch)
4. [5-Fold Stratified Cross-Validation](#4-5-fold-stratified-cross-validation)
5. [AUROC Evaluation & ROC Curves](#5-auroc-evaluation--roc-curves)
6. [Validating on External Datasets](#6-validating-on-external-datasets)
7. [Statistical Significance Testing (ANOVA)](#7-statistical-significance-testing-anova)
8. [Python API Integration](#8-python-api-integration)
9. [Frequently Asked Questions & Troubleshooting](#9-frequently-asked-questions--troubleshooting)

---

## 1. Prerequisites & Quick Setup

SafeStep requires **Python 3.10+**. Follow these simple steps:

```bash
# 1. Clone the repository and enter directory
git clone https://github.com/SafeStep/Safestep-Models.git
cd Safestep-Models

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env
```

---

## 2. SafeStep Models Overview

| Model | Target | Description |
|---|---|---|
| **`cart`** | AI-Generated Text | Hybrid model combining DistilBERT contextual embeddings + stylistic heuristics fused via XGBoost. |
| **`jaw`** | AI-Generated Images | EfficientNet-B3 backbone fused with FFT spectral analysis and Gabor texture features. |
| **`phishing`** | Phishing Emails & URLs | 3-stage escalation pipeline: Attack (TF-IDF + LR) $\rightarrow$ Beast (XGBoost) $\rightarrow$ Warhammer (DistilBERT). |

---

## 3. Training Models from Scratch

### Quick Demo Training (No data needed)
If you don't have large training datasets downloaded yet, test the complete training pipeline using synthetic demo data:

```bash
# Train CART in demo mode
python train.py cart --demo

# Or using the convenient train.sh shell helper
./train.sh demo cart
```

### Full Training on Real Data
To train on real data stored in `data/`:

```bash
# Train CART (text detector)
python train.py cart --data-dir data/

# Train JAW (image detector) with GPU acceleration
python train.py jaw --device cuda --data-dir data/

# Train Phishing detector (auto-downloads public HF datasets on first run)
python train.py phishing
```

---

## 4. 5-Fold Stratified Cross-Validation

Cross-validation splits the dataset into 5 balanced folds, iteratively training on 4 folds and evaluating on the 5th to prevent data leakage and ensure generalizability.

### Run Cross-Validation via CLI:
```bash
# Run 5-fold cross-validation on CART
python train.py cart --cross-val 5

# Run 10-fold cross-validation on JAW
python train.py jaw --cross-val 10

# Shortcut flag
python train.py phishing -cv 5
```

### Sample Output:
```text
                     5-Fold Cross-Validation Metrics (CART)                     
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Fold       ┃ Accuracy   ┃ Precision  ┃ Recall     ┃ F1          ┃ AUROC      ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Fold 1     │ 0.8950     │ 0.8850     │ 0.9050     │ 0.8950      │ 0.9350     │
│ Fold 2     │ 0.9050     │ 0.8950     │ 0.9150     │ 0.9050      │ 0.9450     │
│ Fold 3     │ 0.9150     │ 0.9050     │ 0.9250     │ 0.9150      │ 0.9550     │
│ Fold 4     │ 0.9250     │ 0.9150     │ 0.9350     │ 0.9250      │ 0.9650     │
│ Fold 5     │ 0.9350     │ 0.9250     │ 0.9450     │ 0.9350      │ 0.9750     │
├────────────┼────────────┼────────────┼────────────┼─────────────┼────────────┤
│ Mean ± Std │ 0.9150 ±   │ 0.9050 ±   │ 0.9250 ±   │ 0.9150 ±    │ 0.9550 ±   │
│            │ 0.0158     │ 0.0158     │ 0.0158     │ 0.0158      │ 0.0158     │
└────────────┴────────────┴────────────┴────────────┴─────────────┴────────────┘
```

---

## 5. AUROC Evaluation & ROC Curves

Area Under the Receiver Operating Characteristic (AUROC) measures classification discrimination across all classification thresholds, bound strictly between `0.0` and `1.0`.

### CLI Output & Export:
When evaluating models or running cross-validation, AUROC is calculated automatically. You can export ROC curve points (FPR, TPR, thresholds) to JSON or CSV:

```bash
# Evaluate and export curve points to JSON
python train.py cart --eval-external path/to/dataset.csv --export-roc outputs/cart_roc.json

# Export to CSV
python train.py cart --eval-external path/to/dataset.csv --export-roc outputs/cart_roc.csv
```

### Visual ASCII ROC Plot:
SafeStep renders an in-terminal ASCII ROC curve for rapid inspection without needing a graphical display:
```text
Receiver Operating Characteristic (ROC) Curve:
  TPR ^
 1.0 |****************************************
     |*                                  ...  
     |*                                ..     
     |*                             ...       
     |*                          ...          
 0.5 |*                  ..                   
     |*            ...                        
     |*      ...                              
 0.0 |*.                                      
     +----------------------------------------
      0.0                0.5               1.0
             FPR (False Positive Rate)        
  Legend: * = ROC Curve, . = Random Baseline (AUC = 0.50)
```

---

## 6. Validating on External Datasets

SafeStep features an automated **External Dataset Ingester** that supports `.csv`, `.jsonl`, `.json`, and image folder structures.

### Supported File Formats:
1. **CSV files** (e.g. `benchmark.csv`):
   ```csv
   text,label
   "Sample human written text",0
   "Sample AI generated text",1
   ```
2. **JSONL files** (e.g. `benchmark.jsonl`):
   ```json
   {"content": "Human written message", "target": "human"}
   {"content": "Generated text from LLM", "target": "ai"}
   ```

### Flexible Column Names & Label Aliases
You don't need to rename your columns! SafeStep automatically detects:
- **Text/Image Columns**: `text`, `content`, `message_body`, `body`, `message`, `email`, `url`, `image`, `path`, etc.
- **Label Columns**: `label`, `is_bot`, `target`, `class`, `category`, `status`, etc.
- **Label Aliases**:
  - `0`: `human`, `clean`, `authentic`, `real`, `ham`, `safe`
  - `1`: `ai`, `chatgpt`, `synthetic`, `fake`, `phishing`, `malicious`

### Run External Evaluation:
```bash
python train.py cart --eval-external benchmarks/hc3_benchmark.csv
```

---

## 7. Statistical Significance Testing (ANOVA)

When comparing multiple models, architectures, or cross-validation fold distributions, SafeStep uses **One-Way Analysis of Variance (ANOVA)** to compute the $F$-statistic and $p$-value at significance threshold $\alpha = 0.05$.

### Run ANOVA with Cross-Validation:
```bash
python train.py cart --cross-val 5 --significance-test
```

### Standalone ANOVA Comparison:
```bash
# Run comparison across model variants
python train.py --significance-test

# Or directly via module execution
python -m training.significance
```

### Interpreting ANOVA Results:
* **$p < 0.05$ (Significant Difference: YES)**: The performance difference between the models/folds is statistically significant and unlikely due to random variance.
* **$p \ge 0.05$ (Significant Difference: NO)**: The models perform statistically equivalently.
* **Post-Hoc Pairwise Comparisons**: For $K \ge 3$ groups, Bonferroni-corrected pairwise tests isolate which specific pairs differ.

---

## 8. Python API Integration

You can easily import SafeStep's evaluation tools directly into your own scripts:

```python
from training.evaluation import compute_evaluation_metrics, render_ascii_roc_curve
from training.cross_validation import StratifiedCrossValidator
from training.significance import ANOVATester
from training.external import ExternalDatasetIngester

# 1. Compute AUROC & ROC metrics
y_true = [0, 0, 1, 1]
y_prob = [0.15, 0.25, 0.85, 0.90]
metrics = compute_evaluation_metrics(y_true, y_prob)
print(f"AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}")
print(render_ascii_roc_curve(metrics["roc_curve"]))

# 2. 5-Fold Stratified Cross-Validation
cv = StratifiedCrossValidator(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {fold}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")

# 3. Run ANOVA test
scores = {
    "Model_A": [0.85, 0.86, 0.88, 0.87, 0.89],
    "Model_B": [0.93, 0.95, 0.94, 0.96, 0.95],
}
anova = ANOVATester.run_one_way_anova(scores, alpha=0.05, metric_name="AUROC")
print(f"Significant difference? {anova['significant']} (p = {anova['p_value']:.4e})")
```

---

## 9. Frequently Asked Questions & Troubleshooting

### Q: What if my external dataset has corrupted lines?
SafeStep's `ExternalDatasetIngester` automatically logs a warning, skips the unparseable rows, and evaluates the valid samples without crashing.

### Q: Can I run this without a GPU?
Yes! SafeStep automatically detects available devices (`cuda`, `mps`, or `cpu`). You can force CPU execution at any time by specifying `--device cpu`.

### Q: How do I run the automated test suite?
Verify your local setup by running pytest:
```bash
# Run all tests
pytest tests/

# Run the E2E verification test suite
pytest tests/test_e2e_framework.py -v
```
