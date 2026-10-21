# SafeStep: Newcomer's Guide to Training & Evaluation

Welcome to **SafeStep**! This guide walks you through setting up the environment, understanding the underlying detection architectures, training models from scratch, running 5-fold stratified cross-validation, computing AUROC metrics with ROC curve exports, evaluating on external benchmark datasets, and performing statistical significance testing (One-Way ANOVA).

---

## 📋 Table of Contents
1. [Prerequisites & Quick Setup](#1-prerequisites--quick-setup)
2. [SafeStep Models Overview & Design Philosophy](#2-safestep-models-overview--design-philosophy)
3. [Training Models from Scratch](#3-training-models-from-scratch)
4. [5-Fold Stratified Cross-Validation & Mathematical Theory](#4-5-fold-stratified-cross-validation--mathematical-theory)
5. [AUROC Evaluation & ROC Curves](#5-auroc-evaluation--roc-curves)
6. [Validating on External Datasets (Ingestion Engine)](#6-validating-on-external-datasets-ingestion-engine)
7. [Statistical Significance Testing (ANOVA & Bonferroni)](#7-statistical-significance-testing-anova--bonferroni)
8. [Python API Integration Recipes](#8-python-api-integration-recipes)
9. [Frequently Asked Questions & Troubleshooting](#9-frequently-asked-questions--troubleshooting)

---

## 1. Prerequisites & Quick Setup

SafeStep requires **Python 3.10+**. Follow these simple setup steps:

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

## 2. SafeStep Models Overview & Design Philosophy

Monolithic deep learning models often fail in security contexts due to adversarial perturbations, shortcut learning, and high inference latency. SafeStep solves this with **hybrid architectures** and **confidence escalation**:

| Model | Target Domain | Underlying Design | Why This Outperforms Baselines |
|---|---|---|---|
| **`cart`** | AI-Generated Text | DistilBERT contextual embeddings + 12 stylistic heuristics fused via XGBoost | Pure transformers are vulnerable to paraphrasing and prompt injection; hand-engineered heuristics (unigram entropy perplexity, burstiness, punctuation regularity) provide robust invariant signals. |
| **`jaw`** | AI-Generated Images | EfficientNet-B3 convolutional backbone fused with 2D-FFT radial spectral energy and Gabor wavelet texture banks | Visual generators (GANs, Diffusion) produce imperceptible frequency grid artifacts and unnatural spectral flatness that are easily detected in the Fourier and Gabor domains even when visually flawless. |
| **`phishing`** | Phishing Emails & URLs | 3-stage escalation cascade: Attack (Char N-gram TF-IDF) $\rightarrow$ Beast (XGBoost) $\rightarrow$ Warhammer (DistilBERT) | High-volume traffic is filtered by ultra-fast lexical ($<0.5$ ms) and structural ($<2$ ms) stages; only genuinely ambiguous samples escalate to GPU-intensive transformers. |

---

## 3. Training Models from Scratch

### Quick Demo Training (Instant Validation)
If you do not have large datasets downloaded, test the entire multi-stage training pipeline instantly using synthetic demo data:

```bash
# Train CART in demo mode
python train.py cart --demo

# Train JAW in demo mode
python train.py jaw --demo

# Train Phishing in demo mode
python train.py phishing --demo -v

# Or using the train.sh shell helper
./train.sh demo all
```

### Full Training on Real Datasets
When training on real datasets stored in `data/`:

```bash
# Train CART (text detector)
python train.py cart --data-dir data/

# Train JAW (image detector) with GPU acceleration
python train.py jaw --device cuda --data-dir data/

# Train Phishing detector (auto-downloads public Hugging Face datasets on first run)
python train.py phishing
```

---

## 4. 5-Fold Stratified Cross-Validation & Mathematical Theory

Cross-validation partitions the dataset into $K$ disjoint subsets (folds). The model iteratively trains on $K-1$ folds and evaluates on the held-out fold, ensuring every sample serves as a test instance exactly once without data leakage.

### The Stratification Condition
Given a binary dataset with positive class prior $p = \frac{N_{pos}}{N}$, random splitting introduces fold-to-fold class distribution variance. **Stratified splitting** preserves the exact class proportion across all folds:

$$\frac{N_{k, pos}}{N_k} \approx \frac{N_{pos}}{N}, \quad \forall k \in \{1, \dots, K\}$$

### Unbiased Sample Variance (Bessel's Correction)
To aggregate fold metrics $\{m_1, m_2, \dots, m_K\}$, SafeStep computes the sample mean and unbiased sample standard deviation ($\text{ddof} = 1$):

$$\bar{\mu} = \frac{1}{K} \sum_{k=1}^K m_k$$

$$s = \sqrt{\frac{1}{K - 1} \sum_{k=1}^K (m_k - \bar{\mu})^2}$$

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

The Receiver Operating Characteristic (ROC) curve evaluates classification performance across all possible decision thresholds $t \in [0, 1]$.

### Area Under the Curve (AUROC) Formulation
Given $M$ unique classification thresholds yielding curve points $(\text{FPR}_1, \text{TPR}_1), \dots, (\text{FPR}_M, \text{TPR}_M)$, AUROC is computed via the **Composite Trapezoidal Riemann Sum**:

$$\text{AUROC} = \sum_{i=1}^{M-1} \left( \text{FPR}_{i+1} - \text{FPR}_i \right) \cdot \frac{\text{TPR}_i + \text{TPR}_{i+1}}{2}$$

AUROC is bounded strictly between $0.0$ and $1.0$:
- $\text{AUROC} = 0.50$: Performance equivalent to random chance.
- $\text{AUROC} = 1.00$: Perfect separation with zero false positives or false negatives.

### CLI Export:
Export exact ROC coordinate points (`fpr`, `tpr`, `thresholds`) to JSON or CSV:

```bash
# Evaluate and export curve points to JSON
python train.py cart --eval-external benchmarks/test_set.csv --export-roc outputs/cart_roc.json

# Export to CSV
python train.py cart --eval-external benchmarks/test_set.csv --export-roc outputs/cart_roc.csv
```

### In-Terminal ASCII ROC Curve:
SafeStep provides instant visual verification in the console without requiring graphical windows:
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

## 6. Validating on External Datasets (Ingestion Engine)

SafeStep features an automated **External Dataset Ingester** (`training/external.py`) that supports `.csv`, `.jsonl`, `.json`, and structured image directories with schema auto-inference.

### Supported File Formats:
1. **CSV files** (`benchmark.csv`):
   ```csv
   text,label
   "Sample human written text",0
   "Sample AI generated text",1
   ```
2. **JSONL files** (`benchmark.jsonl`):
   ```json
   {"content": "Human written message", "target": "human"}
   {"content": "Generated text from LLM", "target": "ai"}
   ```

### Schema Auto-Inference:
You do not need to rename your columns! SafeStep automatically detects:
- **Text/Image Columns**: `text`, `content`, `message_body`, `body`, `message`, `email`, `url`, `image`, `path`, `filename`.
- **Label Columns**: `label`, `is_bot`, `target`, `class`, `category`, `status`.
- **Label Aliases**:
  - Positive ($1$): `ai`, `chatgpt`, `synthetic`, `fake`, `phishing`, `malicious`, `spam`, `1`.
  - Negative ($0$): `human`, `clean`, `authentic`, `real`, `ham`, `safe`, `legitimate`, `0`.

### Run External Evaluation:
```bash
python train.py cart --eval-external benchmarks/hc3_benchmark.csv
```

---

## 7. Statistical Significance Testing (ANOVA & Bonferroni)

When comparing multiple models, architectures, or cross-validation distributions, SafeStep uses **One-Way Analysis of Variance (ANOVA)** (`training/significance.py`) to compute the $F$-statistic and $p$-value at significance threshold $\alpha = 0.05$.

### Mathematical Formulation
Total variation decomposes into between-group sum of squares ($SS_{between}$) and within-group sum of squares ($SS_{within}$):

$$SS_{between} = \sum_{j=1}^k n_j (\bar{x}_j - \bar{x}_{\cdot\cdot})^2, \quad df_{between} = k - 1$$

$$SS_{within} = \sum_{j=1}^k \sum_{i=1}^{n_j} (x_{ij} - \bar{x}_j)^2, \quad df_{within} = N_{total} - k$$

$$F = \frac{MS_{between}}{MS_{within}} = \frac{SS_{between} / df_{between}}{SS_{within} / df_{within}} \sim \mathcal{F}(df_{between}, df_{within})$$

The $p$-value is derived from the cumulative Fisher-Snedecor distribution:

$$p = 1 - F_{cdf}(F; df_{between}, df_{within})$$

### Post-Hoc Pairwise Comparisons (Bonferroni Correction)
When comparing $k \ge 3$ models, SafeStep conducts pairwise Welch's $t$-tests across all $m = \frac{k(k-1)}{2}$ pairs, correcting for family-wise error inflation:

$$\alpha_{bonf} = \frac{\alpha}{m}, \quad p_{adj} = \min(1.0, m \cdot p_{raw})$$

### Run ANOVA via CLI:
```bash
# Run 5-fold cross-validation with ANOVA testing across folds
python train.py cart --cross-val 5 --significance-test

# Standalone ANOVA module execution
python -m training.significance
```

---

## 8. Python API Integration Recipes

You can import SafeStep's evaluation and detection modules directly into your Python scripts:

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

# 2. Stratified Cross-Validation Splitting
import numpy as np
X = np.arange(100)
y = np.array([0] * 50 + [1] * 50)

cv = StratifiedCrossValidator(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {fold}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")

# 3. One-Way ANOVA Testing Across Models
scores = {
    "Baseline_Transformer": [0.82, 0.84, 0.83, 0.81, 0.85],
    "SafeStep_Hybrid":       [0.93, 0.95, 0.94, 0.96, 0.95],
}
anova = ANOVATester.run_one_way_anova(scores, alpha=0.05, metric_name="AUROC")
print(f"Statistically significant? {anova['significant']} (F={anova['f_statistic']:.2f}, p={anova['p_value']:.4e})")
```

---

## 9. Frequently Asked Questions & Troubleshooting

### Q: What happens if an external dataset contains corrupted lines?
The `ExternalDatasetIngester` automatically catches format errors, logs a clear warning with line and row numbers, skips invalid samples, and evaluates all valid records without interrupting execution.

### Q: Can I run training and evaluation on a CPU?
Yes. SafeStep dynamically senses host hardware. If no CUDA or MPS GPU is available, it gracefully configures multi-worker CPU loaders. You can also explicitly force CPU execution via `--device cpu`.

### Q: How do I verify my installation?
Run the full automated test suite using `pytest`:
```bash
pytest tests/ -v
```
All 174 tests should pass cleanly!
