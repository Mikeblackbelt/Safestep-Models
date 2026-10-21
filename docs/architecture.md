# SafeStep Architecture: Technical Specification & System Design

SafeStep is a modular security inspection suite comprising three specialized detectors and an orchestration layer for multimodal content verification:

1. **CART** (`cart/`): Context-Aware Representation for Text (DistilBERT + Stylistic Heuristic Features + XGBoost Meta-Fusion).
2. **JAW** (`jaw/`): Joint Artifact Wavelet Image Classifier (EfficientNet-B3 + FFT Radial Spectral Energy + Gabor Wavelets).
3. **Phishing** (`phishing/`): High-Throughput Confidence Escalation Pipeline (**Attack** $\rightarrow$ **Beast** $\rightarrow$ **Warhammer**).
4. **Unified API & Model Registry** (`api/`): Unified FastAPI service and in-memory Python inference engine with dynamic weight reloading.

```
                               ┌────────────────────────────────────────────────────────┐
                               │                 SafeStep Input Pipeline                │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                   ┌──────────────────────────────────────┼──────────────────────────────────────┐
                   │                                      │                                      │
                   ▼ Text                                 ▼ Image                                ▼ URL / Email
       ┌───────────────────────┐              ┌───────────────────────┐              ┌───────────────────────┐
       │     CART Detector     │              │     JAW Detector      │              │   Phishing Pipeline   │
       │ (DistilBERT + Heur.)  │              │  (EfficientNet + FFT) │              │  (Attack/Beast/War.)  │
       └───────────┬───────────┘              └───────────┬───────────┘              └───────────┬───────────┘
                   │                                      │                                      │
                   └──────────────────────────────────────┼──────────────────────────────────────┘
                                                          ▼
                                      ┌───────────────────────────────────────┐
                                      │        Multimodal Unified Engine      │
                                      │   (Risk Scorer & Consensus Policy)    │
                                      └───────────────────┬───────────────────┘
                                                          ▼
                                      ┌───────────────────────────────────────┐
                                      │         Structured Detection JSON     │
                                      │    (p_ai, risk_score, stage, etc.)    │
                                      └───────────────────────────────────────┘
```

---

## 1. CART Architecture (AI-Generated Text Detection)

CART uses a **hybrid dual-branch architecture**: a deep contextual transformer extracts semantic and discourse-level representations, while an engineered statistical extractor captures surface-level stylistic markers, token frequency variance, and punctuation patterns.

### 1.1 Structural Data Flow

```
Input Text String (up to 512 subwords)
   │
   ├──► DistilBERT Tokenizer ──► input_ids (B, 512), attention_mask (B, 512)
   │        │
   │        ▼
   │    DistilBERT Encoder (6 transformer layers, 12 attention heads, hidden_dim = 768)
   │        │
   │        ▼
   │    Extract [CLS] Token Embedding: h_0 ∈ R^(B x 768)
   │        │
   │        ▼
   │    Classification Head:
   │        Linear(768, 256) ──► ReLU ──► Dropout(p=0.2) ──► Linear(256, 2)
   │        │
   │        ▼
   │    Softmax Function ──► p_neural = [p_human, p_ai] ∈ R^(B x 2)
   │                                  │
   └──► HeuristicExtractor            │
            │                         │
            ▼                         │
        12 Stylistic Metrics ∈ R^(B x 12)
            │                         │
            └───────────┬─────────────┘
                        ▼
            Feature Concatenation Layer: f_fused ∈ R^(B x 14)
                        │
                        ▼
            CARTFusionModel (XGBoost Meta-Classifier, 100 trees, depth 4)
                        │
                        ▼
            Final Calibrated Output: p_ai ∈ [0.0, 1.0], label ∈ {human, ai_generated}
```

### 1.2 Tensor Transformations & Dimension Accounting

| Layer / Component | Input Shape | Output Shape | Parameters / Dimension | Activation / Operation |
|---|---|---|---|---|
| `DistilBERT Tokenizer` | Raw string | `(B, 512)` | WordPiece vocab ($V = 30,522$) | Padding to 512, truncation |
| `DistilBERT Embeddings` | `(B, 512)` | `(B, 512, 768)` | Token + Positional embeddings | LayerNorm + Dropout |
| `6x Transformer Blocks` | `(B, 512, 768)` | `(B, 512, 768)` | 66.4M weights | Multi-head self-attention |
| `[CLS] Pooling` | `(B, 512, 768)` | `(B, 768)` | 0 (Slice `[:, 0, :]`) | First token extraction |
| `Classifier Dense 1` | `(B, 768)` | `(B, 256)` | $768 \times 256 + 256 = 196,864$ | ReLU + Dropout ($p=0.2$) |
| `Classifier Dense 2` | `(B, 256)` | `(B, 2)` | $256 \times 2 + 2 = 514$ | Identity (raw logits) |
| `Neural Softmax` | `(B, 2)` | `(B, 2)` | 0 | $\text{Softmax}(\mathbf{z}, \text{dim}=-1)$ |
| `Heuristic Extractor` | Raw string | `(B, 12)` | 0 (Algorithmic extraction) | Deterministic statistics |
| `Concatenation` | `(B, 2)` & `(B, 12)` | `(B, 14)` | 0 | Matrix concatenation |
| `XGBoost Fusion` | `(B, 14)` | `(B, 2)` | 100 boosted trees | Logistic loss meta-classification |

### 1.3 Stylistic Heuristic Feature Dictionary

CART extracts 12 deterministic features that remain invariant to language model scaling:
1. `perplexity_proxy`: $2^{H(X)}$ derived from unigram token frequency entropy.
2. `burstiness`: Coefficient of variation ($\sigma / \mu$) of token recurrence.
3. `repetition_ratio`: Ratio of vocabulary $n$-grams ($n \in [2, 5]$) appearing $> 2$ times.
4. `unique_ngram_ratio`: Ratio of unique $n$-grams to total $n$-grams.
5. `max_ngram_repeat`: Highest count of any individual repeating $n$-gram.
6. `punctuation_regularity`: Coefficient of variation of inter-punctuation character spans.
7. `sentence_length_mean`: Mean token count per sentence.
8. `sentence_length_std`: Standard deviation of sentence lengths.
9. `capitalization_consistency`: Proportion of sentences beginning with an uppercase letter.
10. `rare_word_ratio`: Proportion of single-occurrence words (hapax legomena).
11. `transition_word_density`: Frequency of formulaic connective words per total words.
12. `avg_word_length`: Mean character length per word.

---

## 2. JAW Architecture (AI-Generated Image Detection)

JAW (**J**oint **A**rtifact **W**avelet Image Classifier) pairs a deep convolutional spatial encoder with a deterministic frequency-domain analysis engine.

### 2.1 Structural Data Flow

```
Input Image Array (H x W x 3)
   │
   ├──► RGB Preprocessing (Resize 224x224, ImageNet Normalization)
   │        │
   │        ▼
   │    EfficientNet-B3 Convolutional Backbone (MBConv blocks with squeeze-and-excitation)
   │        │
   │        ▼
   │    Global Average Pooling ──► v_spatial ∈ R^(B x 1536)
   │                                  │
   └──► Luminance Conversion (Y = 0.299R + 0.587G + 0.114B)
            │
            ▼
        FrequencyFeatureExtractor:
            ├─► 2D-FFT Radial Spectral Bands (8 bands, normalized energy) ──► R^8
            ├─► Spectral Shape (Normalized Peak Frequency + Spectral Flatness) ──► R^2
            └─► Gabor Wavelet Filter Bank (4 scales x 4 orientations convolved) ──► R^16
            │
            ▼
        v_freq ∈ R^(B x 26)
            │
            └─────────────────────────┬────────────────────────┘
                                      ▼
                      Concatenation: v_fused ∈ R^(B x 1562)
                                      │
                                      ▼
                      Dense Layer 1: Linear(1562, 512) ──► ReLU ──► Dropout(p=0.2)
                                      │
                                      ▼
                      Dense Layer 2: Linear(512, 128)  ──► ReLU ──► Dropout(p=0.2)
                                      │
                                      ▼
                      Logit Layer: Linear(128, 2)
                                      │
                                      ▼
                      Softmax Output: [p_authentic, p_ai]
```

### 2.2 Tensor Transformations & Dimension Accounting

| Component / Layer | Input Shape | Output Shape | Parameters / Dimension | Operational Details |
|---|---|---|---|---|
| `Input RGB Image` | Arbitrary `(H, W, 3)` | `(B, 3, 224, 224)` | 0 | Bilinear resize, ImageNet $\mu, \sigma$ |
| `EfficientNet-B3` | `(B, 3, 224, 224)` | `(B, 1536, 7, 7)` | 10.7M parameters | Depthwise separable convolutions |
| `Adaptive Avg Pool`| `(B, 1536, 7, 7)` | `(B, 1536)` | 0 | Spatial reduction across $7 \times 7$ grid |
| `Grayscale Convert`| Arbitrary `(H, W, 3)` | `(B, H, W)` | 0 | $Y = 0.299R + 0.587G + 0.114B$ |
| `2D FFT & Shift` | `(B, H, W)` | `(B, H, W)` complex | 0 | 2D-DFT + origin centering |
| `Radial Energy` | Centered magnitude | `(B, 8)` | 0 | 8 concentric radial rings, $L_1$ normalized |
| `Spectral Shape` | Centered magnitude | `(B, 2)` | 0 | Peak radial index + Wiener flatness |
| `Gabor Wavelet Bank`| Grayscale image | `(B, 16)` | 0 (16 static kernels) | 4 scales $\times$ 4 angles 2D convolutions |
| `Feature Concat` | `(B, 1536)` & `(B, 26)` | `(B, 1562)` | 0 | Vector concatenation |
| `Fusion Dense 1` | `(B, 1562)` | `(B, 512)` | $1562 \times 512 + 512 = 800,256$ | ReLU + Dropout ($p=0.2$) |
| `Fusion Dense 2` | `(B, 512)` | `(B, 128)` | $512 \times 128 + 128 = 65,664$ | ReLU + Dropout ($p=0.2$) |
| `Fusion Classifier`| `(B, 128)` | `(B, 2)` | $128 \times 2 + 2 = 258$ | Linear projection to class logits |

---

## 3. Phishing Escalation Pipeline Architecture

Rather than executing computationally expensive transformer inferences on all incoming emails and URLs, SafeStep routes traffic through a **dynamic confidence-gated escalation cascade**.

### 3.1 Three-Stage Pipeline Topology

```
Incoming Message / URL
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Attack (Ultra-fast Lexical Filter)            │
│   • Sublinear character n-grams (n=2..5, max 50,000)   │
│   • L2-regularized Logistic Regression (C=4.0)         │
│   • Latency: < 0.5 ms per sample                       │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
         Does p_phishing fall inside uncertainty band?
                   0.10 < p_phishing < 0.90
                        │
            No ─────────┴───────── Yes
            │                       │
            ▼                       ▼
    TERMINATE               ┌────────────────────────────────────────────────────────┐
    Verdict: Attack         │ Stage 2: Beast (Structural Meta-Classifier)            │
    (Stage = "attack")      │   • Concatenates Attack probabilities (2-d)            │
                            │   • 18 Structural URL & content indicators (18-d)      │
                            │   • XGBoost classifier (200 trees, depth 5)            │
                            │   • Latency: ~2.0 ms per sample                        │
                            └───────────────────────┬────────────────────────────────┘
                                                    │
                                                    ▼
                                     Does p_phishing fall inside uncertainty band?
                                               0.20 < p_phishing < 0.80
                                                    │
                                        No ─────────┴───────── Yes (and Warhammer ready)
                                        │                       │
                                        ▼                       ▼
                                TERMINATE               ┌────────────────────────────────────────────────────────┐
                                Verdict: Beast          │ Stage 3: Warhammer (Deep Semantic Transformer)         │
                                (Stage = "beast")       │   • DistilBERT contextual self-attention (max len 256) │
                                                        │   • Understands deceptive pretexts & impersonation     │
                                                        │   • Latency: 15-40 ms per sample                       │
                                                        └───────────────────────┬────────────────────────────────┘
                                                                                │
                                                                                ▼
                                                                        TERMINATE
                                                                        Verdict: Warhammer
                                                                        (Stage = "warhammer")
```

### 3.2 Stage Specifications & Uncertainty Boundaries

| Stage Name | Underlying Model | Input Features | Feature Dim | Execution Condition | Latency Profile |
|---|---|---|---|---|---|
| **Attack** | Scikit-learn Logistic Regression | Character $n$-grams (`char_wb`, $2\text{--}5$) | 50,000 | All samples | $< 0.5$ ms (CPU) |
| **Beast** | XGBoost Classifier | Attack probabilities $\oplus$ 18 indicators | 20 | $0.10 < p_{att} < 0.90$ | $\sim 2.0$ ms (CPU) |
| **Warhammer** | DistilBERT Transformer | Raw token sequence (max length 256) | 768 | $0.20 < p_{bst} < 0.80$ | $15\text{--}40$ ms (CPU/GPU) |

### 3.3 The 18 Structural Phishing Indicators

1. **`url_count`**: Total URLs identified in the text.
2. **`ip_url_ratio`**: Proportion of URLs using direct IPv4 addresses instead of hostnames.
3. **`shortener_url_ratio`**: Proportion of URLs utilizing URL shortening domains (`bit.ly`, `tinyurl.com`, `t.co`).
4. **`punycode_url_ratio`**: Proportion of domains containing `"xn--"` homograph disguises.
5. **`suspicious_tld_ratio`**: Proportion of URLs utilizing high-risk TLDs (`.xyz`, `.top`, `.tk`, `.click`, `.zip`).
6. **`at_in_url_ratio`**: Proportion of URLs containing `"@"`, routing browser credentials.
7. **`https_ratio`**: Ratio of secure HTTPS links to total links.
8. **`max_url_length`**: Maximum character length across all embedded URLs.
9. **`max_subdomain_depth`**: Highest count of subdomains observed across links.
10. **`max_domain_hyphens`**: Highest count of hyphens in a domain (common in spoofing).
11. **`url_digit_ratio`**: Proportion of numerical characters across all URLs (DGA signature).
12. **`urgency_density`**: Frequency of coercive terms ("immediately", "suspended", "urgent").
13. **`credential_density`**: Frequency of auth harvesting words ("password", "login", "SSN").
14. **`money_density`**: Frequency of financial lures ("wire", "refund", "invoice", "bitcoin").
15. **`exclamation_density`**: Count of exclamation marks normalized by total words.
16. **`uppercase_ratio`**: Proportion of capitalized alphabetic characters.
17. **`has_html_form`**: Boolean flag indicating presence of `<form>` or `<input>` tags.
18. **`log_length`**: Natural logarithm $\ln(1 + \text{text\_length})$.

---

## 4. Multimodal Unified Detection Engine

The SafeStep REST API exposes `POST /api/v1/detect/unified`, fusing textual and visual analyses into a consolidated threat assessment.

### 4.1 Fusion Logic & Risk Scoring
When evaluating a multimodal sample with text $T$ and image $I$:
1. **CART Evaluation:** Produces $p_{ai}^{text}$.
2. **JAW Evaluation:** Produces $p_{ai}^{image}$.
3. **Phishing Evaluation (Optional):** Produces $p_{phishing}$ and escalation `stage`.
4. **Risk Score Formulation:**
   $$\text{Base Score} = \max\left(p_{ai}^{text}, p_{ai}^{image}\right)$$
   If phishing detection is active and real weights are loaded:
   $$\text{Risk Score} = \max\left(\text{Base Score}, p_{phishing}\right)$$

### 4.2 Categorical Risk Levels
- **HIGH RISK** ($\text{Risk Score} \ge 0.80$): Immediate synthetic or malicious alert.
- **MEDIUM RISK** ($0.50 \le \text{Risk Score} < 0.80$): Flagged for human review.
- **LOW RISK** ($\text{Risk Score} < 0.50$): Content deemed authentic and benign.

---

## 5. Checkpoint Layout & Serialization Specifications

Each model persists its learned weights into dedicated subdirectories under `checkpoints/`:

```
checkpoints/
├── cart/
│   ├── neural_model.pt       # PyTorch state_dict for DistilBERT + classification head
│   └── fusion_model.json      # Serialized XGBoost JSON tree structure
├── jaw/
│   └── jaw_model.pt          # PyTorch state_dict for JAWModel (EfficientNet-B3 + MLP)
└── phishing/
    ├── attack.joblib          # Joblib binary: TfidfVectorizer + LogisticRegression Pipeline
    ├── beast.json             # Serialized XGBoost JSON tree structure
    └── warhammer.pt          # PyTorch state_dict for DistilBERT classifier (optional)
```

### 5.1 Dynamic Weight Reloading (Zero Downtime)
SafeStep detectors support live reloading via HTTP `POST /models/{model_id}/reload`. The model registry:
1. Validates that the requested checkpoint path sits securely within the approved base directory.
2. Constructs a fresh detector instance and loads weight binaries in isolated memory.
3. Swaps the internal pointer atomically, ensuring zero dropped requests during retraining cycles.
