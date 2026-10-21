# Training SafeStep Models: Architecture, Optimization & Mathematical Foundations

SafeStep provides unified, reproducible, and hardware-aware training pipelines for three production detection systems:
1. **CART** (`cart/`): Context-Aware Representation for Text (DistilBERT + Stylistic Heuristic Features + XGBoost Meta-Classifier).
2. **JAW** (`jaw/`): Joint Artifact Wavelet Image Classifier (EfficientNet-B3 + FFT Radial Spectral Energy + Gabor Wavelet Texture Banks).
3. **Phishing** (`phishing/`): Multi-tier Confidence Escalation Pipeline (**Attack** $\rightarrow$ **Beast** $\rightarrow$ **Warhammer**).

All training workflows are orchestrated through `train.py` (or wrappers `train.sh` and `train.ps1`), which manages data ingestion, device placement, multi-phase optimization, stratified cross-validation, and statistical significance testing.

---

## Table of Contents
- [1. Global Training Framework & Mathematical Formulations](#1-global-training-framework--mathematical-foundations)
  - [1.1 Optimization Objectives & Loss Formulations](#11-optimization-objectives--loss-formulations)
  - [1.2 Optimizer Mechanics (AdamW with Decoupled Weight Decay)](#12-optimizer-mechanics-adamw-with-decoupled-weight-decay)
  - [1.3 Learning Rate Scheduling & Warmup Schedules](#13-learning-rate-scheduling--warmup-schedules)
  - [1.4 Gradient Regularization & Early Stopping Mechanics](#14-gradient-regularization--early-stopping-mechanics)
- [2. CART Training Pipeline (AI-Generated Text Detection)](#2-cart-training-pipeline-ai-generated-text-detection)
  - [2.1 Multi-Stage Architecture Flow](#21-multi-stage-architecture-flow)
  - [2.2 Stage 1: Frozen Neural Encoder Training](#22-stage-1-frozen-neural-encoder-training)
  - [2.3 Stage 2: End-to-End Fine-Tuning](#23-stage-2-end-to-end-fine-tuning)
  - [2.4 Heuristic Feature Extraction Mathematics](#24-heuristic-feature-extraction-mathematics)
  - [2.5 Stage 3: XGBoost Meta-Fusion Training](#25-stage-3-xgboost-meta-fusion-training)
- [3. JAW Training Pipeline (AI-Generated Image Detection)](#3-jaw-training-pipeline-ai-generated-image-detection)
  - [3.1 Multi-Branch Architecture Flow](#31-multi-branch-architecture-flow)
  - [3.2 Preprocessing & Spatial Feature Extraction](#32-preprocessing--spatial-feature-extraction)
  - [3.3 Frequency-Domain Feature Extraction Mathematics](#33-frequency-domain-feature-extraction-mathematics)
  - [3.4 Two-Stage Training Protocol](#34-two-stage-training-protocol)
- [4. Phishing Escalation Pipeline Training](#4-phishing-escalation-pipeline-training)
  - [4.1 The Confidence Escalation Paradigm](#41-the-confidence-escalation-paradigm)
  - [4.2 Stage 1: Attack (Sublinear Char N-gram TF-IDF + Logistic Regression)](#42-stage-1-attack-sublinear-char-n-gram-tf-idf--logistic-regression)
  - [4.3 Stage 2: Beast (XGBoost Meta-Model + 18 Structural Indicators)](#43-stage-2-beast-xgboost-meta-model--18-structural-indicators)
  - [4.4 Stage 3: Warhammer (Contextual Semantic Transformer)](#44-stage-3-warhammer-contextual-semantic-transformer)
  - [4.5 Leak-Free Staged Training Protocol](#45-leak-free-staged-training-protocol)
- [5. Cross-Validation, Evaluation & Statistical Significance Mathematics](#5-cross-validation-evaluation--statistical-significance-mathematics)
  - [5.1 Leak-Free Stratified K-Fold Cross-Validation](#51-leak-free-stratified-k-fold-cross-validation)
  - [5.2 Classification Metrics & AUROC Riemann Sum](#52-classification-metrics--auroc-riemann-sum)
  - [5.3 One-Way Analysis of Variance (ANOVA) & Pairwise Tests](#53-one-way-analysis-of-variance-anova--pairwise-tests)
- [6. CLI Command Reference & Hardware Execution](#6-cli-command-reference--hardware-execution)
  - [6.1 Flag Reference](#61-flag-reference)
  - [6.2 Shell Wrappers (`train.sh` & `train.ps1`)](#62-shell-wrappers-trainsh--trainps1)
  - [6.3 Hardware Planner & Throughput Modeling](#63-hardware-planner--throughput-modeling)
  - [6.4 Checkpoint Layout & Hot Reloading](#64-checkpoint-layout--hot-reloading)

---

## 1. Global Training Framework & Mathematical Formulations

Training deep neural representations alongside gradient-boosted decision trees requires distinct mathematical optimization strategies. Neural branches minimize empirical risk over continuous manifolds via stochastic gradient descent, while tree ensembles minimize regularized loss functions via iterative second-order Taylor approximations.

### 1.1 Optimization Objectives & Loss Formulations

For neural classification heads in CART, JAW, and Warhammer, the primary training criterion is **Categorical Cross-Entropy** over mini-batches of size $B$. Given ground-truth one-hot label vector $\mathbf{y}_i \in \{0, 1\}^C$ (where $C=2$ for binary detection: authentic vs. synthetic, or legitimate vs. phishing) and raw model logit output vector $\mathbf{z}_i \in \mathbb{R}^C$:

$$\mathcal{L}_{CE}(\mathbf{z}_i, \mathbf{y}_i) = - \sum_{c=1}^C y_{i, c} \log(\hat{p}_{i, c})$$

where the predicted probability $\hat{p}_{i, c}$ is computed via the **Softmax function**:

$$\hat{p}_{i, c} = \frac{\exp(z_{i, c})}{\sum_{j=1}^C \exp(z_{i, j})}$$

For binary classification ($C=2$), let $y_i \in \{0, 1\}$ denote the binary target (with $1$ indicating AI-generated or phishing) and $\hat{p}_i = \hat{p}_{i, 1}$ denote the positive class probability. The loss reduces to **Binary Cross-Entropy (BCE)**:

$$\mathcal{L}_{BCE}(\hat{p}_i, y_i) = - \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

The batch empirical risk minimized during backpropagation across batch $\mathcal{B}$ is:

$$\mathcal{L}_{\mathcal{B}} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathcal{L}_{CE}(\mathbf{z}_i, \mathbf{y}_i)$$

The gradient with respect to logit $z_{i, c}$ exhibits the canonical error residual form:

$$\frac{\partial \mathcal{L}_{\mathcal{B}}}{\partial z_{i, c}} = \frac{1}{|\mathcal{B}|} \left( \hat{p}_{i, c} - y_{i, c} \right)$$

---

### 1.2 Optimizer Mechanics (AdamW with Decoupled Weight Decay)

All neural backbones in SafeStep use the **AdamW** optimizer (Loshchilov & Hutter, 2019). Unlike classical Adam—which mixes $L_2$ weight penalty into the running gradient moments—AdamW decouples weight decay from the moving averages of the gradients, preventing weights with large historical gradients from under-decaying.

At optimization time-step $t$, let $g_t = \nabla_\theta \mathcal{L}_{\mathcal{B}}(\theta_{t-1})$ be the stochastic gradient with respect to trainable parameter tensor $\theta$. AdamW maintains exponentially decaying first and second moment estimates:

1. **First Moment (Momentum):**
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   *(SafeStep default: $\beta_1 = 0.9$)*

2. **Second Uncentered Moment (Variance):**
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   *(SafeStep default: $\beta_2 = 0.999$)*

3. **Bias Corrections:**
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

4. **Decoupled Parameter Update:**
   $$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
   *(SafeStep defaults: $\epsilon = 10^{-8}$, $\lambda = \text{weight\_decay} = 0.01$)*

Here, $\lambda \theta_{t-1}$ directly contracts the parameter vector proportionally to the learning rate $\eta_t$, independent of gradient scaling $\sqrt{\hat{v}_t}$.

---

### 1.3 Learning Rate Scheduling & Warmup Schedules

SafeStep employs two specialized scheduling strategies depending on the backbone dynamics:

#### A. Linear Warmup with Linear Decay (CART & Phishing Warhammer)
Transformer architectures suffer from unstable early gradient updates when pre-trained weights encounter randomly initialized classification heads. SafeStep applies a linear warm-up phase over $T_{warmup}$ steps, followed by linear decay to zero over the remaining horizon $T_{total}$:

$$\eta(t) = \begin{cases} 
\eta_{base} \cdot \frac{t}{\max(1, T_{warmup})} & \text{if } t < T_{warmup} \\
\eta_{base} \cdot \max\left(0, \frac{T_{total} - t}{\max(1, T_{total} - T_{warmup})}\right) & \text{if } t \ge T_{warmup}
\end{cases}$$

For CART and Warhammer fine-tuning:
- $T_{warmup} = 100$ steps (configurable via `warmup_steps`).
- $T_{total} = \lceil N_{train} / B \rceil \times E_{finetune}$.
- $\eta_{base} = 2 \times 10^{-5}$ (`learning_rate`).

#### B. Cosine Annealing Schedule (JAW Fine-Tuning)
For vision models (EfficientNet-B3), smooth cyclical deceleration prevents destructive feature degradation in early convolutional blocks. JAW uses Cosine Annealing over epoch index $e \in [0, E_{finetune}]$:

$$\eta(e) = \eta_{min} + \frac{1}{2} (\eta_{base} - \eta_{min}) \left( 1 + \cos\left( \frac{e}{E_{finetune}} \pi \right) \right)$$

where $\eta_{min} = 0$ and $\eta_{base} = 10^{-5}$ (`lr_finetune`).

```
Learning Rate Schedule Comparison:
LR ^
   |        Linear Warmup + Linear Decay (CART)
   |       /\
   |      /  \
   |     /    \
   |    /      \
   |   /        \
   |  /          \
   | /            \
   +-------------------------------------> Step
   
LR ^
   |---..   Cosine Annealing (JAW)
   |     ''..
   |         ''..
   |             ''..
   |                 ''..
   |                     ''---..
   +-------------------------------------> Epoch
```

---

### 1.4 Gradient Regularization & Early Stopping Mechanics

#### Gradient Norm Clipping
To stabilize backpropagation through deep transformer self-attention layers and dense fusion projections, SafeStep clips parameter gradients by global Euclidean $L_2$ norm:

$$\|\mathbf{g}\|_2 = \sqrt{\sum_{p \in \Theta} \sum_j (g_{p, j})^2}$$

If $\|\mathbf{g}\|_2 > M$ (where $M = 1.0$, `gradient_clip_norm`):

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{M}{\|\mathbf{g}\|_2}$$

#### Validation Early Stopping
To prevent overfitting during full backbone fine-tuning, trainers track validation accuracy $A_{val}^{(e)}$ at the conclusion of each epoch $e$:

$$e^* = \arg\max_{e \le t} A_{val}^{(e)}$$

A patience counter $p_t$ increments if $A_{val}^{(t)} \le A_{val}^{(e^*)}$. If $p_t \ge P$ (default $P = 3$, `early_stopping_patience`), training aborts early, conserving compute and restoring the best checkpoint.

---

## 2. CART Training Pipeline (AI-Generated Text Detection)

CART (**C**ontext-**A**ware **R**epresentation for **T**ext) pairs contextual representations from pre-trained language models with hand-engineered stylistic markers, using gradient boosting to resolve edge cases that fool pure neural transformers.

### 2.1 Multi-Stage Architecture Flow

```
Raw Text x_i
    │
    ├──────────────────────────────────────────────────────┐
    ▼                                                      ▼
DistilBERT Tokenizer (max_length=512)            HeuristicExtractor (12 features)
    │                                                      │
    ▼                                                      │
DistilBERT Transformer Backbone (6 layers, 768-d)          │
    │                                                      │
    ▼                                                      │
[CLS] Token Embedding h_0 ∈ R^768                          │
    │                                                      │
    ▼                                                      │
Linear Head (768 -> 256 -> ReLU -> Dropout -> 2)           │
    │                                                      │
    ▼                                                      │
Softmax Probabilities [p_human, p_ai]                      │
    │                                                      │
    └───────────────────────┬──────────────────────────────┘
                            ▼
               Feature Concatenation (14-d)
                            │
                            ▼
               XGBoost Fusion Meta-Classifier
                            │
                            ▼
                 Final Calibrated p(ai)
```

---

### 2.2 Stage 1: Frozen Neural Encoder Training

In Stage 1, all parameters of the DistilBERT transformer encoder $\theta_{enc}$ are frozen:

$$\nabla_{\theta_{enc}} \mathcal{L} = \mathbf{0}, \quad \forall p \in \theta_{enc}: p.\text{requires\_grad} = \text{False}$$

Only the dense classification head $\theta_{head}$ is optimized:

$$\theta_{head} = \{ \mathbf{W}_1 \in \mathbb{R}^{256 \times 768}, \mathbf{b}_1 \in \mathbb{R}^{256}, \mathbf{W}_2 \in \mathbb{R}^{2 \times 256}, \mathbf{b}_2 \in \mathbb{R}^2 \}$$

$$\mathbf{h}_{cls} = \text{Transformer}(\mathbf{x})[:, 0, :] \in \mathbb{R}^{768}$$
$$\mathbf{h}_1 = \text{Dropout}(\text{ReLU}(\mathbf{W}_1 \mathbf{h}_{cls} + \mathbf{b}_1), p=0.2)$$
$$\mathbf{z} = \mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2 \in \mathbb{R}^2$$

- **Optimizer:** AdamW with high learning rate $\eta = 10^{-3}$, weight decay $\lambda = 0.01$.
- **Epochs:** `num_epochs_frozen` (default: 2).
- **Purpose:** Warms up the random classification projections to align with DistilBERT's contextual manifold without corrupting pre-trained attention weights.

---

### 2.3 Stage 2: End-to-End Fine-Tuning

In Stage 2, the encoder is unfrozen:

$$\forall p \in \theta_{enc}: p.\text{requires\_grad} = \text{True}$$

- **Optimizer:** AdamW with reduced learning rate $\eta = 2 \times 10^{-5}$ (`learning_rate`), $\lambda = 0.01$.
- **Scheduler:** Linear warmup (100 steps) with linear decay.
- **Epochs:** `num_epochs_finetune` (default: 10) with early stopping patience $P = 3$.
- **Backbone Update:** The self-attention matrices ($W_Q, W_K, W_V$) throughout all 6 transformer layers adapt to detect long-range syntax anomalies and neural generation artifacts.

---

### 2.4 Heuristic Feature Extraction Mathematics

While transformers capture contextual semantics, they frequently miss subtle stylistic distributions, token-frequency variances, and punctuation regularity. SafeStep computes 12 handcrafted statistical features:

#### 1. Perplexity Proxy (Unigram Self-Entropy)
AI generators minimize token perplexity, leading to more predictable unigram frequency distributions. Given word frequency counts $c(w)$ over vocabulary $V$ with $N = \sum_{w \in V} c(w)$:

$$p(w) = \frac{c(w)}{N}$$

The unigram Shannon self-entropy is:

$$H(X) = - \sum_{w \in V} p(w) \log_2\left(p(w) + \epsilon\right), \quad \epsilon = 10^{-12}$$

The **Perplexity Proxy** transforms entropy back to an effective vocabulary choice count:

$$\text{Perplexity Proxy} = 2^{H(X)}$$

*Human writing exhibits higher token self-entropy, yielding elevated perplexity proxy values.*

#### 2. Burstiness (Frequency Coefficient of Variation)
Human text features "bursty" word recurrence: rare topical nouns recur in clusters, whereas AI models sample tokens more uniformly. Burstiness is defined as the Coefficient of Variation ($c_v$) of token counts:

$$\mu_c = \frac{1}{|V|} \sum_{w \in V} c(w), \quad \sigma_c = \sqrt{\frac{1}{|V|} \sum_{w \in V} \left(c(w) - \mu_c\right)^2}$$

$$\text{Burstiness} = \frac{\sigma_c}{\mu_c}$$

#### 3. Repetition Ratios ($N$-gram Analysis)
For $n \in [n_{min}, n_{max}]$ (default $n \in [2, 5]$), let $\mathcal{N}$ denote the multiset of all extracted character/word $n$-grams, and $\mathcal{U}$ denote the set of unique $n$-grams:

$$\text{Unique N-gram Ratio} = \frac{|\mathcal{U}|}{|\mathcal{N}|}$$

Let $\tau$ be the repetition threshold (`repetition_threshold = 2`):

$$\text{Repetition Ratio} = \frac{|\{ g \in \mathcal{U} : \text{count}(g) > \tau \}|}{|\mathcal{U}|}$$

$$\text{Max N-gram Repeat} = \max_{g \in \mathcal{U}} \text{count}(g)$$

#### 4. Punctuation Regularity
Measures the variance in character spacing between punctuation marks $(\text{.} , \text{;} \text{:} \text{!} \text{?})$. Let $\mathbf{p} = [p_1, p_2, \dots, p_k]$ be the character indices of punctuation marks. The inter-punctuation intervals are:

$$\delta_i = p_{i+1} - p_i, \quad i \in \{1, \dots, k-1\}$$

$$\text{Punctuation Regularity} = \frac{\sigma_\delta}{\mu_\delta} = \frac{\sqrt{\frac{1}{k-1}\sum_{i=1}^{k-1}(\delta_i - \bar{\delta})^2}}{\bar{\delta}}$$

*Low punctuation regularity variance indicates formal, rigidly periodic sentence structure characteristic of synthetic text.*

#### 5. Sentence Length Dynamics
For sentences $S = [s_1, \dots, s_m]$, let $L_i$ be the word count of sentence $s_i$:

$$\mu_L = \frac{1}{m}\sum_{i=1}^m L_i, \quad \sigma_L = \sqrt{\frac{1}{m}\sum_{i=1}^m (L_i - \mu_L)^2}$$

#### 6. Capitalization Consistency
Fraction of sentences beginning with an uppercase letter:

$$C_{cap} = \frac{1}{m} \sum_{i=1}^m \mathbb{I}(\text{is\_upper}(s_i[0]))$$

#### 7. Rare-Word Ratio (Hapax Legomena Ratio)
Fraction of vocabulary words that appear exactly once in the passage:

$$R_{rare} = \frac{|\{ w \in V : c(w) = 1 \}|}{|V|}$$

#### 8. Transition Word Density & Average Word Length
$$\text{Transition Density} = \frac{\sum_{t \in \mathcal{T}} \text{count}(t)}{\sum_{w \in V} c(w)}$$

where $\mathcal{T} = \{\text{"however"}, \text{"therefore"}, \text{"furthermore"}, \text{"moreover"}, \dots\}$.

$$\text{Avg Word Length} = \frac{1}{\sum_{w \in V} c(w)} \sum_{w \in V} c(w) \cdot \text{len}(w)$$

---

### 2.5 Stage 3: XGBoost Meta-Fusion Training

#### Leak-Free Stacking Protocol
To avoid the meta-classifier memorizing neural prediction confidence on samples the neural backbone was trained on, Stage 3 executes **honest stacking**:
- DistilBERT is trained exclusively on $\mathcal{D}_{train}$.
- The XGBoost fusion model is fit strictly on validation split $\mathcal{D}_{val}$.
- Inputs to XGBoost: $\mathbf{f}_i = [\hat{p}_{human}, \hat{p}_{ai}] \oplus \mathbf{h}_{heuristic} \in \mathbb{R}^{14}$.

#### Mathematical Objective of Gradient Boosted Decision Trees
At boosting step $m \in \{1, \dots, M\}$ (where $M = 100$, `fusion_n_estimators`), XGBoost adds a regression tree $f_m(\mathbf{x})$ that minimizes the penalized objective:

$$\mathcal{L}^{(m)} = \sum_{i=1}^{N_{val}} l\left(y_i, \hat{y}_i^{(m-1)} + f_m(\mathbf{f}_i)\right) + \Omega(f_m)$$

where $l(y, \hat{y})$ is the binary logistic loss:

$$l(y, \hat{y}) = y \ln(1 + e^{-\hat{y}}) + (1 - y) \ln(1 + e^{\hat{y}})$$

and the tree complexity penalty is:

$$\Omega(f_m) = \gamma T + \frac{1}{2}\lambda_{xgb} \sum_{j=1}^T w_j^2$$

($T$ is the number of leaves, $w_j$ is the output score of leaf $j$, $\gamma$ controls minimum split gain, and $\lambda_{xgb} = 1.0$ is $L_2$ leaf regularization).

Using a second-order Taylor expansion around $\hat{y}_i^{(m-1)}$:

$$\tilde{\mathcal{L}}^{(m)} \approx \sum_{i=1}^{N_{val}} \left[ l(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(\mathbf{f}_i) + \frac{1}{2} h_i f_m^2(\mathbf{f}_i) \right] + \Omega(f_m)$$

where first- and second-order loss derivatives are:

$$g_i = \partial_{\hat{y}^{(m-1)}} l(y_i, \hat{y}^{(m-1)}) = \hat{p}_i^{(m-1)} - y_i$$
$$h_i = \partial^2_{\hat{y}^{(m-1)}} l(y_i, \hat{y}^{(m-1)}) = \hat{p}_i^{(m-1)} (1 - \hat{p}_i^{(m-1)})$$

For leaf node $j$ containing sample index set $I_j = \{i : \mathbf{f}_i \in \text{leaf}_j\}$, the optimal leaf weight $w_j^*$ and split gain are:

$$w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda_{xgb}}$$

$$\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda_{xgb}} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda_{xgb}} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda_{xgb}} \right] - \gamma$$

Tree growth continues recursively up to `fusion_max_depth = 4`.

---

## 3. JAW Training Pipeline (AI-Generated Image Detection)

JAW (**J**oint **A**rtifact **W**avelet Image Classifier) detects synthetic imagery by fusing high-level spatial semantic features from an EfficientNet-B3 convolutional backbone with low-level spectral energy and multi-scale wavelet textures.

### 3.1 Multi-Branch Architecture Flow

```
Input Image (H x W x 3)
   │
   ├───────────────────────────────────────────────────────┐
   ▼                                                       ▼
RGB Preprocessing (Resize 224x224, ImageNet Norm)     Grayscale Luminance Y = 0.299R + 0.587G + 0.114B
   │                                                       │
   ▼                                                       ▼
EfficientNet-B3 Backbone                              FrequencyFeatureExtractor
   │                                                       │
   ▼                                                       ├─► 2D-FFT Radial Spectral Bands (8-d)
Global Average Pooling                                     ├─► Spectral Shape: Peak Freq + Flatness (2-d)
   │                                                       └─► Gabor Wavelet Filter Bank (16-d)
   ▼                                                       │
1536-d Spatial Feature Vector                              ▼
   │                                                  26-d Frequency Feature Vector
   └───────────────────────┬───────────────────────────────┘
                           ▼
               Concatenation Layer (1562-d)
                           │
                           ▼
          Dense Projection 1: Linear(1562, 512) -> ReLU -> Dropout(0.2)
                           │
                           ▼
          Dense Projection 2: Linear(512, 128)  -> ReLU -> Dropout(0.2)
                           │
                           ▼
          Classification Logits: Linear(128, 2)
                           │
                           ▼
                Softmax [p_authentic, p_ai]
```

---

### 3.2 Preprocessing & Spatial Feature Extraction

#### Spatial Branch Pipeline
1. **Resizing:** Bilinear interpolation to $(H, W) = (224, 224)$.
2. **Standardization:** Input pixel values $x \in [0, 255]$ are scaled to $[0, 1]$ and normalized using ImageNet channel-wise statistics:
   $$z_{c, i, j} = \frac{x_{c, i, j} / 255 - \mu_c}{\sigma_c}$$
   $$\boldsymbol{\mu} = [0.485, 0.456, 0.406], \quad \boldsymbol{\sigma} = [0.229, 0.224, 0.225]$$
3. **Backbone:** EfficientNet-B3 pre-trained on ImageNet. The final classification layer is replaced with `nn.Identity()`, yielding spatial embedding:
   $$\mathbf{v}_{spatial} = \text{Backbone}(\mathbf{z}) \in \mathbb{R}^{1536}$$

---

### 3.3 Frequency-Domain Feature Extraction Mathematics

Generative models (GANs upsampling via transposed convolution, Latent Diffusion decoders) leave subtle grid-like artifacts and high-frequency discrepancies that are imperceptible in RGB space but clearly visible in the frequency domain.

#### A. 2D Discrete Fourier Transform (2D-DFT) & Spectrum Centering
The image is converted to grayscale luminance $I(x, y) \in [0, 1]$:

$$I(x, y) = 0.299 R(x, y) + 0.587 G(x, y) + 0.114 B(x, y)$$

The 2D Discrete Fourier Transform transforms spatial signals to complex frequency coefficients:

$$F(u, v) = \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} I(x, y) \exp\left( -j 2\pi \left( \frac{ux}{W} + \frac{vy}{H} \right) \right)$$

Applying quadrant shift (`fftshift`) moves the zero-frequency DC component $(u=0, v=0)$ to the matrix center $(c_y, c_x) = (\lfloor H/2 \rfloor, \lfloor W/2 \rfloor)$. The magnitude spectrum is:

$$S(u, v) = |F(u, v)| = \sqrt{\text{Re}(F(u, v))^2 + \text{Im}(F(u, v))^2}$$

#### B. Radial Spectral Energy Distribution
To quantify how energy is distributed from low to high frequencies, SafeStep computes the Euclidean radial distance from the spectral center:

$$r(u, v) = \sqrt{(u - c_y)^2 + (v - c_x)^2}, \quad r_{max} = \max_{u, v} r(u, v)$$

The frequency domain is partitioned into $K = 8$ concentric radial bands (`num_radial_bands = 8`) with linear boundaries:

$$b_k = k \cdot \frac{r_{max}}{K}, \quad k \in \{0, 1, \dots, K\}$$

The spectral energy in band $k$ is the sum of magnitudes within ring $[b_k, b_{k+1})$:

$$E_k = \sum_{(u, v) : b_k \le r(u, v) < b_{k+1}} S(u, v)$$

Normalized radial band energy ensures scale invariance:

$$\tilde{E}_k = \frac{E_k}{\sum_{j=0}^{K-1} E_j + \epsilon}, \quad \mathbf{e}_{radial} = [\tilde{E}_0, \tilde{E}_1, \dots, \tilde{E}_7] \in \mathbb{R}^8$$

#### C. Spectral Shape Properties
SafeStep extracts two global spectral shape descriptors:
1. **Normalized Peak Frequency Radius:**
   $$(u^*, v^*) = \arg\max_{(u, v)} S(u, v), \quad r_{peak} = \frac{r(u^*, v^*)}{r_{max}} \in [0, 1]$$

2. **Spectral Flatness (Wiener Entropy):**
   Measures whether spectral energy is concentrated in isolated peaks (tonal, common in GAN upsampling artifacts) or evenly distributed across all frequencies (flat white noise). Computed as the ratio of geometric mean to arithmetic mean of the power spectrum $P(u, v) = S(u, v)^2 + 10^{-12}$:

   $$\text{Flatness} = \frac{\exp\left( \frac{1}{HW} \sum_{u, v} \ln P(u, v) \right)}{\frac{1}{HW} \sum_{u, v} P(u, v)}$$

#### D. Multi-Scale 2D Gabor Wavelet Texture Bank
Gabor filters act as localized band-pass filters sensitive to specific spatial frequencies and orientations, capturing periodic surface textures. The 2D Gabor kernel is defined as:

$$g(x, y; f, \theta) = \exp\left( -\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2} \right) \cos\left( 2\pi f x' \right)$$

where $(x', y')$ represents rotated spatial coordinates:

$$x' = x \cos \theta + y \sin \theta, \quad y' = -x \sin \theta + y \cos \theta$$

SafeStep samples $S = 4$ frequency scales and $O = 4$ angular orientations (16 total filters):
- Scales $s \in \{0, 1, 2, 3\}$: frequencies $f_s = 0.10 + 0.15 \cdot s \in \{0.10, 0.25, 0.40, 0.55\}$.
- Orientations $o \in \{0, 1, 2, 3\}$: angles $\theta_o = o \cdot \frac{\pi}{4} \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}$.

Each kernel $g_{s, o}$ is convolved with the grayscale image $I$ (downsampled to $128 \times 128$ for efficiency), and the mean absolute response is computed:

$$R_{s, o} = \frac{1}{H' W'} \sum_{x=1}^{H'} \sum_{y=1}^{W'} \left| (I * g_{s, o})(x, y) \right|$$

$$\mathbf{v}_{gabor} = [R_{0,0}, R_{0,1}, \dots, R_{3,3}] \in \mathbb{R}^{16}$$

#### Total Frequency Feature Vector
$$\mathbf{v}_{freq} = \mathbf{e}_{radial} \oplus [r_{peak}, \text{Flatness}] \oplus \mathbf{v}_{gabor} \in \mathbb{R}^{26}$$

---

### 3.4 Two-Stage Training Protocol

#### Stage 1: Frozen Backbone Head Training
- **Backbone State:** Frozen ($\nabla_{\theta_{backbone}} \mathcal{L} = \mathbf{0}$).
- **Trainable Parameters:** Fusion MLP $\theta_{head} \in \mathbb{R}^{(1562 \times 512) + (512 \times 128) + (128 \times 2)}$.
- **Optimizer:** AdamW with $\eta = 10^{-3}$ (`lr_frozen`), $\lambda = 0.01$.
- **Epochs:** `num_epochs_frozen` (default: 3).
- **Objective:** Aligns the randomly initialized 1562-dimensional projection weights with the pre-trained EfficientNet pooling distribution before gradient updates reach convolutional kernels.

#### Stage 2: End-to-End Fine-Tuning
- **Backbone State:** Unfrozen ($\nabla_{\theta_{all}} \mathcal{L} \ne \mathbf{0}$).
- **Optimizer:** AdamW with low learning rate $\eta = 10^{-5}$ (`lr_finetune`), $\lambda = 0.01$.
- **Scheduler:** CosineAnnealingLR across $T_{max} = \text{num\_epochs\_finetune}$ (default: 15).
- **Early Stopping:** Monitored against validation accuracy with patience $P = 3$.

---

## 4. Phishing Escalation Pipeline Training

The SafeStep Phishing Detector employs a **multi-tiered confidence escalation pipeline** designed for high throughput, robust security, and low compute cost. Samples pass through cheap, ultra-fast stages first, escalating to heavier models only when confidence is low.

### 4.1 The Confidence Escalation Paradigm

```
Incoming Text (Email / URL / Message)
                  │
                  ▼
         Stage 1: Attack (Char N-gram TF-IDF + Logistic Regression)
                  │
                  ├─► High Confidence (p < 0.10 or p > 0.90) ──► TERMINATE (Verdict: Attack)
                  │
                  ▼ Uncertain Confidence (0.10 ≤ p ≤ 0.90)
         Stage 2: Beast (XGBoost on Attack Probs + 18 Indicators)
                  │
                  ├─► High Confidence (p < 0.20 or p > 0.80) ──► TERMINATE (Verdict: Beast)
                  │
                  ▼ Uncertain Confidence (0.20 ≤ p ≤ 0.80) [and Warhammer loaded]
         Stage 3: Warhammer (DistilBERT Contextual Transformer)
                  │
                  ▼
         TERMINATE (Verdict: Warhammer)
```

Throughput and latency characteristics across stages:
- **Attack:** $> 2,000$ samples/sec per CPU core; latency $< 0.5$ ms.
- **Beast:** $\sim 500$ samples/sec per CPU core; latency $\sim 2.0$ ms.
- **Warhammer:** $\sim 25$ samples/sec on CPU ($\sim 300$ samples/sec on GPU); latency $15\text{--}40$ ms.

Under typical distributions, $> 85\%$ of samples resolve at Stage 1, $\sim 10\text{--}12\%$ resolve at Stage 2, and $< 3\%$ require Stage 3 transformer inference.

---

### 4.2 Stage 1: Attack (Sublinear Char N-gram TF-IDF + Logistic Regression)

#### Feature Extraction
Attack bypasses word tokenization to defend against adversarial character perturbations (e.g., `"paypa1"`, `"secur1ty"`, `"rn" $\rightarrow$ "m"`). It extracts character $n$-grams within word boundaries (`analyzer="char_wb"`) for $n \in [2, 5]$ with a vocabulary cap of $50,000$ features:

$$\text{tf-idf}(t, d) = \left( 1 + \ln(\text{tf}(t, d)) \right) \cdot \left( \ln\left( \frac{1 + N_D}{1 + \text{df}(t)} \right) + 1 \right)$$

Rows are normalized to unit Euclidean length: $\|\mathbf{x}_d\|_2 = 1$.

#### Optimization Objective
$L_2$-regularized Logistic Regression with balanced class weighting:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{N_{train}} w_{y_i} \ln\left( 1 + \exp\left( -y_i (\mathbf{w}^T \mathbf{x}_i + b) \right) \right)$$

where $C = 4.0$ (`attack_C`), $y_i \in \{-1, +1\}$, and class weights compensate for class imbalance:

$$w_{c} = \frac{N_{train}}{2 \cdot N_c}$$

#### Escalation Decision Rule
Let $p_{att} = \sigma(\mathbf{w}^T \mathbf{x} + b)$ be the predicted probability of phishing:

$$\text{Escalate to Beast} \iff \tau_{att}^{low} < p_{att} < \tau_{att}^{high}$$

where $\tau_{att}^{low} = 0.10$ and $\tau_{att}^{high} = 0.90$.

---

### 4.3 Stage 2: Beast (XGBoost Meta-Model + 18 Structural Indicators)

Beast evaluates a 20-dimensional feature vector combining Attack's output probabilities with 18 hand-engineered structural features:

$$\mathbf{f}_{beast} = [p_{legitimate}^{att}, p_{phishing}^{att}] \oplus \mathbf{v}_{indicators} \in \mathbb{R}^{20}$$

#### Mathematical Definitions of the 18 Structural Indicators
Given text $T$, let $U = [u_1, \dots, u_n]$ be extracted URLs, $H = [\text{host}(u_1), \dots, \text{host}(u_n)]$ be domain hosts, and $W$ be total word count:

| Indicator Feature | Mathematical Definition | Security Rationale |
|---|---|---|
| `url_count` | $n = \|U\|$ | Phishing emails typically embed multiple call-to-action links |
| `ip_url_ratio` | $\frac{1}{n} \sum_{h \in H} \mathbb{I}(h \text{ matches IPv4})$ | Direct IP addresses evade domain reputation filters |
| `shortener_url_ratio` | $\frac{1}{n} \sum_{h \in H} \mathbb{I}(h \in \text{Shorteners})$ | Services like bit.ly obfuscate target destinations |
| `punycode_url_ratio` | $\frac{1}{n} \sum_{h \in H} \mathbb{I}(\text{"xn--" } \in h)$ | Homograph attacks (e.g., Cyrillic 'а' replacing Latin 'a') |
| `suspicious_tld_ratio` | $\frac{1}{n} \sum_{h \in H} \mathbb{I}(\text{tld}(h) \in \text{SusTLDs})$ | Cheap/free TLDs (`.xyz`, `.top`, `.tk`, `.click`) correlate with malicious infrastructure |
| `at_in_url_ratio` | $\frac{1}{n} \sum_{u \in U} \mathbb{I}(\text{"@" } \in u)$ | `@` in URLs forces browser authorization routing |
| `https_ratio` | $\frac{1}{n} \sum_{u \in U} \mathbb{I}(u \text{ starts with https://})$ | Assesses TLS presence across links |
| `max_url_length` | $\max_{u \in U} \text{len}(u)$ | Phishing links embed long encoded base64 tracking strings |
| `max_subdomain_depth` | $\max_{h \in H} \text{count}(h, \text{"."})$ | Deep subdomains mimic legitimate banking portals |
| `max_domain_hyphens` | $\max_{h \in H} \text{count}(h, \text{"-"})$ | Hyphenation disguises brand impersonation (`login-bank.com`) |
| `url_digit_ratio` | $\frac{\sum_{c \in \text{all\_urls}} \mathbb{I}(c \text{ is digit})}{\sum_{c \in \text{all\_urls}} 1}$ | Domain generation algorithms (DGAs) generate numerical patterns |
| `urgency_density` | $\frac{1}{W} \sum_{t \in \text{UrgencyTerms}} \text{count}(T, t)$ | Psychological pressure ("immediately", "suspended", "urgent") |
| `credential_density` | $\frac{1}{W} \sum_{t \in \text{CredTerms}} \text{count}(T, t)$ | Solicits confidential auth tokens ("password", "SSN", "login") |
| `money_density` | $\frac{1}{W} \sum_{t \in \text{MoneyTerms}} \text{count}(T, t)$ | Financial lure terms ("wire", "refund", "bitcoin", "invoice") |
| `exclamation_density` | $\frac{\text{count}(T, \text{"!"})}{W}$ | Punctuation shouting correlates with coercive pretexts |
| `uppercase_ratio` | $\frac{\sum_{c \in \text{letters}} \mathbb{I}(c \text{ is upper})}{\sum_{c \in \text{letters}} 1}$ | Shouting / capitalization anomalies |
| `has_html_form` | $\mathbb{I}(\text{regex}(<\text{form}\|<\text{input}) \in T)$ | Direct credential harvesting forms embedded in payload |
| `log_length` | $\ln(1 + \text{len}(T))$ | Payload size scaling |

#### Beast Model Specification
- **Algorithm:** XGBoost (`n_estimators = 200`, `max_depth = 5`, `learning_rate = 0.1`, `subsample = 0.8`).
- **Escalation Thresholds:** $\tau_{bst}^{low} = 0.20$, $\tau_{bst}^{high} = 0.80$.

---

### 4.4 Stage 3: Warhammer (Contextual Semantic Transformer)

Warhammer runs full transformer self-attention over sequences up to `max_seq_length = 256` using `distilbert-base-uncased`.

#### Architecture
$$\mathbf{h}_{cls} = \text{DistilBERT}(T)[:, 0, :] \in \mathbb{R}^{768}$$
$$\mathbf{z}_{war} = \mathbf{W}_2 \cdot \text{Dropout}(\text{ReLU}(\mathbf{W}_1 \mathbf{h}_{cls} + \mathbf{b}_1)) + \mathbf{b}_2 \in \mathbb{R}^2$$

- **Stage 3a (Frozen Head):** $\eta = 10^{-3}$, 1 epoch.
- **Stage 3b (End-to-End Fine-Tuning):** $\eta = 2 \times 10^{-5}$, 3 epochs, linear warmup over 100 steps, early stopping patience $P = 3$.

---

### 4.5 Leak-Free Staged Training Protocol

The phishing pipeline uses a carefully designed data-flow split to eliminate data leakage:

```
Full Dataset D
      │
      ├──────────────────────────────┬──────────────────────────────┐
      ▼                              ▼                              ▼
D_train (60%)                  D_val (20%)                    D_test (20%)
      │                              │                              │
      ├─► Fits Attack (Stage 1)      │                              │
      │                              │                              │
      │   Evaluates Attack ─────────►│                              │
      │                              ├─► Fits Beast (Stage 2)       │
      │                              │   (Uses unseen Attack probs) │
      │                              │                              │
      ├─► Fits Warhammer (Stage 3)   │                              │
      │   Early stopping on ────────►│                              │
      │                              │                              │
      │                              │   Full Escalation Evaluation─►
      │                              │   (Completely held-out)
```

> [!IMPORTANT]
> **Why Beast Must Train on Validation Data:**
> If Beast trained on $\mathcal{D}_{train}$, Attack's predicted probabilities would reflect severe training-set overconfidence ($\hat{p} \approx 0.0$ or $1.0$). Beast would learn that Attack is infallible, failing to learn appropriate correction weights when Attack is uncertain on real, unseen data. Fitting Beast on $\mathcal{D}_{val}$ guarantees that Beast trains against realistic, uncalibrated Attack errors.

---

## 5. Cross-Validation, Evaluation & Statistical Significance Mathematics

SafeStep includes rigorous validation tooling in `training/cross_validation.py`, `training/evaluation.py`, and `training/significance.py`.

### 5.1 Leak-Free Stratified K-Fold Cross-Validation

Standard random splitting causes label distribution skew when working with imbalanced datasets. Stratified $K$-fold guarantees that every fold $k \in \{1, \dots, K\}$ maintains identical class proportions:

$$\frac{N_{k, c}}{N_k} \approx \frac{N_c}{N}, \quad \forall c \in \{0, 1\}$$

#### Unbiased Sample Standard Deviation
Summary statistics across the $K$ folds use Bessel's correction ($\text{ddof} = 1$):

$$\bar{\mu} = \frac{1}{K} \sum_{k=1}^K m_k$$

$$s = \sqrt{\frac{1}{K - 1} \sum_{k=1}^K (m_k - \bar{\mu})^2}$$

---

### 5.2 Classification Metrics & AUROC Riemann Sum

#### Primary Confusion Metrics
Given predictions $\hat{y}_i \in \{0, 1\}$ and targets $y_i \in \{0, 1\}$:
- $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
- $\text{Precision} = \frac{TP}{TP + FP}$
- $\text{Recall (True Positive Rate, TPR)} = \frac{TP}{TP + FN}$
- $\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$
- $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$

#### Area Under the Receiver Operating Characteristic (AUROC)
The ROC curve maps $\text{TPR}(t)$ against $\text{FPR}(t)$ as classification threshold $t \in [0, 1]$ varies across all unique predicted probabilities. SafeStep computes AUROC using the **Composite Trapezoidal Rule** over sorted points $(\text{FPR}_1, \text{TPR}_1), \dots, (\text{FPR}_M, \text{TPR}_M)$:

$$\text{AUROC} = \sum_{i=1}^{M-1} \left( \text{FPR}_{i+1} - \text{FPR}_i \right) \cdot \frac{\text{TPR}_i + \text{TPR}_{i+1}}{2}$$

AUROC is statistically equivalent to the **Wilcoxon-Mann-Whitney $U$ statistic**:

$$\text{AUROC} = \frac{1}{N_{pos} N_{neg}} \sum_{i \in \text{Pos}} \sum_{j \in \text{Neg}} \left( \mathbb{I}(\hat{p}_i > \hat{p}_j) + \frac{1}{2}\mathbb{I}(\hat{p}_i = \hat{p}_j) \right)$$

It quantifies the probability that a randomly chosen positive sample receives a higher score than a randomly chosen negative sample.

#### Edge Case Handling & Fallbacks
- **Single-Class Ingestion:** If $y_{true}$ contains only one class, AUROC is undefined. SafeStep logs an explicit warning and returns `float("nan")`, preventing silent crashes.
- **Discrete Label Inputs:** When binary predictions $\{0, 1\}$ are passed instead of continuous probabilities, the trapezoidal curve collapses to three points $(0, 0)$, $(\text{FPR}, \text{TPR})$, and $(1, 1)$, correctly reducing to **Balanced Accuracy**:

$$\text{AUROC}_{discrete} = \frac{\text{TPR} + (1 - \text{FPR})}{2} = \frac{1}{2}\left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$$

---

### 5.3 One-Way Analysis of Variance (ANOVA) & Pairwise Tests

When validating whether performance improvements between model architectures or feature ablations are statistically meaningful, SafeStep uses **One-Way ANOVA** (`ANOVATester` in `training/significance.py`).

#### Hypothesis Formulation
For $k$ independent models evaluated across $n_j$ cross-validation folds:
- $H_0: \mu_1 = \mu_2 = \dots = \mu_k$ (all model distributions have identical means)
- $H_1:$ At least one model mean differs significantly.

#### Variance Decomposition
Total variance decomposes into between-group variance ($SS_{between}$) and within-group error variance ($SS_{within}$):

$$SS_{between} = \sum_{j=1}^k n_j (\bar{x}_j - \bar{x}_{\cdot\cdot})^2, \quad df_{between} = k - 1$$

$$SS_{within} = \sum_{j=1}^k \sum_{i=1}^{n_j} (x_{ij} - \bar{x}_j)^2, \quad df_{within} = N_{total} - k$$

where $\bar{x}_j$ is the mean of group $j$, and $\bar{x}_{\cdot\cdot}$ is the grand mean across all $N_{total} = \sum n_j$ samples.

#### Mean Squares & $F$-Statistic
$$MS_{between} = \frac{SS_{between}}{df_{between}}, \quad MS_{within} = \frac{SS_{within}}{df_{within}}$$

$$F = \frac{MS_{between}}{MS_{within}} \sim \mathcal{F}(df_{between}, df_{within})$$

The $p$-value represents the probability of observing an $F$-statistic at least as extreme under $H_0$:

$$p = 1 - F_{cdf}(F; df_{between}, df_{within})$$

If $p < \alpha$ (default $\alpha = 0.05$), $H_0$ is rejected: the performance difference is statistically significant.

#### Post-Hoc Pairwise Testing with Bonferroni Correction
For $k \ge 3$ groups, SafeStep conducts post-hoc Welch's two-sample $t$-tests across all $m = \binom{k}{2} = \frac{k(k-1)}{2}$ unique pairs. To prevent false positives from family-wise error rate inflation, the significance threshold is adjusted:

$$\alpha_{bonf} = \frac{\alpha}{m}$$

$$p_{adj} = \min(1.0, m \cdot p_{raw})$$

---

## 6. CLI Command Reference & Hardware Execution

### 6.1 Flag Reference

```bash
python train.py [cart|jaw|phishing|all] [options]
```

#### Core Arguments
| Flag | Default | Description |
|---|---|---|
| `--model {cart,jaw,phishing,all}` | `all` | Target model. Note: `all` runs CART and JAW concurrently; phishing is trained separately. |
| `--data-dir PATH` | `data` | Root directory for dataset discovery. |
| `--checkpoint-dir PATH` | `checkpoints` | Root directory for saved weights (`checkpoints/cart`, `checkpoints/jaw`, `checkpoints/phishing`). |
| `--device {auto,cpu,cuda,mps}` | `auto` | Execution accelerator. Resolves CUDA on Linux/Windows, MPS on Apple Silicon, or CPU. |
| `--workers N` | `cpu_count // 2` | Parallel data-loading worker processes. |
| `--estimate-only` | off | Calculates hardware-tailored ETA and batch plans without executing training. |
| `--resume PATH` | none | Directory containing checkpoint files to resume training from. |
| `--epochs-frozen N` | config | Number of epochs for frozen backbone head pre-training. |
| `--epochs-finetune N` | config | Number of epochs for full end-to-end backbone fine-tuning. |
| `--batch-size N` | config | Training batch size. |
| `--lr RATE` | config | Fine-tuning learning rate (sets CART/Warhammer fine-tune LR and JAW `lr_finetune`). |
| `-v`, `--verbose` | off | Prints per-epoch loss, validation accuracy, and AUROC metrics to console. |

#### Evaluation & Cross-Validation Arguments
| Flag | Default | Description |
|---|---|---|
| `--cross-val [K]`, `-cv` | `5` | Runs $K$-fold stratified cross-validation reporting mean $\pm$ sample std. |
| `--eval-external PATH` | none | Evaluates a trained model against an external benchmark file (`.csv`, `.jsonl`, `.json`, or image dir). |
| `--export-roc PATH` | none | Exports ROC curve coordinates (`fpr`, `tpr`, `thresholds`) to JSON or CSV. |
| `--significance-test`, `--anova` | off | Runs One-Way ANOVA significance testing at $\alpha=0.05$. |

#### Dashboard, Sweeps & Data Arguments
| Flag | Default | Description |
|---|---|---|
| `--dashboard` | off | Full-screen interactive terminal dashboard with ETA, hardware utilization, and loss charts. |
| `--sweep KEY=V1,V2` | none | Runs hyperparameter sweep across listed keys (e.g. `--sweep lr=1e-5,3e-5`). |
| `--demo` | off | Uses synthetic placeholder data when real dataset directories are missing. |
| `--prepare-data [{cart,jaw,phishing,all}]` | `all` | Generates sample datasets under `--data-dir` and exits. |
| `--download-data [SOURCES...]` | default | Fetches public datasets (`hf_emails`, `hf_urls`, etc.) into `data/phishing/downloads/`. |

---

### 6.2 Shell Wrappers (`train.sh` & `train.ps1`)

The bash (`train.sh`) and PowerShell (`train.ps1`) wrappers automatically detect Python virtual environments (`.venv/bin/python`, `venv/Scripts/python.exe`, etc.) and provide convenience commands:

```bash
# Interactive menu wizard
./train.sh interactive

# Train individual models
./train.sh cart
./train.sh jaw --device cuda
./train.sh phishing

# Run demo training without data
./train.sh demo all

# Cross-validation with statistical significance
./train.sh cart --cross-val 5 --significance-test

# Live training dashboard
./train.sh dashboard
```

On Windows:
```powershell
.\train.ps1 cart
.\train.ps1 jaw -device cuda
.\train.ps1 phishing
```

---

### 6.3 Hardware Planner & Throughput Modeling

When launching training with `--dashboard` or `--estimate-only`, SafeStep uses `training/planner.py` to evaluate local compute capabilities and estimate wall-clock execution time.

#### Throughput Model
$$\text{Estimated Time (seconds)} = \frac{N_{samples} \times E_{total}}{\text{Throughput}_{device}}$$

$$\text{Throughput}_{device} = \text{Base Throughput}_{CPU} \times S_{device}$$

where default benchmarks are:
- **CART Base Throughput:** $\sim 28$ samples/sec on CPU.
- **JAW Base Throughput:** $\sim 14$ samples/sec on CPU.
- **CUDA Speedup ($S_{cuda}$):** $\sim 6.5\times$.
- **Apple Silicon Speedup ($S_{mps}$):** $\sim 3.2\times$.

The hardware planner checks GPU VRAM and CPU core counts, scheduling CART and JAW concurrently only when sufficient host memory and accelerator bandwidth are available.

---

### 6.4 Checkpoint Layout & Hot Reloading

Checkpoints are serialized to `--checkpoint-dir`:

```
checkpoints/
├── cart/
│   ├── neural_model.pt       # PyTorch state_dict for DistilBERT + classifier head
│   └── fusion_model.json      # Serialized XGBoost meta-model
├── jaw/
│   └── jaw_model.pt          # PyTorch state_dict for EfficientNet-B3 + fusion head
└── phishing/
    ├── attack.joblib          # Scikit-learn Pipeline (char n-gram TF-IDF + Logistic Regression)
    ├── beast.json             # Serialized XGBoost meta-model
    └── warhammer.pt          # PyTorch state_dict for DistilBERT classifier (optional)
```

Checkpoints can be hot-reloaded into a running SafeStep API server without downtime:
```bash
curl -X POST http://localhost:8000/models/cart/reload \
     -H "X-Admin-Token: $SAFESTEP_ADMIN_TOKEN"
```
