# Architecture

SafeStep ships three independent detectors. Each package exposes a `*Detector` class (`detector.py`) with `predict()`, `save()` and `load()`, and a `*Trainer` (`train.py`) that fits its components.

| Model | Package | Input | Output keys |
|---|---|---|---|
| CART | `cart/` | raw text | `p_human`, `p_ai`, `label` (`human` / `ai_generated`) |
| JAW | `jaw/` | image paths | `p_authentic`, `p_ai`, `label` (`authentic` / `ai_generated`) |
| Phishing | `phishing/` | email / URL / page text | `p_legitimate`, `p_phishing`, `label` (`legitimate` / `phishing`), `stage` |

Every result also carries the `decision_threshold` that was applied (default `0.5` in each config).

---

## CART: AI-generated text

```
text ──► DistilBERT tokenizer ──► CARTNeuralClassifier ──► softmax [p_human, p_ai] ─┐
   └───► HeuristicExtractor (stylistic features) ─────────────────────────────────────┴─► concat ──► XGBoost (CARTFusionModel) ──► p_ai
```

- **Neural branch** (`cart/model.py`): `distilbert-base-uncased` encoder plus a classifier head (`classifier_hidden_dim=256`, `max_seq_length=512`). Trained in two phases: head-only (`num_epochs_frozen`, default 2), then the full encoder unfrozen (`num_epochs_finetune`, default 10), with early stopping (`early_stopping_patience=3`).
- **Heuristic branch** (`cart/heuristics.py`): perplexity proxy, burstiness, n-gram repetition (`ngram_min..ngram_max` = 2..5), punctuation regularity, sentence-length stats, capitalization consistency, rare-word ratio and transition-word density.
- **Fusion** (`cart/fusion.py`): an XGBoost classifier (`fusion_n_estimators=100`, `fusion_max_depth=4`) over the concatenated neural probabilities and heuristic features.

**Checkpoint files** (`checkpoints/cart/`):

| File | Contents |
|---|---|
| `neural_model.pt` | `state_dict` of the DistilBERT classifier |
| `fusion_model.json` | XGBoost fusion model |

Both files are required. The API only loads CART when both exist.

---

## JAW: AI-generated images

```
image ──► JAWPreprocessor (resize 224, ImageNet norm) ──► EfficientNet-B3 ──► 1536-d spatial features ─┐
   └────► FrequencyFeatureExtractor (grayscale) ─► radial FFT energy + spectral shape + Gabor bank ────┴─► concat ──► MLP 512 ─► 128 ─► 2 logits
```

- **Spatial branch** (`jaw/model.py`): `efficientnet_b3` backbone (ImageNet-pretrained, `pretrained=True`), 1536-d pooled output.
- **Frequency branch** (`jaw/frequency_features.py`): 8 radial spectral-energy bands, 2 spectral-shape values (peak frequency location, spectral flatness), and one mean response per Gabor kernel (4 orientations x 4 scales = 16). That is 26 features with the default config.
- **Fusion head**: an MLP (`fusion_hidden_dim_1=512`, `fusion_hidden_dim_2=128`, `dropout=0.2`) over the concatenated features. Training runs a frozen-backbone phase (`num_epochs_frozen=3`, `lr_frozen=1e-3`) and then a fine-tuning phase (`num_epochs_finetune=15`, `lr_finetune=1e-5`).

**Checkpoint files** (`checkpoints/jaw/`):

| File | Contents |
|---|---|
| `jaw_model.pt` | `state_dict` of the whole `JAWModel` (backbone + fusion head) |

The frequency extractor has no learned weights, so it isn't saved.

---

## Phishing: Attack → Beast → Warhammer

An escalation pipeline. Every sample goes through the cheap stage, and only uncertain samples move on to the more expensive ones.

```
text ──► Attack ──► p in (attack_escalate_low, attack_escalate_high)? ──no──► final (stage=attack)
                         │ yes
                         ▼
          Beast(Attack probs + 18 indicators) ──► p in (beast_escalate_low, beast_escalate_high)? ──no──► final (stage=beast)
                                                       │ yes (and Warhammer is trained/loaded)
                                                       ▼
                                                  Warhammer ──► final (stage=warhammer)
```

| Stage | File | Model | Key config (`phishing/config.py`) |
|---|---|---|---|
| Attack | `phishing/attack.py` | Character n-gram TF-IDF + logistic regression | `attack_ngram_min=2`, `attack_ngram_max=5`, `attack_max_features=50_000`, `attack_C=4.0` |
| Beast | `phishing/beast.py` | XGBoost on Attack's probabilities + indicator features | `beast_n_estimators=200`, `beast_max_depth=5`, `beast_learning_rate=0.1` |
| Warhammer | `phishing/warhammer.py` | Fine-tuned DistilBERT (`distilbert-base-uncased`, `max_seq_length=256`) | `use_warhammer=True`, `num_epochs_frozen=1`, `num_epochs_finetune=3`, `learning_rate=2e-5` |

**Escalation bands.** A stage's verdict is final unless `p(phishing)` falls strictly inside its open band. The defaults are `attack_escalate_low=0.1` / `attack_escalate_high=0.9` and `beast_escalate_low=0.2` / `beast_escalate_high=0.8`. `PhishingDetector.predict(texts, max_stage=...)` can stop the pipeline early at `"attack"` or `"beast"`.

**Indicators** (`phishing/indicators.py`, `FEATURE_NAMES`, 18 features): `url_count`, `ip_url_ratio`, `shortener_url_ratio`, `punycode_url_ratio`, `suspicious_tld_ratio`, `at_in_url_ratio`, `https_ratio`, `max_url_length`, `max_subdomain_depth`, `max_domain_hyphens`, `url_digit_ratio`, `urgency_density`, `credential_density`, `money_density`, `exclamation_density`, `uppercase_ratio`, `has_html_form`, `log_length`.

**Training data flow.** Attack is fit on the train split. Beast is fit on the val split, which Attack never saw, so it learns from honest Attack scores. Warhammer trains on the train split with early stopping on val. Pipeline accuracy reported on val is therefore optimistic.

**Warhammer readiness.** A fresh, untrained Warhammer is never used. It only joins the pipeline once it has been trained or loaded from `warhammer.pt`. Until then, uncertain samples keep Beast's verdict.

**Checkpoint files** (`checkpoints/phishing/`):

| File | Contents | Required |
|---|---|---|
| `attack.joblib` | TF-IDF vectorizer + logistic regression | yes |
| `beast.json` | XGBoost model | yes |
| `warhammer.pt` | DistilBERT classifier `state_dict` | no. Only written when Warhammer was trained, and skipped on load if missing |

The API serves phishing at `/api/v1/detect/phishing` and `/api/v1/detect/phishing/batch`, and in the unified endpoint's `phishing_result` (see [deployment.md](deployment.md)).

---

## Data loading

| Module | Role |
|---|---|
| `training/data.py` | Split loaders used by `train.py`: `load_text_split` (CART), `load_image_split` (JAW), `load_splits` |
| `util/loaders/cart.py`, `jaw.py`, `phishing.py` | Dataset-specific loaders (HC3, GPT-2 output, RAID, MAGE, CIFAKE, GenImage, ArtiFact, Enron, UCI, PhishTank), label normalization and sample-data generators |
| `util/manager.py` | `SafeStepDataManager` facade (`get_cart_data`, `get_jaw_data`, `get_phishing_data`, `prepare_sample_data`, `inspect_directory`) plus the `python -m util.manager` CLI |

See [training.md](training.md#data-layouts) for the directory layouts each loader accepts.

## Serving

`api/` wraps CART and JAW in FastAPI. `api/model_registry.py` lazily builds each detector and loads weights from its checkpoint dir, and `api/service.py` / `api/app.py` expose the endpoints. Details are in [deployment.md](deployment.md).
