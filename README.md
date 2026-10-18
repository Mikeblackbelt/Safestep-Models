# SafeStep Models

Detection models and pipelines for AI-generated text, AI-generated images, and malicious content.

| Model | Package | Detects | Approach |
|-------|---------|---------|----------|
| **CART** | `cart/` | AI-generated text | DistilBERT + stylistic heuristics, fused with XGBoost |
| **JAW** | `jaw/` | AI-generated images | EfficientNet-B3 + FFT spectral / Gabor texture features |
| **Phishing** | `phishing/` | Phishing emails & URLs | Escalation pipeline: **Attack** (char n-gram TF-IDF + logistic regression) → **Beast** (XGBoost on Attack score + URL/urgency/credential indicators) → **Warhammer** (fine-tuned DistilBERT) |

Phishing samples only escalate to the next, more expensive stage while `p(phishing)` sits inside that stage's uncertainty band (`attack_escalate_low/high`, `beast_escalate_low/high` in `PhishingConfig`); each prediction reports which `stage` decided it. Train with `python train.py phishing` (or `--demo`); checkpoints go to `checkpoints/phishing/`.

---

## ⚡ Quickstart in 3 Steps

### Step 1: Install Dependencies

```bash
# Clone and enter repo
cd Safestep-Models

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
cp .env.example .env
```

### Step 2: Prepare Data (or use Instant Demo)

You can either use your own datasets or generate ready-to-train sample data with one command:

```bash
# Generate sample datasets for CART and JAW in data/
python -m train --prepare-data all --data-dir data/
```

*Or skip data preparation entirely and run training in demo mode using `--demo`!*

### Step 3: Run Training

You can use the convenient `./train.sh` terminal wrapper or invoke `python train.py` directly:

```bash
# Easy Terminal Commands:
./train.sh cart                # Train CART (AI-text detector)
./train.sh jaw                 # Train JAW (AI-image detector)
./train.sh phishing            # Train Attack / Beast / Warhammer phishing pipeline
./train.sh all                 # Train CART + JAW in parallel
./train.sh demo                # Test training instantly on synthetic demo data
./train.sh interactive         # Launch interactive step-by-step terminal wizard

# Or standard python commands:
python train.py cart
python train.py jaw
python train.py all --dashboard
```

---

## 📁 Data Loader Guide (Simple & Flexible)

SafeStep features intelligent, multi-format data loaders that accept standard benchmarks, popular downloads (like Kaggle CIFAKE), flat folders, or CSV/JSONL files with **automatic label normalization** and **auto-partitioning**.

### 1. JAW (AI-Generated Image Detection)

JAW analyzes both high-level image semantics (via EfficientNet-B3) and frequency-domain artifacts (via 2D-FFT radial spectral energy and Gabor filter banks).

#### Supported Image Formats
- `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`

#### Supported Directory Structures
You can organize your images in any of the following layouts:

**Option A: Standard Split Layout (Recommended)**
```
data/
├── train/
│   ├── authentic/      # Real images
│   │   ├── 001.jpg
│   │   └── ...
│   └── ai_generated/   # AI-generated images
│       ├── 001.png
│       └── ...
└── val/
    ├── authentic/
    └── ai_generated/
```

**Option B: Kaggle CIFAKE or Real/Fake Folders**
If you download CIFAKE from Kaggle or GitHub, you don't need to rename or rearrange anything!
```
data/
├── train/
│   ├── REAL/           # Auto-normalized to 'authentic'
│   └── FAKE/           # Auto-normalized to 'ai_generated'
└── test/ (or val/)
    ├── REAL/
    └── FAKE/
```

**Option C: Flat / Un-Split Folders (Automatic 80/20 Partition)**
Just put your images into two folders—SafeStep will automatically partition them into 80% train and 20% validation:
```
data/
├── real/               # Auto-mapped to 'authentic'
└── fake/               # Auto-mapped to 'ai_generated'
```

#### Automatic Label Normalization
Folder names are automatically normalized (case-insensitive):

| Target Label | Accepted Folder / Class Names |
|---|---|
| **`authentic`** | `authentic`, `real`, `REAL`, `human`, `nature`, `genuine`, `original`, `photo`, `0` |
| **`ai_generated`** | `ai_generated`, `fake`, `FAKE`, `synthetic`, `ai`, `generated`, `deepfake`, `1` |

#### Built-in Dataset Support

##### CIFAKE (60,000 Real & 60,000 AI Images)
Download from [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) or GitHub, then:
```bash
# Point directly to your extracted CIFAKE folder:
python -m train --model jaw --data-dir /path/to/cifake

# Or import and organize into standard data/jaw format:
python -m util.manager --cifake /path/to/cifake --export data/jaw
```

##### GenImage & ArtiFact
In Python:
```python
from util.loaders.jaw import JAWDatasetLoader

loader = JAWDatasetLoader()
loader.load_genimage("/path/to/genimage")  # Auto-maps generator subfolders and 'nature'/'ai'
loader.load_artifact("/path/to/artifact")  # Auto-maps multi-generator benchmarks
splits = loader.get_splits()
```

##### Loading Any Custom Directory
```python
from util.loaders.jaw import JAWDatasetLoader

loader = JAWDatasetLoader()
loader.load_directory("/path/to/my_images")
splits = loader.get_splits()  # returns {'train': (paths, labels), 'val': (...), 'test': (...)}
```

---

### 2. CART (AI-Generated Text Detection)

CART combines a fine-tuned DistilBERT neural classifier with stylistic/heuristic repetition and entropy extractors, fused with XGBoost.

#### Supported Text Formats

**Option A: Standard texts.txt & labels.txt**
```
data/
├── train/
│   ├── texts.txt       # One text document per line
│   └── labels.txt      # 'human' or 'ai_generated' per line
└── val/
    ├── texts.txt
    └── labels.txt
```

**Option B: CSV Files**
Save `train.csv` and `val.csv` (or single `texts.csv`):
```csv
text,label
"The committee reviewed quarterly finances.","human"
"As an AI language model, I provide balanced views.","ai_generated"
```

**Option C: JSON Lines (JSONL)**
Save `train.jsonl` and `val.jsonl`:
```json
{"text": "Sample human sentence.", "label": "human"}
{"text": "Generated sentence.", "label": "ai_generated"}
```

**Option D: Class Folders**
```
data/
├── human/              # *.txt files (auto-split 80/20 if un-split)
└── ai_generated/       # *.txt files
```

#### Automatic Text Label Normalization

| Target Label | Accepted Aliases |
|---|---|
| **`human`** | `human`, `real`, `authentic`, `genuine`, `original`, `0` |
| **`ai_generated`** | `ai_generated`, `ai`, `fake`, `synthetic`, `chatgpt`, `gpt`, `generated`, `1` |

#### Built-in Text Datasets
- **HC3** (Human ChatGPT Comparison Corpus): `loader.load_hc3("data/hc3.json")`
- **GPT-2 Output Dataset**: `loader.load_gpt2_output("data/gpt2_output.json")`
- **RAID / MAGE**: `loader.load_raid(...)`, `loader.load_mage(...)`
- **CSV / JSONL**: `loader.load_csv("data.csv")`, `loader.load_jsonl("data.jsonl")`

---

### 3. Unified Data Manager CLI (`util.manager`)

Use `util.manager` to inspect datasets, generate samples, or convert folders:

```bash
# Generate sample datasets for CART, JAW, and Phishing in data/
python -m util.manager --sample all --data-dir data/

# Generate sample dataset for JAW only
python -m util.manager --sample jaw --data-dir data/

# Inspect any dataset directory (counts images, texts, CSVs, and classes)
python -m util.manager --inspect data/

# Import CIFAKE and export to standard train/val splits
python -m util.manager --cifake /path/to/cifake --export data/jaw
```

---

## 🚀 Training Steps Made Simple

The `train.py` CLI is parallelized, device-aware, and time-estimated.

### 1. Instant Zero-Setup Demo / Smoke Test (`--demo`)
If you don't have datasets downloaded yet, run with `--demo` to automatically create synthetic samples and test the full training pipeline in seconds:

```bash
# Test JAW training pipeline
python -m train --model jaw --demo

# Test CART training pipeline
python -m train --model cart --demo

# Test both models simultaneously
python -m train --model all --demo
```

### 2. Pre-flight Time and Memory Estimation (`--estimate-only`)
Measure throughput, total epochs, and ETA on your current hardware without training:

```bash
python -m train --model jaw --data-dir data/ --estimate-only
python -m train --model cart --data-dir data/ --estimate-only
```

### 3. Train on GPU, Apple Silicon, or CPU (`--device`)
Device selection is automatic by default, or you can explicitly specify it:

```bash
# CUDA GPU
python -m train --model jaw --data-dir data/ --device cuda

# Apple Silicon (Metal Performance Shaders)
python -m train --model jaw --data-dir data/ --device mps

# CPU (with configurable data workers)
python -m train --model cart --data-dir data/ --device cpu --workers 8
```

### 4. Interactive Live Terminal Dashboard (`--dashboard`)
Launch a full-screen terminal dashboard with live hardware monitors, memory planning, progress bars, throughput metrics, and braille accuracy/loss charts:

```bash
# Live dashboard training both models
python -m train --model all --data-dir data/ --dashboard

# Live dashboard with demo data
python -m train --model all --dashboard --demo

# Preview the parallel hardware plan without training
python -m train --model all --data-dir data/ --dashboard --estimate-only
```

### 5. Hyperparameter Sweeps (`--sweep`)
Run parameter grid sweeps that parallelize sessions across available hardware:

```bash
# Sweep learning rate and batch size for CART
python -m train --model cart --sweep lr=1e-5,3e-5 --sweep batch_size=8,16

# Sweep frozen vs finetune epochs for JAW
python -m train --model jaw --sweep epochs_frozen=2,4 --sweep epochs_finetune=10,15
```

### 6. Resume from Checkpoints (`--resume`)
If training is interrupted, you can resume seamlessly:

```bash
# Resume JAW from checkpoint
python -m train --model jaw --data-dir data/ --resume checkpoints/jaw

# Resume CART from checkpoint
python -m train --model cart --data-dir data/ --resume checkpoints/cart
```

### 7. Run History & Chart Comparisons (`--history`)
Inspect past and partial runs with terminal charts comparing accuracy and loss over time:

```bash
# View all recent training runs
python -m train --history

# View recent JAW runs
python -m train --model jaw --history
```

---

## 💻 Python API Usage

You can also use the data loaders and trainers directly in Python scripts:

### Train JAW in Python
```python
from jaw.config import JAWConfig
from jaw.train import JAWTrainer
from util.manager import SafeStepDataManager

# 1. Load splits
dm = SafeStepDataManager()
splits = dm.get_jaw_data("data/jaw")
train_paths, train_labels = splits["train"]
val_paths, val_labels = splits["val"]

# 2. Train model
config = JAWConfig(train_batch_size=16, num_epochs_frozen=2, num_epochs_finetune=5)
trainer = JAWTrainer(config, device="cuda" if torch.cuda.is_available() else "cpu")
trainer.fit(train_paths, train_labels, val_paths, val_labels)

# 3. Save checkpoint
trainer.detector.save("checkpoints/jaw")
```

### Train CART in Python
```python
from cart.config import CARTConfig
from cart.train import CARTTrainer
from util.manager import SafeStepDataManager

dm = SafeStepDataManager()
splits = dm.get_cart_data("data/cart")
train_texts, train_labels = splits["train"]
val_texts, val_labels = splits["val"]

config = CARTConfig(train_batch_size=16, num_epochs_frozen=1, num_epochs_finetune=3)
trainer = CARTTrainer(config, device="cpu")
trainer.fit(train_texts, train_labels, val_texts, val_labels)

trainer.detector.save("checkpoints/cart")
```

---

## 🌐 Model Serving & Inference

Once models are trained (or using default initialized checkpoints), start the FastAPI service:

```bash
# Launch API server on port 8000
python -m api.cli serve --port 8000
```

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Run Predictions in Python
```python
import api

# Detect AI text
res_text = api.predict_text("Machine learning models are increasingly capable.")
print(f"CART: {res_text.label} (p_ai: {res_text.p_ai:.3f})")

# Detect AI image
res_img = api.predict_image("tests/samples/sample_authentic.png")
print(f"JAW: {res_img.label} (p_ai: {res_img.p_ai:.3f})")
```

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest
```

Or run the self-contained functional verification script:

```bash
python tests/run_tests.py
```
