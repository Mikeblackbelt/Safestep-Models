# Training

All training goes through `train.py`. You can run it as `python train.py ...` or `python -m train ...`, which do the same thing. `train.sh` is a thin bash wrapper around it.

`train.py` loads `.env` from the repo root (via `python-dotenv`) on start-up.

---

## `train.py` flag reference

```
python train.py [cart|jaw|phishing|all] [options]
```

The positional model name is optional. When given, it overrides `--model`.

### Core

| Flag | Default | Description |
|---|---|---|
| `--model {cart,jaw,phishing,all}` | `all` | Model to train. `all` means CART + JAW in parallel and does **not** include phishing. |
| `--data-dir PATH` | `data` | Root directory for training data |
| `--checkpoint-dir PATH` | `checkpoints` | Root for saved checkpoints. Each model writes to `<dir>/cart`, `<dir>/jaw` or `<dir>/phishing` |
| `--device {auto,cpu,cuda,mps}` | `auto` | Compute device |
| `--workers N` | `cpu_count // 2` (min 1) | Parallel data-loading workers |
| `--estimate-only` | off | Print the time estimate and exit without training |
| `--resume PATH` | none | Checkpoint dir to resume from (see [Resume](#resume)) |
| `--epochs-frozen N` | config | Frozen-backbone / head-only epochs |
| `--epochs-finetune N` | config | Full fine-tuning epochs |
| `--batch-size N` | config | Training batch size (`train_batch_size`) |
| `--lr RATE` | config | Learning rate. Sets `learning_rate` for CART and phishing, and `lr_finetune` for JAW (JAW's `lr_frozen` is unchanged) |
| `-v`, `--verbose` | off | Print per-epoch metrics |

For phishing, `--epochs-frozen`, `--epochs-finetune`, `--batch-size` and `--lr` only affect Warhammer.

### Dashboard & history

| Flag | Default | Description |
|---|---|---|
| `--dashboard` | off | Full-screen live dashboard. Plans parallel sessions from device metrics and shows ETA, utilization and accuracy/loss charts |
| `--sweep KEY=V1,V2` | none | Hyperparameter sweep, repeatable. Keys: `lr`, `batch_size`, `epochs_frozen`, `epochs_finetune`, or any config field. Implies `--dashboard` |
| `--max-parallel N` | planner | Cap on concurrent sessions |
| `--history` | off | Show previous and partial runs with charts, then exit. Combine with `--model` to filter |
| `--history-limit N` | `20` | Number of sessions shown by `--history` |
| `--runs-dir PATH` | `runs` | Where run records and logs live |

### Data convenience

| Flag | Default | Description |
|---|---|---|
| `--demo` | off | Use generated sample data when the real dataset is missing |
| `--prepare-data [{cart,jaw,phishing,all}]` | `all` if given bare | Generate sample datasets under `--data-dir` and exit |

`--history` and `--prepare-data` are handled first and exit before any training.

---

## `train.sh` reference

`train.sh` picks a Python interpreter in this order: `$VIRTUAL_ENV/bin/python`, `./.venv/bin/python`, `./venv/bin/python`, `python3`, `python`. On Windows, use `python train.py` directly or run the script from Git Bash/WSL.

| Command | Runs |
|---|---|
| `./train.sh` (no args, interactive terminal) | Interactive menu wizard. Prints help when stdin isn't a TTY |
| `./train.sh interactive` / `wizard` / `menu` | Interactive menu wizard |
| `./train.sh cart\|jaw\|all [opts]` | `train.py --model <m> [opts]` |
| `./train.sh phishing [opts]` | Passes through to `train.py phishing [opts]` |
| `./train.sh demo [cart\|jaw\|all] [opts]` | `train.py --model <m> --demo` (default `all`) |
| `./train.sh prepare\|prep-data\|sample [cart\|jaw\|phishing\|all] [opts]` | `train.py --prepare-data <m> [opts]` (default `all`) |
| `./train.sh inspect [dir]` | `python -m util.manager --inspect <dir>` (default `data`) |
| `./train.sh cifake <dir> [opts]` | `python -m util.manager --cifake <dir> [opts]` |
| `./train.sh dashboard\|dash [opts]` | `train.py --model all --dashboard [opts]` |
| `./train.sh estimate\|est [cart\|jaw\|all] [opts]` | `train.py --model <m> --estimate-only` (default `all`) |
| `./train.sh history\|hist\|logs [opts]` | `train.py --history [opts]` |
| `./train.sh -h\|--help\|help` | Wrapper help followed by `train.py --help` |
| anything else | Passed through as `train.py <args>` |

Notes:

- The interactive wizard only offers CART, JAW and "all". For phishing, use `./train.sh phishing`.
- `demo` and `estimate` only recognize `cart|jaw|all` as their model argument. `./train.sh demo phishing` still works, because `phishing` is forwarded as the positional model.
- `prepare` takes a data dir as a flag, not a positional argument: `./train.sh prepare all --data-dir mydata`. The help text says `prepare [data-dir]`, but a bare path is passed to `train.py` as an invalid model name.

---

## Data layouts

### CART (text): `training/data.py: load_text_split`

The loader looks for a split directory in this order: `{data-dir}/{split}`, `{data-dir}/cart/{split}`, `{data-dir}/cifake/{split}`, `{data-dir}/sample/{split}`, where `split` is `train` or `val`. Inside the first match it tries:

1. `texts.txt` + `labels.txt` (one item per line)
2. any `*.csv` (columns `text`/`content`/`body` and `label`/`target`/`class`)
3. any `*.jsonl` (keys `text`/`content` and `label`/`target`)
4. class folders `{split}/{human,ai_generated}/**/*.txt`

If nothing matches there, it falls back to `{data-dir}/{split}.csv`, then `{data-dir}/{split}.jsonl`, then un-split class folders `{data-dir}/{human,ai_generated}/` (a deterministic 80/20 split, where every 5th file goes to val).

> **Warning:** If no CART data is found, the loader logs a warning and trains on **placeholder sentences**, even without `--demo`. Check the "Loaded N samples" lines before you trust a CART checkpoint.

### JAW (images): `training/data.py: load_image_split`

It uses the same split-dir search with `jaw` as the model subfolder. Every subfolder of the split dir is treated as a class and normalized (`REAL` becomes `authentic`, `FAKE` becomes `ai_generated`, and so on, as listed in the README). Accepted extensions are `.png .jpg .jpeg .webp .bmp .tiff .tif`.

If there's no split dir, un-split class folders under `{data-dir}` (for example `real/` and `fake/`) are auto-partitioned 80/20. If nothing is found and `--demo` isn't set, JAW reports the missing data and doesn't train.

Only `train/` and `val/` are read. A Kaggle CIFAKE download ships `train/` and `test/`, so import it first:

```bash
python -m util.manager --cifake /path/to/cifake --export data/jaw
python train.py jaw --data-dir data/jaw
```

### Phishing: `train.py: _load_phishing_splits`

Root is `{data-dir}/phishing` if it exists, otherwise `{data-dir}`. Sources:

| Source | Path under root | Labels |
|---|---|---|
| Enron spam | `enron_spam/spam/*.txt`, `enron_spam/ham/*.txt` | phishing / legitimate |
| UCI phishing websites | `uci_phishing.data` | both |
| PhishTank | `phishtank.csv` | phishing only |

You need both classes. PhishTank on its own fails with "Need both phishing and legitimate samples". Splits come from `PhishingDatasetLoader.get_splits()`.

### Sample data

```bash
python train.py --prepare-data all --data-dir data      # data/jaw, data/cart, data/phishing
python train.py --prepare-data jaw --data-dir data/jaw  # a single model writes directly into --data-dir (phishing still uses <dir>/phishing)
python -m util.manager --sample all --data-dir data     # same generators via util.manager
python -m util.manager --inspect data                   # count images / texts / CSV / JSONL per folder
```

---

## Demo mode

`--demo` only kicks in when real data is missing:

- **JAW** generates synthetic images in `{data-dir}/.demo_jaw_cache`.
- **Phishing** generates sample emails in `{data-dir}/.demo_phishing_cache` when fewer than two classes were found.
- **CART** always falls back to placeholder text when data is missing, with or without `--demo` (see the warning above).

```bash
python train.py cart --demo
python train.py jaw --demo
python train.py phishing --demo -v
python train.py all --demo
```

---

## Resume

| Model | `--resume` points at | Resumes when this file exists |
|---|---|---|
| `cart` | `checkpoints/cart` | `neural_model.pt` (loads the full detector) |
| `jaw` | `checkpoints/jaw` | `jaw_model.pt` |
| `phishing` | `checkpoints/phishing` | `warhammer.pt` (warm-starts Warhammer only; Attack and Beast always refit) |
| `all` | `checkpoints` (the root) | `<root>/cart/...` and `<root>/jaw/...` |

If the expected file is missing, training silently starts from scratch, so double-check the path. Checkpoints are always written to `--checkpoint-dir/<model>` and overwrite what's there.

---

## Dashboard, sweeps and history

- `--dashboard` and `--sweep` support `cart`, `jaw` and `all`. With `--model phishing` they print a warning and phishing trains normally without them.
- `--history` reads run records from `--runs-dir`. Phishing runs aren't covered yet.
- With `--model all`, the dashboard resumes each model from `<--resume>/<model>`.

```bash
python train.py all --dashboard
python train.py all --dashboard --estimate-only        # preview the hardware plan
python train.py cart --sweep lr=1e-5,3e-5 --sweep batch_size=8,16
python train.py jaw --sweep epochs_frozen=2,4 --max-parallel 2
python train.py --history --history-limit 50
python train.py jaw --history
```
