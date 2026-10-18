"""
Training data loading shared by the CLI, the planner and worker processes.

Supports flexible directory structures:
Text:
  1. Standard:  {data_dir}/{split}/texts.txt + labels.txt (one item per line)
  2. Model subfolder: {data_dir}/cart/{split}/texts.txt + labels.txt
  3. CSV / JSONL:     {data_dir}/{split}.csv or {data_dir}/{split}.jsonl
  4. Class folders:   {data_dir}/{split}/{label}/*.txt or {data_dir}/{label}/*.txt (auto-split)

Images:
  1. Standard:        {data_dir}/{split}/{label}/*.{png,jpg,jpeg,webp,bmp}
  2. Model subfolder: {data_dir}/jaw/{split}/{label}/*.*
  3. CIFAKE / Kaggle: {data_dir}/{split}/{REAL,FAKE}/*.*
  4. Flat folders:    {data_dir}/{real,fake}/*.* or {data_dir}/{authentic,ai_generated}/*.* (auto-split)
"""

import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

from rich.console import Console

from util.loaders.cart import normalize_text_label
from util.loaders.jaw import normalize_image_label

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _make_placeholder_text_data(n: int = 100) -> Tuple[list, list]:
    """Generate placeholder text data for smoke-test runs."""
    texts = [f"Sample training sentence number {i} with distinct phrasing and characteristics." for i in range(n)]
    labels = ["human" if i % 2 == 0 else "ai_generated" for i in range(n)]
    return texts, labels


def _make_placeholder_image_data(cache_dir: Path, split: str, n: int = 30) -> Tuple[list, list]:
    """Generate temporary sample images for quick smoke tests and demo runs."""
    from util.loaders.jaw import JAWDatasetLoader
    demo_dir = cache_dir / ".demo_jaw_cache"
    if not (demo_dir / "train" / "authentic").exists():
        JAWDatasetLoader.create_sample_dataset(demo_dir, samples_per_class=n)
    return load_image_split(demo_dir, split, console=None, allow_demo=False)


def _resolve_split_dir(data_dir: Path, model_sub: str, split: str) -> Optional[Path]:
    """Check common subdirectories for a given split.

    For ``val``, falls back to ``valid/``, ``validation/`` and finally ``test/``
    (e.g. CIFAKE ships only train/ + test/).
    """
    names = [split]
    if split == "val":
        names += ["valid", "validation", "test"]
    roots = [data_dir, data_dir / model_sub, data_dir / "cifake", data_dir / "sample"]
    for name in names:
        for root in roots:
            c = root / name
            if c.exists() and c.is_dir():
                return c
    return None


def load_text_split(
    data_dir: Path,
    split: str,
    console: Optional[Console] = None,
    allow_demo: bool = False,
) -> Tuple[list, list]:
    """
    Load text data for a split. Supports:
      - {split}/texts.txt + labels.txt
      - {split}.csv or {split}/texts.csv (text & label columns)
      - {split}.jsonl
      - {split}/{human,ai_generated}/*.txt
      - Un-split class folders (auto-partitioned 80/20 train/val)
    If no data is found, returns empty lists unless allow_demo=True, in which
    case placeholder sentences are generated for smoke tests.
    """
    data_dir = Path(data_dir)
    split_dir = _resolve_split_dir(data_dir, "cart", split)

    # 1. Standard texts.txt + labels.txt
    if split_dir:
        txt_file = split_dir / "texts.txt"
        label_file = split_dir / "labels.txt"
        if txt_file.exists() and label_file.exists():
            texts = txt_file.read_text(encoding="utf-8", errors="replace").splitlines()
            raw_labels = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
            labels = [normalize_text_label(l) for l in raw_labels]
            if console:
                console.print(f"  [green]✓[/green] Loaded {len(texts):,} {split} samples from {split_dir}")
            return texts, labels

        # 2. Check for CSV / JSONL inside split_dir
        for f in split_dir.glob("*.csv"):
            texts, labels = _read_text_csv(f)
            if texts:
                if console:
                    console.print(f"  [green]✓[/green] Loaded {len(texts):,} {split} samples from {f.name}")
                return texts, labels

        for f in split_dir.glob("*.jsonl"):
            texts, labels = _read_text_jsonl(f)
            if texts:
                if console:
                    console.print(f"  [green]✓[/green] Loaded {len(texts):,} {split} samples from {f.name}")
                return texts, labels

        # 3. Check for subdirectories by class: {split}/{human,ai_generated}/*.txt
        class_subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
        texts, labels = [], []
        for cdir in class_subdirs:
            norm_lbl = normalize_text_label(cdir.name)
            if norm_lbl in ("human", "ai_generated"):
                for txt_p in cdir.glob("**/*.txt"):
                    try:
                        content = txt_p.read_text(encoding="utf-8", errors="ignore").strip()
                        if content:
                            texts.append(content)
                            labels.append(norm_lbl)
                    except OSError:
                        pass
        if texts:
            if console:
                console.print(f"  [green]✓[/green] Loaded {len(texts):,} {split} samples from class folders in {split_dir}")
            return texts, labels

    # 4. Check {data_dir}/{split}.csv or {split}.jsonl
    csv_cand = data_dir / f"{split}.csv"
    if csv_cand.exists():
        texts, labels = _read_text_csv(csv_cand)
        if texts:
            if console:
                console.print(f"  [green]✓[/green] Loaded {len(texts):,} {split} samples from {csv_cand}")
            return texts, labels

    jsonl_cand = data_dir / f"{split}.jsonl"
    if jsonl_cand.exists():
        texts, labels = _read_text_jsonl(jsonl_cand)
        if texts:
            if console:
                console.print(f"  [green]✓[/green] Loaded {len(texts):,} {split} samples from {jsonl_cand}")
            return texts, labels

    # 5. Check un-split root directory {data_dir}/{human,ai_generated}/*.txt (auto 80/20 partition)
    root_classes = [p for p in data_dir.iterdir() if p.is_dir()] if data_dir.is_dir() else []
    if any(normalize_text_label(p.name) in ("human", "ai_generated") for p in root_classes):
        texts, labels = _load_and_partition_text_classes(data_dir, split)
        if texts:
            if console:
                console.print(f"  [green]✓[/green] Auto-partitioned {len(texts):,} {split} text samples from {data_dir}")
            return texts, labels

    if not allow_demo:
        if console:
            console.print(f"  [red]✗[/red] No {split} text data found under [dim]{data_dir}[/dim].")
        return [], []

    if console:
        console.print(
            f"  [yellow]⚠[/yellow]  No {split} text data found under [dim]{data_dir}[/dim] — using placeholder data (demo mode)."
        )
    n = 200 if split == "train" else 50
    return _make_placeholder_text_data(n)


def _read_text_csv(csv_path: Path) -> Tuple[list, list]:
    texts, labels = [], []
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("text") or row.get("content") or row.get("body")
                l = row.get("label") or row.get("target") or row.get("class")
                if t and l:
                    texts.append(t)
                    labels.append(normalize_text_label(l))
    except Exception:
        pass
    return texts, labels


def _read_text_jsonl(jsonl_path: Path) -> Tuple[list, list]:
    texts, labels = [], []
    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                t = row.get("text") or row.get("content")
                l = row.get("label") or row.get("target")
                if t and l:
                    texts.append(t)
                    labels.append(normalize_text_label(l))
    except Exception:
        pass
    return texts, labels


def _load_and_partition_text_classes(data_dir: Path, split: str) -> Tuple[list, list]:
    """Auto-split un-partitioned text class folders into train (80%) and val (20%)."""
    all_texts = []
    all_labels = []
    for cdir in sorted(data_dir.iterdir()):
        if cdir.is_dir():
            norm_lbl = normalize_text_label(cdir.name)
            if norm_lbl in ("human", "ai_generated"):
                files = sorted(cdir.glob("**/*.txt"))
                for p in files:
                    try:
                        c = p.read_text(encoding="utf-8", errors="ignore").strip()
                        if c:
                            all_texts.append(c)
                            all_labels.append(norm_lbl)
                    except OSError:
                        pass
    if not all_texts:
        return [], []

    # Deterministic 80/20 split
    split_texts, split_labels = [], []
    for i, (t, l) in enumerate(zip(all_texts, all_labels)):
        is_val = (i % 5 == 0)
        if split == "val" and is_val:
            split_texts.append(t)
            split_labels.append(l)
        elif split == "train" and not is_val:
            split_texts.append(t)
            split_labels.append(l)

    return split_texts, split_labels


def load_image_split(
    data_dir: Path,
    split: str,
    console: Optional[Console] = None,
    allow_demo: bool = False,
) -> Tuple[list, list]:
    """
    Load image paths and labels for a split.
    Supports:
      - {data_dir}/{split}/{label}/*.{png,jpg,jpeg,webp,bmp}
      - {data_dir}/jaw/{split}/{label}/*.*
      - CIFAKE structure ({data_dir}/{split}/{REAL,FAKE}/*.*)
      - Flat un-split folders ({data_dir}/{real,fake}/*.*, auto-partitioned 80/20)
      - Demo/synthetic generation when allow_demo=True
    """
    data_dir = Path(data_dir)
    split_dir = _resolve_split_dir(data_dir, "jaw", split)

    paths, labels = [], []

    # 1. Load from split_dir if present
    if split_dir and split_dir.exists():
        for label_dir in sorted(split_dir.iterdir()):
            if label_dir.is_dir():
                norm_label = normalize_image_label(label_dir.name)
                for img in sorted(label_dir.glob("**/*")):
                    if img.suffix.lower() in IMAGE_SUFFIXES:
                        paths.append(str(img))
                        labels.append(norm_label)
        if paths:
            if console:
                console.print(f"  [green]✓[/green] Loaded {len(paths):,} {split} images from {split_dir}")
            return paths, labels

    # 2. Check if data_dir itself has un-split class folders (e.g. data/real and data/fake)
    root_subdirs = [p for p in data_dir.iterdir() if p.is_dir()] if data_dir.exists() else []
    recognized_classes = [
        p for p in root_subdirs
        if normalize_image_label(p.name) in ("authentic", "ai_generated")
    ]
    if recognized_classes:
        paths, labels = _load_and_partition_image_classes(recognized_classes, split)
        if paths:
            if console:
                console.print(
                    f"  [green]✓[/green] Auto-partitioned {len(paths):,} {split} images from {data_dir}"
                )
            return paths, labels

    # 3. Demo fallback if explicitly allowed
    if allow_demo:
        if console:
            console.print(f"  [cyan]ℹ[/cyan] Generating sample images for {split} split demo…")
        return _make_placeholder_image_data(data_dir, split)

    # 4. No images found
    if console:
        console.print(
            f"  [yellow]⚠[/yellow]  No {split} images found at [dim]{split_dir or data_dir / split}[/dim].\n"
            f"     [dim]Expected layout: data/{split}/authentic/ and data/{split}/ai_generated/ (or real/ and fake/)[/dim]\n"
            f"     [dim]Tip: pass --demo to test with generated sample data, or run:\n"
            f"       python -m util.manager --sample jaw --data-dir {data_dir}[/dim]"
        )
    return [], []


def _load_and_partition_image_classes(class_dirs: List[Path], split: str) -> Tuple[list, list]:
    """Auto-split un-partitioned image class folders into train (80%) and val (20%)."""
    all_paths = []
    all_labels = []
    for cdir in sorted(class_dirs):
        norm_label = normalize_image_label(cdir.name)
        imgs = [p for p in sorted(cdir.glob("**/*")) if p.suffix.lower() in IMAGE_SUFFIXES]
        for img in imgs:
            all_paths.append(str(img))
            all_labels.append(norm_label)

    if not all_paths:
        return [], []

    # Deterministic 80/20 split
    split_paths, split_labels = [], []
    for i, (p, l) in enumerate(zip(all_paths, all_labels)):
        is_val = (i % 5 == 0)
        if split == "val" and is_val:
            split_paths.append(p)
            split_labels.append(l)
        elif split == "train" and not is_val:
            split_paths.append(p)
            split_labels.append(l)

    return split_paths, split_labels


def load_splits(
    model: str,
    data_dir: Path,
    console: Optional[Console] = None,
    allow_demo: bool = False,
) -> Tuple[list, list, list, list]:
    """Unified loader returning (train_x, train_y, val_x, val_y)."""
    if model == "cart":
        train_x, train_y = load_text_split(data_dir, "train", console, allow_demo=allow_demo)
        val_x, val_y = load_text_split(data_dir, "val", console, allow_demo=allow_demo)
    else:
        train_x, train_y = load_image_split(data_dir, "train", console, allow_demo=allow_demo)
        val_x, val_y = load_image_split(data_dir, "val", console, allow_demo=allow_demo)
    return train_x, train_y, val_x, val_y
