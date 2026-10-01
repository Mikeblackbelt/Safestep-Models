"""
Training data loading shared by the CLI, the planner and worker processes.

Text:   {data_dir}/{split}/texts.txt + labels.txt (one item per line)
Images: {data_dir}/{split}/{label}/*.{png,jpg,jpeg,webp}
"""

from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _make_placeholder_text_data(n: int = 100) -> Tuple[list, list]:
    """Generate trivial placeholder text data for smoke-test runs."""
    texts = [f"Sample training sentence number {i}." for i in range(n)]
    labels = ["human" if i % 2 == 0 else "ai_generated" for i in range(n)]
    return texts, labels


def load_text_split(data_dir: Path, split: str, console: Optional[Console] = None) -> Tuple[list, list]:
    """
    Load {split}/texts.txt and {split}/labels.txt from data_dir.
    Falls back to placeholder data if files are not found.
    """
    txt_file = data_dir / split / "texts.txt"
    label_file = data_dir / split / "labels.txt"
    if txt_file.exists() and label_file.exists():
        texts = txt_file.read_text(encoding="utf-8").splitlines()
        labels = label_file.read_text(encoding="utf-8").splitlines()
        if console:
            console.print(f"  [green]✓[/green] Loaded {len(texts):,} {split} samples from {data_dir / split}")
        return texts, labels
    if console:
        console.print(
            f"  [yellow]⚠[/yellow]  No {split} data found at [dim]{txt_file}[/dim] — using placeholder data."
        )
    n = 200 if split == "train" else 50
    return _make_placeholder_text_data(n)


def load_image_split(data_dir: Path, split: str, console: Optional[Console] = None) -> Tuple[list, list]:
    """
    Expect data_dir/{split}/{label}/*.{png,jpg,jpeg,webp}.
    Returns empty lists if not found.
    """
    split_dir = data_dir / split
    paths, labels = [], []
    if split_dir.exists():
        for label_dir in sorted(split_dir.iterdir()):
            if label_dir.is_dir():
                label = label_dir.name  # directory name = class label
                for img in label_dir.glob("**/*"):
                    if img.suffix.lower() in IMAGE_SUFFIXES:
                        paths.append(str(img))
                        labels.append(label)
        if paths:
            if console:
                console.print(f"  [green]✓[/green] Loaded {len(paths):,} {split} images from {split_dir}")
            return paths, labels

    if console:
        console.print(
            f"  [yellow]⚠[/yellow]  No {split} images found at [dim]{split_dir}[/dim] — skipping (provide real data to train)."
        )
    return [], []


def load_splits(model: str, data_dir: Path, console: Optional[Console] = None) -> Tuple[list, list, list, list]:
    loader = load_text_split if model == "cart" else load_image_split
    train_x, train_y = loader(data_dir, "train", console)
    val_x, val_y = loader(data_dir, "val", console)
    return train_x, train_y, val_x, val_y
