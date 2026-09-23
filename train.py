"""
SafeStep Training CLI
=====================
Parallelized, device-aware training launcher for CART and JAW models.

Usage examples
--------------
# Estimate time only (no training)
python -m train --model cart --data-dir data/ --estimate-only

# Train CART on CPU with 4 parallel data workers
python -m train --model cart --data-dir data/ --workers 4

# Train JAW on CUDA, custom checkpoint dir, verbose
python -m train --model jaw --data-dir data/ --device cuda --checkpoint-dir checkpoints/ -v

# Train both models in sequence
python -m train --model all --data-dir data/ --workers 8

# Resume from existing checkpoint
python -m train --model cart --data-dir data/ --resume checkpoints/cart
"""

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Load .env before any project imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── Rich console ────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich import print as rprint

console = Console()

# ── Project imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import torch

from cart.config import CARTConfig
from cart.detector import CARTDetector
from cart.train import CARTTrainer
from jaw.config import JAWConfig
from jaw.detector import JAWDetector
from jaw.train import JAWTrainer
from util.logging_config import get_logger

logger = get_logger("safestep.train_cli")


# ── Device utilities ─────────────────────────────────────────────────────────────

def detect_device(requested: str = "auto") -> str:
    """Resolve and validate the compute device, printing a clear report."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        console.print("[bold yellow]⚠  CUDA requested but not available — falling back to CPU.[/bold yellow]")
        return "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        console.print("[bold yellow]⚠  MPS requested but not available — falling back to CPU.[/bold yellow]")
        return "cpu"
    return requested


def device_report(device: str, workers: int) -> None:
    """Print a Rich table summarising the detected hardware."""
    table = Table(title="Hardware Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Device", device.upper())
    table.add_row("CPU cores", str(multiprocessing.cpu_count()))
    table.add_row("Data workers", str(workers))

    if device == "cuda":
        gpu_count = torch.cuda.device_count()
        table.add_row("CUDA GPUs", str(gpu_count))
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024 ** 3)
            table.add_row(f"  GPU {i}", f"{props.name}  {mem_gb:.1f} GB")
    elif device == "mps":
        table.add_row("Apple Silicon", "MPS backend enabled")
    else:
        table.add_row("Note", "No GPU detected — training will be slower on CPU")

    torch_ver = torch.__version__
    table.add_row("PyTorch", torch_ver)
    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    console.print(table)


# ── Training time estimation ──────────────────────────────────────────────────────

# Empirical throughput baselines (samples / second) on CPU.
# These are deliberately conservative. GPU multipliers applied below.
_CART_CPU_SAMPLES_PER_SEC = 12.0   # DistilBERT forward on short texts
_JAW_CPU_SAMPLES_PER_SEC  = 4.0    # EfficientNet-B3 on 224×224 images

_GPU_SPEEDUP = 8.0   # rough multiplier vs CPU for CUDA
_MPS_SPEEDUP = 4.0   # rough multiplier vs CPU for MPS


def _effective_throughput(base: float, device: str) -> float:
    if device == "cuda":
        return base * _GPU_SPEEDUP
    if device == "mps":
        return base * _MPS_SPEEDUP
    return base


def estimate_cart_time(
    n_train: int,
    n_val: int,
    config: CARTConfig,
    device: str,
) -> dict:
    """Estimate total CART training wall-time in seconds."""
    tput = _effective_throughput(_CART_CPU_SAMPLES_PER_SEC, device)
    total_epochs = config.num_epochs_frozen + config.num_epochs_finetune
    # Each epoch: forward on train + eval on val
    secs_per_epoch = (n_train / tput) + (n_val / tput) * 0.5
    # Fusion fitting is fast (XGBoost on val set)
    fusion_secs = n_val * 0.001
    total = secs_per_epoch * total_epochs + fusion_secs
    return {
        "total_epochs": total_epochs,
        "frozen_epochs": config.num_epochs_frozen,
        "finetune_epochs": config.num_epochs_finetune,
        "secs_per_epoch": secs_per_epoch,
        "fusion_secs": fusion_secs,
        "total_secs": total,
    }


def estimate_jaw_time(
    n_train: int,
    n_val: int,
    config: JAWConfig,
    device: str,
) -> dict:
    tput = _effective_throughput(_JAW_CPU_SAMPLES_PER_SEC, device)
    total_epochs = config.num_epochs_frozen + config.num_epochs_finetune
    secs_per_epoch = (n_train / tput) + (n_val / tput) * 0.5
    total = secs_per_epoch * total_epochs
    return {
        "total_epochs": total_epochs,
        "frozen_epochs": config.num_epochs_frozen,
        "finetune_epochs": config.num_epochs_finetune,
        "secs_per_epoch": secs_per_epoch,
        "fusion_secs": 0,
        "total_secs": total,
    }


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def print_estimate(model_name: str, n_train: int, n_val: int, estimate: dict) -> None:
    """Print a nicely formatted pre-flight time estimate."""
    table = Table(
        title=f"[bold]{model_name} Training Estimate[/bold]",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Phase", style="dim")
    table.add_column("Details")
    table.add_column("Est. Time", justify="right", style="bold green")

    table.add_row("Dataset", f"train={n_train:,}  val={n_val:,}", "")
    table.add_row(
        f"Frozen epochs ({estimate['frozen_epochs']})",
        f"{estimate['secs_per_epoch']:.1f}s / epoch",
        _fmt_duration(estimate["frozen_epochs"] * estimate["secs_per_epoch"]),
    )
    table.add_row(
        f"Finetune epochs ({estimate['finetune_epochs']})",
        f"{estimate['secs_per_epoch']:.1f}s / epoch",
        _fmt_duration(estimate["finetune_epochs"] * estimate["secs_per_epoch"]),
    )
    if estimate["fusion_secs"] > 0:
        table.add_row("XGBoost fusion fit", "", _fmt_duration(estimate["fusion_secs"]))
    table.add_row(
        "[bold]Total[/bold]",
        f"{estimate['total_epochs']} epochs",
        f"[bold yellow]{_fmt_duration(estimate['total_secs'])}[/bold yellow]",
    )
    console.print(table)


# ── Synthetic placeholder data (used when no real data dir supplied) ─────────────

def _make_placeholder_text_data(n: int = 100) -> tuple[list, list]:
    """Generate trivial placeholder text data for smoke-test runs."""
    texts  = [f"Sample training sentence number {i}." for i in range(n)]
    labels = ["human" if i % 2 == 0 else "ai_generated" for i in range(n)]
    return texts, labels


def _load_text_split(data_dir: Path, split: str) -> tuple[list, list]:
    """
    Load {split}/texts.txt  (one text per line) and
         {split}/labels.txt (one label per line) from data_dir.
    Falls back to placeholder data if files are not found.
    """
    txt_file   = data_dir / split / "texts.txt"
    label_file = data_dir / split / "labels.txt"
    if txt_file.exists() and label_file.exists():
        texts  = txt_file.read_text(encoding="utf-8").splitlines()
        labels = label_file.read_text(encoding="utf-8").splitlines()
        console.print(f"  [green]✓[/green] Loaded {len(texts):,} {split} samples from {data_dir / split}")
        return texts, labels
    else:
        console.print(
            f"  [yellow]⚠[/yellow]  No {split} data found at [dim]{txt_file}[/dim] — using placeholder data."
        )
        n = 200 if split == "train" else 50
        return _make_placeholder_text_data(n)


def _load_image_split(data_dir: Path, split: str) -> tuple[list, list]:
    """
    Expect data_dir/{split}/{label}/*.{png,jpg,jpeg,webp}.
    Falls back to placeholder (empty paths) if not found.
    """
    split_dir = data_dir / split
    paths, labels = [], []
    if split_dir.exists():
        for label_dir in sorted(split_dir.iterdir()):
            if label_dir.is_dir():
                label = label_dir.name  # directory name = class label
                for img in label_dir.glob("**/*"):
                    if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                        paths.append(str(img))
                        labels.append(label)
        if paths:
            console.print(f"  [green]✓[/green] Loaded {len(paths):,} {split} images from {split_dir}")
            return paths, labels

    console.print(
        f"  [yellow]⚠[/yellow]  No {split} images found at [dim]{split_dir}[/dim] — skipping (provide real data to train)."
    )
    return [], []


# ── CART training entry-point ────────────────────────────────────────────────────

def train_cart(
    data_dir: Path,
    checkpoint_dir: Path,
    device: str,
    workers: int,
    estimate_only: bool,
    resume: Optional[Path],
    verbose: bool,
    config_overrides: dict,
) -> bool:
    """Run the full CART training pipeline. Returns True on success."""
    console.print(Rule("[bold blue]CART — AI-Text Detector[/bold blue]"))

    config = CARTConfig(**config_overrides) if config_overrides else CARTConfig()

    # Load splits
    console.print("[dim]Loading training data…[/dim]")
    train_texts, train_labels = _load_text_split(data_dir, "train")
    val_texts,   val_labels   = _load_text_split(data_dir, "val")

    if not train_texts:
        console.print("[red]✗  No training data available for CART. Aborting.[/red]")
        return False

    # Time estimate
    est = estimate_cart_time(len(train_texts), len(val_texts), config, device)
    print_estimate("CART", len(train_texts), len(val_texts), est)

    if estimate_only:
        return True

    console.print(f"\n[bold]Starting CART training[/bold]  device=[cyan]{device}[/cyan]  workers=[cyan]{workers}[/cyan]")
    t0 = time.perf_counter()

    trainer = CARTTrainer(config, device=device)

    # Optionally resume
    if resume and (resume / "neural_model.pt").exists():
        console.print(f"  [cyan]Resuming from checkpoint:[/cyan] {resume}")
        trainer.detector.load(resume)

    # Training with a Rich progress bar wrapping each epoch
    total_epochs = config.num_epochs_frozen + config.num_epochs_finetune

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        epoch_task = progress.add_task("[blue]CART epochs", total=total_epochs)

        # Monkey-patch the trainer's logger to tick the progress bar
        original_info = logger.info

        epoch_counter = [0]
        def _patched_info(msg, *args, **kwargs):
            original_info(msg, *args, **kwargs)
            if "[frozen] epoch" in msg or "[finetune] epoch" in msg:
                epoch_counter[0] += 1
                progress.update(epoch_task, completed=epoch_counter[0])
            if verbose:
                console.print(f"  [dim]{msg}[/dim]")

        import cart.train as _cart_train_mod
        _cart_train_mod.logger.info = _patched_info  # type: ignore[attr-defined]

        try:
            trainer.fit(train_texts, train_labels, val_texts, val_labels)
        finally:
            _cart_train_mod.logger.info = original_info  # type: ignore[attr-defined]

        progress.update(epoch_task, completed=total_epochs)

    # Save
    out_dir = checkpoint_dir / "cart"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.detector.save(out_dir)
    elapsed = time.perf_counter() - t0

    console.print(
        f"\n[bold green]✓  CART training complete[/bold green]  "
        f"saved → [cyan]{out_dir}[/cyan]  "
        f"wall-time: [bold]{_fmt_duration(elapsed)}[/bold]"
    )
    return True


# ── JAW training entry-point ─────────────────────────────────────────────────────

def train_jaw(
    data_dir: Path,
    checkpoint_dir: Path,
    device: str,
    workers: int,
    estimate_only: bool,
    resume: Optional[Path],
    verbose: bool,
    config_overrides: dict,
) -> bool:
    """Run the full JAW training pipeline. Returns True on success."""
    console.print(Rule("[bold magenta]JAW — AI-Image Detector[/bold magenta]"))

    config = JAWConfig(**config_overrides) if config_overrides else JAWConfig()

    # Load splits
    console.print("[dim]Loading training data…[/dim]")
    train_paths, train_labels = _load_image_split(data_dir, "train")
    val_paths,   val_labels   = _load_image_split(data_dir, "val")

    if not train_paths:
        console.print("[red]✗  No training images available for JAW. Aborting.[/red]")
        return False

    # Time estimate
    est = estimate_jaw_time(len(train_paths), len(val_paths), config, device)
    print_estimate("JAW", len(train_paths), len(val_paths), est)

    if estimate_only:
        return True

    console.print(f"\n[bold]Starting JAW training[/bold]  device=[cyan]{device}[/cyan]  workers=[cyan]{workers}[/cyan]")
    t0 = time.perf_counter()

    trainer = JAWTrainer(config, device=device)

    if resume and (resume / "jaw_model.pt").exists():
        console.print(f"  [cyan]Resuming from checkpoint:[/cyan] {resume}")
        trainer.detector.load(resume)

    total_epochs = config.num_epochs_frozen + config.num_epochs_finetune

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        epoch_task = progress.add_task("[magenta]JAW epochs", total=total_epochs)

        original_info = logger.info
        epoch_counter = [0]

        def _patched_info(msg, *args, **kwargs):
            original_info(msg, *args, **kwargs)
            if "[frozen] epoch" in msg or "[finetune] epoch" in msg:
                epoch_counter[0] += 1
                progress.update(epoch_task, completed=epoch_counter[0])
            if verbose:
                console.print(f"  [dim]{msg}[/dim]")

        import jaw.train as _jaw_train_mod
        _jaw_train_mod.logger.info = _patched_info  # type: ignore[attr-defined]

        try:
            trainer.fit(train_paths, train_labels, val_paths, val_labels)
        finally:
            _jaw_train_mod.logger.info = original_info  # type: ignore[attr-defined]

        progress.update(epoch_task, completed=total_epochs)

    out_dir = checkpoint_dir / "jaw"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.detector.save(out_dir)
    elapsed = time.perf_counter() - t0

    console.print(
        f"\n[bold green]✓  JAW training complete[/bold green]  "
        f"saved → [cyan]{out_dir}[/cyan]  "
        f"wall-time: [bold]{_fmt_duration(elapsed)}[/bold]"
    )
    return True


# ── Parallel "train all" launcher ────────────────────────────────────────────────

def _cart_worker(args: dict) -> tuple[str, bool]:
    """Process-pool worker for CART (must be picklable → top-level fn)."""
    try:
        ok = train_cart(**args)
        return "cart", ok
    except Exception as exc:
        return "cart", False


def _jaw_worker(args: dict) -> tuple[str, bool]:
    """Process-pool worker for JAW."""
    try:
        ok = train_jaw(**args)
        return "jaw", ok
    except Exception as exc:
        return "jaw", False


def train_all_parallel(
    data_dir: Path,
    checkpoint_dir: Path,
    device: str,
    workers: int,
    estimate_only: bool,
    resume_cart: Optional[Path],
    resume_jaw: Optional[Path],
    verbose: bool,
) -> None:
    """
    Launch CART and JAW training in parallel subprocesses.

    Note: GPU training cannot be truly parallelised on a single GPU (OOM risk).
    On CPU we use both workers simultaneously; on CUDA/MPS we run sequentially.
    """
    shared = dict(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
        workers=workers,
        estimate_only=estimate_only,
        verbose=verbose,
        config_overrides={},
    )
    cart_args = {**shared, "resume": resume_cart}
    jaw_args  = {**shared, "resume": resume_jaw}

    if device == "cpu" and not estimate_only:
        console.print(
            "\n[bold yellow]Parallel mode:[/bold yellow] "
            "training CART and JAW simultaneously in separate processes.\n"
        )
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_cart_worker, cart_args): "CART",
                pool.submit(_jaw_worker,  jaw_args):  "JAW",
            }
            for fut in as_completed(futures):
                model_name = futures[fut]
                _, ok = fut.result()
                status = "[green]✓ success[/green]" if ok else "[red]✗ failed[/red]"
                console.print(f"  {model_name}: {status}")
    else:
        # Sequential (single GPU or estimate-only)
        if device != "cpu":
            console.print(
                f"\n[dim]Sequential mode on {device.upper()} "
                "(parallel GPU training requires multiple GPUs).[/dim]\n"
            )
        train_cart(**cart_args)
        train_jaw(**jaw_args)


# ── Argument parsing ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train",
        description="SafeStep Training CLI — parallelized, device-aware model trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        choices=["cart", "jaw", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory for training data (default: data/)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory for saved checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel data-loading workers (default: cpu_count // 2)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print training time estimate then exit without training",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint dir to resume from (e.g. checkpoints/cart for CART; checkpoints/ for all)",
    )
    parser.add_argument(
        "--epochs-frozen",
        type=int,
        default=None,
        help="Override number of frozen-backbone epochs",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=None,
        help="Override number of fine-tuning epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-epoch metrics to console",
    )
    return parser


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────────────────────────
    console.print(
        Panel(
            "[bold cyan]SafeStep Training CLI[/bold cyan]\n"
            "[dim]Parallelized · Device-aware · Time-estimated[/dim]",
            border_style="cyan",
        )
    )

    # ── Device resolution ────────────────────────────────────────────────────────
    device = detect_device(args.device)
    device_report(device, args.workers)
    console.print()

    # ── Config overrides ─────────────────────────────────────────────────────────
    cart_overrides: dict = {}
    jaw_overrides:  dict = {}

    if args.epochs_frozen is not None:
        cart_overrides["num_epochs_frozen"] = args.epochs_frozen
        jaw_overrides["num_epochs_frozen"]  = args.epochs_frozen
    if args.epochs_finetune is not None:
        cart_overrides["num_epochs_finetune"] = args.epochs_finetune
        jaw_overrides["num_epochs_finetune"]  = args.epochs_finetune
    if args.batch_size is not None:
        cart_overrides["train_batch_size"] = args.batch_size
        jaw_overrides["train_batch_size"]  = args.batch_size
    if args.lr is not None:
        cart_overrides["learning_rate"] = args.lr
        jaw_overrides["lr_finetune"]    = args.lr

    # ── Dispatch ─────────────────────────────────────────────────────────────────
    if args.model == "cart":
        train_cart(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            workers=args.workers,
            estimate_only=args.estimate_only,
            resume=args.resume,
            verbose=args.verbose,
            config_overrides=cart_overrides,
        )

    elif args.model == "jaw":
        train_jaw(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            workers=args.workers,
            estimate_only=args.estimate_only,
            resume=args.resume,
            verbose=args.verbose,
            config_overrides=jaw_overrides,
        )

    else:  # "all"
        resume_cart = args.resume / "cart" if args.resume else None
        resume_jaw  = args.resume / "jaw"  if args.resume else None
        train_all_parallel(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            workers=args.workers,
            estimate_only=args.estimate_only,
            resume_cart=resume_cart,
            resume_jaw=resume_jaw,
            verbose=args.verbose,
        )

    if args.estimate_only:
        console.print("\n[dim]--estimate-only flag set; no training was performed.[/dim]")


if __name__ == "__main__":
    main()
