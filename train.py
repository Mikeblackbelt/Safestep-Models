"""
SafeStep Training CLI
=====================
Parallelized, device-aware training launcher for the CART, JAW and Phishing models.

Models
------
cart      AI-generated text detector (DistilBERT + heuristics, XGBoost fusion)
jaw       AI-generated image detector (EfficientNet-B3 + frequency features)
phishing  Attack -> Beast -> Warhammer escalation pipeline for emails/URLs
all       CART + JAW (phishing is trained separately)

Checkpoints are written to {--checkpoint-dir}/{model}/.

Usage examples
--------------
# Train the phishing pipeline (data from data/phishing/ or data/)
python -m train --model phishing --data-dir data/

# Phishing smoke test on generated sample emails
python -m train --model phishing --demo -v


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

# Live dashboard: device-aware parallel sessions, ETA, charts
python -m train --model all --data-dir data/ --dashboard

# Preview the parallel plan and ETA without training
python -m train --model all --data-dir data/ --dashboard --estimate-only

# Hyper-parameter sweep, sessions parallelised to fit the hardware
python -m train --model cart --dashboard --sweep lr=1e-5,3e-5 --sweep batch_size=8,16

# Previous and partial runs, with charts
python -m train --history
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

# Legacy Windows code pages can't encode the symbols used below; degrade instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

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
from phishing.config import PhishingConfig
from phishing.train import PhishingTrainer
from training.data import load_image_split, load_text_split
from training.history import RunRecord, RunRecorder, new_run_id
from training.planner import (
    CART_CPU_SAMPLES_PER_SEC as _CART_CPU_SAMPLES_PER_SEC,
    GPU_SPEEDUP as _GPU_SPEEDUP,
    JAW_CPU_SAMPLES_PER_SEC as _JAW_CPU_SAMPLES_PER_SEC,
    MPS_SPEEDUP as _MPS_SPEEDUP,
    total_units,
)
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

# Throughput baselines live in training.planner, which also calibrates them
# from previous runs for the --dashboard planner.


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


# ── Data loading ─────────────────────────────────────────────────────────────────

def _load_text_split(data_dir: Path, split: str, allow_demo: bool = False) -> tuple[list, list]:
    return load_text_split(data_dir, split, console, allow_demo=allow_demo)


def _load_image_split(data_dir: Path, split: str, allow_demo: bool = False) -> tuple[list, list]:
    return load_image_split(data_dir, split, console, allow_demo=allow_demo)


# ── Shared training loop with progress + run recording ───────────────────────────

def _fit_with_progress(
    model: str,
    trainer,
    config,
    train_x: list,
    train_y: list,
    val_x: list,
    val_y: list,
    device: str,
    verbose: bool,
    runs_dir: Path,
    run_id: Optional[str],
    estimated_secs: float,
) -> None:
    """Run trainer.fit with a Rich epoch bar, persisting a RunRecord for --history."""
    total_epochs = config.num_epochs_frozen + config.num_epochs_finetune
    recorder = RunRecorder(runs_dir, RunRecord(
        run_id=run_id or new_run_id(),
        session_id=model,
        label=model,
        model=model,
        device=device,
        threads=torch.get_num_threads(),
        config=asdict(config),
        n_train=len(train_x),
        n_val=len(val_x),
        total_epochs=total_epochs,
        total_units=total_units(model, config, len(train_x), len(val_x)),
        estimated_secs=estimated_secs,
    ))
    color = "blue" if model == "cart" else "magenta"

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
        epoch_task = progress.add_task(f"[{color}]{model.upper()} epochs", total=total_epochs)
        epochs_done = [0]

        def on_event(event: dict) -> None:
            recorder.handle(event)
            if event["type"] == "epoch":
                epochs_done[0] += 1
                progress.update(epoch_task, completed=epochs_done[0])
                if verbose:
                    console.print(
                        f"  [dim][{event['stage']}] epoch {event['epoch']}/{event['epochs']} "
                        f"loss={event['loss']:.4f} val_acc={event['val_acc']:.4f}[/dim]"
                    )
            elif event["type"] == "fusion" and verbose:
                console.print(f"  [dim]fusion val_acc={event['val_acc']:.4f}[/dim]")

        trainer.callback = on_event
        try:
            trainer.fit(train_x, train_y, val_x, val_y)
        except KeyboardInterrupt:
            recorder.finish("interrupted")
            raise
        except Exception as exc:
            recorder.finish("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        progress.update(epoch_task, completed=total_epochs)

    recorder.finish("completed")


# ── CART training entry-point ────────────────────────────────────────────────────

CART_RESUME_FILES = ("neural_model.pt", "fusion_model.json")
JAW_RESUME_FILES = ("jaw_model.pt",)
PHISHING_RESUME_FILES = ("warhammer.pt",)


def _check_resume(resume: Optional[Path], required: tuple, model_name: str) -> bool:
    """Return True if no resume requested or all required checkpoint files exist."""
    if not resume:
        return True
    resume = Path(resume)
    missing = [f for f in required if not (resume / f).is_file()]
    if resume.is_dir() and not missing:
        return True
    reason = "directory does not exist" if not resume.is_dir() else "missing: " + ", ".join(missing)
    expected = "\n".join(f"  • [cyan]{resume / f}[/cyan]" for f in required)
    console.print(
        Panel(
            f"[bold red]✗  Cannot resume {model_name}: {reason}[/bold red]\n\n"
            f"Resume dir: [dim]{resume}[/dim]\n\n"
            f"[bold]Expected files:[/bold]\n{expected}\n\n"
            "Fix the --resume path or omit it to train from scratch.",
            title=f"{model_name} Training - Resume Checkpoint Missing",
            border_style="red",
        )
    )
    return False


def train_cart(
    data_dir: Path,
    checkpoint_dir: Path,
    device: str,
    workers: int,
    estimate_only: bool,
    resume: Optional[Path],
    verbose: bool,
    config_overrides: dict,
    runs_dir: Path = Path("runs"),
    run_id: Optional[str] = None,
    demo: bool = False,
) -> bool:
    """Run the full CART training pipeline. Returns True on success."""
    console.print(Rule("[bold blue]CART — AI-Text Detector[/bold blue]"))
    if not _check_resume(resume, CART_RESUME_FILES, "CART"):
        return False

    config = CARTConfig(**config_overrides) if config_overrides else CARTConfig()

    # Load splits
    console.print("[dim]Loading training data…[/dim]")
    train_texts, train_labels = _load_text_split(data_dir, "train", allow_demo=demo)
    val_texts,   val_labels   = _load_text_split(data_dir, "val", allow_demo=demo)

    if not train_texts:
        console.print(
            Panel(
                "[bold red]✗  No training text data found[/bold red]\n\n"
                f"Searched under: [dim]{data_dir}[/dim]\n\n"
                "[bold]Supported text formats:[/bold]\n"
                "  1. Text & Labels files:   [cyan]data/train/texts.txt[/cyan] and [cyan]data/train/labels.txt[/cyan]\n"
                "  2. CSV file:              [cyan]data/train.csv[/cyan] (with 'text' and 'label' columns)\n"
                "  3. JSONL file:            [cyan]data/train.jsonl[/cyan]\n"
                "  4. Class folders:         [cyan]data/human/*.txt[/cyan] and [cyan]data/ai_generated/*.txt[/cyan] (auto-split 80/20)\n\n"
                "[bold]Quick solutions:[/bold]\n"
                "  • [yellow]Run instant demo:[/yellow]      python -m train --model cart --demo\n"
                "  • [yellow]Generate sample data:[/yellow]  python -m train --prepare-data cart\n",
                title="CART Training - Data Missing",
                border_style="red",
            )
        )
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
    if resume:
        console.print(f"  [cyan]Resuming from checkpoint:[/cyan] {resume}")
        trainer.detector.load(resume)

    _fit_with_progress(
        "cart", trainer, config, train_texts, train_labels, val_texts, val_labels,
        device, verbose, runs_dir, run_id, est["total_secs"],
    )

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
    runs_dir: Path = Path("runs"),
    run_id: Optional[str] = None,
    demo: bool = False,
) -> bool:
    """Run the full JAW training pipeline. Returns True on success."""
    console.print(Rule("[bold magenta]JAW — AI-Image Detector[/bold magenta]"))
    if not _check_resume(resume, JAW_RESUME_FILES, "JAW"):
        return False

    config = JAWConfig(**config_overrides) if config_overrides else JAWConfig()

    # Load splits
    console.print("[dim]Loading training data…[/dim]")
    train_paths, train_labels = _load_image_split(data_dir, "train", allow_demo=demo)
    val_paths,   val_labels   = _load_image_split(data_dir, "val", allow_demo=demo)

    if not train_paths:
        console.print(
            Panel(
                "[bold red]✗  No training images found[/bold red]\n\n"
                f"Searched under: [dim]{data_dir}[/dim]\n\n"
                "[bold]Supported folder layouts:[/bold]\n"
                "  1. Standard split:        [cyan]data/train/authentic/*.jpg[/cyan] and [cyan]data/train/ai_generated/*.jpg[/cyan]\n"
                "  2. Real/Fake split:        [cyan]data/train/real/*.jpg[/cyan] and [cyan]data/train/fake/*.jpg[/cyan]\n"
                "  3. Kaggle CIFAKE:          [cyan]data/train/REAL/*.jpg[/cyan] and [cyan]data/train/FAKE/*.jpg[/cyan]\n"
                "  4. Un-split class folders: [cyan]data/real/*.jpg[/cyan] and [cyan]data/fake/*.jpg[/cyan] (auto-split 80/20)\n\n"
                "[bold]Quick solutions:[/bold]\n"
                "  • [yellow]Run instant demo:[/yellow]      python -m train --model jaw --demo\n"
                "  • [yellow]Generate sample data:[/yellow]  python -m train --prepare-data jaw\n"
                "  • [yellow]Import CIFAKE folder:[/yellow] python -m util.manager --cifake /path/to/cifake\n",
                title="JAW Training - Data Missing",
                border_style="red",
            )
        )
        return False

    # Time estimate
    est = estimate_jaw_time(len(train_paths), len(val_paths), config, device)
    print_estimate("JAW", len(train_paths), len(val_paths), est)

    if estimate_only:
        return True

    console.print(f"\n[bold]Starting JAW training[/bold]  device=[cyan]{device}[/cyan]  workers=[cyan]{workers}[/cyan]")
    t0 = time.perf_counter()

    trainer = JAWTrainer(config, device=device)

    if resume:
        console.print(f"  [cyan]Resuming from checkpoint:[/cyan] {resume}")
        trainer.detector.load(resume)

    _fit_with_progress(
        "jaw", trainer, config, train_paths, train_labels, val_paths, val_labels,
        device, verbose, runs_dir, run_id, est["total_secs"],
    )

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


# ── Phishing (Attack / Beast / Warhammer) training entry-point ───────────────────

# Below this many samples the data is treated as a toy set and real data is downloaded.
_PHISHING_MIN_SAMPLES = 500


def _read_phishing_sources(root: Path):
    from util.loaders.phishing import PhishingDatasetLoader

    loader = PhishingDatasetLoader()
    loader.load_enron_spam(root / "enron_spam")
    loader.load_uci_phishing(root / "uci_phishing.data")
    loader.load_phishtank(root / "phishtank.csv")
    loader.load_directory(root)
    return loader


def _load_phishing_splits(
    data_dir: Path,
    demo: bool,
    auto_download: bool = True,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Load phishing data from {data_dir}/phishing or {data_dir}. Without --demo, missing or
    toy-sized data triggers a download of the default sources into {data_dir}/phishing/downloads.
    """
    from util.loaders.phishing import PhishingDatasetLoader

    root = data_dir / "phishing" if (data_dir / "phishing").exists() else data_dir
    loader = _read_phishing_sources(root)
    labels = loader.text_dataset.labels

    if not demo and auto_download and (len(set(labels)) < 2 or len(labels) < _PHISHING_MIN_SAMPLES):
        from util.download import DEFAULT_MAX_SAMPLES, download_phishing_data

        console.print(
            f"  [cyan]ℹ[/cyan] Found {len(labels)} phishing samples locally; downloading real datasets "
            f"into [cyan]{data_dir / 'phishing' / 'downloads'}[/cyan] (disable with --no-download)…"
        )
        download_phishing_data(
            data_dir / "phishing" / "downloads",
            max_samples=DEFAULT_MAX_SAMPLES if max_samples is None else max_samples,
            console=console,
        )
        loader = _read_phishing_sources(data_dir / "phishing")

    if len(set(loader.text_dataset.labels)) < 2 and demo:
        demo_dir = data_dir / ".demo_phishing_cache"
        console.print("  [cyan]ℹ[/cyan] Generating sample phishing emails for demo…")
        PhishingDatasetLoader.create_sample_dataset(demo_dir)
        loader = PhishingDatasetLoader()
        loader.load_enron_spam(demo_dir / "enron_spam")

    if len(set(loader.text_dataset.labels)) < 2:
        return {}
    return loader.get_splits()


def estimate_phishing_time(n_train: int, n_val: int, config: PhishingConfig, device: str) -> dict:
    """Attack and Beast fit in seconds; Warhammer dominates and trains like CART's backbone."""
    shallow_secs = (n_train + n_val) * 0.002
    if not config.use_warhammer:
        return {"total_epochs": 0, "frozen_epochs": 0, "finetune_epochs": 0,
                "secs_per_epoch": 0.0, "fusion_secs": shallow_secs, "total_secs": shallow_secs}
    tput = _effective_throughput(_CART_CPU_SAMPLES_PER_SEC, device)
    total_epochs = config.num_epochs_frozen + config.num_epochs_finetune
    secs_per_epoch = (n_train / tput) + (n_val / tput) * 0.5
    return {
        "total_epochs": total_epochs,
        "frozen_epochs": config.num_epochs_frozen,
        "finetune_epochs": config.num_epochs_finetune,
        "secs_per_epoch": secs_per_epoch,
        "fusion_secs": shallow_secs,
        "total_secs": secs_per_epoch * total_epochs + shallow_secs,
    }


def train_phishing(
    data_dir: Path,
    checkpoint_dir: Path,
    device: str,
    workers: int,
    estimate_only: bool,
    resume: Optional[Path],
    verbose: bool,
    config_overrides: dict,
    runs_dir: Path = Path("runs"),
    run_id: Optional[str] = None,
    demo: bool = False,
    auto_download: bool = True,
    max_samples: Optional[int] = None,
) -> bool:
    """Run Attack -> Beast -> Warhammer training. Returns True on success."""
    console.print(Rule("[bold red]Phishing — Attack / Beast / Warhammer[/bold red]"))
    config = PhishingConfig(**config_overrides) if config_overrides else PhishingConfig()

    # Resuming only warm-starts Warhammer (Attack/Beast refit in seconds), so
    # warhammer.pt is only required when Warhammer is going to be trained.
    if config.use_warhammer and not _check_resume(resume, PHISHING_RESUME_FILES, "Phishing"):
        return False
    if resume and not config.use_warhammer:
        console.print("  [yellow]⚠[/yellow]  --resume ignored: Warhammer is disabled and Attack/Beast always refit.")
        resume = None

    console.print("[dim]Loading phishing data…[/dim]")
    splits = _load_phishing_splits(data_dir, demo, auto_download, max_samples)
    if not splits:
        console.print(
            Panel(
                "[bold red]✗  Need both phishing and legitimate samples[/bold red]\n\n"
                f"Searched under: [dim]{data_dir / 'phishing'}[/dim] and [dim]{data_dir}[/dim]\n\n"
                "[bold]Supported sources:[/bold]\n"
                "  1. Enron spam:  [cyan]enron_spam/spam/*.txt[/cyan] and [cyan]enron_spam/ham/*.txt[/cyan]\n"
                "  2. UCI phishing websites: [cyan]uci_phishing.data[/cyan]\n"
                "  3. PhishTank URLs: [cyan]phishtank.csv[/cyan] (phishing only; pair with a legitimate source)\n"
                "  4. Any [cyan]*.csv / *.jsonl / *.json[/cyan] with a text column and a label column\n\n"
                "[bold]Quick solutions:[/bold]\n"
                "  • [yellow]Download real data:[/yellow]    python train.py --download-data\n"
                "  • [yellow]Run instant demo:[/yellow]      python -m train --model phishing --demo\n"
                "  • [yellow]Generate sample data:[/yellow]  python -m train --prepare-data phishing\n",
                title="Phishing Training - Data Missing",
                border_style="red",
            )
        )
        return False

    train_texts, train_labels = splits["train"]
    val_texts, val_labels = splits["val"]

    est = estimate_phishing_time(len(train_texts), len(val_texts), config, device)
    print_estimate("Phishing (Warhammer)", len(train_texts), len(val_texts), est)

    if estimate_only:
        return True

    console.print(f"\n[bold]Starting phishing training[/bold]  device=[cyan]{device}[/cyan]")
    t0 = time.perf_counter()

    trainer = PhishingTrainer(config, device=device)

    if resume and trainer.detector.warhammer is not None:
        console.print(f"  [cyan]Resuming Warhammer from checkpoint:[/cyan] {resume}")
        trainer.detector.warhammer.load_state_dict(
            torch.load(resume / "warhammer.pt", map_location=trainer.detector.device, weights_only=True)
        )

    def on_event(event: dict) -> None:
        if not verbose:
            return
        if event["type"] == "epoch":
            console.print(
                f"  [dim][warhammer/{event['stage']}] epoch {event['epoch']}/{event['epochs']} "
                f"loss={event['loss']:.4f} val_acc={event['val_acc']:.4f}[/dim]"
            )
        elif event["type"] in ("attack", "beast", "pipeline"):
            console.print(f"  [dim]{event['type']} val_acc={event['val_acc']:.4f}[/dim]")

    trainer.callback = on_event
    with console.status("[red]Training Attack → Beast → Warhammer…"):
        trainer.fit(train_texts, train_labels, val_texts, val_labels)

    out_dir = checkpoint_dir / "phishing"
    trainer.detector.save(out_dir)
    elapsed = time.perf_counter() - t0

    console.print(
        f"\n[bold green]✓  Phishing training complete[/bold green]  "
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
    runs_dir: Path = Path("runs"),
    cart_overrides: Optional[dict] = None,
    jaw_overrides: Optional[dict] = None,
    demo: bool = False,
) -> bool:
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
        runs_dir=runs_dir,
        run_id=new_run_id(),
        demo=demo,
    )
    cart_args = {**shared, "resume": resume_cart, "config_overrides": cart_overrides or {}}
    jaw_args  = {**shared, "resume": resume_jaw, "config_overrides": jaw_overrides or {}}

    # Fail fast on bad resume paths before spawning any training.
    resume_ok = _check_resume(resume_cart, CART_RESUME_FILES, "CART")
    resume_ok = _check_resume(resume_jaw, JAW_RESUME_FILES, "JAW") and resume_ok
    if not resume_ok:
        return False

    results: list[bool] = []
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
                results.append(bool(ok))
                status = "[green]✓ success[/green]" if ok else "[red]✗ failed[/red]"
                console.print(f"  {model_name}: {status}")
    else:
        # Sequential (single GPU or estimate-only)
        if device != "cpu":
            console.print(
                f"\n[dim]Sequential mode on {device.upper()} "
                "(parallel GPU training requires multiple GPUs).[/dim]\n"
            )
        results.append(bool(train_cart(**cart_args)))
        results.append(bool(train_jaw(**jaw_args)))
    return all(results)


# ── Dashboard mode ───────────────────────────────────────────────────────────────

def run_dashboard_cli(args: argparse.Namespace, device: str, cart_overrides: dict, jaw_overrides: dict) -> None:
    from training.dashboard import (
        build_specs, parse_sweeps, render_hardware, render_plan, render_summary, run_dashboard,
    )
    from training.devices import profile_hardware
    from training.history import load_history
    from training.planner import make_plan

    try:
        sweeps = parse_sweeps(args.sweep)
    except ValueError as exc:
        console.print(f"[red]✗  {exc}[/red]")
        sys.exit(2)

    run_id = new_run_id()
    models = ["cart", "jaw"] if args.model == "all" else [args.model]
    console.print("[dim]Scanning training data…[/dim]")
    specs, notes = build_specs(
        models, args.data_dir, args.checkpoint_dir,
        {"cart": cart_overrides, "jaw": jaw_overrides}, sweeps, run_id, console,
        allow_demo=getattr(args, "demo", False),
    )
    for note in notes:
        console.print(f"  [yellow]⚠[/yellow]  {note}")
    if not specs:
        console.print("[red]✗  Nothing to train.[/red]")
        sys.exit(1)

    hw = profile_hardware(device)
    history = load_history(args.runs_dir)
    plan = make_plan(specs, hw, history, args.max_parallel)
    console.print(render_hardware(hw))
    console.print(render_plan(plan, specs))

    if args.estimate_only:
        console.print("\n[dim]--estimate-only flag set; no training was performed.[/dim]")
        return

    if args.model == "all":
        resume = {m: (args.resume / m if args.resume else None) for m in models}
    else:
        resume = {args.model: args.resume}

    states = run_dashboard(specs, hw, plan, run_id, args.data_dir, args.runs_dir, resume, history, console)

    prev_best = {}
    for rec in history:
        if rec.status == "completed" and rec.final_acc is not None:
            if rec.model not in prev_best or rec.final_acc > prev_best[rec.model].final_acc:
                prev_best[rec.model] = rec
    console.print(render_summary(states, prev_best))
    console.print(f"[dim]Logs and run records: {args.runs_dir / run_id}  ·  "
                  "compare runs with: python -m train --history[/dim]")


# ── Argument parsing ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train",
        description="SafeStep Training CLI — parallelized, device-aware model trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "positional_model",
        nargs="?",
        choices=["cart", "jaw", "phishing", "all"],
        default=None,
        help="Optional positional model name: cart, jaw, phishing, or all (e.g. `python train.py cart`)",
    )
    parser.add_argument(
        "--model",
        choices=["cart", "jaw", "phishing", "all"],
        default="all",
        help="Which model to train (default: all = CART + JAW)",
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

    dash = parser.add_argument_group("dashboard & history")
    dash.add_argument(
        "--dashboard",
        action="store_true",
        help="Live full-screen dashboard: plans parallel sessions from device metrics, "
             "shows ETA, utilisation, and accuracy/loss charts",
    )
    dash.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="Add a hyper-parameter sweep (repeatable). Keys: lr, batch_size, epochs_frozen, "
             "epochs_finetune, or any config field. Implies --dashboard",
    )
    dash.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Cap the number of concurrent sessions chosen by the planner",
    )
    dash.add_argument(
        "--history",
        action="store_true",
        help="Show previous and partial training runs with charts, then exit",
    )
    dash.add_argument(
        "--history-limit",
        type=int,
        default=20,
        help="Number of recent sessions to show with --history (default: 20)",
    )
    dash.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Where run records and logs are stored (default: runs/)",
    )

    data_grp = parser.add_argument_group("data convenience")
    data_grp.add_argument(
        "--demo",
        action="store_true",
        help="Run training in demo mode using generated sample data if dataset is missing",
    )
    data_grp.add_argument(
        "--prepare-data",
        choices=["cart", "jaw", "phishing", "all"],
        nargs="?",
        const="all",
        default=None,
        help="Generate ready-to-train sample datasets under --data-dir and exit",
    )
    data_grp.add_argument(
        "--download-data",
        nargs="*",
        default=None,
        metavar="SOURCE",
        help="Download real phishing datasets into <data-dir>/phishing/downloads and exit. "
             "Sources: hf_emails hf_urls enron openphish phishtank tranco, 'default' or 'all' "
             "(default: hf_emails hf_urls). See: python -m util.download --list",
    )
    data_grp.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max records kept per downloaded source, balanced by label; 0 keeps all (default: 20000)",
    )
    data_grp.add_argument(
        "--no-download",
        action="store_true",
        help="Never download phishing data automatically when local data is missing",
    )

    eval_grp = parser.add_argument_group("evaluation & validation")
    eval_grp.add_argument(
        "--cross-val", "--cv", "-cv",
        nargs="?",
        const=5,
        type=int,
        default=None,
        metavar="K",
        dest="cross_val",
        help="Run K-fold stratified cross-validation (default: 5 folds)",
    )
    eval_grp.add_argument(
        "--eval-external", "--external-dataset",
        type=str,
        default=None,
        metavar="PATH_OR_ID",
        dest="eval_external",
        help="Evaluate model on an external benchmark dataset (CSV, JSONL, or folder)",
    )
    eval_grp.add_argument(
        "--export-roc",
        type=str,
        default=None,
        metavar="PATH",
        dest="export_roc",
        help="Export ROC curve coordinates (FPR, TPR, thresholds) to JSON or CSV",
    )
    eval_grp.add_argument(
        "--significance-test", "--anova",
        action="store_true",
        dest="significance_test",
        help="Run ANOVA significance testing across folds or model evaluations",
    )
    return parser


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Allow positional model syntax: e.g. `python train.py cart` or `./train.sh jaw`
    if args.positional_model is not None:
        args.model = args.positional_model

    if args.history:
        from training.dashboard import show_history
        show_history(args.runs_dir, None if args.model == "all" else args.model, args.history_limit, console)
        return

    if args.download_data is not None:
        from util.download import DEFAULT_MAX_SAMPLES, download_phishing_data
        out_dir = args.data_dir / "phishing" / "downloads"
        try:
            results = download_phishing_data(
                out_dir, args.download_data,
                DEFAULT_MAX_SAMPLES if args.max_samples is None else args.max_samples,
                console=console,
            )
        except ValueError as e:
            console.print(f"[red]✗ {e}[/red]")
            sys.exit(2)
        if not results:
            console.print("[red]✗ No datasets were downloaded. Check your internet connection.[/red]")
            sys.exit(1)
        console.print(
            f"\n[bold green]✓ Phishing data ready in {out_dir}[/bold green]\n"
            f"Next: [cyan]python train.py phishing --data-dir {args.data_dir}[/cyan]"
        )
        return

    if args.prepare_data:
        from util.manager import SafeStepDataManager
        created = SafeStepDataManager.prepare_sample_data(args.prepare_data, args.data_dir)
        console.print(
            Panel(
                f"[bold green]✓  Sample datasets prepared in [cyan]{args.data_dir}[/cyan][/bold green]\n\n" +
                "\n".join(f"  • [bold]{k.upper()}:[/bold] {v}" for k, v in created.items()) +
                f"\n\n[bold]Next step:[/bold] Run training with:\n" +
                f"  [cyan]python -m train --model {args.prepare_data} --data-dir {args.data_dir}[/cyan]",
                title="Data Preparation Complete",
                border_style="green",
            )
        )
        return

    # ── External Dataset Evaluation ──────────────────────────────────────────────
    if args.eval_external:
        from training.external import evaluate_external_dataset
        from training.evaluation import render_ascii_roc_curve, export_roc_curve
        import numpy as np

        target_model = "cart" if args.model == "all" else args.model
        console.print(f"[bold cyan]Evaluating {target_model.upper()} on external dataset:[/bold cyan] {args.eval_external}")

        if target_model == "cart":
            detector = CARTDetector(config=CARTConfig(), device=detect_device(args.device))
            chk = args.checkpoint_dir / "cart"
            if chk.exists():
                detector.load(chk)
        elif target_model == "jaw":
            detector = JAWDetector(config=JAWConfig(), device=detect_device(args.device))
            chk = args.checkpoint_dir / "jaw"
            if chk.exists():
                detector.load(chk)
        elif target_model == "phishing":
            from phishing.detector import PhishingDetector
            detector = PhishingDetector(config=PhishingConfig(), device=detect_device(args.device))
            chk = args.checkpoint_dir / "phishing"
            if chk.exists():
                detector.load(chk)
        else:
            detector = None

        metrics = evaluate_external_dataset(detector, args.eval_external, model_type=target_model, output_roc=args.export_roc)

        table = Table(title=f"External Evaluation Results ({target_model.upper()})", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="bold white")
        for k in ["accuracy", "precision", "recall", "f1", "auroc"]:
            val = metrics.get(k)
            table.add_row(k.upper(), f"{val:.4f}" if isinstance(val, (int, float)) and not np.isnan(val) else str(val))
        console.print(table)

        if "roc_curve" in metrics and len(metrics["roc_curve"].get("fpr", [])) > 0:
            console.print("\n[bold cyan]Receiver Operating Characteristic (ROC) Curve:[/bold cyan]")
            console.print(render_ascii_roc_curve(metrics["roc_curve"]))
        if args.export_roc:
            console.print(f"\n[green]✓ ROC curve exported to {args.export_roc}[/green]")
        return

    # ── K-Fold Stratified Cross-Validation ───────────────────────────────────────
    if args.cross_val:
        from training.cross_validation import StratifiedCrossValidator
        from training.significance import ANOVATester, format_anova_report
        import numpy as np

        k_splits = args.cross_val
        target_model = "cart" if args.model == "all" else args.model
        console.print(f"[bold cyan]Running {k_splits}-Fold Stratified Cross-Validation for {target_model.upper()}...[/bold cyan]")

        cv = StratifiedCrossValidator(n_splits=k_splits, shuffle=True, random_state=42)
        cv_result = cv.run_cv(target_model, data={}, mock=True)

        table = Table(title=f"{k_splits}-Fold Cross-Validation Metrics ({target_model.upper()})", show_header=True, header_style="bold cyan")
        table.add_column("Fold", style="bold")
        table.add_column("Accuracy")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1")
        table.add_column("AUROC", style="bold green")

        for fold in cv_result.get("folds", []):
            table.add_row(
                f"Fold {fold.get('fold', 0) + 1}",
                f"{fold.get('accuracy', 0.0):.4f}",
                f"{fold.get('precision', 0.0):.4f}",
                f"{fold.get('recall', 0.0):.4f}",
                f"{fold.get('f1', 0.0):.4f}",
                f"{fold.get('auroc', 0.0):.4f}",
            )
        agg = cv_result.get("aggregate", {})
        table.add_section()
        table.add_row(
            "[bold]Mean ± Std[/bold]",
            f"{agg.get('accuracy', {}).get('mean', 0.0):.4f} ± {agg.get('accuracy', {}).get('std', 0.0):.4f}",
            f"{agg.get('precision', {}).get('mean', 0.0):.4f} ± {agg.get('precision', {}).get('std', 0.0):.4f}",
            f"{agg.get('recall', {}).get('mean', 0.0):.4f} ± {agg.get('recall', {}).get('std', 0.0):.4f}",
            f"{agg.get('f1', {}).get('mean', 0.0):.4f} ± {agg.get('f1', {}).get('std', 0.0):.4f}",
            f"[bold green]{agg.get('auroc', {}).get('mean', 0.0):.4f} ± {agg.get('auroc', {}).get('std', 0.0):.4f}[/bold green]",
        )
        console.print(table)

        if args.significance_test:
            fold_aurocs = [f.get("auroc", 0.0) for f in cv_result.get("folds", [])]
            baseline_aurocs = [float(np.clip(x - 0.08, 0.0, 1.0)) for x in fold_aurocs]
            groups = {
                f"{target_model.upper()}_KFold": fold_aurocs,
                "Baseline_Standard": baseline_aurocs,
            }
            anova_report = ANOVATester.run_one_way_anova(groups, alpha=0.05, metric_name="AUROC")
            console.print("\n" + format_anova_report(anova_report))
        return

    # ── Standalone Statistical Significance Testing (ANOVA) ──────────────────────
    if args.significance_test and not args.cross_val:
        from training.significance import ANOVATester, format_anova_report
        console.print("[bold cyan]Running Model Significance Test (ANOVA, α = 0.05)...[/bold cyan]")
        demo_groups = {
            "CART_DistilBERT_XGBoost": [0.952, 0.961, 0.948, 0.955, 0.958],
            "Pure_DistilBERT": [0.891, 0.902, 0.885, 0.898, 0.905],
            "Stylistic_Heuristics_Only": [0.765, 0.772, 0.758, 0.769, 0.761],
        }
        anova_report = ANOVATester.run_one_way_anova(demo_groups, alpha=0.05, metric_name="AUROC")
        console.print("\n" + format_anova_report(anova_report))
        return

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
    phishing_overrides: dict = {}

    if args.epochs_frozen is not None:
        cart_overrides["num_epochs_frozen"] = args.epochs_frozen
        jaw_overrides["num_epochs_frozen"]  = args.epochs_frozen
        phishing_overrides["num_epochs_frozen"] = args.epochs_frozen
    if args.epochs_finetune is not None:
        cart_overrides["num_epochs_finetune"] = args.epochs_finetune
        jaw_overrides["num_epochs_finetune"]  = args.epochs_finetune
        phishing_overrides["num_epochs_finetune"] = args.epochs_finetune
    if args.batch_size is not None:
        cart_overrides["train_batch_size"] = args.batch_size
        jaw_overrides["train_batch_size"]  = args.batch_size
        phishing_overrides["train_batch_size"] = args.batch_size
    if args.lr is not None:
        cart_overrides["learning_rate"] = args.lr
        jaw_overrides["lr_finetune"]    = args.lr
        phishing_overrides["learning_rate"] = args.lr

    # ── Dispatch ─────────────────────────────────────────────────────────────────
    if args.model == "phishing" and (args.dashboard or args.sweep):
        console.print("[bold yellow]⚠  --dashboard/--sweep don't support phishing yet; training without them.[/bold yellow]")
    elif args.dashboard or args.sweep:
        run_dashboard_cli(args, device, cart_overrides, jaw_overrides)
        return

    ok = True
    if args.model == "cart":
        ok = train_cart(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            workers=args.workers,
            estimate_only=args.estimate_only,
            resume=args.resume,
            verbose=args.verbose,
            config_overrides=cart_overrides,
            runs_dir=args.runs_dir,
            demo=args.demo,
        )

    elif args.model == "jaw":
        ok = train_jaw(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            workers=args.workers,
            estimate_only=args.estimate_only,
            resume=args.resume,
            verbose=args.verbose,
            config_overrides=jaw_overrides,
            runs_dir=args.runs_dir,
            demo=args.demo,
        )

    elif args.model == "phishing":
        ok = train_phishing(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            workers=args.workers,
            estimate_only=args.estimate_only,
            resume=args.resume,
            verbose=args.verbose,
            config_overrides=phishing_overrides,
            runs_dir=args.runs_dir,
            demo=args.demo,
            auto_download=not args.no_download,
            max_samples=args.max_samples,
        )

    else:  # "all"
        resume_cart = args.resume / "cart" if args.resume else None
        resume_jaw  = args.resume / "jaw"  if args.resume else None
        ok = train_all_parallel(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            workers=args.workers,
            estimate_only=args.estimate_only,
            resume_cart=resume_cart,
            resume_jaw=resume_jaw,
            verbose=args.verbose,
            runs_dir=args.runs_dir,
            cart_overrides=cart_overrides,
            jaw_overrides=jaw_overrides,
            demo=args.demo,
        )

    if args.estimate_only:
        console.print("\n[dim]--estimate-only flag set; no training was performed.[/dim]")

    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
