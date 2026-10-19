"""
Automatic dataset downloads for the phishing model.

Each source is fetched over HTTPS, cached under ``<out>/raw/`` and normalized
into ``<out>/<source>.jsonl`` with one ``{"text": ..., "label": ...}`` record
per line (label is ``phishing`` or ``legitimate``). PhishingDatasetLoader
picks those files up automatically, so after downloading you just run
``python train.py phishing``.

    python -m util.download                       # default sources into data/phishing/downloads
    python -m util.download --sources all          # every source
    python -m util.download --list                 # show what is available
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from util.logging_config import get_logger

logger = get_logger(__name__)

USER_AGENT = "SafeStep-Models/1.0 (+dataset downloader)"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
DEFAULT_MAX_SAMPLES = 20000
DEFAULT_SOURCES = ["hf_emails", "hf_urls"]

Record = Tuple[str, str]  # (text, label)


def _hf_url(repo: str, filename: str) -> str:
    return f"{HF_ENDPOINT}/datasets/{repo}/resolve/main/{filename}"


def _label01(value) -> Optional[str]:
    if str(value).strip() in ("1", "1.0", "True", "true"):
        return "phishing"
    if str(value).strip() in ("0", "0.0", "False", "false"):
        return "legitimate"
    return None


# ── Parsers: raw file -> (text, label) records ──────────────────────────────────

def _parse_json_list(path: Path) -> Iterator[Record]:
    with open(path, "r", encoding="utf-8") as f:
        for row in json.load(f):
            label = _label01(row.get("label"))
            text = str(row.get("text") or "").strip()
            if label and text:
                yield text, label


def _parse_jsonl(path: Path) -> Iterator[Record]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            label = _label01(row.get("label"))
            text = str(row.get("text") or "").strip()
            if label and text:
                yield text, label


def _parse_url_lines(path: Path) -> Iterator[Record]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            url = line.strip()
            if url and not url.startswith("#"):
                yield url, "phishing"


def _parse_phishtank(path: Path) -> Iterator[Record]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            if url:
                yield url, "phishing"


def _parse_tranco(path: Path) -> Iterator[Record]:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                parts = line.strip().split(",")
                if len(parts) == 2 and parts[1]:
                    yield f"https://{parts[1]}/", "legitimate"


SOURCES: Dict[str, dict] = {
    "hf_emails": {
        "description": "Phishing + legitimate emails (Hugging Face ealvaradob/phishing-dataset, ~18k, Apache-2.0)",
        "url": _hf_url("ealvaradob/phishing-dataset", "texts.json"),
        "raw": "hf_emails_texts.json",
        "parse": _parse_json_list,
        "labels": "both",
    },
    "hf_urls": {
        "description": "Phishing + legitimate URLs (Hugging Face ealvaradob/phishing-dataset, ~800k, Apache-2.0)",
        "url": _hf_url("ealvaradob/phishing-dataset", "urls.json"),
        "raw": "hf_urls.json",
        "parse": _parse_json_list,
        "labels": "both",
    },
    "enron": {
        "description": "Enron spam/ham emails (Hugging Face SetFit/enron_spam, ~31k)",
        "url": _hf_url("SetFit/enron_spam", "train.jsonl"),
        "raw": "enron_spam_train.jsonl",
        "parse": _parse_jsonl,
        "labels": "both",
    },
    "openphish": {
        "description": "Live phishing URLs (openphish.com free feed, phishing only)",
        "url": "https://openphish.com/feed.txt",
        "raw": "openphish_feed.txt",
        "parse": _parse_url_lines,
        "labels": "phishing",
    },
    "phishtank": {
        "description": "Verified phishing URLs (phishtank.com, phishing only, rate limited)",
        "url": "https://data.phishtank.com/data/online-valid.csv",
        "raw": "phishtank_online_valid.csv",
        "parse": _parse_phishtank,
        "labels": "phishing",
    },
    "tranco": {
        "description": "Top legitimate domains as URLs (tranco-list.eu, legitimate only)",
        "url": "https://tranco-list.eu/top-1m.csv.zip",
        "raw": "tranco_top1m.csv.zip",
        "parse": _parse_tranco,
        "labels": "legitimate",
    },
}


# ── Download + normalize ────────────────────────────────────────────────────────

def _fetch(url: str, dest: Path, progress: Optional[Callable[[int, Optional[int]], None]] = None) -> None:
    """Stream url to dest atomically (a .part file is renamed on success)."""
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith(HF_ENDPOINT):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as out:
        total = resp.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    tmp.replace(dest)


def _sample(records: Iterable[Record], max_samples: int, seed: int = 42) -> List[Record]:
    """Keep at most max_samples records, balanced across labels when both are present."""
    by_label: Dict[str, List[str]] = {}
    for text, label in records:
        by_label.setdefault(label, []).append(text)
    rng = random.Random(seed)
    out: List[Record] = []
    per_label = max_samples // len(by_label) if max_samples and by_label else 0
    for label, texts in by_label.items():
        if per_label and len(texts) > per_label:
            texts = rng.sample(texts, per_label)
        out.extend((t, label) for t in texts)
    rng.shuffle(out)
    return out


def download_source(
    name: str,
    out_dir: Path,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    force: bool = False,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Tuple[Path, Dict[str, int]]:
    """Download one source and write <out_dir>/<name>.jsonl. Returns (path, label counts)."""
    spec = SOURCES[name]
    out_dir = Path(out_dir)
    raw_path = out_dir / "raw" / spec["raw"]
    out_path = out_dir / f"{name}.jsonl"

    if out_path.exists() and not force:
        counts: Dict[str, int] = {}
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                label = json.loads(line)["label"]
                counts[label] = counts.get(label, 0) + 1
        return out_path, counts

    if force or not raw_path.exists():
        logger.info(f"Downloading {name} from {spec['url']}")
        _fetch(spec["url"], raw_path, progress)

    records = _sample(spec["parse"](raw_path), max_samples)
    counts = {}
    tmp = out_path.with_name(out_path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for text, label in records:
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
            counts[label] = counts.get(label, 0) + 1
    tmp.replace(out_path)
    return out_path, counts


def resolve_sources(requested: Optional[List[str]]) -> List[str]:
    if not requested:
        return list(DEFAULT_SOURCES)
    names: List[str] = []
    for item in requested:
        for part in item.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part == "all":
                names.extend(SOURCES)
            elif part == "default":
                names.extend(DEFAULT_SOURCES)
            elif part in SOURCES:
                names.append(part)
            else:
                raise ValueError(f"Unknown source '{part}'. Choose from: {', '.join(SOURCES)}, default, all")
    return list(dict.fromkeys(names))


def download_phishing_data(
    out_dir: Path,
    sources: Optional[List[str]] = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    force: bool = False,
    console=None,
) -> Dict[str, Dict[str, int]]:
    """Download the requested sources, printing progress. Failed sources are reported and skipped."""
    from rich.console import Console
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

    console = console or Console()
    results: Dict[str, Dict[str, int]] = {}
    for name in resolve_sources(sources):
        console.print(f"[bold]{name}[/bold] [dim]{SOURCES[name]['description']}[/dim]")
        try:
            with Progress(
                TextColumn("  "), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                console=console, transient=True,
            ) as bar:
                task = bar.add_task(name, total=None)
                path, counts = download_source(
                    name, out_dir, max_samples, force,
                    progress=lambda done, total: bar.update(task, completed=done, total=total),
                )
            summary = ", ".join(f"{v:,} {k}" for k, v in sorted(counts.items()))
            console.print(f"  [green]✓[/green] {path}  ({summary})")
            results[name] = counts
        except Exception as e:  # network errors, rate limits, format changes
            console.print(f"  [red]✗ {name} failed:[/red] {e}")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download phishing training datasets")
    parser.add_argument("--sources", nargs="*", default=None,
                        help=f"Sources to fetch (default: {','.join(DEFAULT_SOURCES)}). Use 'all' for every source")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Data root; files go to <data-dir>/phishing/downloads (default: data)")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help=f"Max records kept per source, balanced by label; 0 keeps everything (default: {DEFAULT_MAX_SAMPLES})")
    parser.add_argument("--force", action="store_true", help="Re-download and rebuild even if files exist")
    parser.add_argument("--list", action="store_true", help="List available sources and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in SOURCES.items():
            mark = "*" if name in DEFAULT_SOURCES else " "
            print(f" {mark} {name:<10} {spec['description']}")
        print("\n * = downloaded by default")
        return 0

    try:
        results = download_phishing_data(args.data_dir / "phishing" / "downloads",
                                         args.sources, args.max_samples, args.force)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
