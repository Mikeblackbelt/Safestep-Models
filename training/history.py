"""
Persistent run history.

Every training session writes `<runs_dir>/<run_id>/<session_id>.json`, owned
by the process doing the training (single writer). The record is rewritten
after every epoch and on a heartbeat, so a crash or Ctrl+C leaves a usable
partial record behind. A record still marked "running" whose process is gone
is reported as "interrupted".
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

# Relative cost of one validation sample vs one training sample (forward only).
EVAL_WEIGHT = 0.5

_HEARTBEAT_SECS = 15.0
_STALE_SECS = 15 * 60


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class EpochRecord:
    stage: str
    epoch: int
    loss: float
    val_acc: float
    elapsed: float  # seconds since the session started training


@dataclass
class RunRecord:
    run_id: str
    session_id: str
    label: str
    model: str
    status: str = "running"  # running | completed | failed | interrupted
    pid: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: Optional[float] = None
    device: str = "cpu"
    threads: int = 1
    concurrent: int = 1  # sessions sharing the device when this one started
    overrides: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    total_epochs: int = 0
    total_units: float = 0.0  # planned work, in training-sample equivalents
    units_done: float = 0.0
    units_per_sec: Optional[float] = None
    estimated_secs: Optional[float] = None
    epochs: List[EpochRecord] = field(default_factory=list)
    fusion_val_acc: Optional[float] = None
    early_stopped: bool = False
    checkpoint: Optional[str] = None
    error: Optional[str] = None

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def best_val_acc(self) -> Optional[float]:
        accs = [e.val_acc for e in self.epochs if e.val_acc == e.val_acc]  # drop NaN
        return max(accs) if accs else None

    @property
    def final_acc(self) -> Optional[float]:
        """Headline accuracy: fusion accuracy for CART when available."""
        return self.fusion_val_acc if self.fusion_val_acc is not None else self.best_val_acc

    @property
    def duration(self) -> float:
        end = self.finished_at or self.updated_at
        return max(0.0, end - self.started_at)

    @property
    def device_kind(self) -> str:
        return self.device.split(":")[0]

    @property
    def is_partial(self) -> bool:
        return self.status in ("interrupted", "failed", "running")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        d = dict(d)
        d["epochs"] = [EpochRecord(**e) for e in d.get("epochs", [])]
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


def _pid_alive(pid: int) -> bool:
    try:
        return pid > 0 and psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def load_history(runs_dir: Path, model: Optional[str] = None) -> List[RunRecord]:
    """All run records, oldest first, with dead "running" sessions marked interrupted."""
    records: List[RunRecord] = []
    if not runs_dir.exists():
        return records
    for path in sorted(runs_dir.glob("*/*.json")):
        try:
            rec = RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        if model and rec.model != model:
            continue
        if rec.status == "running" and (
            not _pid_alive(rec.pid) or time.time() - rec.updated_at > _STALE_SECS
        ):
            rec.status = "interrupted"
        records.append(rec)
    records.sort(key=lambda r: r.started_at)
    return records


class RunRecorder:
    """Turns trainer progress events into a persisted RunRecord."""

    def __init__(self, runs_dir: Path, record: RunRecord):
        self.record = record
        self.path = runs_dir / record.run_id / f"{record.session_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._first_batch_t: Optional[float] = None
        self._last_save = 0.0
        now = time.time()
        record.pid = record.pid or os.getpid()
        record.started_at = record.started_at or now
        record.updated_at = now
        self.save()

    def handle(self, event: Dict[str, Any]) -> None:
        rec = self.record
        etype = event.get("type")
        now = time.time()
        if etype == "batch":
            if self._first_batch_t is None:
                self._first_batch_t = now
            weight = 1.0 if event.get("phase") == "train" else EVAL_WEIGHT
            rec.units_done += event.get("samples", 0) * weight
            elapsed = now - self._first_batch_t
            if elapsed > 1.0:
                rec.units_per_sec = rec.units_done / elapsed
        elif etype == "epoch":
            rec.epochs.append(EpochRecord(
                stage=event["stage"],
                epoch=event["epoch"],
                loss=float(event["loss"]),
                val_acc=float(event["val_acc"]),
                elapsed=now - rec.started_at,
            ))
        elif etype == "early_stop":
            rec.early_stopped = True
        elif etype == "stage" and event.get("stage") == "fusion":
            rec.units_done += EVAL_WEIGHT * rec.n_val  # fusion inference pass emits no batches
        elif etype == "fusion":
            rec.fusion_val_acc = float(event["val_acc"])

        rec.updated_at = now
        if etype in ("epoch", "early_stop", "fusion", "stage") or now - self._last_save > _HEARTBEAT_SECS:
            self.save()

    def finish(self, status: str, error: Optional[str] = None, checkpoint: Optional[str] = None) -> None:
        self.record.status = status
        self.record.error = error
        self.record.checkpoint = checkpoint or self.record.checkpoint
        self.record.finished_at = self.record.updated_at = time.time()
        self.save()

    def save(self) -> None:
        self._last_save = time.time()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self.record), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)
