"""
A single training session, run in its own process.

The worker redirects stdout/stderr to `<runs_dir>/<run_id>/<session_id>.log`
(so library output never garbles the dashboard), pins its torch thread count
to the slot the planner assigned, streams trainer events to the parent over a
multiprocessing queue, and persists its own RunRecord.
"""

import os
import sys
import traceback
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional

from training.data import load_splits
from training.history import RunRecord, RunRecorder

# CLI-friendly aliases → per-model config field names
_ALIASES = {
    "lr": {"cart": "learning_rate", "jaw": "lr_finetune"},
    "batch_size": {"cart": "train_batch_size", "jaw": "train_batch_size"},
    "epochs_frozen": {"cart": "num_epochs_frozen", "jaw": "num_epochs_frozen"},
    "epochs_finetune": {"cart": "num_epochs_finetune", "jaw": "num_epochs_finetune"},
}


def config_class(model: str):
    if model == "cart":
        from cart.config import CARTConfig
        return CARTConfig
    from jaw.config import JAWConfig
    return JAWConfig


def resolve_override(model: str, key: str) -> Optional[str]:
    """Map an alias or raw field name to the model's config field (None if unsupported)."""
    key = key.replace("-", "_")
    field_name = _ALIASES.get(key, {}).get(model, key)
    names = {f.name for f in fields(config_class(model))}
    return field_name if field_name in names else None


def coerce_value(model: str, field_name: str, raw: str) -> Any:
    """Cast a CLI string to the type of the config field's default."""
    default = next(f.default for f in fields(config_class(model)) if f.name == field_name)
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


def make_config(model: str, overrides: Dict[str, Any]):
    return config_class(model)(**overrides)


def checkpoint_saved_files(model: str) -> str:
    return "neural_model.pt" if model == "cart" else "jaw_model.pt"


def run_session(job: Dict[str, Any], queue) -> None:
    """Process entry point. `job` is a plain dict so it pickles under spawn."""
    sid = job["session_id"]
    runs_dir = Path(job["runs_dir"])
    log_path = runs_dir / job["run_id"] / f"{sid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", buffering=1, encoding="utf-8")
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    except OSError:
        pass
    sys.stdout = sys.stderr = log

    def emit(event: Dict[str, Any]) -> None:
        try:
            queue.put({"session_id": sid, **event})
        except Exception:  # noqa: BLE001 - parent gone; keep training and recording
            pass

    emit({"type": "status", "status": "loading", "pid": os.getpid()})
    recorder: Optional[RunRecorder] = None
    try:
        import torch

        threads = int(job["threads"])
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(max(1, threads // 2))
        except RuntimeError:
            pass

        model = job["model"]
        config = make_config(model, job["overrides"])
        train_x, train_y, val_x, val_y = load_splits(model, Path(job["data_dir"]))
        if not train_x:
            raise RuntimeError(f"no training data for {model.upper()} under {job['data_dir']}")

        recorder = RunRecorder(runs_dir, RunRecord(
            run_id=job["run_id"],
            session_id=sid,
            label=job["label"],
            model=model,
            device=job["device"],
            threads=threads,
            concurrent=int(job["concurrent"]),
            overrides=job["overrides"],
            config=asdict(config),
            n_train=len(train_x),
            n_val=len(val_x),
            total_epochs=config.num_epochs_frozen + config.num_epochs_finetune,
            total_units=float(job["total_units"]),
            estimated_secs=job.get("estimated_secs"),
        ))

        def on_event(event: Dict[str, Any]) -> None:
            recorder.handle(event)
            emit(event)

        if model == "cart":
            from cart.train import CARTTrainer as Trainer
        else:
            from jaw.train import JAWTrainer as Trainer
        trainer = Trainer(config, device=job["device"], callback=on_event)

        resume = job.get("resume")
        if resume and (Path(resume) / checkpoint_saved_files(model)).exists():
            trainer.detector.load(Path(resume))
            print(f"Resumed from {resume}")

        emit({"type": "status", "status": "training"})
        trainer.fit(train_x, train_y, val_x, val_y)

        emit({"type": "status", "status": "saving"})
        out_dir = Path(job["checkpoint_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.detector.save(out_dir)

        recorder.finish("completed", checkpoint=str(out_dir))
        emit({"type": "finished", "status": "completed", "checkpoint": str(out_dir)})
    except KeyboardInterrupt:
        if recorder:
            recorder.finish("interrupted")
        emit({"type": "finished", "status": "interrupted"})
    except BaseException as exc:  # noqa: BLE001 - report every failure to the dashboard
        traceback.print_exc()
        if recorder:
            recorder.finish("failed", error=f"{type(exc).__name__}: {exc}")
        emit({"type": "finished", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        log.flush()
