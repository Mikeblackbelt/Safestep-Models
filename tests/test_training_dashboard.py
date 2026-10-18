"""
Unit tests for the training dashboard building blocks: planner, run history,
sweep expansion, and terminal charts. None of these train a model.
"""

import json
import time
from pathlib import Path

import pytest
from rich.console import Console

from cart.config import CARTConfig
from jaw.config import JAWConfig
from training.charts import LineChart, Series, bar
from training.dashboard import build_specs, parse_sweeps
from training.devices import HardwareProfile
from training.history import RunRecord, RunRecorder, load_history
from training.planner import (
    SessionSpec,
    calibrate,
    make_plan,
    peak_memory_gb,
    predict_throughput,
    projected_finish,
)


def _cpu(cores=8, ram_free=64.0) -> HardwareProfile:
    return HardwareProfile("cpu", cores, cores * 2, ram_free * 2, ram_free)


def _spec(sid, model="cart", n_train=1000, n_val=200, **overrides) -> SessionSpec:
    cfg = (CARTConfig if model == "cart" else JAWConfig)(**overrides)
    return SessionSpec(sid, sid, model, cfg, overrides, n_train, n_val, f"checkpoints/{sid}")


# ── planner ──────────────────────────────────────────────────────────────────

def test_equal_sessions_run_in_parallel():
    plan = make_plan([_spec("a"), _spec("b")], _cpu(cores=8), history=[])
    assert plan.parallel == 2
    assert plan.slots[0].threads == 4
    assert plan.makespan < plan.sequential_secs


def test_short_session_does_not_force_parallelism():
    # a tiny job alongside a huge one: splitting cores would slow the big one down
    specs = [_spec("big", n_train=100_000), _spec("tiny", n_train=100)]
    plan = make_plan(specs, _cpu(cores=8), history=[])
    assert plan.parallel == 1
    assert plan.order[0] == "big"


def test_memory_limits_concurrency_and_warns():
    specs = [_spec("a"), _spec("b")]
    need = peak_memory_gb("cart", specs[0].config)
    # room for exactly one session (budget = 80% of free RAM)
    plan = make_plan(specs, _cpu(cores=8, ram_free=need / 0.8 * 1.5), history=[])
    assert plan.parallel == 1
    assert not plan.warnings

    small = make_plan([_spec("a")], _cpu(ram_free=3.0), history=[])
    assert small.warnings and "--batch-size 2" in small.warnings[0]

    tiny = make_plan([_spec("a")], _cpu(ram_free=1.0), history=[])
    assert tiny.warnings and "even at batch size 1" in tiny.warnings[0]


def test_max_parallel_cap():
    specs = [_spec(str(i)) for i in range(4)]
    assert make_plan(specs, _cpu(cores=16), history=[], max_parallel=2).parallel <= 2


def test_calibration_from_history_changes_estimate():
    spec = _spec("a")
    base = make_plan([spec], _cpu(), history=[]).estimates["a"]
    rec = RunRecord("r1", "cart", "cart", "cart", device="cpu", threads=8, units_per_sec=100.0,
                    units_done=5000, config={"max_seq_length": 512})
    calibrated = make_plan([spec], _cpu(), history=[rec]).estimates["a"]
    assert base.source == "baseline"
    assert calibrated.source.startswith("calibrated")
    assert calibrated.units_per_sec == pytest.approx(100.0)  # same threads → same measured rate


def test_throughput_scales_sublinearly_with_threads():
    cfg = CARTConfig()
    t4 = predict_throughput("cart", cfg, "cpu", 4)
    t8 = predict_throughput("cart", cfg, "cpu", 8)
    assert t4 < t8 < 2 * t4


def test_calibrate_ignores_other_devices():
    rec = RunRecord("r", "s", "s", "cart", device="cuda:0", units_per_sec=500.0, units_done=5000)
    assert calibrate([rec], "cart", "cpu") is None


def test_projected_finish_fills_idle_slots():
    assert projected_finish([10.0], 2, [5.0, 5.0]) == pytest.approx(10.0)
    assert projected_finish([], 1, [3.0, 4.0]) == pytest.approx(7.0)


# ── live dashboard state ─────────────────────────────────────────────────────

def test_live_eta_recalibrates_queued_sessions(tmp_path: Path):
    from training.dashboard import Dashboard
    from training.devices import DeviceMonitor

    specs = [_spec("a"), _spec("b"), _spec("j", model="jaw")]
    plan = make_plan(specs, _cpu(), history=[], max_parallel=1)
    dash = Dashboard("run", plan, specs, DeviceMonitor("cpu"), [], tmp_path)
    first = dash.states[plan.order[0]]
    queued = [s for s in dash.states.values() if s is not first]
    before = {s.spec.session_id: dash.queued_secs(s) for s in queued}

    # first session measured at half its planned throughput
    first.status = "training"
    first.first_batch_t = time.time() - 100
    first.units_done = first.planned_rate * 0.5 * 100

    for s in queued:
        assert dash.queued_secs(s) == pytest.approx(before[s.spec.session_id] * 2, rel=0.05)

    # a finished session's rate stays fixed instead of decaying with wall time
    first.first_batch_t = time.time() - 10_000
    first.finished_t = first.first_batch_t + 100
    assert first.rate == pytest.approx(first.planned_rate * 0.5)


# ── history ──────────────────────────────────────────────────────────────────

def test_recorder_persists_epochs_and_detects_interrupted(tmp_path: Path):
    rec = RunRecord("run1", "cart", "cart", "cart", n_val=10, total_epochs=2, pid=999_999_999)
    recorder = RunRecorder(tmp_path, rec)
    recorder.handle({"type": "batch", "phase": "train", "samples": 16, "stage": "frozen", "epoch": 1})
    recorder.handle({"type": "epoch", "stage": "frozen", "epoch": 1, "loss": 0.7, "val_acc": 0.6})

    saved = json.loads((tmp_path / "run1" / "cart.json").read_text())
    assert saved["epochs"][0]["val_acc"] == 0.6

    # never finished and its pid is dead → reported as a partial (interrupted) run
    [loaded] = load_history(tmp_path)
    assert loaded.status == "interrupted"
    assert loaded.is_partial
    assert loaded.best_val_acc == 0.6


def test_completed_run_uses_fusion_accuracy(tmp_path: Path):
    recorder = RunRecorder(tmp_path, RunRecord("run2", "cart", "cart", "cart", n_val=10))
    recorder.handle({"type": "epoch", "stage": "finetune", "epoch": 1, "loss": 0.5, "val_acc": 0.8})
    recorder.handle({"type": "fusion", "stage": "fusion", "epoch": 0, "val_acc": 0.9})
    recorder.finish("completed")
    [loaded] = load_history(tmp_path)
    assert loaded.status == "completed"
    assert loaded.final_acc == 0.9


def test_load_history_filters_by_model_and_sorts(tmp_path: Path):
    for i, model in enumerate(["jaw", "cart", "cart"]):
        r = RunRecorder(tmp_path, RunRecord(f"run{i}", model, model, model, started_at=time.time() - 100 + i))
        r.finish("completed")
    carts = load_history(tmp_path, model="cart")
    assert [r.run_id for r in carts] == ["run1", "run2"]


# ── sweeps ───────────────────────────────────────────────────────────────────

def test_parse_sweeps():
    assert parse_sweeps(["lr=1e-5, 3e-5"]) == [("lr", ["1e-5", "3e-5"])]
    with pytest.raises(ValueError):
        parse_sweeps(["lr"])


def test_build_specs_expands_grid(tmp_path: Path):
    sweeps = parse_sweeps(["lr=1e-5,3e-5", "batch_size=8,16"])
    specs, notes = build_specs(["cart"], tmp_path, tmp_path / "ckpt", {"cart": {}}, sweeps, "run", Console(quiet=True), allow_demo=True)
    assert len(specs) == 4
    assert {s.config.learning_rate for s in specs} == {1e-5, 3e-5}
    assert {s.config.train_batch_size for s in specs} == {8, 16}
    assert len({s.session_id for s in specs}) == 4
    assert all("sweeps" in s.checkpoint_dir for s in specs)


def test_build_specs_skips_models_without_data(tmp_path: Path):
    specs, notes = build_specs(["jaw"], tmp_path, tmp_path, {}, [], "run", Console(quiet=True))
    assert specs == [] and "skipped" in notes[0]


# ── charts ───────────────────────────────────────────────────────────────────

def test_line_chart_renders_requested_height():
    console = Console(width=60, record=True, color_system=None)
    chart = LineChart([Series("a", [1, 2, 3], [0.2, 0.5, 0.9])], y_range=(0, 1), height=8)
    console.print(chart)
    lines = console.export_text().rstrip("\n").split("\n")
    assert len(lines) == 8
    assert any(ch for ch in "".join(lines) if 0x2800 < ord(ch) <= 0x28FF)  # braille dots drawn


def test_line_chart_empty_message():
    console = Console(width=40, record=True, color_system=None)
    console.print(LineChart([], empty_message="nothing yet", height=3))
    assert "nothing yet" in console.export_text()


def test_bar_width():
    assert len(bar(0.5, 10).plain) == 10
    assert len(bar(1.5, 10).plain) == 10
