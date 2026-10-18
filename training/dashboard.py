"""
Live training dashboard and run-history viewer.

`run_dashboard()` plans sessions against the measured hardware, launches them
into worker processes as slots free up, and renders a full-screen Rich view:
session progress with live ETAs, device utilisation, accuracy/loss charts
(overlaid with the best previous run), and an event log. `show_history()`
prints previous and partial runs with charts.
"""

import itertools
import math
import multiprocessing as mp
import queue as queue_mod
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from training.charts import PALETTE, LineChart, Series, bar, sparkline
from training.data import load_splits
from training.devices import DeviceMonitor, HardwareProfile
from training.history import EVAL_WEIGHT, RunRecord, load_history
from training.planner import Plan, SessionSpec, Slot, fmt_duration, projected_finish
from training.session import coerce_value, make_config, resolve_override, run_session

_STAGE_SHORT = {"frozen": "frz", "finetune": "tune"}
TERMINAL = {"completed", "failed", "interrupted", "cancelled", "skipped"}
STATUS_STYLE = {
    "queued": "grey62",
    "starting": "cyan",
    "loading": "cyan",
    "training": "bold blue",
    "saving": "magenta",
    "completed": "bold green",
    "failed": "bold red",
    "interrupted": "yellow",
    "cancelled": "grey50",
    "skipped": "grey50",
    "running": "bold blue",
}


# ── Session construction ─────────────────────────────────────────────────────

def parse_sweeps(raw: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """["lr=1e-5,3e-5", "batch_size=8,16"] → [("lr", ["1e-5", "3e-5"]), ...]"""
    sweeps = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--sweep expects KEY=V1,V2,… (got {item!r})")
        key, values = item.split("=", 1)
        vals = [v.strip() for v in values.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"--sweep {key} has no values")
        sweeps.append((key.strip(), vals))
    return sweeps


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-").lower()


def build_specs(
    models: Sequence[str],
    data_dir: Path,
    checkpoint_dir: Path,
    base_overrides: Dict[str, Dict[str, Any]],
    sweeps: Sequence[Tuple[str, List[str]]],
    run_id: str,
    console: Console,
    allow_demo: bool = False,
) -> Tuple[List[SessionSpec], List[str]]:
    """Expand models × sweep grid into session specs. Returns (specs, notes)."""
    specs: List[SessionSpec] = []
    notes: List[str] = []
    for model in models:
        train_x, _, val_x, _ = load_splits(model, data_dir, console, allow_demo=allow_demo)
        if not train_x:
            notes.append(f"{model.upper()}: no training data under {data_dir} — skipped")
            continue

        grid: List[Tuple[str, List[str]]] = []
        for key, values in sweeps:
            field_name = resolve_override(model, key)
            if field_name is None:
                notes.append(f"{model.upper()}: ignoring sweep key {key!r} (not a {model.upper()} config field)")
                continue
            grid.append((field_name, values))

        combos = list(itertools.product(*[vals for _, vals in grid])) if grid else [()]
        for combo in combos:
            overrides = dict(base_overrides.get(model, {}))
            parts = []
            for (field_name, _), raw in zip(grid, combo):
                overrides[field_name] = coerce_value(model, field_name, raw)
                parts.append(f"{field_name}={raw}")
            label = f"{model}[{','.join(parts)}]" if parts else model
            sid = _slug(label)
            ckpt = checkpoint_dir / model / "sweeps" / run_id / sid if parts else checkpoint_dir / model
            specs.append(SessionSpec(
                session_id=sid,
                label=label,
                model=model,
                config=make_config(model, overrides),
                overrides=overrides,
                n_train=len(train_x),
                n_val=len(val_x),
                checkpoint_dir=str(ckpt),
            ))
    return specs, notes


# ── Static renderables (pre-flight / summaries) ──────────────────────────────

def render_hardware(hw: HardwareProfile) -> Table:
    t = Table(title="Hardware", box=box.SIMPLE_HEAD, header_style="bold cyan", title_justify="left")
    t.add_column("Property", style="dim")
    t.add_column("Value", style="bold")
    t.add_row("Device", hw.device.upper())
    t.add_row("CPU cores", f"{hw.physical_cores} physical / {hw.logical_cores} logical")
    t.add_row("RAM", f"{hw.ram_available_gb:.1f} GB free of {hw.ram_total_gb:.1f} GB")
    for g in hw.gpus:
        t.add_row(f"GPU {g.index}", f"{g.name} · {g.free_gb:.1f} GB free of {g.total_gb:.1f} GB")
    return t


def render_plan(plan: Plan, specs: Sequence[SessionSpec]) -> Group:
    t = Table(title="Training plan", box=box.SIMPLE_HEAD, header_style="bold magenta", title_justify="left")
    for col, just in (("Session", "left"), ("Train/Val", "right"), ("Epochs", "right"), ("Peak mem", "right"),
                      ("Est. rate", "right"), ("Est. time", "right"), ("Source", "left")):
        t.add_column(col, justify=just)
    for sid in plan.order:
        spec = next(s for s in specs if s.session_id == sid)
        est = plan.estimates[sid]
        epochs = spec.config.num_epochs_frozen + spec.config.num_epochs_finetune
        t.add_row(
            Text(spec.label),
            f"{spec.n_train:,}/{spec.n_val:,}",
            str(epochs),
            f"{est.mem_gb:.1f} GB",
            f"{est.units_per_sec:.1f}/s",
            fmt_duration(est.secs),
            Text(est.source, style="green" if est.source.startswith("calibrated") else "dim"),
        )

    c = Table(box=box.SIMPLE_HEAD, header_style="bold", title="Parallelism options", title_justify="left")
    c.add_column("Concurrent")
    c.add_column("Threads each", justify="right")
    c.add_column("Makespan", justify="right")
    c.add_column("")
    for cand in plan.candidates:
        chosen = cand.parallel == plan.parallel
        c.add_row(str(cand.parallel), str(cand.threads), fmt_duration(cand.makespan),
                  Text("◀ chosen", style="bold green") if chosen else "")

    slot = plan.slots[0]
    saved = plan.sequential_secs - plan.makespan
    summary = Text.assemble(
        ("Plan: ", "bold"),
        f"{plan.parallel} concurrent session{'s' if plan.parallel != 1 else ''} on {plan.device.upper()}, "
        f"{slot.threads} thread{'s' if slot.threads != 1 else ''} each · memory budget {plan.mem_budget_gb:.1f} GB\n",
        ("ETA: ", "bold"),
        (fmt_duration(plan.makespan), "bold yellow"),
        f"  (sequential {fmt_duration(plan.sequential_secs)}"
        + (f", saves {fmt_duration(saved)}" if saved > 1 else "") + ")",
    )
    parts: List[Any] = [t, c, summary]
    for w in plan.warnings:
        parts.append(Text(f"⚠  {w}", style="yellow"))
    return Group(*parts)


# ── Live state ───────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    spec: SessionSpec
    est_secs: float
    total_units: float
    status: str = "queued"
    planned_units: float = 0.0
    slot: Optional[Slot] = None
    process: Any = None
    pid: Optional[int] = None
    stage: Optional[str] = None
    epoch: int = 0
    stage_epochs: int = 0
    batch: int = 0
    n_batches: int = 0
    units_done: float = 0.0
    loss_ema: Optional[float] = None
    started_t: Optional[float] = None
    first_batch_t: Optional[float] = None
    finished_t: Optional[float] = None
    acc_points: List[Tuple[float, float]] = field(default_factory=list)
    loss_points: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=800))
    fusion_acc: Optional[float] = None
    error: Optional[str] = None
    color: str = PALETTE[0]

    @property
    def global_epoch_offset(self) -> int:
        return self.spec.config.num_epochs_frozen if self.stage == "finetune" else 0

    @property
    def progress(self) -> float:
        if self.status == "completed":
            return 1.0
        return min(1.0, self.units_done / self.total_units) if self.total_units else 0.0

    @property
    def rate(self) -> Optional[float]:
        """Measured units/sec (frozen once the session finishes)."""
        if self.first_batch_t is None:
            return None
        elapsed = (self.finished_t or time.time()) - self.first_batch_t
        return self.units_done / elapsed if elapsed > 3 and self.units_done > 0 else None

    @property
    def planned_rate(self) -> float:
        return self.planned_units / max(self.est_secs, 1e-9)

    @property
    def best_acc(self) -> Optional[float]:
        accs = [a for _, a in self.acc_points if a == a]
        return max(accs) if accs else None

    def remaining_secs(self, speed_ratio: float = 1.0) -> float:
        """Seconds left: measured rate once available, else the (live-corrected) plan."""
        if self.status in TERMINAL:
            return 0.0
        rate = self.rate or self.planned_rate * speed_ratio
        return max(0.0, self.total_units - self.units_done) / max(rate, 1e-9)


class Dashboard:
    def __init__(
        self,
        run_id: str,
        plan: Plan,
        specs: Sequence[SessionSpec],
        monitor: DeviceMonitor,
        history: Sequence[RunRecord],
        runs_dir: Path,
    ):
        self.run_id = run_id
        self.plan = plan
        self.monitor = monitor
        self.runs_dir = runs_dir
        self.t0 = time.time()
        self.events: Deque[Text] = deque(maxlen=200)
        self.states: Dict[str, SessionState] = {}
        for i, sid in enumerate(plan.order):
            spec = next(s for s in specs if s.session_id == sid)
            est = plan.estimates[sid]
            self.states[sid] = SessionState(spec, est.secs, est.units, planned_units=est.units,
                                            color=PALETTE[i % len(PALETTE)])
        # best previous completed run per model, for overlay + comparison
        self.prev_best: Dict[str, RunRecord] = {}
        for rec in history:
            if rec.status == "completed" and rec.final_acc is not None:
                cur = self.prev_best.get(rec.model)
                if cur is None or rec.final_acc > (cur.final_acc or -1):
                    self.prev_best[rec.model] = rec
        self.stopping = False

    # ── events ───────────────────────────────────────────────────────────────
    def log(self, msg: str, style: str = "") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.events.append(Text.assemble((stamp + " ", "grey50"), (msg, style)))

    def apply(self, ev: Dict[str, Any]) -> None:
        st = self.states.get(ev.get("session_id", ""))
        if st is None:
            return
        etype = ev.get("type")
        name = st.spec.label
        if etype == "status":
            st.status = ev["status"]
            st.pid = ev.get("pid", st.pid)
            if ev["status"] == "training":
                self.log(f"{name}: model ready, training started", "cyan")
        elif etype == "stage":
            st.stage = ev["stage"]
            st.stage_epochs = ev.get("epochs", 0)
            st.epoch, st.batch = 0, 0
            if st.stage == "fusion":
                st.units_done += EVAL_WEIGHT * st.spec.n_val
                self.log(f"{name}: fitting XGBoost fusion", "magenta")
            else:
                self.log(f"{name}: stage {st.stage} ({st.stage_epochs} epochs)")
        elif etype == "batch":
            now = time.time()
            if st.first_batch_t is None:
                st.first_batch_t = now
            weight = 1.0 if ev.get("phase") == "train" else EVAL_WEIGHT
            st.units_done += ev.get("samples", 0) * weight
            st.epoch = ev.get("epoch", st.epoch)
            if ev.get("phase") == "train":
                st.batch, st.n_batches = ev["batch"], ev["n_batches"]
                loss = float(ev["loss"])
                st.loss_ema = loss if st.loss_ema is None else 0.9 * st.loss_ema + 0.1 * loss
                x = st.global_epoch_offset + (st.epoch - 1) + st.batch / max(1, st.n_batches)
                st.loss_points.append((x, st.loss_ema))
        elif etype == "epoch":
            x = st.global_epoch_offset + ev["epoch"]
            st.acc_points.append((x, float(ev["val_acc"])))
            self.log(f"{name}: {ev['stage']} epoch {ev['epoch']}/{ev['epochs']}  "
                     f"loss {ev['loss']:.4f}  val_acc {ev['val_acc']:.4f}")
        elif etype == "early_stop":
            fusion = EVAL_WEIGHT * st.spec.n_val if st.spec.model == "cart" else 0.0
            st.total_units = st.units_done + fusion
            self.log(f"{name}: early stopping (patience exceeded)", "yellow")
        elif etype == "fusion":
            st.fusion_acc = float(ev["val_acc"])
            self.log(f"{name}: fusion val_acc {st.fusion_acc:.4f}", "magenta")
        elif etype == "finished":
            st.status = ev["status"]
            st.finished_t = time.time()
            st.error = ev.get("error")
            style = {"completed": "bold green", "failed": "bold red"}.get(st.status, "yellow")
            detail = f" → {ev['checkpoint']}" if ev.get("checkpoint") else (f": {st.error}" if st.error else "")
            self.log(f"{name}: {st.status}{detail}", style)

    # ── ETA ──────────────────────────────────────────────────────────────────
    def speed_ratio(self, model: str) -> float:
        """
        Measured ÷ planned throughput, learned live from sessions that have started.
        Prefers sessions of the same model; falls back to any model (machine-wide speed).
        """
        by_model: Dict[str, List[float]] = {}
        for s in self.states.values():
            if s.rate:
                by_model.setdefault(s.spec.model, []).append(s.rate / s.planned_rate)
        if model in by_model:
            return median(by_model[model])
        pooled = [r for rs in by_model.values() for r in rs]
        return median(pooled) if pooled else 1.0

    def queued_secs(self, st: SessionState) -> float:
        return st.est_secs / max(self.speed_ratio(st.spec.model), 1e-9)

    def overall_eta(self) -> float:
        running = [s for s in self.states.values() if s.status not in TERMINAL and s.status != "queued"]
        queued = [s for s in self.states.values() if s.status == "queued"]
        remaining = [s.remaining_secs(self.speed_ratio(s.spec.model)) for s in running]
        return projected_finish(remaining, self.plan.parallel, [self.queued_secs(s) for s in queued])

    def overall_progress(self) -> float:
        total = sum(s.total_units for s in self.states.values() if s.status not in ("skipped", "cancelled"))
        done = sum(min(s.units_done, s.total_units) if s.status != "completed" else s.total_units
                   for s in self.states.values() if s.status not in ("skipped", "cancelled"))
        return done / total if total else 0.0

    # ── rendering ────────────────────────────────────────────────────────────
    def _header(self) -> Panel:
        elapsed = time.time() - self.t0
        prog = self.overall_progress()
        done = sum(1 for s in self.states.values() if s.status in TERMINAL)
        line = Text.assemble(
            ("SafeStep Training ", "bold cyan"), (f"run {self.run_id}", "grey62"),
            "   elapsed ", (fmt_duration(elapsed), "bold"),
            "   ETA ", (fmt_duration(self.overall_eta()) if not self.stopping else "stopping…", "bold yellow"),
            "   planned ", (fmt_duration(self.plan.makespan), "grey62"),
            f"   sessions {done}/{len(self.states)}  ",
        )
        line.append_text(bar(prog, 24, "green"))
        line.append(f" {prog * 100:.0f}%")
        return Panel(line, box=box.HEAVY, border_style="cyan", padding=(0, 1))

    def _sessions(self) -> Panel:
        t = Table(box=box.SIMPLE_HEAD, expand=True, header_style="bold", pad_edge=False, show_edge=False,
                  padding=(0, 1), collapse_padding=True)
        t.add_column("Session", no_wrap=True, overflow="ellipsis", min_width=14)
        t.add_column("Slot", style="grey62", no_wrap=True)
        t.add_column("Status", no_wrap=True)
        t.add_column("Epoch", no_wrap=True)
        t.add_column("Progress", no_wrap=True, min_width=15)
        t.add_column("Loss", justify="right", no_wrap=True)
        t.add_column("Val", justify="right", no_wrap=True)
        t.add_column("Prev", justify="right", style="grey62", no_wrap=True)
        t.add_column("Rate", justify="right", no_wrap=True)
        t.add_column("Proc", justify="right", style="grey62", no_wrap=True)
        t.add_column("ETA", justify="right", style="yellow", no_wrap=True)
        for st in self.states.values():
            slot = f"{st.slot.device}×{st.slot.threads}" if st.slot else "—"
            if st.stage == "fusion":
                epoch = "fusion"
            elif st.stage:
                epoch = f"{_STAGE_SHORT.get(st.stage, st.stage)} {st.epoch}/{st.stage_epochs}"
            else:
                epoch = "—"
            prog = bar(st.progress, 10, st.color)
            prog.append(f" {st.progress * 100:3.0f}%")
            acc = st.fusion_acc if st.fusion_acc is not None else st.best_acc
            prev = self.prev_best.get(st.spec.model)
            proc = self.monitor.process(st.pid) if st.pid and st.status not in TERMINAL else None
            eta = "—"
            if st.status == "queued":
                eta = f"~{fmt_duration(self.queued_secs(st))}"
            elif st.status not in TERMINAL:
                eta = fmt_duration(st.remaining_secs(self.speed_ratio(st.spec.model)))
            elif st.finished_t and st.started_t:
                eta = Text(f"✓ {fmt_duration(st.finished_t - st.started_t)}", style="grey62")
            t.add_row(
                Text("● ", style=st.color) + Text(st.spec.label),
                slot,
                Text(st.status, style=STATUS_STYLE.get(st.status, "")),
                epoch,
                prog,
                f"{st.loss_ema:.3f}" if st.loss_ema is not None else "—",
                f"{acc:.3f}" if acc is not None else "—",
                f"{prev.final_acc:.3f}" if prev and prev.final_acc is not None else "—",
                f"{st.rate:.1f}/s" if st.rate else "—",
                f"{proc.cpu_pct:.0f}%·{proc.rss_gb:.1f}G" if proc else "—",
                eta,
            )
        return Panel(t, title="Sessions", border_style="blue", title_align="left")

    def _acc_chart(self) -> Panel:
        series = []
        for model, rec in self.prev_best.items():
            if any(s.spec.model == model for s in self.states.values()) and rec.epochs:
                frozen = rec.config.get("num_epochs_frozen", 0)
                xs = [e.epoch + (frozen if e.stage == "finetune" else 0) for e in rec.epochs]
                series.append(Series(f"best {model} ({rec.run_id})", xs, [e.val_acc for e in rec.epochs],
                                     color="grey50", dotted=True))
        for st in self.states.values():
            if st.acc_points:
                series.append(Series(st.spec.label, [x for x, _ in st.acc_points], [y for _, y in st.acc_points],
                                     color=st.color))
        chart = LineChart(series, y_range=(0.0, 1.0), y_fmt=lambda v: f"{v:.2f}", x_fmt=lambda v: f"ep {v:.0f}",
                          empty_message="accuracy appears after the first epoch")
        return Panel(chart, title="Validation accuracy · per epoch", border_style="green", title_align="left")

    def _loss_chart(self) -> Panel:
        series = [
            Series(st.spec.label, [x for x, _ in st.loss_points], [y for _, y in st.loss_points], color=st.color)
            for st in self.states.values() if st.loss_points
        ]
        chart = LineChart(series, y_fmt=lambda v: f"{v:.3f}", x_fmt=lambda v: f"ep {v:.1f}",
                          empty_message="loss appears once training starts")
        return Panel(chart, title="Training loss · EMA per batch", border_style="red", title_align="left")

    def _device_bars(self) -> Panel:
        samples = self.monitor.samples
        rows: List[Any] = []
        if samples:
            s = samples[-1]
            cpu = Text("CPU ", style="bold")
            cpu.append_text(bar(s.cpu_pct / 100, 12, "#4e9fe6"))
            cpu.append(f" {s.cpu_pct:5.1f}% ")
            cpu.append_text(sparkline([x.cpu_pct for x in samples], 8, color="#4e9fe6"))
            rows.append(cpu)
            ram = Text("RAM ", style="bold")
            ram.append_text(bar(s.ram_pct / 100, 12, "#f28e2b"))
            ram.append(f" {s.ram_used_gb:4.1f}/{self.monitor_total_ram:.0f} GB")
            rows.append(ram)
            for g in s.gpus:
                used = g.total_gb - g.free_gb
                line = Text(f"GPU{g.index}", style="bold")
                line.append_text(bar(used / g.total_gb if g.total_gb else 0, 12, "#59a14f"))
                line.append(f" {used:.1f}/{g.total_gb:.0f} GB")
                if g.util_pct is not None:
                    line.append(f" · {g.util_pct:.0f}% util")
                rows.append(line)
        else:
            rows.append(Text("sampling…", style="dim"))
        for row in rows:
            row.no_wrap, row.overflow = True, "ellipsis"
        return Panel(Group(*rows), title="Device", border_style="cyan", title_align="left")

    def _device_chart(self) -> Panel:
        samples = list(self.monitor.samples)
        series = []
        if samples:
            now = samples[-1].t
            xs = [s.t - now for s in samples]
            series.append(Series("CPU %", xs, [s.cpu_pct for s in samples], color="#4e9fe6"))
            series.append(Series("RAM %", xs, [s.ram_pct for s in samples], color="#f28e2b"))
            n_gpus = len(samples[-1].gpus)
            for i in range(n_gpus):
                utils = [s.gpus[i].util_pct if i < len(s.gpus) and s.gpus[i].util_pct is not None else math.nan
                         for s in samples]
                if any(u == u for u in utils):
                    series.append(Series(f"GPU{i} %", xs, utils, color=PALETTE[2 + i % 6]))
        chart = LineChart(series, y_range=(0, 100), y_fmt=lambda v: f"{v:.0f}", x_fmt=lambda v: f"{-v:.0f}s ago" if v < -0.5 else "now")
        return Panel(chart, title="Utilisation history", border_style="cyan", title_align="left")

    def _plan_panel(self) -> Panel:
        slot = self.plan.slots[0]
        lines = [
            Text.assemble(("Parallel ", "bold"), f"{self.plan.parallel} × {slot.threads} threads on {self.plan.device.upper()}"),
            Text.assemble(("Planned  ", "bold"), fmt_duration(self.plan.makespan),
                          (f"  (seq {fmt_duration(self.plan.sequential_secs)})", "grey62")),
            Text.assemble(("Budget   ", "bold"), f"{self.plan.mem_budget_gb:.1f} GB"),
            Text.assemble(("Logs     ", "bold"), (str(self.runs_dir / self.run_id), "grey62")),
        ]
        for w in self.plan.warnings[:2]:
            lines.append(Text(f"⚠ {w}", style="yellow"))
        return Panel(Group(*lines), title="Plan", border_style="magenta", title_align="left")

    def _events(self) -> Panel:
        return Panel(_TailLog(self.events), title="Events", border_style="grey50", title_align="left")

    @property
    def monitor_total_ram(self) -> float:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3

    def __rich__(self) -> Layout:
        n_gpus = len(self.monitor.samples[-1].gpus) if self.monitor.samples else 0
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=3),
            Layout(name="body"),
            Layout(Text(" Ctrl+C stops all sessions · partial results are kept · "
                        "view later with: python -m train --history", style="grey50"), name="footer", size=1),
        )
        layout["body"].split_row(Layout(name="main", ratio=3), Layout(name="side", ratio=1, minimum_size=38))
        layout["main"].split_column(
            Layout(self._sessions(), name="sessions", size=min(len(self.states), 12) + 4),
            Layout(name="charts"),
        )
        layout["charts"].split_row(Layout(self._acc_chart()), Layout(self._loss_chart()))
        layout["side"].split_column(
            Layout(self._device_bars(), size=4 + n_gpus),
            Layout(self._device_chart(), size=11),
            Layout(self._plan_panel(), size=6 + min(2, len(self.plan.warnings))),
            Layout(self._events()),
        )
        return layout


class _TailLog:
    """Shows the newest log lines that fit in the available height."""

    def __init__(self, lines: Sequence[Text]):
        self.lines = lines

    def __rich_console__(self, console, options):
        height = options.height or 10
        for line in list(self.lines)[-height:]:
            line = line.copy()
            line.truncate(options.max_width, overflow="ellipsis")
            yield line


# ── Orchestration ────────────────────────────────────────────────────────────

def run_dashboard(
    specs: List[SessionSpec],
    hw: HardwareProfile,
    plan: Plan,
    run_id: str,
    data_dir: Path,
    runs_dir: Path,
    resume: Dict[str, Optional[Path]],
    history: Sequence[RunRecord],
    console: Console,
) -> Dict[str, SessionState]:
    ctx = mp.get_context("spawn")
    events = ctx.Queue()
    monitor = DeviceMonitor(hw.device)
    dash = Dashboard(run_id, plan, specs, monitor, history, runs_dir)
    pending = [dash.states[sid] for sid in plan.order]
    free_slots: List[Slot] = list(plan.slots)
    running: List[SessionState] = []

    def launch(st: SessionState, slot: Slot) -> None:
        job = {
            "run_id": run_id,
            "session_id": st.spec.session_id,
            "label": st.spec.label,
            "model": st.spec.model,
            "overrides": st.spec.overrides,
            "data_dir": str(data_dir),
            "checkpoint_dir": st.spec.checkpoint_dir,
            "runs_dir": str(runs_dir),
            "device": slot.device,
            "threads": slot.threads,
            "concurrent": slot.share,
            "total_units": st.total_units,
            "estimated_secs": st.est_secs,
            "resume": str(resume[st.spec.model]) if resume.get(st.spec.model) else None,
        }
        proc = ctx.Process(target=run_session, args=(job, events), name=f"safestep-{st.spec.session_id}")
        proc.start()
        st.process, st.pid, st.slot = proc, proc.pid, slot
        st.status, st.started_t = "starting", time.time()
        running.append(st)
        dash.log(f"{st.spec.label}: launched on {slot.device} with {slot.threads} threads (pid {proc.pid})")

    def drain() -> None:
        while True:
            try:
                dash.apply(events.get_nowait())
            except queue_mod.Empty:
                return

    def reap() -> None:
        for st in list(running):
            if st.process.is_alive():
                continue
            st.process.join()
            drain()
            if st.status not in TERMINAL:
                st.status = "interrupted" if dash.stopping else "failed"
                st.error = st.error or f"worker exited with code {st.process.exitcode}"
                st.finished_t = time.time()
                dash.log(f"{st.spec.label}: {st.status} ({st.error})", "red")
            running.remove(st)
            free_slots.append(st.slot)

    monitor.sample()
    last_sample = 0.0
    with Live(dash, console=console, screen=True, refresh_per_second=4, redirect_stderr=False) as live:
        try:
            while pending or running:
                while pending and free_slots and not dash.stopping:
                    launch(pending.pop(0), free_slots.pop(0))
                drain()
                reap()
                if time.time() - last_sample >= 1.0:
                    monitor.sample()
                    last_sample = time.time()
                time.sleep(0.2)
        except KeyboardInterrupt:
            dash.stopping = True
            for st in pending:
                st.status = "cancelled"
            pending.clear()
            dash.log("Ctrl+C — stopping sessions and saving partial results…", "bold yellow")
            deadline = time.time() + 20
            try:
                while running and time.time() < deadline:
                    drain()
                    reap()
                    live.refresh()
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
            for st in running:
                st.process.terminate()
            for st in list(running):
                st.process.join(5)
            reap()
        drain()
        live.refresh()
    return dash.states


def render_summary(states: Dict[str, SessionState], prev_best: Dict[str, RunRecord]) -> Group:
    t = Table(title="Run summary", box=box.SIMPLE_HEAD, header_style="bold", title_justify="left")
    for col in ("Session", "Status", "Best val acc", "Fusion acc", "vs prev best", "Time", "Estimated", "Rate"):
        t.add_column(col, justify="left" if col in ("Session", "Status") else "right")
    series = []
    for st in states.values():
        acc = st.fusion_acc if st.fusion_acc is not None else st.best_acc
        prev = prev_best.get(st.spec.model)
        delta = ""
        if acc is not None and prev and prev.final_acc is not None:
            d = acc - prev.final_acc
            delta = Text(f"{d:+.4f}", style="green" if d >= 0 else "red")
        took = (st.finished_t - st.started_t) if st.finished_t and st.started_t else None
        t.add_row(
            Text("● ", style=st.color) + Text(st.spec.label),
            Text(st.status, style=STATUS_STYLE.get(st.status, "")),
            f"{st.best_acc:.4f}" if st.best_acc is not None else "—",
            f"{st.fusion_acc:.4f}" if st.fusion_acc is not None else "—",
            delta,
            fmt_duration(took),
            fmt_duration(st.est_secs),
            f"{st.rate:.1f}/s" if st.rate else "—",
        )
        if st.acc_points:
            series.append(Series(st.spec.label, [x for x, _ in st.acc_points], [y for _, y in st.acc_points],
                                 color=st.color))
    parts: List[Any] = [t]
    for st in states.values():
        if st.error:
            parts.append(Text(f"✗ {st.spec.label}: {st.error}", style="red"))
    if series:
        parts.append(Panel(LineChart(series, y_range=(0, 1), height=12), title="Validation accuracy",
                           border_style="green", title_align="left", height=14))
    return Group(*parts)


# ── History view ─────────────────────────────────────────────────────────────

def show_history(runs_dir: Path, model: Optional[str], limit: int, console: Console) -> None:
    records = load_history(runs_dir, model)
    if not records:
        console.print(f"[yellow]No training runs recorded under {runs_dir} yet.[/yellow]")
        console.print("[dim]Runs are recorded automatically by `python -m train` (with or without --dashboard).[/dim]")
        return
    shown = records[-limit:]

    t = Table(title=f"Training history · {len(shown)} of {len(records)} sessions", box=box.SIMPLE_HEAD,
              header_style="bold", title_justify="left")
    for col, just in (("Run", "left"), ("Session", "left"), ("Status", "left"), ("Epochs", "right"),
                      ("Best val", "right"), ("Fusion", "right"), ("Duration", "right"), ("Est.", "right"),
                      ("Rate", "right"), ("Device", "left"), ("Data", "right")):
        t.add_column(col, justify=just, no_wrap=True)
    for rec in reversed(shown):
        status = rec.status if rec.status != "interrupted" else "partial"
        style = STATUS_STYLE.get(rec.status, "")
        t.add_row(
            rec.run_id,
            Text(rec.label),
            Text(status + (" ⏹" if rec.early_stopped else ""), style=style),
            f"{len(rec.epochs)}/{rec.total_epochs}",
            f"{rec.best_val_acc:.4f}" if rec.best_val_acc is not None else "—",
            f"{rec.fusion_val_acc:.4f}" if rec.fusion_val_acc is not None else "—",
            fmt_duration(rec.duration),
            fmt_duration(rec.estimated_secs),
            f"{rec.units_per_sec:.1f}/s" if rec.units_per_sec else "—",
            f"{rec.device}×{rec.threads}",
            f"{rec.n_train:,}",
        )
    console.print(t)
    partial = [r for r in shown if r.is_partial]
    if partial:
        console.print(f"[yellow]{len(partial)} partial session(s)[/yellow] "
                      "[dim](interrupted or failed — metrics up to the last completed epoch are kept; "
                      "dotted in the charts)[/dim]")
    for rec in shown:
        if rec.status == "failed" and rec.error:
            console.print(f"  [red]✗ {rec.run_id} {rec.label}:[/red] [dim]{rec.error}[/dim]")

    models = sorted({r.model for r in shown})
    for m in models:
        recs = [r for r in shown if r.model == m and r.epochs][-6:]
        if not recs:
            continue
        series = []
        for j, rec in enumerate(recs):
            frozen = rec.config.get("num_epochs_frozen", 0)
            xs = [e.epoch + (frozen if e.stage == "finetune" else 0) for e in rec.epochs]
            label = f"{rec.run_id[4:13]} {rec.label}" + (" (partial)" if rec.is_partial else "")
            series.append(Series(label, xs, [e.val_acc for e in rec.epochs],
                                 color=PALETTE[j % len(PALETTE)], dotted=rec.is_partial))
        console.print(Panel(LineChart(series, y_range=(0, 1), height=12),
                            title=f"{m.upper()} · validation accuracy per epoch (last {len(recs)} sessions)",
                            border_style="green", title_align="left", height=14))

    # final accuracy across runs, one line per model
    trend = []
    for i, m in enumerate(models):
        recs = [r for r in shown if r.model == m and r.final_acc is not None]
        if len(recs) >= 1:
            trend.append(Series(m.upper(), list(range(1, len(recs) + 1)), [r.final_acc for r in recs],
                                color=PALETTE[i % len(PALETTE)]))
    if trend:
        console.print(Panel(LineChart(trend, y_range=(0, 1), height=10, x_fmt=lambda v: f"#{v:.0f}"),
                            title="Best accuracy per session over time", border_style="blue",
                            title_align="left", height=12))

    rates = []
    for i, m in enumerate(models):
        recs = [r for r in shown if r.model == m and r.units_per_sec]
        if recs:
            rates.append(Series(f"{m.upper()} samples/s", list(range(1, len(recs) + 1)),
                                [r.units_per_sec for r in recs], color=PALETTE[(i + 2) % len(PALETTE)]))
    if rates:
        console.print(Panel(LineChart(rates, height=8, y_fmt=lambda v: f"{v:.1f}", x_fmt=lambda v: f"#{v:.0f}"),
                            title="Measured throughput per session (feeds ETA calibration)",
                            border_style="cyan", title_align="left", height=10))
