"""
Load estimation and parallel session planning.

For each session we estimate
  * peak memory (weights + grads + AdamW state + activations at the batch size),
  * total work in "units" (training-sample equivalents, eval samples weighted),
  * throughput in units/sec — calibrated from previous runs on the same device
    kind when history exists, otherwise from conservative baselines.

The planner then simulates every feasible degree of parallelism (limited by
RAM / VRAM and cores), splitting CPU threads between concurrent sessions, and
picks the schedule with the shortest makespan. Thread scaling is sub-linear,
so running two sessions on half the cores each usually beats running them
back-to-back — but not when one session is much shorter than the other, which
is exactly what the simulation catches.
"""

import heapq
import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from training.devices import HardwareProfile
from training.history import EVAL_WEIGHT, RunRecord

# ── Throughput model ──────────────────────────────────────────────────────────

# Empirical CPU baselines (training samples / sec) at the reference thread count
# and default workload (CART seq_len 512, JAW 224px). Deliberately conservative;
# replaced by calibrated numbers after the first real run.
CART_CPU_SAMPLES_PER_SEC = 12.0
JAW_CPU_SAMPLES_PER_SEC = 4.0
_BASELINE = {"cart": CART_CPU_SAMPLES_PER_SEC, "jaw": JAW_CPU_SAMPLES_PER_SEC}
_REF_THREADS = 4

GPU_SPEEDUP = 8.0
MPS_SPEEDUP = 4.0
_SPEEDUP = {"cpu": 1.0, "cuda": GPU_SPEEDUP, "mps": MPS_SPEEDUP}

_THREAD_EXP = 0.8  # throughput ∝ threads^0.8 on CPU
_SHARE_EXP = 0.9   # per-session throughput ∝ 1 / sharers^0.9 on a shared GPU
_MIN_THREADS_PER_SESSION = 2

# ── Memory model (GB) ─────────────────────────────────────────────────────────
# fixed: weights + grads + 2×AdamW moments (16 B/param) + runtime overhead
_CART_FIXED_GB = 1.4    # DistilBERT ~66M params
_JAW_FIXED_GB = 0.6     # EfficientNet-B3 ~12M params
_HOST_OVERHEAD_GB = 2.0  # per-process torch/CUDA context in host RAM (GPU mode)

RAM_BUDGET_FRACTION = 0.80
VRAM_BUDGET_FRACTION = 0.85


def _get(cfg: Any, key: str, default: Any) -> Any:
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def memory_footprint(model: str, cfg: Any) -> Tuple[float, float]:
    """(fixed_gb, per_sample_gb) for a training session."""
    if model == "cart":
        r = _get(cfg, "max_seq_length", 512) / 512
        # hidden activations scale with L, attention maps with L^2
        return _CART_FIXED_GB, 0.12 * r + 0.23 * r * r
    r = (_get(cfg, "image_size", 224) / 224) ** 2
    return _JAW_FIXED_GB, 0.18 * r


def peak_memory_gb(model: str, cfg: Any) -> float:
    fixed, per_sample = memory_footprint(model, cfg)
    return fixed + per_sample * _get(cfg, "train_batch_size", 16)


def max_batch_for(model: str, cfg: Any, budget_gb: float) -> int:
    fixed, per_sample = memory_footprint(model, cfg)
    return max(1, int((budget_gb - fixed) / per_sample))


def workload_scale(model: str, cfg: Any) -> float:
    """Per-sample cost relative to the default config."""
    if model == "cart":
        return (_get(cfg, "max_seq_length", 512) / 512) ** 1.2
    return (_get(cfg, "image_size", 224) / 224) ** 2


def total_units(model: str, cfg: Any, n_train: int, n_val: int) -> float:
    epochs = _get(cfg, "num_epochs_frozen", 0) + _get(cfg, "num_epochs_finetune", 0)
    units = epochs * (n_train + EVAL_WEIGHT * n_val)
    if model == "cart":
        units += EVAL_WEIGHT * n_val  # fusion: one inference pass over val
    return units


def _device_factor(device_kind: str, threads: int, share: int) -> float:
    if device_kind == "cpu":
        return (max(1, threads) / _REF_THREADS) ** _THREAD_EXP
    return 1.0 / max(1, share) ** _SHARE_EXP


def calibrate(history: Sequence[RunRecord], model: str, device_kind: str) -> Optional[Tuple[float, int]]:
    """Normalised throughput from up to 5 recent runs: (units/sec at reference, n_runs)."""
    norms = []
    for rec in reversed(history):
        if rec.model != model or rec.device_kind != device_kind or not rec.units_per_sec:
            continue
        if rec.units_done < 32:  # too little signal
            continue
        factor = _device_factor(device_kind, rec.threads, rec.concurrent)
        norms.append(rec.units_per_sec * workload_scale(model, rec.config) / factor)
        if len(norms) == 5:
            break
    return (median(norms), len(norms)) if norms else None


def predict_throughput(
    model: str,
    cfg: Any,
    device_kind: str,
    threads: int,
    share: int = 1,
    calibration: Optional[Tuple[float, int]] = None,
) -> float:
    ref = calibration[0] if calibration else _BASELINE[model] * _SPEEDUP[device_kind]
    return ref * _device_factor(device_kind, threads, share) / workload_scale(model, cfg)


# ── Planning ──────────────────────────────────────────────────────────────────

@dataclass
class SessionSpec:
    session_id: str
    label: str
    model: str
    config: Any  # CARTConfig | JAWConfig
    overrides: Dict[str, Any]
    n_train: int
    n_val: int
    checkpoint_dir: str


@dataclass
class Slot:
    device: str  # "cpu" | "cuda:0" | "mps"
    threads: int
    share: int = 1  # sessions sharing this physical device

    @property
    def kind(self) -> str:
        return self.device.split(":")[0]


@dataclass
class SessionEstimate:
    mem_gb: float
    units: float
    units_per_sec: float
    secs: float
    source: str  # "baseline" | "calibrated (n runs)"


@dataclass
class Candidate:
    parallel: int
    threads: int
    makespan: float


@dataclass
class Plan:
    device: str
    slots: List[Slot]
    estimates: Dict[str, SessionEstimate]
    order: List[str]  # launch order (longest first)
    makespan: float
    sequential_secs: float
    candidates: List[Candidate]
    mem_budget_gb: float
    warnings: List[str] = field(default_factory=list)

    @property
    def parallel(self) -> int:
        return len(self.slots)


def schedule_makespan(
    jobs: Sequence[Tuple[str, Any]],
    slots: Sequence[Slot],
    duration,
    busy_until: Optional[Sequence[float]] = None,
) -> Tuple[float, List[str]]:
    """
    Longest-processing-time list scheduling. `duration(job, slot)` gives the
    job's runtime on that slot; `busy_until` seeds slots already occupied.
    Returns (makespan, job ids in launch order).
    """
    if not slots:
        return 0.0, []
    free = list(busy_until) if busy_until else [0.0] * len(slots)
    ordered = sorted(jobs, key=lambda j: duration(j, slots[0]), reverse=True)
    for job in ordered:
        best = min(range(len(slots)), key=lambda i: free[i] + duration(job, slots[i]))
        free[best] += duration(job, slots[best])
    return max(free), [j[0] for j in ordered]


def _fit_count(mems: Sequence[float], budget: float) -> int:
    """How many sessions are guaranteed to fit together (largest ones first)."""
    total, count = 0.0, 0
    for m in sorted(mems, reverse=True):
        if total + m > budget:
            break
        total += m
        count += 1
    return max(1, count)


def _make_slots(hw: HardwareProfile, parallel: int) -> List[Slot]:
    cores = hw.physical_cores
    threads = max(1, cores // parallel)
    if hw.device == "cuda":
        n_gpus = max(1, len(hw.gpus))
        devices = [f"cuda:{i % n_gpus}" for i in range(parallel)]
        return [Slot(d, threads, share=devices.count(d)) for d in devices]
    return [Slot(hw.device, threads) for _ in range(parallel)]


def make_plan(
    specs: Sequence[SessionSpec],
    hw: HardwareProfile,
    history: Sequence[RunRecord],
    max_parallel: Optional[int] = None,
) -> Plan:
    device = hw.device
    warnings: List[str] = []
    mems = {s.session_id: peak_memory_gb(s.model, s.config) for s in specs}
    units = {s.session_id: total_units(s.model, s.config, s.n_train, s.n_val) for s in specs}
    calib = {m: calibrate(history, m, device) for m in {s.model for s in specs}}
    by_id = {s.session_id: s for s in specs}

    # ── memory-bound concurrency ───────────────────────────────────────────
    ram_budget = RAM_BUDGET_FRACTION * hw.ram_available_gb
    if device == "cuda":
        n_gpus = max(1, len(hw.gpus))
        vram_free = [g.free_gb for g in hw.gpus] or [float("inf")]
        mem_budget = VRAM_BUDGET_FRACTION * min(vram_free)
        per_gpu = _fit_count(list(mems.values()), mem_budget)
        host_cap = max(1, int(ram_budget // _HOST_OVERHEAD_GB))
        mem_cap = min(per_gpu * n_gpus, host_cap)
    elif device == "mps":
        mem_budget = ram_budget  # unified memory; one session keeps it predictable
        mem_cap = 1
    else:
        mem_budget = ram_budget
        mem_cap = _fit_count(list(mems.values()), mem_budget)

    for s in specs:
        if mems[s.session_id] > mem_budget:
            fixed, per_sample = memory_footprint(s.model, s.config)
            hint = (
                f"try --batch-size {max_batch_for(s.model, s.config, mem_budget)}"
                if fixed + per_sample <= mem_budget
                else "won't fit even at batch size 1; free memory or expect heavy swapping"
            )
            warnings.append(
                f"{s.label}: needs ~{mems[s.session_id]:.1f} GB but only {mem_budget:.1f} GB is available — {hint}"
            )

    core_cap = max(1, hw.physical_cores // _MIN_THREADS_PER_SESSION) if device == "cpu" else len(specs)
    kmax = max(1, min(len(specs), mem_cap, core_cap, max_parallel or len(specs)))

    def duration(job, slot: Slot) -> float:
        sid = job[0]
        spec = by_id[sid]
        ups = predict_throughput(spec.model, spec.config, slot.kind, slot.threads, slot.share, calib[spec.model])
        return units[sid] / ups

    jobs = [(s.session_id, None) for s in specs]
    candidates: List[Candidate] = []
    best: Optional[Tuple[float, int, List[str]]] = None
    for k in range(1, kmax + 1):
        slots = _make_slots(hw, k)
        makespan, order = schedule_makespan(jobs, slots, duration)
        candidates.append(Candidate(k, slots[0].threads, makespan))
        # prefer fewer sessions unless parallelism buys at least 3%
        if best is None or makespan < best[0] * 0.97:
            best = (makespan, k, order)

    assert best is not None
    makespan, k, order = best
    slots = _make_slots(hw, k)
    single = _make_slots(hw, 1)[0]

    estimates: Dict[str, SessionEstimate] = {}
    for s in specs:
        slot = slots[0]
        ups = predict_throughput(s.model, s.config, slot.kind, slot.threads, slot.share, calib[s.model])
        c = calib[s.model]
        estimates[s.session_id] = SessionEstimate(
            mem_gb=mems[s.session_id],
            units=units[s.session_id],
            units_per_sec=ups,
            secs=units[s.session_id] / ups,
            source=f"calibrated ({c[1]} run{'s' if c[1] != 1 else ''})" if c else "baseline",
        )

    sequential = sum(duration((s.session_id, None), single) for s in specs)
    return Plan(
        device=device,
        slots=slots,
        estimates=estimates,
        order=order,
        makespan=makespan,
        sequential_secs=sequential,
        candidates=candidates,
        mem_budget_gb=mem_budget,
        warnings=warnings,
    )


def projected_finish(running_remaining: Sequence[float], n_slots: int, queued_secs: Sequence[float]) -> float:
    """Seconds until everything finishes, given busy slots and queued job durations."""
    free = sorted(list(running_remaining) + [0.0] * max(0, n_slots - len(running_remaining)))
    heapq.heapify(free)
    for secs in sorted(queued_secs, reverse=True):
        heapq.heappush(free, heapq.heappop(free) + secs)
    return max(free) if free else 0.0


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "—"
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m"
