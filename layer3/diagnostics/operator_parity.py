"""Offline operator-parity harness (codegen ↔ proxy).

PURPOSE
-------
The proxy evaluator (``layer2.evaluator_vectorized``) is the GP *objective
function*; the generated QuantConnect algorithm (``layer3.codegen``) is the
*serving* environment. When a proxy-good strategy realises a different signal in
QC, the divergence lives in exactly one of two places:

  (1) OPERATOR / EVAL-SCHEDULE skew — the bar-by-bar Lag/Delta/Cross/GT logic, or
      *when* each tree is evaluated (codegen evals entry only when flat, exit only
      when in-position, so nested Lag/Delta buffers go gappy; the proxy evaluates
      both trees continuously over every bar). This is testable OFFLINE.

  (2) TERMINAL-VALUE skew — VIXChange / RealizedVol30m_5m / … computed live in QC
      differ from the proxy's l1_minute_scalars. This needs QC logging.

This harness ISOLATES (1) from (2): it feeds the codegen's REAL operator methods
(extracted from the actually-generated file — zero re-implementation drift) the
proxy's OWN per-bar terminal values, and checks whether the resulting entry/exit
booleans match the proxy's vectorized signal arrays. If they match here, the
divergence is 100% terminal-VALUE (locus 2 → QC). If they DON'T, we have found an
offline-fixable codegen bug and the exact bar/terminal where it bites.

No QC slots are consumed. Reusable across ALL templates via ``StrategySpec``.
"""
from __future__ import annotations

import sys
import types
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from layer2.evaluator_vectorized import (
    VectorizedTreeEvaluator,
    prepare_terminal_data,
    ENTRY_CUTOFF_MTC,
    EOD_FORCE_CLOSE_MTC,
)
from layer2.grammar import from_sexpr
from layer2.templates import template_by_name
from layer2.terminal_stats import compute_norm_stats_from_data
from layer3.codegen import generate_qc_algorithm

MAX_TRADES_PER_DAY = 8  # both proxy (evaluator_vectorized:1126) and codegen
MIN_BARS_IN_TRADE = 15
MAX_BARS_IN_TRADE = 330


# ---------------------------------------------------------------------------
# Fake AlgorithmImports so the generated module can be exec'd offline.
# The generated code only references QC names (Resolution, CBOE, …) INSIDE
# method bodies we never call (Initialize / OnData / _open_position); the only
# import/def-time reference is the base class ``QCAlgorithm``. So a permissive
# stub module is sufficient to DEFINE the class and call its pure operator
# methods (_lag/_delta/_lag_expr/_eval_entry/…), which touch only self._buffers,
# self.MAX_LAG, self._curr/prev_eval and the stdlib (math, deque).
# ---------------------------------------------------------------------------
class _Dummy:
    """Permissive stand-in for any QC symbol referenced at class-def time."""
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Dummy()
    def __getattr__(self, _n): return _Dummy()


def _install_fake_algorithm_imports() -> None:
    if "AlgorithmImports" in sys.modules:
        return
    mod = types.ModuleType("AlgorithmImports")
    names = [
        "QCAlgorithm", "Resolution", "OptionRight", "Market", "CBOE", "Symbol",
        "SecurityType", "ConstantFeeModel", "NullBuyingPowerModel", "TimeSpan",
        "OrderStatus", "BrokerageName", "AccountType", "DataNormalizationMode",
        "InsightDirection", "Insight", "PortfolioTarget", "SetFeeModel",
    ]
    for n in names:
        setattr(mod, n, _Dummy if n == "QCAlgorithm" else _Dummy())
    # Stable enum-likes: equality must hold across accesses (QC enums are stable;
    # the catch-all _Dummy returns a fresh object each time, which breaks
    # `contract.Right == OptionRight.Call` filters in offline exec).
    mod.OptionRight = types.SimpleNamespace(Call="call", Put="put")
    mod.SecurityType = types.SimpleNamespace(IndexOption="indexoption", Index="index")
    mod.OrderStatus = types.SimpleNamespace(Filled="filled", Submitted="submitted")
    mod.__all__ = names
    sys.modules["AlgorithmImports"] = mod


# ---------------------------------------------------------------------------
# Strategy spec
# ---------------------------------------------------------------------------
@dataclass
class StrategySpec:
    strategy_id: str
    template_name: str
    entry_sexpr: str
    exit_sexpr: str
    size_sexpr: str
    delta_sexpr: Optional[str] = None
    is_credit: bool = True  # for the scheduled trade reconstruction (no pricing)


# ---------------------------------------------------------------------------
# Build a bare codegen instance whose REAL methods we can call offline.
# ---------------------------------------------------------------------------
def build_codegen_instance(spec: StrategySpec, window: Tuple[str, str],
                           norm_serial: Dict) -> object:
    _install_fake_algorithm_imports()
    code = generate_qc_algorithm(
        strategy_id=spec.strategy_id, template_name=spec.template_name,
        entry_sexpr=spec.entry_sexpr, exit_sexpr=spec.exit_sexpr,
        size_sexpr=spec.size_sexpr, delta_sexpr=spec.delta_sexpr,
        start_date=window[0], end_date=window[1], base_contracts=1,
        norm_stats=norm_serial,
    )
    ns: Dict = {}
    exec(compile(code, f"<gen:{spec.strategy_id}>", "exec"), ns)
    cls = next(v for k, v in ns.items()
               if isinstance(v, type) and k.startswith("GP_"))
    inst = cls.__new__(cls)            # bypass Initialize / QC base __init__
    inst._buffers = {}
    # Per-trade UnrealizedProfitPct history (mirror Initialize) — required by the
    # _lag/_delta UPP special-case; without it Lag/Delta(UnrealizedProfitPct)
    # AttributeErrors through this harness.
    inst._upp_hist = deque(maxlen=inst.MAX_LAG + 1)
    # Position state (mirror Initialize default) — the _lag/_delta UPP branch gates
    # on it (UPP is the 0 baseline while flat). Tests set it True to read _upp_hist.
    inst._position_open = False
    inst._prev_eval = {}
    inst._curr_eval = {}
    inst._bar_count = 0
    return inst, code


# ---------------------------------------------------------------------------
# Codegen buffer discipline — faithfully mirrors codegen.py:_on_bar
# • day boundary: clear ALL self._buffers (P0-7b), reset prev/curr_eval
# • append this bar's terminal values to self._buffers (561-567)
# • append DeltaSpread1/5 separately (578-581)
# • rotate prev/curr_eval after eval (731-732)
# `s` carries the PROXY's per-bar terminal values (the controlled input), so any
# resulting boolean difference is attributable to OPERATORS / EVAL SCHEDULE only.
# ---------------------------------------------------------------------------
def _new_day_reset(inst) -> None:
    for b in inst._buffers.values():
        b.clear()
    inst._prev_eval = {}
    inst._curr_eval = {}


def _append_terminal_buffers(inst, s: Dict[str, float]) -> None:
    for k, v in s.items():
        if k.startswith("_raw_") or k in ("DeltaSpread1", "DeltaSpread5"):
            continue
        inst._buffers.setdefault(k, deque(maxlen=inst.MAX_LAG + 1)).append(v)
    for ds in ("DeltaSpread1", "DeltaSpread5"):
        if ds in s:
            inst._buffers.setdefault(ds, deque(maxlen=inst.MAX_LAG + 1)).append(s[ds])


def _scalar_names(td: Dict[str, np.ndarray]) -> List[str]:
    return [k for k, v in td.items() if isinstance(v, np.ndarray) and v.ndim == 1]


# ---------------------------------------------------------------------------
# Harness A — CONTINUOUS operator check.
# Evaluate BOTH trees every bar with continuous buffers, on the proxy's terminal
# values, and compare to the proxy's vectorized signal arrays. Isolates the pure
# operator semantics (Lag/Delta/Cross/GT) from the deployment schedule. This is
# DELIBERATELY schedule-free — no WARMUP gate, no position gating — because its
# job is to test the operators in isolation; the proxy ground-truth it compares
# against is itself evaluated continuously over all bars. The deployment schedule
# (warmup + entry-when-flat / exit-when-in-position) is modelled separately in
# run_scheduled_codegen. The match-rate is reported over the in-window mask only,
# which excludes the warmup days.
# ---------------------------------------------------------------------------
def run_continuous(inst, td: Dict[str, np.ndarray],
                   dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    names = _scalar_names(td)
    n = len(dates)
    e_out = np.zeros(n, dtype=np.float64)
    x_out = np.zeros(n, dtype=np.float64)
    inst._buffers = {}
    inst._prev_eval = {}
    inst._curr_eval = {}
    prev_date = None
    for i in range(n):
        d = dates[i]
        if prev_date is not None and d != prev_date:
            _new_day_reset(inst)
        prev_date = d
        s = {k: float(td[k][i]) for k in names}
        _append_terminal_buffers(inst, s)
        try:
            e_out[i] = 1.0 if inst._eval_entry(s) else 0.0
        except Exception:
            e_out[i] = np.nan
        try:
            x_out[i] = 1.0 if inst._eval_exit(s) else 0.0
        except Exception:
            x_out[i] = np.nan
        inst._prev_eval = dict(inst._curr_eval)
        inst._curr_eval = {}
    return e_out, x_out


# ---------------------------------------------------------------------------
# Harness B — SCHEDULED (deployed) reconstruction.
# Reproduce the codegen's REAL eval schedule: entry only when flat, exit only when
# in-position (after min-hold). This makes nested-Lag/Delta expr-buffers GAPPY,
# exactly as deployed. Returns the list of entry-bar indices.
# Stop-loss is intentionally EXCLUDED (needs pricing); both this and the proxy
# reconstruction below model exits via signal / max-hold / EOD only, so the
# comparison isolates SIGNAL + schedule, not PnL.
# ---------------------------------------------------------------------------
def run_scheduled_codegen(inst, td: Dict[str, np.ndarray], dates: np.ndarray,
                          mtc: np.ndarray) -> List[int]:
    """Faithful model of the FIXED codegen _on_bar schedule (2026-05-31): ALL
    trees are evaluated EVERY bar (continuous Lag/Delta expr-buffers), and only
    the ACTION is gated by position state. This mirrors codegen.py:_on_bar after
    the parity fix; pre-fix it evaluated entry only when flat / exit only when
    in-position, which gapped the buffers and over-fired."""
    names = _scalar_names(td)
    n = len(dates)
    inst._buffers = {}
    inst._prev_eval = {}
    inst._curr_eval = {}
    prev_date = None
    in_pos = False
    bars_held = 0
    pending = False
    trades_today = 0
    entries: List[int] = []
    for i in range(n):
        d = dates[i]
        new_day = prev_date is not None and d != prev_date
        if new_day:
            _new_day_reset(inst)
            trades_today = 0
            pending = False
            in_pos = False  # day-boundary force close (proxy:1152)
        prev_date = d
        s = {k: float(td[k][i]) for k in names}
        _append_terminal_buffers(inst, s)          # terminal buffers ALWAYS update
        inst._bar_count += 1
        # WARMUP gate — mirror codegen.py:_on_bar, which appends terminal buffers
        # but skips ALL tree evaluation while _bar_count < WARMUP_BARS (so expr
        # buffers do not fill during warmup). bar_count persists across days, so
        # this bites only the very first WARMUP_BARS of the whole run.
        if inst._bar_count < getattr(inst, "WARMUP_BARS", 0):
            inst._prev_eval = dict(inst._curr_eval)
            inst._curr_eval = {}
            continue
        # eval ALL trees every bar -> continuous expr-buffers (the fix)
        try:
            sig_entry = bool(inst._eval_entry(s))
        except Exception:
            sig_entry = False
        try:
            sig_exit = bool(inst._eval_exit(s))
        except Exception:
            sig_exit = False
        try:
            inst._eval_size(s)                     # keep size buffer continuous
            inst._eval_delta(s)                    # keep delta buffer continuous
        except Exception:
            pass
        if not in_pos:
            if mtc[i] <= ENTRY_CUTOFF_MTC:
                pending = False
            elif trades_today >= MAX_TRADES_PER_DAY:
                pending = False
            elif pending:
                pending = False
                in_pos = True
                bars_held = 0
                trades_today += 1
                entries.append(i)
            elif sig_entry:
                pending = True
        else:
            bars_held += 1
            if bars_held >= MAX_BARS_IN_TRADE:
                in_pos = False
            elif bars_held >= MIN_BARS_IN_TRADE and sig_exit:
                in_pos = False
        inst._prev_eval = dict(inst._curr_eval)
        inst._curr_eval = {}
    return entries


def run_scheduled_proxy(entry_sig: np.ndarray, exit_sig: np.ndarray,
                        dates: np.ndarray, mtc: np.ndarray) -> List[int]:
    """Same gates on the proxy's precomputed (continuous) signal arrays."""
    n = len(dates)
    prev_date = None
    in_pos = False
    bars_held = 0
    pending = False
    trades_today = 0
    entries: List[int] = []
    for i in range(n):
        d = dates[i]
        if prev_date is not None and d != prev_date:
            trades_today = 0
            pending = False
            in_pos = False
        prev_date = d
        if not in_pos:
            if mtc[i] <= ENTRY_CUTOFF_MTC:
                pending = False
            elif trades_today >= MAX_TRADES_PER_DAY:
                pending = False
            elif pending:
                pending = False
                in_pos = True
                bars_held = 0
                trades_today += 1
                entries.append(i)
            elif entry_sig[i] > 0.5:
                pending = True
        else:
            bars_held += 1
            if bars_held >= MAX_BARS_IN_TRADE:
                in_pos = False
            elif bars_held >= MIN_BARS_IN_TRADE and exit_sig[i] > 0.5:
                in_pos = False
    return entries


# ---------------------------------------------------------------------------
# Top-level diagnose
# ---------------------------------------------------------------------------
def diagnose(spec: StrategySpec, parquet: str, window: Tuple[str, str],
             train_end: str, warmup_start: Optional[str] = None) -> Dict:
    from layer2.io import load_minute_parquet
    df, _ = load_minute_parquet(parquet)
    norm = compute_norm_stats_from_data(df[df["date"] <= train_end])
    norm_serial = {k: [float(v[0]), float(v[1])] + ([v[2]] if len(v) >= 3 else [])
                   for k, v in norm.items()}

    lo = warmup_start or window[0]
    sl = df[(df["date"] >= lo) & (df["date"] <= window[1])].reset_index(drop=True)
    td = prepare_terminal_data(sl, norm_stats_override=norm)
    dates = sl["date"].astype(str).values
    mtc = (sl["MinutesToClose"].values if "MinutesToClose" in sl.columns
           else np.full(len(sl), 999.0))
    n = len(sl)

    # within-day positions (mirror vectorized_backtest:990-1003)
    within = np.zeros(n, dtype=np.int64)
    day_mask = np.ones(n, dtype=np.float64)
    day_mask[0] = 0.0
    pos = 0
    for i in range(1, n):
        if dates[i] != dates[i - 1]:
            day_mask[i] = 0.0
            pos = 0
        else:
            pos += 1
        within[i] = pos

    ev = VectorizedTreeEvaluator(td, n, day_boundary_mask=day_mask,
                                 within_day_pos=within)
    entry_sig = np.nan_to_num(ev.evaluate(from_sexpr(spec.entry_sexpr)).astype(float))
    exit_sig = np.nan_to_num(ev.evaluate(from_sexpr(spec.exit_sexpr)).astype(float))

    inst, code = build_codegen_instance(spec, window, norm_serial)

    # A: continuous operator parity
    e_cont, x_cont = run_continuous(inst, td, dates)

    in_win = np.array([d >= window[0] for d in dates])  # exclude warmup days
    e_match = (e_cont > 0.5) == (entry_sig > 0.5)
    x_match = (x_cont > 0.5) == (exit_sig > 0.5)
    e_rate = float(np.mean(e_match[in_win]))
    x_rate = float(np.mean(x_match[in_win]))
    first_div = None
    div_idx = np.where(~e_match & in_win)[0]
    if len(div_idx):
        j = int(div_idx[0])
        first_div = {
            "bar": j, "date": dates[j], "within_day": int(within[j]),
            "proxy_entry": float(entry_sig[j]), "codegen_entry": float(e_cont[j]),
        }

    # B: scheduled (deployed) reconstruction
    cg_entries = run_scheduled_codegen(inst, td, dates, mtc)
    px_entries = run_scheduled_proxy(entry_sig, exit_sig, dates, mtc)

    def _per_day(entries):
        from collections import Counter
        c = Counter(dates[i][:10] for i in entries if dates[i] >= window[0])
        return c

    cg_days = _per_day(cg_entries)
    px_days = _per_day(px_entries)
    shared = set(cg_days) & set(px_days)

    return {
        "strategy_id": spec.strategy_id,
        "n_bars": n,
        "continuous_entry_match_rate": e_rate,
        "continuous_exit_match_rate": x_rate,
        "first_entry_divergence": first_div,
        "scheduled_proxy_entries": len(px_entries),
        "scheduled_codegen_entries": len(cg_entries),
        "scheduled_proxy_active_days": len(px_days),
        "scheduled_codegen_active_days": len(cg_days),
        "scheduled_shared_days": len(shared),
        "scheduled_proxy_per_active_day": (
            len(px_entries) / max(len(px_days), 1)),
        "scheduled_codegen_per_active_day": (
            len(cg_entries) / max(len(cg_days), 1)),
        "px_days": dict(px_days),
        "cg_days": dict(cg_days),
    }
