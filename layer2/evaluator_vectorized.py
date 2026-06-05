"""Vectorized GP tree evaluator — evaluates expression trees over NumPy arrays.

PnL computation uses Edgeworth-corrected option pricing (Bandi, Fusari &
Reno 2024) with empirical IV surface interpolation from 11-point delta grid
when grid IVs are available; falls back to Black-Scholes with parametric vol
skew (slope=-0.15) when grid data is absent. Delta-to-strike with $5 grid +
separation, gross-value sizing (debit/ratio-backspread uses max(gross, debit)),
leverage cap, calibrated delta-dependent costs, margin gate (debit/ratio-
backspread uses wing width), equity floor at 10% notional. M2M tracks from
raw entry value (phantom loss fix).

Session boundary force-close, daily trade cap, 1-bar execution delay, and
EOD timing all match QC backtest behavior (5 gap fixes).

L2 grammar fix: enables 1-minute resolution evaluation (405 bars/day x 882
days = 357K bars) by replacing the per-bar recursive tree walker with
vectorized array operations. Supports Conditions A-D:
  A (scalar-only): 1-D terminal arrays only
  B (probes): + probe scalar terminals (PredRV15, etc.)
  C (embeddings): + 2-D typed-vector terminals (N, 384) + embedding operators
  D (full): all of the above

Each terminal -> NumPy array over all bars.
GT(A, B) -> A > B element-wise.
Lag(term, k) -> np.roll(array, k) with NaN fill.
CrossAbove(A, B) -> (A > B) & (shift(A,1) <= shift(B,1)).
IfThenElse(cond, t, f) -> np.where(cond, t, f).
EmbNorm(vec) -> L2 norm per bar (N,D) -> (N,).
EmbProj(vec, k) -> PCA projection per bar (N,D) -> (N,).

The backtester position loop remains sequential (state-dependent), but
processes vectorized entry/exit signal arrays.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from layer2.grammar import (
    FuncNode, GType, Node, TermNode, to_str,
)
from layer2.terminal_stats import NORMALIZED_TERMINALS, normalize
from layer2.evaluator import BacktestResult, Trade, _safe_real, _load_pca_bases_cached


# ---------------------------------------------------------------------------
# Change A: Per-terminal noise injection for proxy-QC robustness
# ---------------------------------------------------------------------------
# Per-terminal noise sigma calibrated from proxy-QC terminal comparison
# (compare_proxy_qc_terminals.py, Jan 2025 diagnostic). Values are MAE in
# normalized space between proxy's spline/derived terminals and QC's
# contract-level/CBOE terminals. Adding Gaussian noise at these magnitudes
# prevents GP from overfitting to proxy-specific terminal values that diverge
# from QC reality.
_TERMINAL_NOISE_SIGMA = {
    "ATM_IV": 0.19,           # spline vs chain IV difference
    "ATM_IV_5m": 0.19,
    "VIXSpot": 1.15,          # CBOE VIX vs derived VIX
    "VIXChange": 0.59,        # IV-scale measurement noise
    "RealizedVol30m": 0.52,
    "RealizedVol30m_5m": 0.36,
    "RawSpread": 0.39,        # spline vs single-contract spread
    "RawSpread_5m": 0.51,
    "PutCallSkew": 1.44,      # spline vs contract skew
    "OvernightGap": 0.95,     # price source difference
    "GridReliability": 0.73,
    "SessionReturn": 0.95,
    "VIXTermSlope": 3.01,     # CBOE vs futures
    "VIXMean5d": 1.70,
    "RV5d": 1.31,
    "IVRVGap5d": 2.22,
    "SessionPosition": 0.31,
    "SPXReturn3d": 0.06,      # well-matched, low noise
    # Deterministic/derived terminals — no proxy-QC divergence
    "BarOfDay": 0.0,
    "MinutesToClose": 0.0,
    "DeltaSpread1": 0.0,      # derived from RawSpread (already noised)
    "DeltaSpread5": 0.0,      # derived from RawSpread (already noised)
    "ThetaUrgency": 0.0,      # deterministic: 1/sqrt(MTC), no proxy-QC divergence
    # IV grid columns: these are raw grid IVs used for spline interpolation.
    # They come from the same L1 Parquet as ATM_IV (no separate QC computation),
    # so proxy-QC divergence is already captured by ATM_IV noise. Adding noise
    # here corrupts the pricing surface without improving robustness.
    "IV_ATM": 0.0, "IV_ATMp": 0.0, "IV_ATMc": 0.0,
    "IV_5Dp": 0.0, "IV_5Dc": 0.0,
    "IV_10Dp": 0.0, "IV_10Dc": 0.0,
    "IV_25Dp": 0.0, "IV_25Dc": 0.0,
    "IV_40Dp": 0.0, "IV_40Dc": 0.0,
    # SPXClose: price level, not a GP signal terminal. No proxy-QC divergence.
    "SPXClose": 0.0,
}
# Default for terminals not in the calibration dict. Set to 0.0 because
# all terminals that GP can access are either (a) in the dict above with
# calibrated sigmas, or (b) structural (IV grid, SPXClose) with sigma=0.
# Any terminal hitting this default is either a test artifact or a new
# terminal that was added without noise calibration — sigma=0 is the
# safe default (no uncalibrated noise).
_DEFAULT_NOISE_SIGMA = 0.0
# MAE -> Gaussian-sigma conversion (2026-06-01 audit): the stored noise values are
# MAEs; for a normal, MAE ≈ 0.8·σ, so multiply by 1/0.8 = 1.25 to inject the correct
# std. (The stored values are also STALE — see _add_terminal_noise docstring.)
_MAE_TO_SIGMA = 1.25

# Per-terminal DIRECTIONAL bias (mean offset) for G11 (MH3, 2026-06-01 holistic
# review). The proxy↔QC terminal gap is NOT zero-mean: a SYSTEMATIC offset (e.g. QC's
# live option-chain IV running consistently above/below the History()-built parquet)
# crosses entry/exit thresholds in a way symmetric jitter cannot. This dict carries the
# measured per-terminal mean offset (normalized units); _add_terminal_noise shifts the
# terminal by it BEFORE adding the symmetric sigma noise, so G11 tests robustness to the
# REAL (biased) gap, not just variance. EMPTY until recalibrated against a current-
# codegen QC terminal trace (the same trace that refreshes the STALE sigmas) — until
# then offset=0 and G11 remains a (weaker) variance-only screen. Populate together with
# _TERMINAL_NOISE_SIGMA from the fresh QC trace before enabling G11 in a confirmatory run.
_TERMINAL_NOISE_OFFSET: dict = {}
_DEFAULT_NOISE_OFFSET = 0.0

# Maximum spread width ($40) matching QC codegen's $40 cap. GP strategies
# with wider spreads get rejected in QC but pass in proxy, creating false
# positive strategies. (Change C)
MAX_SPREAD_WIDTH = 40.0

# Minimum per-leg bid-ask crossing cost ($/share), modeled (2026-06-01 audit). Real
# SPX option spreads are a roughly fixed dollar width, so a spread-crossing cost that
# is purely a fraction of EXTRINSIC value collapses to ~0 on deep-ITM exits — too
# optimistic on the losing tail. This floors the crossing. NOT QC-measured (no
# bid-ask export); conservative-leaning. Tighten when QC bid-ask data is available.
_MIN_SPREAD_COST_PER_LEG = 0.10

# Sortino guard (2026-05-31, adversarially-confirmed fix): the downside-deviation
# 1e-9 floor + the zero-down-day 10.0 free-pass let few-trade strategies with
# few/no losing DAYS post astronomical Sortinos (17 run strategies > 1000, one
# at 3662 on 37 trades) that dominated objective-1. Require a minimum number of
# losing days to trust the downside deviation, and cap |Sortino| at a plausible
# bound. (The Sharpe std 1e-9 floor was checked and does NOT bind in practice —
# daily aggregation over ~250 days neutralizes it — so it is left unchanged.)
_MAX_SORTINO = 10.0       # ann-Sortino > 10 is a floor artifact, not signal
_MIN_DOWNSIDE_OBS = 5     # need >= 5 losing DAYS for a trustworthy downside dev
# M3 fix (2026-06-02): when there are too few losing DAYS to estimate the
# downside deviation empirically, floor the downside-dev denominator at this
# fraction of the OVERALL (symmetric) daily-return std — a magnitude-tied floor.
# This keeps Sortino a genuine, BOUNDED downside measure that is INDEPENDENT of
# Sharpe (which divides by the full symmetric std), instead of the old
# clip(sharpe) fallback (perfectly collinear with Sharpe) or the 1e-9 artifact
# floor (manufactured astronomical Sortinos). 1/3 keeps the denominator strictly
# below the symmetric std so a downside-light series still scores a HIGHER (but
# capped) Sortino than Sharpe, while a left-skewed series' realized downside dev
# exceeds the floor and dominates, driving Sortino BELOW Sharpe — the correct,
# independent risk ordering.
_DOWNSIDE_DEV_STD_FRACTION = 1.0 / 3.0
# 2026-06-02 (finding #2): subtractive penalty for a vestigial (>90% max-hold)
# exit tree. Replaces the old HARD -1e6 sentinel that killed the legitimate
# hold-to-close theta-harvest strategy (a selective-entry credit spread held to
# settlement — the canonical 0DTE play). Subtractive (not -1e6, not multiplicative)
# preserves NSGA gradient and lets a genuinely profitable strategy survive on its
# real Sharpe while pushing a hold-to-LOSS degenerate below zero.
_MAX_HOLD_VESTIGIAL_PENALTY = 1.0

# Stop-loss execution discount: QC fills stop-loss orders at worse prices
# than BS mid. This 0.85x factor tightens the proxy's stop-loss trigger
# to match QC's effective stop-loss level, preventing GP from evolving
# strategies that rely on recovery between proxy's wider stop and QC's
# tighter stop. Calibrated from QC v5 reconciliation data. (Change D)
STOP_LOSS_EXECUTION_DISCOUNT = 0.85

# HIGH M1-fin (2026-06-01 holistic review): adverse-fill markup on the REALIZED
# stop-loss PnL. A stop fills DURING the fast adverse move that triggered it, so the
# actual fill is worse than the BS-mid the proxy marks (close_val). Distinct from the
# _sl_slippage on exit_cost (which only widens the bid-ask spread crossing): this
# models the mid itself filling against us. Replaces a DEAD 1.15x `unrealised` markup that
# was discarded before the realized PnL was computed (so stop losses were optimistic).
# 1.10 is conservative vs the QC v5 reconciliation (avg loss 2.3x proxy); the
# remaining gap is intentionally left to the spread slippage + credit haircuts.
STOP_LOSS_FILL_MARKUP = 1.10

# EOD time gates — SINGLE SOURCE OF TRUTH for the shared proxy<->QC trading
# window (R1/R2 parity fix, 2026-05-29). MinutesToClose is measured to the SPXW
# PM-settlement at 16:15 ET (research_collector.py:626-629), NOT the 16:00 cash
# close — so an mtc threshold maps to a wall-clock time as (16:15 - mtc):
# EOD_FORCE_CLOSE_MTC = 25 -> 16:15 - 25m = 15:50 ET force-close
# ENTRY_CUTOFF_MTC = 30 -> 16:15 - 30m = 15:45 ET last new entry
# These are pinned to codegen's wall-clock gates (codegen.py:509-510) by
# tests/test_codegen_exec_parity.py so the training (proxy) and serving (QC)
# environments enter/exit at the SAME clock time (Sculley et al., 2015 — any
# divergence is training-serving skew). Chosen (user-approved 2026-05-29) to keep
# BOTH environments OUT of the 16:00-16:15 SPXW settlement window, where BS
# pricing is least reliable (gamma explosion + the settlement auction). The
# prior values (mtc<=10 / mtc<=15 == 16:05 / 16:00 ET) ran ~15 min later than
# QC's 16:00 market-hours guard and were asymmetrically proxy-optimistic.
EOD_FORCE_CLOSE_MTC = 25.0
ENTRY_CUTOFF_MTC = 30.0

# Position-aware exit terminal (Phase-2, docs/viable_run_plan_2026_06_03.md).
# UnrealizedProfitPct = (curr_position_value - entry_net_value) / |entry_net_value|
# = the fraction of the structure's MAX PROFIT captured so far, computed PER
# BAR while in a position. For a CREDIT spread entry_net_value < 0 (the credit
# received) and curr_position_value rises toward 0 as the spread decays, so the
# ratio runs ≈0 at entry → +1 at max profit (spread worthless) → negative when
# the spread moves against you. For a DEBIT spread entry_net_value > 0 and the
# SAME formula gives (current_value - debit)/debit. The n_contracts and the $100
# multiplier cancel in the ratio, so this matches QC's unrealised_pnl /
# |entry_credit| exactly (the codegen parity bridge, layer3/codegen.py).
# It is POSITION-DEPENDENT: undefined when flat (then 0.0) and impossible to
# precompute as a vectorized array. The terminal is seeded as an all-zeros array
# (the flat-state / entry-tree value) and the LIVE per-bar value is substituted
# into exit-tree evaluation by _eval_exit_tree_with_upp during the in-position
# loop — only for exit trees that actually reference it (tree_references_upp).
# RAW (un-normalized, NOT in TERMINAL_NORM_STATS): the fraction is already a
# natural bounded scale, so GT(UnrealizedProfitPct, EphReal(0.5)) literally means
# "captured ≥ 50% of max profit".
UNREALIZED_PROFIT_PCT = "UnrealizedProfitPct"
# Denominator floor: |entry_net_value| can be a few cents for a near-worthless
# spread; floor it so the ratio cannot blow up (and a 0-credit entry → 0.0).
_UPP_DENOM_FLOOR = 1e-6


def _add_terminal_noise(terminal_data: dict, rng: np.random.RandomState,
                        noise_scale: float = 1.0) -> dict:
    """Add calibrated Gaussian noise to terminal values for robustness.

    Noise magnitudes are per-terminal, from measured proxy-QC divergence (MAE in
    normalized space). Used ONLY by the G11 transferability gate (OFF by default).

    ⚠ TWO AUDIT CAVEATS (2026-06-01), relevant only when G11 is enabled in a
    confirmatory run:
      1. MAE != sigma. For a normal, MAE ≈ 0.8·σ, so the raw MAE UNDER-states the
         divergence's std. We multiply by _MAE_TO_SIGMA (=1.25) below to inject the
         correct std.
      2. The _TERMINAL_NOISE_SIGMA values are STALE: measured on a since-fixed codegen
         (ThetaUrgency/VIXTermSlope/Sqrt/VIXChange bugs) and an OVERTURNED ATM_IV
         divergence theory (it was a collector-BS-IV vs QC-spline model artifact, now
         fixed by P0-4). RECALIBRATE against current-codegen proxy↔QC terminal MAE
         (requires a QC terminal-trace export) BEFORE relying on G11 in production.
         The directional-bias MECHANISM now exists (_TERMINAL_NOISE_OFFSET, applied
         below) so the gap's systematic component — which crosses thresholds that
         symmetric noise cannot — can be modeled; but the offset dict is EMPTY until
         the same QC-trace recalibration populates it, so today G11 is still a ZERO-
         MEAN variance-only screen (a weaker claim than true OOS transfer). Treat its
         verdict as advisory until BOTH the sigmas and the offsets are recalibrated.

    Args:
        terminal_data: dict of terminal_name -> numpy array (modified in-place
            on a COPY of the values, original arrays are not modified).
        rng: numpy RandomState for reproducible noise.
        noise_scale: multiplier on all sigmas (0.0 = no noise, 1.0 = calibrated).

    Returns:
        New dict with noisy terminal values (original arrays untouched).
    """
    if noise_scale <= 0:
        return terminal_data
    noisy = {}
    for name, values in terminal_data.items():
        if name.startswith('_'):
            # Skip internal keys
            noisy[name] = values
            continue
        if values.ndim != 1:
            # Skip 2-D typed-vector embeddings (noise on 384-d vectors
            # requires different calibration — embeddings don't have
            # proxy-QC divergence data).
            noisy[name] = values
            continue
        # _MAE_TO_SIGMA converts the stored per-terminal MAE to the Gaussian std
        # (MAE ≈ 0.8·σ for a normal). See the docstring's audit caveats.
        sigma = _TERMINAL_NOISE_SIGMA.get(name, _DEFAULT_NOISE_SIGMA) * _MAE_TO_SIGMA * noise_scale
        # MH3 (2026-06-01): apply the measured DIRECTIONAL bias (mean offset) too, so
        # G11 tests robustness to the real (biased) proxy↔QC gap, not just variance.
        # offset=0 (default) until recalibrated → reduces to the prior zero-mean noise.
        offset = _TERMINAL_NOISE_OFFSET.get(name, _DEFAULT_NOISE_OFFSET) * noise_scale
        if sigma > 0:
            noise = rng.normal(0, sigma, size=values.shape).astype(values.dtype)
            noisy[name] = values + offset + noise
        elif offset != 0.0:
            noisy[name] = values + offset
        else:
            noisy[name] = values
    return noisy

# Alias for use in vectorized EmbProj — strict (raises RuntimeError if no bases)
def _load_pca_bases_for_vectorized():
    return _load_pca_bases_cached(raise_on_missing=True)


# ---------------------------------------------------------------------------
# Vectorized tree evaluation
# ---------------------------------------------------------------------------

def _safe_shift(arr: np.ndarray, k: int) -> np.ndarray:
    """Shift array by k positions, filling with 0.0 at the start."""
    if k <= 0:
        return arr
    result = np.zeros_like(arr)
    result[k:] = arr[:-k]
    return result


def _session_lag_daily(values: np.ndarray, dates: np.ndarray,
                       prior_value: "Optional[float]" = None) -> np.ndarray:
    """Replace every row's value with the PRIOR trading session's value (the daily
    series lagged one session).

    Removes the 1-trading-day LOOKAHEAD in the daily CBOE VIX terminals: the
    collector stored day-D's VIX *close* (research_collector.py:666), so using it
    for an intraday day-D entry decision peeks at the 4pm value. A live QC backtest
    only ever has the last DELIVERED daily bar intraday = day-(D-1)'s close
    (Securities[vix].Price). Lagging one session makes the proxy use that same
    realistic value, removing the lookahead AND creating proxy<->QC parity (the
    codegen is already correct). Confirmed 2026-05-31: QC VIXChange[D] == proxy
    VIXChange[D-1] on 7/9 days. Assumes `values` is daily-constant (VIX family).

    prior_value: the value of the session IMMEDIATELY BEFORE this array's first
    day, taken from the FULL corpus (it lives in the walk-forward embargo gap, so
    it is a real, already-closed value — no lookahead). Supplied for walk-forward
    SLICES so the first slice day uses the true prior close instead of keeping its
    own same-day (lookahead) value. None → the first day keeps its own value (only
    safe at the very start of the corpus, where no prior session exists)."""
    out = values.copy()
    n = len(values)
    if n == 0:
        return out
    day_start = 0
    prev_day_val = prior_value
    while day_start < n:
        d = dates[day_start]
        day_end = day_start
        while day_end < n and dates[day_end] == d:
            day_end += 1
        if prev_day_val is not None:
            out[day_start:day_end] = prev_day_val
        prev_day_val = values[day_end - 1]  # last bar of THIS day -> prior for next
        day_start = day_end
    return out


def prior_session_vix(full_df: pd.DataFrame, slice_first_date: str,
                      cols=("VIXSpot", "VIXTermSlope")) -> "Dict[str, float]":
    """Look up the daily VIX-family values of the session IMMEDIATELY BEFORE
    `slice_first_date` in the FULL (un-sliced) corpus, for `_session_lag_daily`'s
    `prior_value`. That prior session lives in the walk-forward embargo gap — it is
    a real already-closed value, so using it is realistic (no lookahead). Returns
    {} if there is no prior session (slice starts at the corpus start)."""
    if "date" not in full_df.columns:
        return {}
    dates_str = full_df["date"].astype(str)
    all_dates = sorted(dates_str.unique())
    sfd = str(slice_first_date)
    if sfd not in all_dates:
        # caller passed a date not in the corpus; nearest prior by ordering
        priors = [d for d in all_dates if d < sfd]
        if not priors:
            return {}
        prev_date = priors[-1]
    else:
        idx = all_dates.index(sfd)
        if idx == 0:
            return {}
        prev_date = all_dates[idx - 1]
    prev_rows = full_df[dates_str == prev_date]
    out: "Dict[str, float]" = {}
    for c in cols:
        if c in full_df.columns and len(prev_rows):
            out[c] = float(prev_rows[c].values[-1])  # last bar of prior session
    return out


class VectorizedTreeEvaluator:
    """Evaluate a GP expression tree over arrays of terminal values.

    The terminal_data dict maps terminal names to 1-D NumPy arrays (one
    value per bar). All operations are element-wise over the bar dimension.
    """

    def __init__(self, terminal_data: Dict[str, np.ndarray], n_bars: int,
                 day_boundary_mask: Optional[np.ndarray] = None,
                 within_day_pos: Optional[np.ndarray] = None,
                 fold_recenter_stats: Optional[Dict] = None,
                 fold_id: Optional[str] = None):
        self.terminal_data = terminal_data
        self.n_bars = n_bars
        # Fold-recenter for EmbProj (B/C/D consistency)
        self._fold_recenter_stats = fold_recenter_stats
        self._fold_id = fold_id
        # _not_day_start: 0.0 at first bar of each day, 1.0 elsewhere.
        if day_boundary_mask is not None:
            self._not_day_start = day_boundary_mask
        else:
            self._not_day_start = np.ones(n_bars, dtype=np.float64)
            self._not_day_start[0] = 0.0
        # _within_day_pos: 0-indexed bar position within each day.
        # Used by Lag/Delta to mask out positions where the shift would
        # reach into the previous day (matching original evaluator's
        # buffer-reset-at-session-boundary behavior).
        if within_day_pos is not None:
            self._within_day_pos = within_day_pos
        else:
            self._within_day_pos = np.arange(n_bars, dtype=np.int64)

    def evaluate(self, node: Node) -> np.ndarray:
        """Evaluate a tree, returning a 1-D array of length n_bars.

        BOOL nodes return float arrays (1.0=True, 0.0=False).
        REAL nodes return float arrays.
        SIDE/REGIME nodes return object arrays of enum values.
        """
        if isinstance(node, TermNode):
            return self._eval_terminal(node)
        return self._eval_function(node)

    def _eval_terminal(self, node: TermNode) -> np.ndarray:
        if node.ret_type == GType.REAL:
            # Ephemeral / literal REAL constants
            if node.value is not None:
                return np.full(self.n_bars, float(node.value), dtype=np.float64)
            # Named terminal — look up in data
            arr = self.terminal_data.get(node.name)
            if arr is not None:
                return arr.astype(np.float64)
            # Missing terminal → zeros. Log once per terminal name to detect
            # silent phantom signals (grammar references column not in data).
            if not hasattr(self, '_warned_missing'):
                self._warned_missing = set()
            if node.name not in self._warned_missing:
                self._warned_missing.add(node.name)
                import warnings
                warnings.warn(
                    f"Terminal '{node.name}' not in terminal_data — returning zeros. "
                    f"GP trees referencing this terminal will see constant 0.0.",
                    stacklevel=2,
                )
            return np.zeros(self.n_bars, dtype=np.float64)
        # Typed-vector terminals (EMB_GRID, EMB_VIX, etc.) → (N, D) arrays
        if node.ret_type.name.startswith("EMB_"):
            arr = self.terminal_data.get(node.name)
            if arr is not None and arr.ndim == 2:
                return arr.astype(np.float64)
            # Missing vector data → zero matrix (N, 384 default)
            return np.zeros((self.n_bars, 384), dtype=np.float64)
        if node.ret_type == GType.INT:
            return np.full(self.n_bars, int(node.value) if node.value is not None else 1,
                          dtype=np.float64)
        if node.ret_type == GType.SIDE:
            # Return array of Side enum values
            return np.full(self.n_bars, node.value, dtype=object)
        if node.ret_type == GType.REGIME:
            return np.full(self.n_bars, node.value, dtype=object)
        return np.zeros(self.n_bars, dtype=np.float64)

    def _eval_function(self, node: FuncNode) -> np.ndarray:
        name = node.name
        ch = node.children

        # Temporal operators — respect session boundaries.
        # Original evaluator resets buffer at day start, so Lag(X, k) returns
        # 0.0 when within_day_pos < k. We mask those positions.
        if name == "Lag":
            lag = int(np.clip(self.evaluate(ch[1]).flat[0], 0, 30))
            # F3 fix: evaluate ANY child expression, not just TermNode
            arr = self.evaluate(ch[0])
            if arr.ndim == 1:
                shifted = _safe_shift(arr, lag)
                shifted[self._within_day_pos < lag] = 0.0
                return shifted
            return arr  # 2-D vector lag handled by EmbLag

        if name == "Delta":
            lag = int(np.clip(self.evaluate(ch[1]).flat[0], 1, 30))
            # F3 fix: evaluate ANY child expression, not just TermNode
            arr = self.evaluate(ch[0])
            if arr.ndim == 1:
                shifted = _safe_shift(arr, lag)
                mask = self._within_day_pos >= lag
                return np.where(mask, arr - shifted, 0.0)
            return np.zeros(self.n_bars, dtype=np.float64)

        # CrossAbove / CrossBelow
        # Must respect session boundaries: first bar of each day has no
        # "previous bar" (original evaluator returns None → False).
        # Use self._day_boundary_mask to zero out crossings at day starts.
        if name == "CrossAbove":
            a = self.evaluate(ch[0]).astype(np.float64)
            b = self.evaluate(ch[1]).astype(np.float64)
            prev_a = _safe_shift(a, 1)
            prev_b = _safe_shift(b, 1)
            signal = ((a > b) & (prev_a <= prev_b)).astype(np.float64)
            signal *= self._not_day_start  # zero out day-boundary false crossings
            return signal

        if name == "CrossBelow":
            a = self.evaluate(ch[0]).astype(np.float64)
            b = self.evaluate(ch[1]).astype(np.float64)
            prev_a = _safe_shift(a, 1)
            prev_b = _safe_shift(b, 1)
            signal = ((a < b) & (prev_a >= prev_b)).astype(np.float64)
            signal *= self._not_day_start
            return signal

        # Comparison / Boolean
        if name == "GT":
            return (self.evaluate(ch[0]).astype(np.float64) >
                    self.evaluate(ch[1]).astype(np.float64)).astype(np.float64)
        if name == "LT":
            return (self.evaluate(ch[0]).astype(np.float64) <
                    self.evaluate(ch[1]).astype(np.float64)).astype(np.float64)
        if name == "AND":
            return ((self.evaluate(ch[0]).astype(np.float64) > 0.5) &
                    (self.evaluate(ch[1]).astype(np.float64) > 0.5)).astype(np.float64)
        if name == "OR":
            return ((self.evaluate(ch[0]).astype(np.float64) > 0.5) |
                    (self.evaluate(ch[1]).astype(np.float64) > 0.5)).astype(np.float64)
        if name == "NOT":
            return (self.evaluate(ch[0]).astype(np.float64) <= 0.5).astype(np.float64)

        # Arithmetic
        if name == "Add":
            return self.evaluate(ch[0]).astype(np.float64) + self.evaluate(ch[1]).astype(np.float64)
        if name == "Sub":
            return self.evaluate(ch[0]).astype(np.float64) - self.evaluate(ch[1]).astype(np.float64)
        if name == "Mul":
            return self.evaluate(ch[0]).astype(np.float64) * self.evaluate(ch[1]).astype(np.float64)
        if name == "Div":
            a = self.evaluate(ch[0]).astype(np.float64)
            b = self.evaluate(ch[1]).astype(np.float64)
            return a / np.sqrt(1.0 + b * b)
        if name == "Sqrt":
            return np.sqrt(np.abs(self.evaluate(ch[0]).astype(np.float64)))

        # Conditional
        if name in ("IfThenElse", "IfSide"):
            cond = self.evaluate(ch[0]).astype(np.float64) > 0.5
            then_val = self.evaluate(ch[1])
            else_val = self.evaluate(ch[2])
            return np.where(cond, then_val, else_val)

        # -- Typed-vector embedding operators (B/C/D conditions) --
        # These operate on (N, D) arrays from typed-vector terminals.

        if name.startswith("EmbNorm_"):
            v = self.evaluate(ch[0])  # (N, D)
            if v.ndim == 2:
                return np.linalg.norm(v, axis=1)  # (N,) L2 norm per bar
            return np.zeros(self.n_bars, dtype=np.float64)

        if name.startswith("EmbCos_"):
            va = self.evaluate(ch[0])  # (N, D)
            vb = self.evaluate(ch[1])  # (N, D)
            if va.ndim == 2 and vb.ndim == 2:
                dot = np.sum(va * vb, axis=1)  # (N,)
                na = np.linalg.norm(va, axis=1)
                nb = np.linalg.norm(vb, axis=1)
                denom = na * nb
                denom = np.where(denom < 1e-8, 1.0, denom)
                return dot / denom
            return np.zeros(self.n_bars, dtype=np.float64)

        if name.startswith("EmbSub_"):
            va = self.evaluate(ch[0])  # (N, D)
            vb = self.evaluate(ch[1])  # (N, D)
            if va.ndim == 2 and vb.ndim == 2:
                return va - vb  # (N, D)
            return np.zeros((self.n_bars, 384), dtype=np.float64)

        if name.startswith("EmbLag_"):
            v = self.evaluate(ch[0])  # (N, D) typed vector
            lag = max(0, min(int(self.evaluate(ch[1]).flat[0]), 30))
            if v.ndim == 2 and lag > 0:
                shifted = np.zeros_like(v)
                shifted[lag:] = v[:-lag]
                # Day-boundary mask: zero out where lag crosses day boundary
                mask = self._within_day_pos >= lag  # (N,) bool
                shifted[~mask] = 0.0
                return shifted
            return v if v.ndim == 2 else np.zeros((self.n_bars, 384), dtype=np.float64)

        if name.startswith("EmbProj_"):
            v = self.evaluate(ch[0])  # (N, D)
            if v.ndim != 2:
                return np.zeros(self.n_bars, dtype=np.float64)
            # Parse: EmbProj_EMB_{GROUP}_{K}
            suffix = name[len("EmbProj_"):]
            _dot = suffix.rfind("_")
            try:
                k = int(suffix[_dot + 1:])
                group = suffix[:_dot]
            except (ValueError, IndexError):
                return np.zeros(self.n_bars, dtype=np.float64)
            # Load PCA bases
            bases = _load_pca_bases_for_vectorized()
            if bases is None or group not in bases:
                return np.zeros(self.n_bars, dtype=np.float64)
            g = bases[group]
            components = g["components"]  # (K_group, D)
            if k >= components.shape[0]:
                return np.zeros(self.n_bars, dtype=np.float64)
            mean_ = g["mean_"]  # (D,)
            pc_std = g["pc_std_"]  # (K_group,)
            # Project: (N, D) @ (D,) → (N,) with centering + Mahalanobis
            # Zero-vector guard: bars with no encoder output (positions 0-58)
            # must return 0.0, not (-mean) @ component (structural constant bug,
            # same as v9 Bug 4 in the original evaluator).
            is_zero = ~np.any(v, axis=1)  # (N,) True for zero-vector bars
            centered = v - mean_[np.newaxis, :]
            raw = centered @ components[k]  # (N,)
            raw[is_zero] = 0.0
            mahal = raw / float(pc_std[k]) if pc_std[k] > 1e-12 else raw
            # Fold-recenter: apply per-fold per-PC (mean, std) correction
            # from fold_recenter_stats.npz. Without this, val/test EmbProj
            # values drift from train distribution (RF-A: 2.85-6.19 sigma).
            if hasattr(self, '_fold_recenter_stats') and self._fold_recenter_stats:
                fold_id = getattr(self, '_fold_id', None)
                # IH1 (2026-06-01 holistic review): FAIL LOUD on a fold-scheme mismatch.
                # The walk-forward run uses a 4-fold rolling scheme (experiment.py),
                # while fold_recenter_stats.npz was keyed to a 3-fold train/val/test
                # split — so a walk-forward fold_id (e.g. "1") is absent from the stats
                # and the per-PC drift correction was SILENTLY skipped, leaving EmbProj
                # values uncorrected (the 2.85-6.19σ val/test drift this exists to
                # remove). Require the key rather than no-op. Only reachable for C/D
                # (embeddings); scalar/probe runs pass fold_recenter_stats=None.
                if not fold_id or fold_id not in self._fold_recenter_stats:
                    raise RuntimeError(
                        f"fold-recenter stats provided but fold_id={fold_id!r} not in "
                        f"keys {sorted(self._fold_recenter_stats.keys())}. Regenerate "
                        f"fold_recenter_stats.npz keyed to the SAME walk-forward fold "
                        f"scheme as the run before evaluating Conditions C/D. (IH1.)"
                    )
                grp_stats = self._fold_recenter_stats[fold_id].get(group)
                if grp_stats and k < len(grp_stats["mean"]):
                    mahal = (mahal - float(grp_stats["mean"][k])) / float(grp_stats["std"][k])
            return mahal

        # Regime operators (InRegime, RegimeIs)
        # Compare the PredRegime terminal (per-bar regime label) against
        # the regime literal from the child node.
        if name == "InRegime":
            regime_arr = self.terminal_data.get("PredRegime",
                                              np.zeros(self.n_bars, dtype=np.float64))
            target = self.evaluate(node.children[0])
            return (regime_arr == target).astype(np.float64)
        if name == "RegimeIs":
            # Arity 2: compare two regime expressions (not PredRegime lookup)
            left = self.evaluate(node.children[0])
            right = self.evaluate(node.children[1])
            return (left == right).astype(np.float64)

        # Unknown function — warn and return zeros to surface grammar/evaluator desync
        warnings.warn(f"VectorizedTreeEvaluator: unknown function '{name}', returning zeros",
                      stacklevel=2)
        return np.zeros(self.n_bars, dtype=np.float64)


# ---------------------------------------------------------------------------
# Position-aware exit-tree evaluation (UnrealizedProfitPct)
# ---------------------------------------------------------------------------

def tree_references_upp(node: Node) -> bool:
    """True iff the tree references the UnrealizedProfitPct terminal.

    Used to gate the per-bar scalar re-evaluation: exit trees that do NOT use
    the position-aware terminal keep the fast precomputed vectorized signal
    (UnrealizedProfitPct == 0 baseline everywhere); only trees that DO use it
    pay the per-in-position-bar scalar walk.
    """
    if node is None:
        return False
    if isinstance(node, TermNode):
        return node.name == UNREALIZED_PROFIT_PCT
    return any(tree_references_upp(c) for c in node.children)


def _eval_node_scalar(
    node: Node,
    i: int,
    terminal_data: Dict[str, np.ndarray],
    within_day_pos: np.ndarray,
    not_day_start: np.ndarray,
    upp_value: float,
    upp_hist: List[float],
) -> float:
    """Evaluate an expression tree at a SINGLE bar index ``i``.

    Mirrors VectorizedTreeEvaluator semantics exactly for the scalar (one-bar)
    case, but substitutes ``upp_value`` for the UnrealizedProfitPct terminal so
    the exit tree sees the CURRENT bar's true unrealized profit. Returns a
    python float (BOOL → 1.0/0.0, REAL → value). SIDE/REGIME enums are returned
    as-is (only reachable inside IfSide/RegimeIs, which compare them).

    Temporal operators on the per-bar UnrealizedProfitPct value read from
    ``upp_hist`` (the values observed so far in THIS trade, most-recent last),
    so Lag/Delta/Cross on it are correct WITHIN a trade and never leak across
    trades (upp_hist is cleared on each new entry). Temporal operators on every
    OTHER terminal read the precomputed array at index ``i`` (position-
    independent, identical to the vectorized pass) with the same day-boundary
    masking. Embedding (EMB_*) operators are not reachable from an exit BOOL
    tree (REAL-typed scalar grammar) and are intentionally unsupported here.
    """
    # Terminal
    if isinstance(node, TermNode):
        if node.name == UNREALIZED_PROFIT_PCT:
            return float(upp_value)
        if node.ret_type == GType.REAL:
            if node.value is not None:  # ephemeral / literal constant
                return float(node.value)
            arr = terminal_data.get(node.name)
            if arr is not None and arr.ndim == 1:
                return float(arr[i])
            return 0.0
        if node.ret_type == GType.INT:
            return float(int(node.value)) if node.value is not None else 1.0
        # SIDE / REGIME literals — return the enum value for equality comparison
        if node.ret_type in (GType.SIDE, GType.REGIME):
            return node.value
        return 0.0

    name = node.name
    ch = node.children

    # --- Temporal: read the lagged value at the right index ---
    def _lagged_scalar(child: Node, lag: int) -> float:
        """Value of ``child`` ``lag`` bars before bar i (0.0 if it crosses the
        day boundary / trade start, matching the vectorized day-mask)."""
        if lag <= 0:
            return _eval_node_scalar(child, i, terminal_data, within_day_pos,
                                     not_day_start, upp_value, upp_hist)
        if within_day_pos[i] < lag:
            return 0.0  # would reach into the prior day → masked to 0.0
        if isinstance(child, TermNode) and child.name == UNREALIZED_PROFIT_PCT:
            # Per-trade history (upp_hist[-1] is the current bar's value).
            if lag < len(upp_hist):
                return float(upp_hist[-1 - lag])
            return 0.0  # before this trade started
        if isinstance(child, TermNode) and child.ret_type == GType.REAL:
            if child.value is not None:
                return float(child.value)  # constant — lag is identity
            arr = terminal_data.get(child.name)
            if arr is not None and arr.ndim == 1:
                return float(arr[i - lag])
            return 0.0
        # FuncNode child: the vectorized path shifts the already-evaluated array,
        # i.e. it evaluates the sub-expression at bar (i-lag). UnrealizedProfitPct
        # nested inside a lagged FuncNode is not produced by the seed/grammar in
        # practice; evaluate the sub-expression at i-lag with upp=0 for that bar
        # (the only correctness-preserving scalar option without a full per-bar
        # upp array — and it matches "no position-state leakage").
        if i - lag >= 0:
            return _eval_node_scalar(child, i - lag, terminal_data, within_day_pos,
                                     not_day_start, 0.0, upp_hist)
        return 0.0

    if name == "Lag":
        lag = int(np.clip(_eval_node_scalar(ch[1], i, terminal_data, within_day_pos,
                                            not_day_start, upp_value, upp_hist), 0, 30))
        return _lagged_scalar(ch[0], lag)

    if name == "Delta":
        lag = int(np.clip(_eval_node_scalar(ch[1], i, terminal_data, within_day_pos,
                                            not_day_start, upp_value, upp_hist), 1, 30))
        if within_day_pos[i] < lag:
            return 0.0
        cur = _eval_node_scalar(ch[0], i, terminal_data, within_day_pos,
                                not_day_start, upp_value, upp_hist)
        return cur - _lagged_scalar(ch[0], lag)

    if name in ("CrossAbove", "CrossBelow"):
        a = _eval_node_scalar(ch[0], i, terminal_data, within_day_pos,
                              not_day_start, upp_value, upp_hist)
        b = _eval_node_scalar(ch[1], i, terminal_data, within_day_pos,
                              not_day_start, upp_value, upp_hist)
        prev_a = _lagged_scalar(ch[0], 1)
        prev_b = _lagged_scalar(ch[1], 1)
        if not_day_start[i] <= 0.0:  # first bar of day → no crossing
            return 0.0
        if name == "CrossAbove":
            return 1.0 if (a > b and prev_a <= prev_b) else 0.0
        return 1.0 if (a < b and prev_a >= prev_b) else 0.0

    # --- Comparison / Boolean ---
    if name == "GT":
        return 1.0 if (_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
                       > _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)) else 0.0
    if name == "LT":
        return 1.0 if (_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
                       < _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)) else 0.0
    if name == "AND":
        return 1.0 if (_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist) > 0.5
                       and _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist) > 0.5) else 0.0
    if name == "OR":
        return 1.0 if (_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist) > 0.5
                       or _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist) > 0.5) else 0.0
    if name == "NOT":
        return 1.0 if (_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist) <= 0.5) else 0.0

    # --- Arithmetic ---
    if name == "Add":
        return (_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
                + _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist))
    if name == "Sub":
        return (_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
                - _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist))
    if name == "Mul":
        return (_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
                * _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist))
    if name == "Div":
        a = _eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
        b = _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
        return a / math.sqrt(1.0 + b * b)  # analytic quotient (matches vectorized)
    if name == "Sqrt":
        return math.sqrt(abs(_eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)))

    # --- Conditional ---
    if name in ("IfThenElse", "IfSide"):
        cond = _eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist) > 0.5
        branch = ch[1] if cond else ch[2]
        return _eval_node_scalar(branch, i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)

    # --- Regime operators ---
    if name == "InRegime":
        regime_arr = terminal_data.get("PredRegime")
        cur_regime = float(regime_arr[i]) if regime_arr is not None and regime_arr.ndim == 1 else 0.0
        target = _eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
        target_v = target.value if hasattr(target, "value") else target
        return 1.0 if cur_regime == target_v else 0.0
    if name == "RegimeIs":
        left = _eval_node_scalar(ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
        right = _eval_node_scalar(ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist)
        return 1.0 if left == right else 0.0

    # EMB_* operators are unreachable from a scalar BOOL exit tree (the only path
    # that uses UnrealizedProfitPct); treat any residual as a no-signal 0.0.
    return 0.0


def _exit_signal_at_bar(
    exit_tree: Node,
    i: int,
    terminal_data: Dict[str, np.ndarray],
    within_day_pos: np.ndarray,
    not_day_start: np.ndarray,
    upp_value: float,
    upp_hist: List[float],
) -> bool:
    """Boolean exit signal at bar ``i`` with the live UnrealizedProfitPct value.

    NaN-safe (a NaN from div/sqrt edge cases → False, matching the vectorized
    ``np.where(isnan, 0.0)`` + ``> 0.5`` path).
    """
    v = _eval_node_scalar(exit_tree, i, terminal_data, within_day_pos,
                          not_day_start, upp_value, upp_hist)
    if isinstance(v, float) and math.isnan(v):
        return False
    return float(v) > 0.5


# ---------------------------------------------------------------------------
# Vectorized backtester
# ---------------------------------------------------------------------------

def prepare_terminal_data(
    data: pd.DataFrame,
    normalize_terminals: bool = True,
    norm_stats_override: "Optional[Dict[str, Tuple[float, float, str]]]" = None,
    lag_daily_vix: bool = True,
    vix_prior: "Optional[Dict[str, float]]" = None,
) -> Dict[str, np.ndarray]:
    """Convert a DataFrame to terminal_data dict for vectorized evaluation.

    Applies normalization to REAL terminals. When norm_stats_override is
    provided, uses those (center, scale, method) per terminal instead of
    the frozen constants in terminal_stats.py. This enables per-fold
    normalization to prevent look-ahead bias.

    lag_daily_vix (default True): lag the daily CBOE VIX terminals (VIXSpot,
    VIXTermSlope, and the VIXChange synthesis) one trading session to remove the
    1-day LOOKAHEAD of using day-D's VIX close for an intraday day-D decision. A
    live QC backtest only has the prior session's close intraday, so this also
    creates proxy<->QC parity. VIXMean5d/SPXReturn3d already use prior days only
    and are left untouched (they read the ORIGINAL closes via _vixspot_orig).
    """
    terminal_data: Dict[str, np.ndarray] = {}
    from layer2.io import L1_TERMINAL_COLUMNS, TYPED_VECTOR_COLUMNS

    # --- VIX lookahead remediation (2026-05-31) ---------------------------------
    # Lag the daily VIX-family columns one session BEFORE they are normalized in
    # the main loop / consumed by the VIXChange synthesis. Preserve the ORIGINAL
    # VIXSpot closes for VIXMean5d, which is already lookahead-free (prior days).
    _vixspot_orig = (data["VIXSpot"].values.astype(np.float64).copy()
                     if "VIXSpot" in data.columns else None)
    if lag_daily_vix and "date" in data.columns:
        _dates_lag = data["date"].values
        _to_lag = [c for c in ("VIXSpot", "VIXTermSlope") if c in data.columns]
        if _to_lag:
            data = data.copy()
            for _c in _to_lag:
                # vix_prior supplies the prior session's close (from the embargo
                # gap of the full corpus) so the FIRST slice day uses the true
                # prior value, not its own same-day close — closing the residual
                # first-slice-day lookahead (code review 2026-05-31).
                _prior = vix_prior.get(_c) if vix_prior else None
                data[_c] = _session_lag_daily(
                    data[_c].values.astype(np.float64), _dates_lag,
                    prior_value=_prior)

    # Helper: resolve normalization stats from override (per-fold) or frozen global.
    # Used by both the main loop and synthesized terminal blocks.
    def _get_norm(name: str) -> tuple:
        if norm_stats_override and name in norm_stats_override:
            return norm_stats_override[name]
        from layer2.terminal_stats import TERMINAL_NORM_STATS
        return TERMINAL_NORM_STATS.get(name, (0.0, 1.0, "standard"))

    for col in data.columns:
        if col in ("date", "window_idx", "bar_position"):
            continue
        # Typed-vector columns → (N, D) arrays for B/C/D conditions
        if col in TYPED_VECTOR_COLUMNS:
            sample = data[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)) and len(sample) > 1:
                # Convert list-of-lists to (N, D) array — use float32 to halve memory
                # (encoder outputs are float32 natively; 10 cols × 357K × 384 × 4B = ~5.5 GB)
                vec_arr = np.array(data[col].tolist(), dtype=np.float32)
                terminal_data[col] = vec_arr
            continue
        try:
            arr = data[col].values.astype(np.float64)
        except (ValueError, TypeError):
            continue  # skip non-numeric columns
        # C4 fix: forward-fill ATM_IV=0 bars BEFORE normalization.
        # 8.6% of bars have ATM_IV=0 (empty grid / no quotes). Without
        # forward-fill, these normalize to -2.13 creating a false "IV crash"
        # signal that the GP can learn to exploit. The signal is a data
        # artifact, not a market condition.
        if col == "ATM_IV":
            zero_mask = arr == 0.0
            if zero_mask.any() and "date" in data.columns:
                dates = data["date"].values
                for j in range(len(arr)):
                    if zero_mask[j] and j > 0 and str(dates[j]) == str(dates[j-1]):
                        arr[j] = arr[j-1]  # forward-fill within day
        if normalize_terminals and col in NORMALIZED_TERMINALS:
            center, scale, _ = _get_norm(col)
            if scale > 1e-12:
                arr = (arr - center) / scale
                # OOD clamping: clip normalized values to [-5, +5] to prevent
                # extreme outliers (flash crashes, data errors) from producing
                # nonsensical tree evaluations. 5σ covers 99.99994% of normal
                # distribution; anything beyond is almost certainly OOD.
                arr = np.clip(arr, -5.0, 5.0)
        terminal_data[col] = arr

    # Synthesize BarOfDay if not present
    if "BarOfDay" not in terminal_data and "MinutesToClose" in data.columns:
        mtc = data["MinutesToClose"].values.astype(np.float64)
        # Derive bar cadence from data: for 1-min bars, consecutive mtc
        # values differ by 1; for 5-min bars, they differ by 5.
        if "date" in data.columns and len(data) > 2:
            dates = data["date"].values
            # Find first intra-day consecutive pair
            _cadence = 1.0
            for _j in range(1, min(len(data), 100)):
                if str(dates[_j]) == str(dates[_j - 1]):
                    _diff = abs(mtc[_j - 1] - mtc[_j])
                    # Skip artifact pairs where MTC jumps by >100 (e.g., the
                    # MTC=0 close-bar artifact at session start → MTC=403).
                    # Real cadence diffs are 1, 5, 15, or 30.
                    if 0.1 < _diff < 100:
                        _cadence = _diff
                        break
            if _cadence not in (1.0, 5.0, 15.0, 30.0):
                import warnings
                warnings.warn(
                    f"Unusual bar cadence detected: {_cadence} min. "
                    f"Expected 1/5/15/30. BarOfDay may be miscalibrated.",
                    stacklevel=2,
                )
        else:
            _cadence = 5.0  # fallback
        # Use actual max MTC from data to derive session length, not hardcoded 390.
        # SPX 0DTE sessions are 9:30-16:15 ET = 405 min (MTC goes from ~404 to 0).
        if len(mtc) == 0 or np.all(np.isnan(mtc)):
            _session_len = 390.0
        else:
            _session_len = float(np.nanmax(mtc))
        if _session_len < 100:
            _session_len = 390.0  # safety fallback for very short test data
        bar_of_day = (_session_len - mtc) / _cadence
        if normalize_terminals and "BarOfDay" in NORMALIZED_TERMINALS:
            from layer2.terminal_stats import TERMINAL_NORM_STATS
            center, scale, _ = _get_norm("BarOfDay")
            if scale > 1e-12:
                bar_of_day = (bar_of_day - center) / scale
        terminal_data["BarOfDay"] = bar_of_day

    # Synthesize ThetaUrgency: 1/sqrt(raw_MTC) captures the nonlinear theta
    # decay specific to 0DTE options. BS ATM theta scales as 1/sqrt(T).
    # The GP cannot construct this from existing terminals because:
    # (a) Div is analytic quotient, not true division
    # (b) normalization makes MTC negative in the afternoon
    # (c) 4-node cost is 27% of 15-node budget
    # This is a sufficiency restoration (Koza 1992), not researcher bias.
    # Kommenda et al. (2020): injecting known physics as terminals improves
    # GP without overfitting. Virgolin et al. (2021): building blocks for
    # known functional forms improve both convergence and solution quality.
    if "MinutesToClose" in data.columns:
        raw_mtc = data["MinutesToClose"].values.astype(np.float64)
        theta_urgency = 1.0 / np.sqrt(np.maximum(raw_mtc, 1.0))
        if normalize_terminals and "ThetaUrgency" in NORMALIZED_TERMINALS:
            center, scale, _ = _get_norm("ThetaUrgency")
            if scale > 1e-12:
                theta_urgency = (theta_urgency - center) / scale
        terminal_data["ThetaUrgency"] = theta_urgency

    # Synthesize 5-bar smoothed terminals from RAW (pre-normalization) values,
    # then normalize the smoothed result. This ensures the 5m terminals are
    # properly N(0,1) distributed, unlike the previous approach that smoothed
    # already-normalized values (producing std~0.45 instead of ~1.0).
    for base_col in ("ATM_IV", "RealizedVol30m", "RawSpread"):
        smooth_name = f"{base_col}_5m"
        if smooth_name not in terminal_data and base_col in data.columns:
            # Use RAW data, not terminal_data[base_col] which is already normalized.
            # For ATM_IV: forward-fill zeros within each day (matching the C4 fix
            # applied to the base terminal) to prevent zero-IV data artifacts from
            # corrupting the 5-bar rolling average.
            raw_vals = data[base_col].values.astype(np.float64).copy()
            if base_col == "ATM_IV" and "date" in data.columns:
                _dates_ff = data["date"].values
                for _j in range(len(raw_vals)):
                    if raw_vals[_j] < 1e-6 and _j > 0 and str(_dates_ff[_j]) == str(_dates_ff[_j-1]):
                        raw_vals[_j] = raw_vals[_j - 1]
            smoothed = np.zeros_like(raw_vals)
            if "date" in data.columns:
                dates = data["date"].values
                day_start = 0
                for j in range(len(raw_vals)):
                    if j > 0 and str(dates[j]) != str(dates[j-1]):
                        day_start = j
                    window_start = max(day_start, j - 4)
                    smoothed[j] = np.mean(raw_vals[window_start:j+1])
            else:
                for j in range(len(raw_vals)):
                    window_start = max(0, j - 4)
                    smoothed[j] = np.mean(raw_vals[window_start:j+1])
            # Normalize with 5m-specific stats (same raw-scale as base terminal)
            if normalize_terminals and smooth_name in NORMALIZED_TERMINALS:
                from layer2.terminal_stats import TERMINAL_NORM_STATS
                center, scale, _ = _get_norm(smooth_name)
                if scale > 1e-12:
                    smoothed = (smoothed - center) / scale
                    smoothed = np.clip(smoothed, -5.0, 5.0)
            terminal_data[smooth_name] = smoothed

    # H2 fix: Synthesize VIXChange (day-over-day VIX change) so the GP gets
    # inter-day VIX dynamics directly, instead of wasting search capacity on
    # Delta(VIXSpot, k) which always returns 0 within a day (VIX is daily).
    if "VIXChange" not in terminal_data and "VIXSpot" in data.columns:
        vix = data["VIXSpot"].values.astype(np.float64)
        vix_change = np.zeros_like(vix)
        if "date" in data.columns:
            dates = data["date"].values
            prev_day_vix = vix[0]
            for j in range(len(vix)):
                if j > 0 and str(dates[j]) != str(dates[j-1]):
                    prev_day_vix = vix[j - 1]  # last bar of previous day
                vix_change[j] = vix[j] - prev_day_vix
        if normalize_terminals and "VIXChange" in NORMALIZED_TERMINALS:
            from layer2.terminal_stats import TERMINAL_NORM_STATS
            center, scale, _ = _get_norm("VIXChange")
            if scale > 1e-12:
                vix_change = (vix_change - center) / scale
                vix_change = np.clip(vix_change, -5.0, 5.0)
        terminal_data["VIXChange"] = vix_change

    # Synthesize inter-day memory terminals (constant within each day, updated
    # at session open). These give GP the ability to condition on multi-day
    # regime state — critical for detecting sustained regime shifts like the
    # Apr-Sep 2025 period where single-day scalars couldn't distinguish
    # "normal calm" from "persistently hostile."
    if "date" in data.columns:
        dates = data["date"].values
        _unique_dates = []
        _day_starts = []
        _prev_d = None
        for j in range(len(dates)):
            _d = str(dates[j])
            if _d != _prev_d:
                _unique_dates.append(_d)
                _day_starts.append(j)
                _prev_d = _d
        _n_days = len(_unique_dates)

        # Shared day-close array (used by SPXReturn3d and RV5d).
        # _day_closes[d] = the 16:00-ET cash-session close of day d (the bar at
        # MinutesToClose==15 relative to the 16:15 SPXW settlement). Available at
        # session open of day d+1, NOT at open of day d.
        #
        # G11 RV5d parity (2026-06-01): this used to be the LAST bar of the day
        # (prices[_day_starts[d+1]-1], MTC=0, ~16:14). QC cannot capture a 16:00+
        # price — its market-hours guard (codegen.py `t.hour >= 16` returns early)
        # freezes `_last_spx_price` at the 15:59 bar, so QC's daily close sat
        # ~1.85pt/day off the proxy's last bar, inflating RV5d. The SPX cash index
        # flatlines after 16:00 (settlement is at 16:15 but the index stops
        # printing at the 16:00 close), so the proxy's old last bar == the 16:00
        # close ON CALM DAYS but diverges whenever a late tick prints. Anchoring
        # BOTH sides to the explicit 16:00 bar (MTC==15) makes the daily-close
        # DEFINITION identical and computable on each side; the QC capture is
        # moved to the same 16:00 bar in codegen (see _on_bar daily-close roll).
        price_col = next(
            (c for c in ("SPXClose", "spx_close", "close") if c in data.columns), None)
        _day_closes = None
        _daily_rets = None
        if price_col:
            prices = data[price_col].values.astype(np.float64)
            _mtc_all = (data["MinutesToClose"].values.astype(np.float64)
                        if "MinutesToClose" in data.columns else None)

            def _day_close_idx(d):
                d_start = _day_starts[d]
                d_end = _day_starts[d + 1] if d < _n_days - 1 else len(prices)
                if _mtc_all is not None and d_end > d_start:
                    seg = _mtc_all[d_start:d_end]
                    # The 16:00 cash close is the bar at MTC==15. Pick the bar
                    # whose MTC is closest to 15 (ties -> earliest, i.e. the first
                    # 16:00 print), falling back to the last bar if MTC is absent.
                    j = int(np.argmin(np.abs(seg - 15.0)))
                    return d_start + j
                return d_end - 1

            _day_closes = np.array([prices[_day_close_idx(d)] for d in range(_n_days)])
            _daily_rets = np.diff(np.log(np.maximum(_day_closes, 1.0)))

        # SPXReturn3d: 3-day cumulative SPX return (close[d-1] vs close[d-4]).
        # Available at session open of day d (uses only prior closes).
        if _day_closes is not None and "SPXReturn3d" not in terminal_data:
            spx_ret3 = np.zeros(len(prices))
            for d in range(_n_days):
                d_start = _day_starts[d]
                d_end = _day_starts[d+1] if d < _n_days - 1 else len(prices)
                if d >= 4 and _day_closes[d-4] > 0:
                    val = (_day_closes[d-1] - _day_closes[d-4]) / _day_closes[d-4]
                else:
                    val = 0.0
                spx_ret3[d_start:d_end] = val
            if normalize_terminals and "SPXReturn3d" in NORMALIZED_TERMINALS:
                center, scale, _ = _get_norm("SPXReturn3d")
                if scale > 1e-12:
                    spx_ret3 = (spx_ret3 - center) / scale
                    spx_ret3 = np.clip(spx_ret3, -5.0, 5.0)
            terminal_data["SPXReturn3d"] = spx_ret3

        # VIXMean5d: 5-day rolling mean of VIXSpot (prior 5 session opens).
        # Uses days d-5..d-1 (NOT current day d) to avoid look-ahead. It is ALREADY
        # lookahead-free, so it reads the ORIGINAL (un-lagged) closes — double-
        # lagging via the VIX-lookahead remediation would make it one day staler
        # than necessary while adding no realism.
        if "VIXSpot" in data.columns and "VIXMean5d" not in terminal_data:
            vix = _vixspot_orig if _vixspot_orig is not None else data["VIXSpot"].values.astype(np.float64)
            _day_vix = np.array([vix[_day_starts[d]] for d in range(_n_days)])
            vix_mean5 = np.zeros(len(vix))
            for d in range(_n_days):
                d_start = _day_starts[d]
                d_end = _day_starts[d+1] if d < _n_days - 1 else len(vix)
                if d >= 5:
                    window = _day_vix[d-5:d]  # 5 prior days, NOT including today
                    vix_mean5[d_start:d_end] = float(np.mean(window))
                else:
                    vix_mean5[d_start:d_end] = _day_vix[0]  # default to first day
            if normalize_terminals and "VIXMean5d" in NORMALIZED_TERMINALS:
                center, scale, _ = _get_norm("VIXMean5d")
                if scale > 1e-12:
                    vix_mean5 = (vix_mean5 - center) / scale
                    vix_mean5 = np.clip(vix_mean5, -5.0, 5.0)
            terminal_data["VIXMean5d"] = vix_mean5

        # RV5d: 5-day realized vol from prior daily returns (no look-ahead).
        # Uses _daily_rets[d-6:d-1] = 5 returns ending at close[d-1].
        # All available at session open of day d.
        _rv5_raw = None  # raw (unnormalized) for IVRVGap5d
        if _daily_rets is not None and "RV5d" not in terminal_data:
            rv5 = np.zeros(len(prices))
            _rv5_raw = np.zeros(_n_days)  # per-day raw values for IVRVGap5d
            for d in range(_n_days):
                d_start = _day_starts[d]
                d_end = _day_starts[d+1] if d < _n_days - 1 else len(prices)
                if d >= 6:
                    window = _daily_rets[d-6:d-1]  # 5 returns, all prior to day d
                    raw_val = float(np.std(window) * np.sqrt(252))
                else:
                    raw_val = 0.15  # long-run SPX realized vol default
                _rv5_raw[d] = raw_val
                rv5[d_start:d_end] = raw_val
            if normalize_terminals and "RV5d" in NORMALIZED_TERMINALS:
                center, scale, _ = _get_norm("RV5d")
                if scale > 1e-12:
                    rv5 = (rv5 - center) / scale
                    rv5 = np.clip(rv5, -5.0, 5.0)
            terminal_data["RV5d"] = rv5

        # IVRVGap5d: 5-day rolling (mean IV - RV5d). Uses raw RV5d values.
        # Positive = IV rich (favorable for credit selling).
        if ("ATM_IV" in data.columns and _rv5_raw is not None
                and "IVRVGap5d" not in terminal_data):
            iv = data["ATM_IV"].values.astype(np.float64)
            # Sample the SETTLED post-open ATM_IV (bar+6), not the bar-0 opening
            # print: QC captures the day's ATM_IV at the first reliable bar
            # (_SESSION_IV_WARMUP=6, codegen.py), and the opening-print IV carries
            # wide first-minute spreads. Aligning the sample bar removes the
            # systematic IVRVGap5d proxy↔QC offset. (2026-06-01)
            _IV_SETTLE = 6
            _day_iv = np.array([
                iv[min(_day_starts[d] + _IV_SETTLE,
                       (_day_starts[d + 1] - 1) if d < _n_days - 1 else len(iv) - 1)]
                for d in range(_n_days)])
            ivrvgap = np.zeros(len(iv))
            for d in range(_n_days):
                d_start = _day_starts[d]
                d_end = _day_starts[d+1] if d < _n_days - 1 else len(iv)
                if d >= 5:
                    iv_window = _day_iv[d-5:d]  # 5 prior days
                    ivrvgap[d_start:d_end] = float(np.mean(iv_window)) - _rv5_raw[d]
                else:
                    ivrvgap[d_start:d_end] = 0.0
            if normalize_terminals and "IVRVGap5d" in NORMALIZED_TERMINALS:
                center, scale, _ = _get_norm("IVRVGap5d")
                if scale > 1e-12:
                    ivrvgap = (ivrvgap - center) / scale
                    ivrvgap = np.clip(ivrvgap, -5.0, 5.0)
            terminal_data["IVRVGap5d"] = ivrvgap

    # Synthesize SessionReturn: cumulative SPX return since session open.
    # Vilkov (2023, SSRN 4641356): conditional 10:00 ET entry rules deliver
    # economically meaningful net performance. First-30-minute return
    # predicts rest-of-day (Gao et al. 2018, JFE). This terminal gives the
    # GP access to intraday momentum/mean-reversion signals.
    if "SessionReturn" not in terminal_data:
        price_col = next(
            (c for c in ("SPXClose", "spx_close", "close") if c in data.columns),
            None,
        )
        if price_col and "date" in data.columns:
            prices = data[price_col].values.astype(np.float64)
            dates = data["date"].values
            session_return = np.zeros(len(prices), dtype=np.float64)
            session_open = prices[0]
            prev_date = str(dates[0])
            for j in range(len(prices)):
                cur_date = str(dates[j])
                if cur_date != prev_date:
                    session_open = prices[j]
                    prev_date = cur_date
                if session_open > 0:
                    session_return[j] = (prices[j] - session_open) / session_open
            # Normalize SessionReturn to ~N(0,1) using terminal_stats
            if normalize_terminals and "SessionReturn" in NORMALIZED_TERMINALS:
                from layer2.terminal_stats import TERMINAL_NORM_STATS
                center, scale, _ = _get_norm("SessionReturn")
                if scale > 1e-12:
                    session_return = (session_return - center) / scale
                    session_return = np.clip(session_return, -5.0, 5.0)
            terminal_data["SessionReturn"] = session_return

    # Synthesize binary regime indicators from PredRegime (conditions B/D).
    # Replaces the raw PredRegime terminal (categorical {0,1,2,3}) with three
    # binary indicators that the GP can use without wasting search budget on
    # redundant thresholds. Values are 0.0 or 1.0, normalized to {-1, +1}.
    if "PredRegime" in terminal_data:
        pred_regime = terminal_data["PredRegime"]
        # PredRegime is already in raw space (discrete 0-3) — derive indicators
        # before removing it from terminal_data.
        from layer2.terminal_stats import TERMINAL_NORM_STATS
        for ind_name, threshold in [("RegimeAboveLow", 1.0),
                                     ("RegimeIsHigh", 2.0),
                                     ("RegimeIsPremium", 3.0)]:
            if ind_name not in terminal_data:
                raw_ind = (pred_regime >= threshold).astype(np.float64)
                if normalize_terminals and ind_name in TERMINAL_NORM_STATS:
                    center, scale, _ = _get_norm(ind_name)
                    if scale > 1e-12:
                        raw_ind = (raw_ind - center) / scale
                terminal_data[ind_name] = raw_ind
        # Keep raw PredRegime as integer for InRegime operator consumption
        # (InRegime compares against regime enum values 0-3). Remove the
        # NORMALIZED float version so it can't be used as a GP terminal leaf.
        terminal_data["PredRegime"] = pred_regime.astype(np.int32)

    # UnrealizedProfitPct: position-aware exit terminal. Seed an all-zeros array
    # — this is BOTH the flat-state value (entry trees / idle bars see 0.0) AND
    # the vectorized baseline. The LIVE per-bar value is substituted into
    # exit-tree evaluation during the in-position loop (vectorized_backtest); the
    # zeros here also satisfy the experiment.py "every grammar terminal has data
    # backing" hardening check. Intentionally NOT normalized (RAW fraction; not in
    # NORMALIZED_TERMINALS), so the OOD [-5,+5] clamp does not touch it.
    if UNREALIZED_PROFIT_PCT not in terminal_data:
        terminal_data[UNREALIZED_PROFIT_PCT] = np.zeros(len(data), dtype=np.float64)

    return terminal_data


# ---------------------------------------------------------------------------
# Level B: Dynamic leg construction from delta_tree output
# ---------------------------------------------------------------------------

def _compute_dynamic_legs(template, delta_value: float):
    """Compute leg deltas from delta_tree output for Level B templates.

    Args:
        template: Template with delta_range set
        delta_value: delta_tree output clamped to [0, 1]

    Returns:
        List of (option_type, delta, qty_sign, ratio) tuples for each leg,
        with deltas computed from the template's delta_range + delta_value.
        Returns None if template has no delta_range (V1 template).
    """
    if template.delta_range is None:
        return None  # V1 template: use fixed legs

    min_d, max_d = template.delta_range

    if template.delta_fixed:
        # IB: short legs fixed at ATM (0.50), delta_tree controls wing delta
        wing_delta = min_d + delta_value * (max_d - min_d)
        wing_delta = max(0.05, wing_delta)  # floor at 5-delta
        # Build IB legs: short ATM call+put, long wing call+put
        return [
            ("call", +0.50, -1, 1),          # short ATM call
            ("call", +wing_delta, +1, 1),     # long OTM call wing
            ("put",  -0.50, -1, 1),           # short ATM put
            ("put",  -wing_delta, +1, 1),     # long OTM put wing
        ]

    short_delta = min_d + delta_value * (max_d - min_d)
    wing_offset = template.wing_offset or 0.15
    long_delta = max(short_delta - wing_offset, 0.05)  # floor at 5-delta

    n_legs = len(template.legs)
    if n_legs == 4:
        # IC: symmetric call+put wings
        return [
            ("call", +short_delta, -1, 1),    # short OTM call
            ("call", +long_delta, +1, 1),     # long further-OTM call
            ("put",  -short_delta, -1, 1),    # short OTM put
            ("put",  -long_delta, +1, 1),     # long further-OTM put
        ]
    elif n_legs == 2:
        # 2-leg spread: BPC, BCC, or ratio backspread.
        # Preserve template leg ratios (ratio=1 for verticals, ratio=2 for backspreads).
        leg0 = template.legs[0]
        leg1 = template.legs[1]
        sign = -1 if leg0.option_type == "put" else +1
        return [
            (leg0.option_type, sign * short_delta, leg0.qty_sign, leg0.ratio),
            (leg1.option_type, sign * long_delta, leg1.qty_sign, leg1.ratio),
        ]
    elif n_legs == 3:
        # 3-leg ratio structure: middle (2x) at short_delta, wings at offsets
        leg0_type = template.legs[0].option_type
        sign = -1 if leg0_type == "put" else +1
        lower_wing_delta = short_delta + wing_offset
        upper_wing_delta = max(short_delta - 0.15, 0.05)  # was 0.05 — $10 profit zone too narrow for 0DTE
        return [
            (leg0_type, sign * lower_wing_delta, +1, 1),  # lower wing (long)
            (leg0_type, sign * short_delta, -1, 2),        # middle (short 2x)
            (leg0_type, sign * upper_wing_delta, +1, 1),   # upper wing (long)
        ]
    return None  # shouldn't reach here


def _delta_haircut_factor(base_retention: float, abs_delta: float) -> float:
    """Delta-dependent credit retention. Near-ATM options have worse slippage.

    Near-ATM (high delta) = MORE haircut = LOWER retention.
    base_retention is the fraction of credit retained (0.90 = 10% haircut).

    retention(delta) = base * (1 - 0.5 * max(0, |delta| - 0.20) / 0.20)
    20d → base (unchanged), 30d → base*0.75, 40d → base*0.50
    Clamped to [0.50, base] to prevent extreme penalties.
    """
    penalty = 0.5 * max(0.0, abs_delta - 0.20) / 0.20
    return max(0.50, base_retention * (1.0 - penalty))


# Module-level constant: per-template credit factors (hoisted from inner loop).
# served_credit = raw_BS_credit * _BASE_CREDIT_FACTORS[tpl] * regime_mult (1465);
# calibrates the proxy's theoretical mid-price premium to real QC fills (you cross
# the bid-ask, MMs take edge). (_delta_haircut_factor above is currently dead code.)
# Recalibrated 2026-05-30 against 2,080 full-range QC mechanical trades (credit
# collector, low-vol regime n=377; (internal doc)).
# The credit-sellers were ~9-17% too pessimistic vs measured QC fills; raised to the
# robust low-regime implied factors. IB kept (3% diff = noise); RPB kept (its implied
# 1.37 is an artifact of sign-unstable raw + dynamic-leg mismatch — needs a paired
# join, not a median ratio). Regime mults (0.85/0.70) held — high-vol n=7 too thin.
# Recalibrated 2026-06-01 on the CLAMPED proxy (B2-fin IV upper clamp) vs the cached
# Jan-2025 QC fills (scripts/r5_credit_factor_check.py, minute+grid Edgeworth path,
# 10:00 ET / mtc=375). implied = median(QC_credit)/median(proxy_raw_credit).
_BASE_CREDIT_FACTORS = {
    "bull_put_credit": 0.83,   # was 0.88; implied 0.831
    "bear_call_credit": 0.83,  # was 0.90; implied 0.825 (proxy over-credited)
    "iron_condor": 0.85,       # was 0.89; implied 0.850
    "iron_butterfly": 0.76,    # was 0.81; implied 0.762
    "ratio_put_backspread": 0.92,  # was 0.80; implied 0.923 — the IV clamp removed the prior 1.37 phantom artifact
}

# Sizing-GROSS realism factors (Residual 1, 2026-05-31). _gross_pos_value returns
# the theoretical mid-price SUM of |leg values|; real QC fills (a) cross the bid-
# ask and (b) fill the cheap OTM long wing far below the Edgeworth/BS surface (a
# 0DTE 10-delta option is near-worthless intraday yet the surface still prices it).
# Gross is therefore over-stated MORE than net — the wing SUBTRACTS in net (so the
# credit factor barely sees it) but ADDS in gross — so the gross haircut is a
# SEPARATE, deeper correction than _BASE_CREDIT_FACTORS. Applied to the SIZING
# basis ONLY (abs_val), never to PnL/credit. It is Sharpe-neutral *wherever the
# abs_val term binds linearly* (a constant scale on n -> on returns); it is a
# NO-OP where the concentration/margin cap binds (large-size strategies), because
# there n is set by shared strike-width margin, not gross -- so there is no gross-
# driven gap to correct in that regime (H1, code review 2026-05-31). The fix
# therefore restores proxy<->QC per-trade n parity exactly in the small-n /
# fractional-size regime where the ~1.67x gross gap actually lives (e.g. bcc_f2,
# size=Delta(ATM_IV_5m,1), n=1-9 << conc cap).
# Calibrated 2026-05-31 from the full-range (Oct2024-May2026) gross-collector run
# (credit_collector.py entry_gross recording), low-vol regime n=378/template, the
# SAME 2,080-trade mechanical dataset + method that set _BASE_CREDIT_FACTORS:
# implied = median(QC_gross) / median(proxy_raw_gross) at the standardized template
# legs (25/10-delta etc.). See scripts/r5_gross_factor_check.py --full-range and
# docs/diagnostics/qc_gross_medians.json. (A thin n=17 strategy-specific bcc probe
# read 0.60, but it conflated strike-placement with fill realism; the robust
# template-level bcc factor is 0.86. Code-review M4 / 2026-05-31.)
# Recalibrated 2026-06-01 on the CLAMPED proxy vs cached Jan-2025 QC low-regime gross
# medians (scripts/r5_gross_factor_check.py). implied = QC_low / proxy_raw_low_median.
_GROSS_REALISM_FACTORS = {
    "bull_put_credit": 0.71,       # was 0.65; QC 752.5 / proxy_low 1065.6
    "bear_call_credit": 0.72,      # was 0.86; QC 592.5 / proxy_low 820.1 (was over-grossed)
    "iron_condor": 0.70,           # was 0.71; QC 1332.5 / proxy_low 1909.3
    "iron_butterfly": 0.65,        # was 0.74; QC 2960.0 / proxy_low 4520.0
    "ratio_put_backspread": 0.70,  # was 0.75; QC 2200.0 / proxy_low 3156.7
}


def vectorized_backtest(
    entry_tree: Node,
    exit_tree: Node,
    size_tree: Node,
    data: pd.DataFrame,
    template,
    delta_tree: Optional[Node] = None,     # Level B: 4th tree for continuous delta
    cost_multiplier: float = 1.0,          # Cost sensitivity sweep parameter
    terminal_data: Optional[Dict[str, np.ndarray]] = None,
    notional: float = 1000.0,
    fee_per_leg: float = 2.50,
    warmup_bars: int = 15,   # Change E: reduced from 30; bars 0-14 have valid
                              # terminal data via forward-fill (ATM_IV, RV30m,
                              # DeltaSpread5). 15 bars = first 15 minutes,
                              # enough for 5-bar smoothed terminals to warm up.
    min_bars_in_trade: int = 15,
    max_bars_in_trade: int = 330,
    default_iv: float = 0.20,
    stop_loss_credit_multiple: float = 2.5,  # calibrated 2026-05-15: QC median 1.8-2.2x
    fold_recenter_stats: Optional[Dict] = None,
    fold_id: Optional[str] = None,
    _override_entry_signals: Optional[np.ndarray] = None,  # for random-entry baseline
    _override_exit_signals: Optional[np.ndarray] = None,   # for random-entry baseline (hold-to-EOD)
    _sizing_log: Optional[list] = None,  # calibration: per-trade sizing recorder (no behavior change)
) -> BacktestResult:
    """Run vectorized tree evaluation + sequential position loop.

    Step 1: Vectorized — evaluate entry/exit/size trees over all bars at once.
    Step 2: Sequential — process position state bar by bar using the signal arrays.

    This is ~30-50x faster than the recursive evaluator for large datasets
    (the tree evaluation dominates, not the position loop).

    Args:
        stop_loss_credit_multiple: For credit spreads, exit when unrealised
            loss exceeds this multiple of the credit received.  E.g. 2.0
            means stop when loss > 2× credit.  Without this, a $20-wide
            IC needs 93% win rate; with 2× stop, break-even is ~67%.
            Set to 0 or None to disable.
    """
    from layer2.evaluator import (
        MultiLegOptionsBacktester, SimpleBacktester,
    )
    from layer2.grammar import Side as GSide

    n_bars = len(data)

    # Step 1: Vectorized tree evaluation
    if terminal_data is None:
        terminal_data = prepare_terminal_data(data)

    # Build day-boundary mask and within-day position array
    day_mask = np.ones(n_bars, dtype=np.float64)
    within_day_pos = np.zeros(n_bars, dtype=np.int64)
    if "date" in data.columns:
        dates = data["date"].values
        day_mask[0] = 0.0
        pos = 0
        for i in range(1, n_bars):
            if str(dates[i]) != str(dates[i - 1]):
                day_mask[i] = 0.0
                pos = 0
            else:
                pos += 1
            within_day_pos[i] = pos
    else:
        within_day_pos = np.arange(n_bars, dtype=np.int64)

    evaluator = VectorizedTreeEvaluator(
        terminal_data, n_bars,
        day_boundary_mask=day_mask,
        within_day_pos=within_day_pos,
        fold_recenter_stats=fold_recenter_stats,
        fold_id=fold_id,
    )
    if _override_entry_signals is not None:
        entry_signals = _override_entry_signals.astype(np.float64)
    elif entry_tree is not None:
        entry_signals = evaluator.evaluate(entry_tree).astype(np.float64)
    else:
        entry_signals = np.zeros(n_bars, dtype=np.float64)
    if _override_exit_signals is not None:
        exit_signals = _override_exit_signals.astype(np.float64)
    elif exit_tree is not None:
        exit_signals = evaluator.evaluate(exit_tree).astype(np.float64)
    else:
        exit_signals = np.zeros(n_bars, dtype=np.float64)
    # Position-aware exit terminal (UnrealizedProfitPct): when the exit tree
    # references it, the precomputed `exit_signals` above used the FLAT (0.0)
    # baseline for it, so the per-bar decision must be re-evaluated with the
    # live unrealized-profit fraction inside the in-position loop. Gate on the
    # tree actually using it (FuncNode override-only path) so trees that don't
    # keep the fast vectorized signal. An _override_exit_signals (random-entry
    # baseline) is a fixed array and is never re-evaluated.
    _exit_uses_upp = (
        _override_exit_signals is None
        and exit_tree is not None
        and tree_references_upp(exit_tree)
    )
    size_signals = evaluator.evaluate(size_tree).astype(np.float64)

    # NaN guard: tree evaluation can produce NaN from division-by-zero, invalid
    # sqrt, etc. NaN > 0.5 returns False, silently disabling entry/exit signals.
    # Replace NaN with 0.0 (no-signal) to make the behavior explicit.
    entry_signals = np.where(np.isnan(entry_signals), 0.0, entry_signals)
    exit_signals = np.where(np.isnan(exit_signals), 0.0, exit_signals)
    size_signals = np.where(np.isnan(size_signals), 0.0, size_signals)

    # Clamp size to [0, 1] as before. The drawdown-exploit (GP minimizes size
    # to minimize drawdown) is fixed by normalizing drawdown by avg position
    # size in the fitness computation, NOT by flooring size here. Flooring just
    # moves the convergence point from 0.004% to 15%.
    size_signals = np.clip(size_signals, 0.0, 1.0)

    # Level B: evaluate delta_tree if present
    if delta_tree is not None:
        delta_signals = evaluator.evaluate(delta_tree).astype(np.float64)
        # NaN guard first (sigmoid of NaN is still NaN)
        delta_signals = np.where(np.isnan(delta_signals), 0.0, delta_signals)
        # Sigmoid mapping instead of hard clip [0, 1]. With N(0,1) terminals,
        # hard clip kills ~50% of values (all negatives → 0.0 = minimum delta).
        # Sigmoid(x) maps the full real line to (0, 1) smoothly, so negative
        # terminal values produce meaningful sub-0.5 deltas instead of all
        # collapsing to 0.0. This gives GP a useful gradient for delta exploration.
        delta_signals = 1.0 / (1.0 + np.exp(-delta_signals))
    else:
        delta_signals = None

    # Step 2: Sequential position loop with REAL PnL computation
    # Pre-extract columns as arrays for fast access (no data.iloc[i])
    price_col = next(
        (c for c in ("SPXClose", "spx_close", "close") if c in data.columns),
        "SPXClose"
    )
    iv_col = next(
        (c for c in ("ATM_IV", "atm_iv", "ATMIV") if c in data.columns), None
    )
    # Price arrays for option pricing — use RAW values from DataFrame.
    # The DataFrame contains raw (un-normalized) values. Normalization for
    # tree evaluation is handled separately by prepare_terminal_data().
    # Do NOT de-normalize here — the values are already raw.
    spot_arr = data[price_col].values.astype(np.float64) if price_col in data.columns else np.zeros(n_bars)
    iv_arr = data[iv_col].values.astype(np.float64) if iv_col and iv_col in data.columns else np.full(n_bars, default_iv)
    mtc_arr = data["MinutesToClose"].values.astype(np.float64) if "MinutesToClose" in data.columns else np.full(n_bars, 60.0)
    # ATM spread for calibrated cost model (raw dollars per share, not normalized)
    spread_arr = data["RawSpread"].values.astype(np.float64) if "RawSpread" in data.columns else np.full(n_bars, 0.015)
    # Fix ATM_IV=0 bars (8.6% of corpus): forward-fill from last valid IV,
    # fallback to default_iv. Zero IV causes bogus option pricing.
    _last_valid_iv = default_iv
    for _i in range(n_bars):
        if iv_arr[_i] > 0.001:
            _last_valid_iv = iv_arr[_i]
        else:
            iv_arr[_i] = _last_valid_iv
    # Clamp extreme IV for pricing (880 bars have ATM_IV > 1.0, Sep-Oct 2022).
    # Raw IV = 5.0 (500% annualized) produces unrealistic BS prices that
    # dominate Sharpe on those dates. Terminal evaluation uses normalized+clamped
    # values (OOD clamp [-5,+5]), but pricing uses raw IV. Cap at 1.0 (100%
    # annualized) which is already extreme for SPX.
    iv_arr = np.clip(iv_arr, 0.0, 1.0)
    date_arr = data["date"].values if "date" in data.columns else None

    # Extract grid IV arrays for empirical IV surface interpolation.
    # grid_iv_matrix: (n_bars, 11) or None if grid IV columns are absent.
    _has_grid_ivs = all(c in data.columns for c in GRID_IV_COLUMNS)
    if _has_grid_ivs:
        grid_iv_matrix = np.column_stack(
            [data[c].values.astype(np.float64) for c in GRID_IV_COLUMNS]
        )  # (n_bars, 11)
    else:
        grid_iv_matrix = None

    bar_returns = np.zeros(n_bars)
    trades: List[Trade] = []
    equity = 0.0
    # BLOCKER B3 (2026-06-01 holistic review): explicit ruin liquidation. A cash /
    # Reg-T account cannot lose >100%. Once cumulative realized equity ≤ −1.0 the
    # account is blown up; halt ALL further entries for the window (the available_
    # capital floor below already zeroes new margin, but the entry_n_contracts=1
    # fallbacks can bypass it — this guarantees the halt on every sizing path). The
    # trade that crossed −100% is the realistic overshoot; equity then stays ruined.
    _ruined = False

    # Position state
    in_position = False
    entry_bar = 0
    entry_spot = 0.0
    prev_position_value = 0.0  # defensive init (set on entry, updated per-bar M2M)
    prev_spot = 0.0            # defensive init (set on entry for simple backtester path)
    strikes: List[float] = []
    entry_net_value = 0.0
    # UnrealizedProfitPct per-trade history (current bar's value is appended each
    # in-position bar; CLEARED on every entry so Lag/Delta/Cross on it never leak
    # across trades). Only populated when _exit_uses_upp.
    upp_hist: List[float] = []
    raw_entry_val = 0.0  # raw (unhaircutted) entry value for M2M tracking
    entry_n_contracts = 0
    entry_cost = 0.0
    # LOW (2026-06-01 holistic review): per-trade init of the debit sizing/stop-loss
    # basis. entry_gross is the gross position value at entry; it is read in the debit
    # stop-loss branch (_debit_risk_basis = max(entry_gross, entry_net_value)) and was
    # previously a loop-persistent local assigned ONLY at entry — a stale value from a
    # prior trade could leak into a later trade's debit-stop if any future entry path
    # skipped the assignment. Initialize it here with the other entry-state vars.
    entry_gross = 0.0

    has_legs = hasattr(template, 'legs') and template.legs
    active_legs = template.legs  # Level B: overridden per-trade by delta_tree
    _entry_short_delta = None    # Level B: short delta at entry. Formerly drove the
    # delta-dependent stop base; that base is now the evolved stop_mult gene
    # (see the credit stop-loss block). Retained as recorded state for
    # diagnostics and a possible future state-dependent stop_tree.

    # Hoist credit factor lookup out of hot loop (code review blocker).
    # Template name and base factor are constant for the entire backtest call.
    _tpl_base = template.name.replace("_standard", "").replace("_narrow", "").replace("_wide", "")
    _base_credit_factor = _BASE_CREDIT_FACTORS.get(_tpl_base, 0.80)
    _gross_realism = _GROSS_REALISM_FACTORS.get(_tpl_base, 1.0)  # Residual-1 sizing haircut

    # Fix #2: Daily trade cap (matches QC codegen). Raised to 8 to enable
    # post-stop re-entry patterns (stopped at 11am → re-enter afternoon).
    MAX_TRADES_PER_DAY = 8
    trades_today = 0

    # Fix #3: 1-bar execution delay (signal at bar N, fill at bar N+1)
    pending_entry = False
    pending_size = 0.0

    _prev_date = None
    for i in range(n_bars):
        spot = spot_arr[i]
        iv = max(iv_arr[i], 0.01)
        mtc = mtc_arr[i]
        # Per-bar grid IVs and Edgeworth params — only compute when needed
        # (in-position M2M or entry pricing). Idle bars skip this entirely.
        # Saves ~7% of total runtime (154K bars, only ~1K need surface params).
        _bar_grid = grid_iv_matrix[i] if grid_iv_matrix is not None else None
        _bar_rho, _bar_beta = 0.0, 0.0
        _surface_computed = False

        # Session boundary
        if date_arr is not None:
            _cur_date = str(date_arr[i])
            if _prev_date is not None and _cur_date != _prev_date:
                # Fix #2: Reset daily trade counter at session boundary
                trades_today = 0
                pending_entry = False  # cancel pending entry across day boundary
                if in_position:
                    # Force close at day boundary using last bar's values at mtc=0 (expiry).
                    # M2M returns accumulated during hold used actual mtc values. The final
                    # delta from prev_position_value (at actual mtc) to close_val (at mtc=0)
                    # captures the remaining time-value decay to expiry.
                    prev_s = spot_arr[i-1]
                    if np.isnan(prev_s) or prev_s <= 0:
                        # Data gap at session boundary — skip force-close pricing,
                        # use entry_net_value as close value (assume flat exit).
                        prev_s = spot_arr[max(0, i-2)] if i >= 2 else spot_arr[0]
                        if np.isnan(prev_s) or prev_s <= 0:
                            prev_s = 5000.0  # last resort fallback
                    prev_iv = max(iv_arr[i-1], 0.01)
                    if has_legs:
                        # MEDIUM (2026-06-01 holistic review): price the settlement
                        # close_val on the SAME empirical-IV + Edgeworth surface the
                        # in-hold marks (prev_position_value) used, for pricing
                        # consistency in the final-bar M2M delta. Use bar i-1's grid
                        # (the last in-position bar, which set prev_position_value).
                        # NOTE: mtc=0 → _edgeworth/_empirical short-circuit to pure
                        # intrinsic, so this is numerically a no-op TODAY; it removes
                        # the plain-BS/surface mismatch and is robust if the
                        # settlement mtc ever becomes positive.
                        _prev_grid = grid_iv_matrix[i-1] if grid_iv_matrix is not None else None
                        if _prev_grid is not None:
                            _prev_rho, _prev_beta = _estimate_surface_params(_prev_grid, prev_iv)
                        else:
                            _prev_rho, _prev_beta = 0.0, 0.0
                        close_val = _net_pos_value(active_legs, strikes,
                                                   prev_s, 0.0, prev_iv,
                                                   grid_ivs=_prev_grid,
                                                   rho_t=_prev_rho, beta_t=_prev_beta)
                        # Settlement: European cash-settled, OTM expires free
                        exit_cost = _settlement_exit_cost(
                            active_legs, strikes, prev_s, prev_iv,
                            fee_per_leg) * cost_multiplier
                        pnl = ((close_val - entry_net_value) - entry_cost - exit_cost) * entry_n_contracts
                        # bar_returns: final M2M delta (to mtc=0) + exit cost
                        final_m2m = (close_val - prev_position_value) * entry_n_contracts
                        bar_returns[i-1] += final_m2m / notional
                        bar_returns[i-1] -= exit_cost * entry_n_contracts / notional
                    else:
                        pnl = (prev_s - entry_spot) * entry_n_contracts
                        # Simple path: final M2M delta to bar_returns for consistency
                        bar_returns[i-1] += (prev_s - prev_spot) * entry_n_contracts / notional
                    equity += pnl / notional
                    if equity <= -1.0:
                        _ruined = True  # B3: account blown up (≤ −100%)
                    trades.append(Trade(entry_bar, i-1, GSide.NEUTRAL,
                                        entry_spot, prev_s, pnl, i-1 - entry_bar,
                                        exit_reason="session"))
                    in_position = False
            _prev_date = _cur_date

        if i < warmup_bars:
            continue

        if spot <= 0 or np.isnan(spot):
            continue

        if in_position:
            bars_held = i - entry_bar

            # Per-bar mark-to-market return (matches original evaluator)
            if has_legs:
                # Lazy surface param computation — only on bars that need pricing
                if not _surface_computed and _bar_grid is not None:
                    _bar_rho, _bar_beta = _estimate_surface_params(_bar_grid, iv)
                    _surface_computed = True
                curr_val = _net_pos_value(active_legs, strikes, spot, mtc, iv,
                                          grid_ivs=_bar_grid, rho_t=_bar_rho, beta_t=_bar_beta)
                bar_pnl = (curr_val - prev_position_value) * entry_n_contracts
                prev_position_value = curr_val
            else:
                bar_pnl = (spot - prev_spot) * entry_n_contracts
                prev_spot = spot
            bar_returns[i] += bar_pnl / notional  # normalize by notional

            # Position-aware UnrealizedProfitPct (live, this bar). The fraction of
            # MAX PROFIT captured = (curr_val - entry_net_value)/|entry_net_value|
            # (credit: entry_net_value<0, curr_val→0 as the spread decays → +1 at
            # max profit; debit: same formula on the debit paid). Uses the HAIRCUT
            # entry_net_value (the realized credit, == the stop-loss block's
            # `credit_received`) so the entry haircut shows as the small negative
            # start. Only computed for legged structures with the terminal in use.
            if _exit_uses_upp:
                if has_legs and abs(entry_net_value) > _UPP_DENOM_FLOOR:
                    upp_now = (curr_val - entry_net_value) / abs(entry_net_value)
                else:
                    upp_now = 0.0  # no credit basis (simple path / ~0 credit)
                upp_hist.append(upp_now)

            # Stop-loss / max-loss check.
            # Credits: time-dependent stop-loss (delta-dependent base × time factor).
            # Debits: exit when unrealised loss > 80% of debit paid, OR
            # still underwater after 40 bars (theta makes recovery
            # improbable for 0DTE; analyst recommendation).
            _exit_by_stop = False
            if has_legs and stop_loss_credit_multiple and stop_loss_credit_multiple > 0:
                # M2M unrealised from RAW entry value (consistent with
                # prev_position_value fix). The haircut cost is captured
                # in the final PnL at exit, not in per-bar M2M.
                unrealised = (curr_val - raw_entry_val) - entry_cost
                if template.is_credit and entry_net_value < -0.01:
                    # Time-dependent credit stop-loss: tighter as τ→0.
                    #
                    # GP STOP GENE (Phase 1, docs/viable_run_plan_2026_06_03.md):
                    # `stop_loss_credit_multiple` IS the per-individual evolved
                    # base `stop_mult` (gp_engine.Individual.stop_mult), threaded
                    # in by fitness.py. It is now the base in BOTH the V1 and the
                    # Level-B (delta_tree-active) paths — it REPLACES the former
                    # hardcoded delta-dependent base. The time-dependent factor
                    # below and the execution discount remain the "adjustments on
                    # top of" the evolved base, per the plan. (The retired
                    # heuristic was `(1.5 + short_delta×2.0)`: 15d→1.8×, 25d→2.0×,
                    # 40d→2.3×. With the gene the GP discovers the level directly,
                    # including 0.0 = hold-to-expiry, which the `> 0` gate above
                    # disables entirely. A future state-dependent `stop_tree`
                    # could re-introduce a delta/τ-conditional shape.)
                    # Time adjustment (continuous in mtc):
                    # At open (mtc~400): base × 1.25 (recovery possible).
                    # Near force-close (mtc~25 == 15:50): base × ~0.78; the 0.75
                    # floor (mtc→0, gamma concentrated) is asymptotic — the
                    # position force-closes at EOD_FORCE_CLOSE_MTC first.
                    _base_mult = stop_loss_credit_multiple * STOP_LOSS_EXECUTION_DISCOUNT
                    # Change D: 0.85x execution discount applied above.
                    # QC fills stop-loss orders at worse prices than BS mid,
                    # so the effective stop-loss triggers earlier. Without this
                    # discount, proxy allows recovery between its wider stop
                    # and QC's tighter stop, creating false positive strategies.
                    # Calibrated from QC v5 reconciliation (avg loss $391 vs
                    # proxy $172, 2.3x gap driven partly by stop level mismatch).
                    _time_factor = 0.75 + 0.50 * min(mtc / 400.0, 1.0)
                    _time_adjusted_mult = _base_mult * _time_factor
                    credit_received = abs(entry_net_value)
                    if unrealised < -_time_adjusted_mult * credit_received:
                        _exit_by_stop = True
                        # (M1-fin: the adverse-selection penalty that used to be a DEAD
                        # 1.15x markup on `unrealised` here is now applied to the
                        # REALIZED PnL at the exit below — STOP_LOSS_FILL_MARKUP — where
                        # it actually affects the booked loss. `unrealised` past this
                        # point is unused on the credit path.)
                elif not template.is_credit and entry_net_value > 0.01:
                    # Debit max-loss: loss > 80% of risk basis.
                    # Risk basis = max(entry_gross, net_debit) — same as sizing.
                    # For simple debit spreads: gross ≈ net, so this ≈ 80% of debit.
                    # For ratio backspreads: gross >> net (e.g. $18 vs $2), so the
                    # stop allows normal intraday gamma fluctuations instead of
                    # exiting on $1.60 noise. Without this, ratio structures get
                    # stopped before the breakout payoff can develop.
                    _debit_risk_basis = max(entry_gross, entry_net_value)
                    if unrealised < -0.80 * _debit_risk_basis:
                        _exit_by_stop = True
                    # Debit time-decay gate: still underwater after N bars.
                    # Ratio backspreads and 3-leg debit structures need to hold
                    # for gamma payoff. Any structure with ratio > 1 legs or 3+
                    # legs gets extended tolerance (240 bars ≈ 4 hours).
                    _has_ratio = any(l.ratio > 1 for l in active_legs)
                    _debit_underwater_bars = 240 if (template.n_legs >= 3 or _has_ratio) else 40
                    if bars_held >= _debit_underwater_bars and unrealised < 0:
                        _exit_by_stop = True

            # Determine exit reason (priority: stop > signal > max_hold > eod)
            # Position-aware path: re-evaluate the exit tree at THIS bar with the
            # live UnrealizedProfitPct (the vectorized exit_signals[i] used the
            # 0.0 flat baseline for it). upp_hist[-1] is this bar's value.
            if _exit_uses_upp:
                _exit_fired = _exit_signal_at_bar(
                    exit_tree, i, terminal_data, within_day_pos, day_mask,
                    upp_hist[-1] if upp_hist else 0.0, upp_hist,
                )
            else:
                _exit_fired = exit_signals[i] > 0.5
            _exit_by_signal = (bars_held >= min_bars_in_trade and _exit_fired)
            _exit_by_max_hold = (bars_held >= max_bars_in_trade)
            # R1/R2 parity (2026-05-29): EOD force-close at mtc <= EOD_FORCE_CLOSE_MTC.
            # mtc is measured to the 16:15 SPXW settle, so mtc<=25 == 15:50 ET,
            # matching codegen's wall-clock force-close exactly (codegen.py:510) and
            # staying out of the 16:00-16:15 settlement window. (Was mtc<=10 == 16:05
            # ET, ~15 min later than QC's 16:00 hard-guard and proxy-optimistic: the
            # proxy banked EOD theta + late holds that codegen never realized.)
            _exit_by_eod = (mtc <= EOD_FORCE_CLOSE_MTC)
            should_exit = _exit_by_stop or _exit_by_signal or _exit_by_max_hold or _exit_by_eod
            if should_exit:
                if _exit_by_stop:
                    _reason = "stop_loss"
                elif _exit_by_signal:
                    _reason = "signal"
                elif _exit_by_max_hold:
                    _reason = "max_hold"
                else:
                    _reason = "eod"
                # Exit cost: settlement (EOD) uses reduced cost; mid-session
                # exits (signal, stop, max_hold) pay full spread crossing.
                # Stop-loss exits get time-dependent slippage (1.2-1.5x) —
                # when SPX is moving fast enough to trigger a stop, fills are
                # worse than BS mid (gamma explosion near short strikes, wide
                # spreads in fast markets). Vol-conditional 1.25x on top.
                # E4 fix: EOD exit at mtc<=EOD_FORCE_CLOSE_MTC (15:50 ET) is an
                # ACTIVE close (market still liquid), not settlement. Use
                # spread-crossing cost. _settlement_exit_cost only at the session
                # boundary (mtc=0, line 870), which the force-close pre-empts.
                if has_legs:
                    exit_cost = _entry_costs(
                        active_legs, strikes, spot, mtc, iv,
                        fee_per_leg,
                        atm_spread=spread_arr[i],
                        grid_ivs=_bar_grid, rho_t=_bar_rho, beta_t=_bar_beta,
                    ) * cost_multiplier
                    if _reason == "stop_loss":
                        # Time-dependent slippage: worse fills later in day
                        # as gamma concentrates near ATM and spreads widen.
                        # Reduced from 1.5-2.0x to 1.2-1.5x (2026-05-19):
                        # prior calibration included pull-to-ATM correction
                        # that double-counted strike placement error. With
                        # pull-to-ATM removed, the credit factors alone
                        # capture the proxy-to-QC gap.
                        # Base 1.2 + 0.3 × (1 - hours_remaining / 6.5)
                        # → 1.20 at open, 1.50 near close.
                        hours_rem = max(mtc / 60.0, 0.0)
                        _sl_slippage = 1.2 + 0.3 * (1.0 - min(hours_rem / 6.5, 1.0))
                        # Regime-conditional slippage: QC v5 reconciliation
                        # showed avg loss $391 vs proxy $172 (2.3x gap). High-vol
                        # months (Jan/Apr/Oct 2025) had much worse fills.
                        # Calibration: low-vol fills are close to model, high-vol
                        # fills slip 25-50% more due to wider bid-ask and faster
                        # delta moves during stop-loss execution.
                        if iv >= 0.30:
                            _sl_slippage *= 1.50  # crisis: extreme slippage
                        elif iv >= 0.20:
                            _sl_slippage *= 1.25  # high vol: notable slippage
                        exit_cost *= _sl_slippage
                else:
                    exit_cost = 0.0
                # Total trade PnL (all values in $/share; scale by n_contracts for total)
                if has_legs:
                    close_val = _net_pos_value(active_legs, strikes, spot, mtc, iv,
                                               grid_ivs=_bar_grid, rho_t=_bar_rho, beta_t=_bar_beta)
                    pnl = ((close_val - entry_net_value) - entry_cost - exit_cost) * entry_n_contracts
                else:
                    pnl = (spot - entry_spot) * entry_n_contracts
                # M1-fin (2026-06-01): adverse-fill markup on the REALIZED stop loss —
                # a stop fills during the fast adverse move at a price worse than the
                # BS-mid close_val. Only on stop exits and only when it IS a loss
                # (a stop should never book a profit; the guard is defensive).
                # BLOCKER (2026-06-01 holistic review): the markup multiplies `pnl`
                # (→ equity + trade.pnl) but the per-bar Sharpe basis (bar_returns)
                # only books exit_cost below, so the extra adverse-fill loss was
                # INVISIBLE to Sharpe/Sortino (computed entirely from bar_returns).
                # Book the markup delta into bar_returns[i] too so the realized stop
                # penalty reaches fitness AND sum(bar_returns) still reconciles to
                # sum(trade.pnl)/notional. Order matters: finalize `pnl` (post-markup)
                # BEFORE computing its bar_returns contribution. `_markup_delta` is the
                # ADDED loss (pnl_post - pnl_pre = pnl - pnl/MARKUP, negative for a loss).
                if _reason == "stop_loss" and pnl < 0.0:
                    pnl *= STOP_LOSS_FILL_MARKUP
                    _markup_delta = pnl - pnl / STOP_LOSS_FILL_MARKUP  # = pnl_post - pnl_pre (< 0)
                    bar_returns[i] += _markup_delta / notional  # pnl already ×n_contracts
                bar_returns[i] -= exit_cost * entry_n_contracts / notional
                equity += pnl / notional
                if equity <= -1.0:
                    _ruined = True  # B3: account blown up (≤ −100%)
                trades.append(Trade(entry_bar, i, GSide.NEUTRAL,
                                    entry_spot, spot, pnl, bars_held,
                                    exit_reason=_reason))
                in_position = False
        else:
            # R1/R2 parity (2026-05-29): no new entries once mtc <= ENTRY_CUTOFF_MTC.
            # mtc to the 16:15 settle, so mtc<=30 == 15:45 ET, matching codegen's
            # entry cutoff (codegen.py:509). (Was mtc<=15 == 16:00 ET, ~15 min later
            # than QC and proxy-optimistic — extra late entries QC never took.)
            if mtc <= ENTRY_CUTOFF_MTC:
                pending_entry = False
                continue
            # B3 (2026-06-01): ruined account (equity ≤ −100%) takes no further
            # entries — the rest of the window is dead (a blown-up cash account).
            if _ruined:
                pending_entry = False
                continue
            # Fix #2: Daily trade cap
            if trades_today >= MAX_TRADES_PER_DAY:
                pending_entry = False
                continue
            # Fix #3: 1-bar execution delay — signal at bar N, fill at bar N+1.
            # QC has ~1 min latency from signal eval to fill. In proxy, entering
            # at the same bar as the signal gives an unrealistic "peek" at the
            # exact price that triggered the condition.
            if pending_entry:
                size = pending_size
                pending_entry = False
            elif entry_signals[i] > 0.5:
                pending_entry = True
                pending_size = max(0.0, min(1.0, size_signals[i]))
                continue  # will execute next bar
            else:
                continue
            if size > 1e-6 and has_legs:
                # Level B: compute dynamic legs from delta_tree output
                if delta_signals is not None and template.delta_range is not None:
                    _dyn = _compute_dynamic_legs(template, delta_signals[i])
                    if _dyn is not None:
                        from layer2.templates import Leg as _Leg
                        active_legs = tuple(
                            _Leg(ot, d, qty_sign=qs, ratio=r)
                            for ot, d, qs, r in _dyn
                        )
                        _entry_short_delta = abs(_dyn[0][1])
                    else:
                        active_legs = template.legs
                        _entry_short_delta = None  # fallback: V1 fixed stop/haircut
                else:
                    active_legs = template.legs
                    _entry_short_delta = None  # V1: use fixed haircut/stop-loss

                # Compute strikes for this entry
                strikes = [
                    MultiLegOptionsBacktester._delta_to_strike(
                        spot, leg.delta_target, iv, mtc
                    ) for leg in active_legs
                ]
                strikes = MultiLegOptionsBacktester._enforce_strike_separation(
                    strikes, active_legs
                )
                # Change C: Spread width cap $40 — matches QC codegen's
                # $40 maximum spread width. GP strategies with wider spreads
                # pass in proxy but get rejected in QC, creating false
                # positive strategies that fail on deployment.
                # Use same leg-pair pairing as the margin gate below:
                # 4-leg (IC/IB): pairs are (0,1) and (2,3).
                # 2-3 leg: consecutive pairs.
                if len(strikes) >= 2:
                    if len(strikes) == 4:
                        _max_width = max(abs(strikes[0] - strikes[1]),
                                         abs(strikes[2] - strikes[3]))
                    else:
                        _max_width = max(
                            abs(strikes[j] - strikes[j+1])
                            for j in range(len(strikes) - 1)
                        )
                    if _max_width > MAX_SPREAD_WIDTH:
                        pending_entry = False
                        continue  # skip this trade — spread too wide for QC
                # Entry value and sizing.
                # Credit haircut: BS/Edgeworth model overstates 0DTE credits
                # vs real fills. Call 25%, put 15% haircut. Evidence:
                # - Bandi, Fusari & Reno (2024, JoF): Edgeworth model
                # prices within bid-ask for only 80% of 0DTE options.
                # - Muravyev & Pearson (2020): true option values closer
                # to bids than asks (mid-price overstates seller proceeds).
                # - Call > put spread asymmetry from dealer delta-hedging
                # costs (documented SPX microstructure pattern).
                # Afternoon penalty (10% after 13:00 ET): theta decay
                # accelerates faster than BS implies near expiry.
                if not _surface_computed and _bar_grid is not None:
                    _bar_rho, _bar_beta = _estimate_surface_params(_bar_grid, iv)
                    _surface_computed = True
                # Edgeworth KEPT for entry credit (2026-05-15 verification):
                # With realistic SPX skew (rho<0, beta>0), Edgeworth REDUCES
                # credit by ~12% — makes proxy more conservative, not less.
                # Disabling it would inflate proxy credit, widening the QC gap.
                raw_entry_val = _net_pos_value(active_legs, strikes, spot, mtc, iv,
                                               grid_ivs=_bar_grid, rho_t=_bar_rho, beta_t=_bar_beta)
                if raw_entry_val < -0.01:  # credit spread
                    # Per-leg-pair haircut: call-side credit gets 25%, put-side 15%.
                    # For IC/IB (mixed call+put), split by leg type to avoid
                    # penalizing the put side at the call rate.
                    call_credit = 0.0
                    put_credit = 0.0
                    for leg, strike_k in zip(active_legs, strikes):
                        if leg.qty_sign >= 0:
                            continue  # long legs don't contribute credit
                        if _bar_grid is not None:
                            leg_iv = _empirical_iv(_bar_grid, leg.delta_target, iv)
                            lv = _edgeworth_option_value(spot, strike_k, mtc, leg_iv,
                                                          leg.option_type == "call",
                                                          _bar_rho, _bar_beta)
                        else:
                            leg_iv = _skew_iv(iv, spot, strike_k, mtc)
                            lv = _option_value(spot, strike_k, mtc, leg_iv,
                                                leg.option_type == "call")
                        short_val = abs(leg.qty_sign * leg.ratio * lv)
                        if leg.option_type == "call":
                            call_credit += short_val
                        else:
                            put_credit += short_val
                    # Regime-conditional credit factor. Base factors at module
                    # level (_BASE_CREDIT_FACTORS), template lookup hoisted
                    # above loop (_base_credit_factor). Only regime multiplier
                    # is per-bar (depends on entry IV).
                    #
                    # Regime adjustment from raw ATM_IV at entry bar.
                    # Calibrated 2026-05-22 from QC v5 reconciliation:
                    # high-vol months had 0-25% WR, calm months 70-85%.
                    if iv < 0.10:
                        _regime_mult = 1.00  # low vol: base calibration
                    elif iv < 0.20:
                        _regime_mult = 1.00  # normal: base calibration
                    elif iv < 0.30:
                        _regime_mult = 0.85  # high vol: wider spreads
                    else:
                        _regime_mult = 0.70  # crisis: extreme widening
                    _credit_factor = _base_credit_factor * _regime_mult
                    haircut_credit = (call_credit + put_credit) * _credit_factor
                    # Reconstruct entry_net_value: raw credit minus haircut
                    # raw_entry_val is negative (credit); scale proportionally
                    if call_credit + put_credit > 0.01:
                        haircut_ratio = haircut_credit / (call_credit + put_credit)
                    else:
                        haircut_ratio = 0.85
                    entry_net_value = raw_entry_val * haircut_ratio
                elif raw_entry_val > 0.01:  # debit spread
                    # Fill slippage: 10% for 2-leg directional debits,
                    # 5% for 3+ leg butterfly structures (partial offset
                    # from the broken wing reduces net slippage).
                    _debit_slippage = 1.05 if template.n_legs >= 3 else 1.10
                    entry_net_value = raw_entry_val * _debit_slippage
                else:
                    entry_net_value = raw_entry_val
                entry_gross = _gross_pos_value(active_legs, strikes, spot, mtc, iv,
                                               grid_ivs=_bar_grid, rho_t=_bar_rho, beta_t=_bar_beta)
                # Sizing basis: for credits, use gross (sum of absolute leg values).
                # For debits / ratio backspreads, use the LARGER of gross and
                # debit paid. Previous code used only debit paid ($3-5 for a
                # 3-leg debit), causing massive oversizing (89-116 contracts on
                # $1000 notional). A 2× short middle leg creates intra-day gamma
                # exposure far exceeding the debit — $6+/contract swings
                # wipe out the account in 2-3 trades, permanently blocking
                # all subsequent entries. Using gross captures the actual
                # intra-day risk exposure for sizing. (2026-05-19 fix)
                # Residual-1: haircut the model gross to real-fill gross for the
                # SIZING basis only (PnL/credit untouched). _gross_realism is a
                # constant per-template scale (Sharpe-neutral) that aligns proxy n
                # with QC's chain-priced (b)-sizing. Hoisted lookup _gross_realism.
                if template.is_credit:
                    abs_val = max(entry_gross * _gross_realism, 2.0)
                else:
                    abs_val = max(entry_gross * _gross_realism, abs(entry_net_value), 2.0)
                # Ratio adjustment: structures with ratio>1 legs (e.g., RPB
                # sell 1, buy 2) have more actual contracts per unit than
                # standard 2-leg spreads. Scale abs_val by total_ratio/2
                # so sizing reflects the true exposure per unit.
                # Standard spread: total_ratio=2 → multiplier=1 (unchanged).
                # RPB: total_ratio=3 → multiplier=1.5 (50% fewer units).
                _total_ratio = sum(l.ratio for l in active_legs)
                # FIX (2026-05-31): the ratio-adjustment + 5% "unlimited-risk" cap
                # below apply ONLY to genuine ratio structures (a leg with ratio>1,
                # e.g. RPB's 1-sell:2-buy). The previous `_total_ratio > 2` ALSO
                # fired for any 4-leg DEFINED-RISK spread (IC/IB sum 4 unit ratios
                # > 2), where the 5% cap used the meaningless cross-side strike gap
                # and forced n=1 — silently sizing every IC/IB to a single contract.
                # IC/IB are defined-risk (no unlimited tail), so the cap must not
                # apply. Gate on a real ratio leg instead.
                _has_ratio_leg = any(l.ratio > 1 for l in active_legs)
                if _has_ratio_leg:
                    abs_val *= _total_ratio / 2.0
                entry_n_contracts = min(
                    int(notional * size / abs_val),
                    int(notional / 2.0)  # MAX_CONTRACTS
                )
                # Hard cap for ratio structures: max 5% of notional at risk
                # per trade. Ratio backspreads have asymmetric risk (unlimited
                # between strikes) that gross doesn't capture. Defined-risk 4-leg
                # spreads (IC/IB) do NOT have this tail, so the cap must not apply
                # (see _has_ratio_leg above).
                if _has_ratio_leg:
                    _max_risk = notional * 0.05
                    if len(strikes) >= 2:
                        _max_width = max(abs(strikes[j] - strikes[j+1])
                                         for j in range(len(strikes)-1))
                        _max_by_risk = max(1, int(_max_risk / max(_max_width, 1.0)))
                        entry_n_contracts = min(entry_n_contracts, _max_by_risk)
                if entry_n_contracts < 1:
                    entry_n_contracts = 1
                # Leverage cap
                if entry_n_contracts * abs_val > 2 * notional:
                    entry_n_contracts = max(1, int(2 * notional / abs_val))
                # Margin-rejection gate: max_loss for defined-risk spreads
                # = max spread width within each leg pair × $100
                # Reject if total margin exceeds available capital (notional + equity).
                # This matches QC's behavior where insufficient margin rejects orders.
                if len(strikes) >= 2:
                    # Paired spread widths: (0,1), (2,3) for 4-leg; consecutive for 2-3 leg.
                    # 4-leg structures (IC, IB) have two independent spreads — consecutive
                    # indices would include the cross-side gap, hugely overestimating margin.
                    if len(strikes) == 4:
                        max_spread_width = max(abs(strikes[0] - strikes[1]),
                                               abs(strikes[2] - strikes[3]))
                    elif len(strikes) == 3 and not template.is_credit:
                        # 3-leg debit structure: use the wider wing width for
                        # margin, not just the debit paid. A 2x short
                        # middle leg creates intra-day gamma exposure far
                        # exceeding the debit. At expiry max loss = debit,
                        # but intra-day M2M can swing $6+/contract when IV
                        # spikes. Using debit ($3-5) as margin → 89+ contracts
                        # → account blowup in 2-3 trades. (2026-05-19 fix)
                        max_spread_width = max(abs(strikes[j] - strikes[j+1])
                                               for j in range(len(strikes)-1))
                    else:
                        max_spread_width = max(abs(strikes[j] - strikes[j+1])
                                               for j in range(len(strikes)-1))
                    # Units: max_spread_width is in $/share (e.g. $20 for
                    # a 20-pt IC). The entire evaluator works in $/share —
                    # notional, entry_cost, PnL are all per-share. No SPX
                    # ×100 multiplier here; that would convert to $/contract
                    # while comparing against $/share capital, blocking all
                    # trades with wider (correct) wings.
                    # Credit offset: real brokerages reduce margin by the
                    # credit received (the credit is collateral). For an IC
                    # with $25 wings and $6 credit, effective margin = $19.
                    if entry_net_value < -0.01:
                        margin_per_contract = max(0.0, max_spread_width - abs(entry_net_value))
                    else:
                        margin_per_contract = max_spread_width
                    # REALISTIC LIQUIDATION (2026-06-01, adversarial-audit fix): floor
                    # available_capital at 0, NOT at 10% of notional. The old 0.10 floor
                    # let the account keep trading on a ruined book, AMPLIFYING equity to
                    # -800%/-1450% (a cash account cannot lose >100%). That artifact (a)
                    # censored a usable proxy-side ruin signal, (b) inflated max_drawdown_
                    # uncapped (adj_dd ~29 vs ~6) which drives the soft drawdown penalty,
                    # and (c) made the proxy↔QC magnitude gap look "irreducible". With a 0
                    # floor, available_capital -> 0 as equity -> -100%, the margin gate
                    # below halts new entries, and equity bottoms near -100% (= QC margin
                    # liquidation). A blown-up strategy then correctly loses the rest of
                    # the window's opportunity (it is ruined) — the prior "GP sees only 18
                    # days" concern is moot for a strategy that SHOULD be rejected. Non-
                    # blown-up strategies (equity > -100%) are UNAFFECTED (floor never binds).
                    available_capital = notional * max(1.0 + equity, 0.0)
                    if margin_per_contract * entry_n_contracts > available_capital:
                        # Cap to what margin allows
                        safe_contracts = max(0, int(available_capital / max(margin_per_contract, 1)))
                        if safe_contracts < 1:
                            continue  # skip — insufficient margin
                        entry_n_contracts = safe_contracts
                    # Concentration cap: max 50% of capital in one trade.
                    # Prevents single-trade losses from exceeding 50% even
                    # under stale grid IV or extreme vol conditions.
                    _max_conc_contracts = max(1, int(0.50 * available_capital / max(margin_per_contract, 1)))
                    entry_n_contracts = min(entry_n_contracts, _max_conc_contracts)
                # Entry costs (scaled by cost_multiplier for sensitivity sweep).
                # DOUBLE-COUNT FIX (2026-06-03): for CREDIT-structure ENTRY the per-leg
                # bid-ask spread-crossing is ALREADY in the credit haircut
                # (_BASE_CREDIT_FACTORS = median(QC MarketOrder fill)/median(proxy
                # BS-mid); QC market orders cross the spread — diag_template_agnostic.py),
                # so charging it again here double-counts the entry bid-ask. MEASURED:
                # compound-entry BPC −0.48 → +1.02 once removed. Skip it for credit
                # entry (fee + flat-uncertainty charge stay). Debit entry keeps it
                # (its _debit_slippage adjustment is NOT QC-calibrated — separate scope).
                _entry_is_credit = raw_entry_val < -0.01
                entry_cost = _entry_costs(
                    active_legs, strikes, spot, mtc, iv,
                    fee_per_leg,
                    atm_spread=spread_arr[i],
                    grid_ivs=_bar_grid, rho_t=_bar_rho, beta_t=_bar_beta,
                    skip_spread_crossing=_entry_is_credit,
                ) * cost_multiplier
                in_position = True
                entry_bar = i
                entry_spot = spot
                upp_hist.clear()  # fresh per-trade UnrealizedProfitPct history (no cross-trade leak)
                if _sizing_log is not None:
                    # calibration recorder: the proxy's exact per-trade sizing,
                    # to compare against QC's traded n_contracts (orders API).
                    _sizing_log.append({
                        "entry_bar": int(i),
                        "n_contracts": int(entry_n_contracts),
                        "gross": float(entry_gross),
                        "net": float(entry_net_value),
                        "equity_before": float(equity),
                        "strikes": [float(_s) for _s in strikes],
                    })
                # (entry_iv/entry_mtc assignments removed — write-only)
                # M2M tracks from RAW entry value (not haircutted). The
                # haircut is fill slippage — a sunk cost at entry, not an
                # ongoing mark change. Using haircutted entry_net_value
                # created a phantom loss on bar 1 (24-34% of credit) that
                # tightened effective stop-loss from 2.5× to ~2.0× and
                # drove 55%+ stop-loss exit rates. (2026-05-19 P0 fix)
                prev_position_value = raw_entry_val
                trades_today += 1  # Fix #2: count toward daily cap
                bar_returns[i] -= entry_cost * entry_n_contracts / notional
                # A1 fix (2026-05-31, adversarially verified to machine precision):
                # book the credit HAIRCUT into the Sharpe basis at the entry bar.
                # M2M telescopes from raw_entry_val (prev_position_value, kept
                # unchanged so the stop-loss path is untouched), so the haircut
                # (raw_entry_val - entry_net_value — a cost for a credit) was
                # realized ONLY in trade.pnl/equity, which the fitness never reads
                # → every credit champion's selected Sharpe was optimistic by the
                # full haircut. Booking it here reconciles sum(bar_returns) to
                # sum(trade.pnl)/notional exactly (regression-tested). Sign: the
                # `+= (raw - net)` form is the one that reconciles (verified; the
                # opposite sign does NOT).
                bar_returns[i] += (raw_entry_val - entry_net_value) * entry_n_contracts / notional
            elif size > 1e-6:
                # SimpleBacktester path (no legs)
                entry_n_contracts = 1
                entry_cost = 0.0
                in_position = True
                entry_bar = i
                entry_spot = spot
                prev_spot = spot
                upp_hist.clear()  # fresh per-trade UnrealizedProfitPct history
                trades_today += 1

    # Force close at end
    # F2 fix: end-of-data force-close — only add exit cost to bar_returns,
    # not the full PnL (M2M already accumulated during the hold period).
    if in_position:
        last_spot = spot_arr[-1]
        last_iv = max(iv_arr[-1], 0.01)
        if has_legs:
            # MEDIUM (2026-06-01 holistic review): match the in-hold surface for the
            # final-bar M2M delta (see session-boundary settlement note). Use the last
            # bar's grid (the bar prev_position_value was last marked on). mtc=0 makes
            # this numerically a no-op today (intrinsic-only), but keeps the settlement
            # value on the same pricing surface as the in-hold marks.
            _last_grid = grid_iv_matrix[-1] if grid_iv_matrix is not None else None
            if _last_grid is not None:
                _last_rho, _last_beta = _estimate_surface_params(_last_grid, last_iv)
            else:
                _last_rho, _last_beta = 0.0, 0.0
            close_val = _net_pos_value(active_legs, strikes, last_spot, 0.0, last_iv,
                                       grid_ivs=_last_grid,
                                       rho_t=_last_rho, beta_t=_last_beta)
            # End-of-data = settlement: European cash-settled
            exit_cost = _settlement_exit_cost(
                active_legs, strikes, last_spot, last_iv, fee_per_leg) * cost_multiplier
            pnl = ((close_val - entry_net_value) - entry_cost - exit_cost) * entry_n_contracts
            # Final M2M delta (to mtc=0 settlement) + exit cost
            final_m2m = (close_val - prev_position_value) * entry_n_contracts
            bar_returns[-1] += final_m2m / notional
            bar_returns[-1] -= exit_cost * entry_n_contracts / notional
        else:
            pnl = (last_spot - entry_spot) * entry_n_contracts
        equity += pnl / notional
        trades.append(Trade(entry_bar, n_bars-1, GSide.NEUTRAL,
                            entry_spot, last_spot, pnl, n_bars-1-entry_bar,
                            exit_reason="end_of_data"))

    # Compute statistics — aggregate to DAILY returns for Sharpe.
    # Per-bar Sharpe on 357K bars with massive zero-dilution is statistically
    # unsound: most bars have zero return (not in position), violating the
    # iid assumption that justifies sqrt(T) annualization. Daily aggregation
    # produces T = n_trading_days independent observations, with sqrt(252)
    # annualization (Lo 2002, "The Statistics of Sharpe Ratios").
    bpd = SimpleBacktester._derive_bars_per_day(data)
    _n_days = data["date"].nunique() if "date" in data.columns else max(1, len(data) // bpd)
    daily_returns = []  # initialized here to prevent NameError in conditional Sharpe block
    daily_arr = np.array([], dtype=np.float64)

    if date_arr is not None and len(bar_returns) > 0:
        # Aggregate per-bar returns to daily
        unique_dates = []
        daily_returns = []
        day_start = 0
        for i in range(1, n_bars):
            if str(date_arr[i]) != str(date_arr[i - 1]):
                daily_returns.append(float(np.sum(bar_returns[day_start:i])))
                unique_dates.append(str(date_arr[day_start]))
                day_start = i
        daily_returns.append(float(np.sum(bar_returns[day_start:])))
        daily_arr = np.array(daily_returns, dtype=np.float64)
        annualization = math.sqrt(252.0)
        returns_std = np.std(daily_arr) if len(daily_arr) > 1 else 1e-9
        sharpe = float(np.mean(daily_arr) / max(returns_std, 1e-9) * annualization)
        # Sortino: downside deviation only. Guarded (see _MAX_SORTINO/
        # _MIN_DOWNSIDE_OBS): require enough losing DAYS to trust the downside
        # dev; below that fall back to the (bounded) Sharpe rather than the
        # 1e-9-floor/10.0-free-pass artifact; cap |Sortino| at the plausibility
        # bound either way.
        downside = daily_arr[daily_arr < 0]
        _mean_daily = float(np.mean(daily_arr))
        _overall_std = float(np.std(daily_arr)) if len(daily_arr) > 1 else 0.0
        if len(downside) >= _MIN_DOWNSIDE_OBS:
            dd_dev = float(np.sqrt(np.mean(downside ** 2)))
            sortino = float(np.clip(_mean_daily / max(dd_dev, 1e-9) * annualization,
                                    -_MAX_SORTINO, _MAX_SORTINO))
        else:
            # M3 fix (2026-06-02): too few losing DAYS to trust an empirical
            # downside deviation — but do NOT fall back to clip(sharpe). That
            # made Sortino IDENTICAL to (collinear with) Sharpe, so it carried
            # zero independent downside-risk information in the few-loss regime
            # (which is exactly where a left-skewed tape lives). Instead keep
            # Sortino a genuine DOWNSIDE measure: estimate the downside deviation
            # from whatever losing days exist (semideviation about 0), and floor
            # it at a fraction of the OVERALL daily-return std — a magnitude-tied
            # floor, NOT the 1e-9 artifact floor that manufactured astronomical
            # Sortinos. With a magnitude-tied floor the result stays bounded and
            # distinct from Sharpe (which divides by the full, symmetric std).
            if len(downside) > 0:
                _emp_dd = float(np.sqrt(np.mean(downside ** 2)))
            else:
                _emp_dd = 0.0
            # Floor: 1/3 of the symmetric daily std. For a left-skewed series the
            # realized downside dev exceeds this and dominates -> Sortino is genuinely
            # independent of Sharpe. The floor only binds when there is essentially no
            # observed downside, where it keeps the denominator economically scaled
            # (not 1e-9); in THAT sub-regime Sortino = 3x Sharpe (bounded, rank-
            # identical to Sharpe) — still strictly better than the old 1x-collinear
            # clip(sharpe) and with no astronomical-Sortino artifact.
            _dd_floor = _DOWNSIDE_DEV_STD_FRACTION * _overall_std
            dd_dev = max(_emp_dd, _dd_floor, 1e-9)
            sortino = float(np.clip(_mean_daily / dd_dev * annualization,
                                    -_MAX_SORTINO, _MAX_SORTINO))
    else:
        # Fallback: per-bar (legacy path for data without date column)
        annualization = math.sqrt(252 * bpd) if bpd > 0 else 1.0
        returns_std = np.std(bar_returns) if len(bar_returns) > 1 else 1e-9
        sharpe = float(np.mean(bar_returns) / max(returns_std, 1e-9) * annualization)
        sortino = SimpleBacktester._sortino(bar_returns, bars_per_day=bpd)

    # Skewness and excess kurtosis of daily returns for DSR (Bailey & Lopez
    # de Prado 2014). Computed from the same daily_arr used for Sharpe/Sortino.
    from scipy.stats import skew as _skew_fn, kurtosis as _kurtosis_fn
    _return_skew = 0.0
    _return_kurtosis = 0.0
    if len(daily_arr) > 3:
        _return_skew = float(_skew_fn(daily_arr))
        _return_kurtosis = float(_kurtosis_fn(daily_arr))  # excess kurtosis

    # Max drawdown
    cum_returns = np.cumsum(bar_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = running_max - cum_returns
    max_dd_uncapped = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    # Cap at 1.0 per spec (legacy field); the uncapped value feeds the drawdown
    # objective so a -1x and a -8x drawdown are distinguishable.
    max_dd = min(max_dd_uncapped, 1.0) if notional > 0 else 0.0
    if notional <= 0:
        max_dd_uncapped = 0.0

    # Win rate + profit factor (gross_profit / gross_loss over trades)
    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(trades) if trades else 0.0
    _gross_profit = sum(p for p in pnls if p > 0)
    _gross_loss = -sum(p for p in pnls if p < 0)
    # >=1 means net positive edge; inf-capped at 10 when no losing trades (ALL WINS).
    # NO TRADES -> 0.0 (same value as all-losses); benign because the min_trades gate
    # rejects zero-trade strategies first, but note the conflation in diagnostics.
    profit_factor = (_gross_profit / _gross_loss) if _gross_loss > 1e-9 else (
        10.0 if _gross_profit > 0 else 0.0)

    # Exit utilization: fraction of trades closed by exit signal
    signal_exits = sum(1 for t in trades if t.exit_reason == "signal")
    exit_util = signal_exits / len(trades) if trades else 0.0

    # Max single-trade loss penalty (Red team Exploit 4: Sortino manipulation).
    # Strategies with many tiny wins and one catastrophic loss look great on
    # Sortino (low downside deviation from the average) but are disastrous
    # in live trading. Penalize if worst single trade > 5× average loss.
    if pnls and len(pnls) >= 10:
        losses = [p for p in pnls if p < 0]
        if losses:
            avg_loss = sum(losses) / len(losses)
            worst_loss = min(losses)
            if avg_loss < -1e-6 and worst_loss < 5.0 * avg_loss:
                # Worst loss is >5x the average loss — catastrophic tail
                sharpe *= 0.5
                sortino *= 0.5

    # Snapshot the Sharpe here — AFTER the sign-preserving catastrophic-tail 0.5x
    # above, but BEFORE the exit-degeneracy / max-hold subtractive penalties below —
    # for (a) the conditional-Sharpe-gap (entry selectivity independent of exit
    # quality) and (b) the B3/B4 profit-conditional gates. Those gates test only its
    # SIGN (_raw_sharpe <= 0), which the 0.5x cannot flip, so a profitable strategy
    # that the later max-hold penalty pushes <= 0 is NOT mis-killed by them.
    _raw_sharpe = sharpe

    # Exit-degeneracy / churning gates below penalize the GP from discovering
    # spread-crossing churners (now profit-conditional — see B3/B4).
    # Only count SIGNAL exits as churning — stop_loss exits within min_bars
    # are legitimate risk management, not degenerate spread-crossing.
    # (2026-05-19: 3-leg debit stop-losses in 1-10 bars were being counted as
    # churning, killing 86% of those strategies.)
    if trades and len(trades) >= 5:
        min_bar_exits = sum(1 for t in trades
                           if t.exit_bar - t.entry_bar <= min_bars_in_trade + 1
                           and t.exit_reason == "signal")
        min_bar_frac = min_bar_exits / len(trades)
        if min_bar_frac > 0.80 and _raw_sharpe <= 0.0:
            # HARD GATE: churning strategies get sentinel values.
            # Was multiplicative penalty (sharpe *= 1-frac) but at frac=1.0
            # this produces sharpe=0.0, destroying ALL gradient information
            # and creating a zero-profit Pareto attractor that fills the
            # front with 200+ functionally identical individuals.
            # 2026-06-02 (B3 finding): made PROFIT-CONDITIONAL (mirrors the #7
            # exit-util gate). A profitable fast-signal-exit (e.g. a debit
            # backspread that catches a fast move and locks the gain within
            # min_bars+1 bars) is NOT a spread-crossing churner — it has a
            # POSITIVE Sharpe and must survive. A genuine churner crosses the
            # spread repeatedly for no edge → negative Sharpe → still caught.
            # Gate on _raw_sharpe (the pre-cascade Sharpe saved above), NOT the
            # running `sharpe`, so the max-hold subtractive penalty applied later
            # in this same block cannot push a profitable strategy under the test.
            sharpe = -1e6  # sentinel — hard gate kills UNPROFITABLE churning strategies
            sortino = -1e6
        # Stop-loss churning gate: if >80% of trades are stop_loss exits
        # within min_bars, the entry tree systematically picks positions
        # that immediately fail. This is not risk management — it's a
        # degenerate entry pattern. Prevents gaming the signal-only
        # churning gate above by routing exits through stop_loss.
        early_stop_exits = sum(1 for t in trades
                               if t.exit_bar - t.entry_bar <= min_bars_in_trade + 1
                               and t.exit_reason == "stop_loss")
        early_stop_frac = early_stop_exits / len(trades)
        if early_stop_frac > 0.80 and _raw_sharpe <= 0.0:
            # 2026-06-02 (B3-twin finding): same PROFIT-CONDITIONAL guard as the
            # signal-churn gate above. A degenerate entry that systematically
            # immediately fails has a negative Sharpe and is still sentineled;
            # a strategy whose stops fire early but is net PROFITABLE (the stop
            # is genuine risk management on a winning edge) must survive.
            sharpe = -1e6
            sortino = -1e6
        # Max-hold exit penalty: strategies where >90% of exits are at
        # max_bars_in_trade have vestigial exit trees. For credit spreads,
        # holding to max_bars then forced exit is systematically profitable
        # (theta harvest) regardless of entry quality.
        max_hold_exits = sum(1 for t in trades
                            if t.exit_bar - t.entry_bar >= max_bars_in_trade - 1)
        max_hold_frac = max_hold_exits / len(trades)
        if max_hold_frac > 0.90:
            # >90% max-hold exits = vestigial exit tree. 2026-06-02 (finding #2):
            # was a HARD -1e6 sentinel, which killed the legitimate hold-to-close
            # theta-harvest strategy the comment two lines up admits "is
            # systematically profitable" — a selective-entry credit spread held to
            # settlement is the canonical 0DTE play, NOT a defect. Demote it with a
            # SUBTRACTIVE penalty instead of killing it: this preserves NSGA gradient
            # (no -1e6 flat landscape / zero-attractor — the reason the multiplicative
            # form was rejected in 2026-05-19) and lets a genuinely profitable strategy
            # survive on its real Sharpe while pushing a hold-to-LOSS degenerate below
            # zero. True non-selective "always-enter + hold" degeneracy is still caught
            # by the random-entry-null and fire-rate tautology gates downstream, so
            # softening here does not admit garbage onto the front.
            sharpe -= _MAX_HOLD_VESTIGIAL_PENALTY
            sortino -= _MAX_HOLD_VESTIGIAL_PENALTY

        # Exit-time variability penalty: learned clock detection.
        # If signal exits cluster at a single time-of-day (std < 15 bars
        # = 15 minutes), the exit tree is a clock, not market-adaptive.
        _signal_exit_trades = [t for t in trades if t.exit_reason == "signal"]
        if len(_signal_exit_trades) >= 5:
            _exit_bars_in_day = []
            for t in _signal_exit_trades:
                _exit_bars_in_day.append(
                    evaluator._within_day_pos[min(t.exit_bar, n_bars - 1)])
            _exit_std = float(np.std(_exit_bars_in_day))
            if _exit_std < 3.0 and _raw_sharpe <= 0.0:  # all exits within ~3 minutes
                # Relaxed from 10 bars (2026-05-19): 0DTE theta strategies
                # legitimately exit at consistent times (e.g., 15:30 daily).
                # 10-bar threshold penalized valid time-aware exits.
                # 3 bars catches only truly degenerate clock exits where
                # the exit tree is a trivial time comparison.
                # 2026-06-02 (B4 finding): made PROFIT-CONDITIONAL. A profitable
                # "always close near 15:45" 0DTE exit has std≈0 yet is the
                # canonical legitimate time-aware exit the comment above admits
                # is valid — it must survive. A clock exit that is also a NET
                # LOSER (the trivial time-comparison degeneracy with no edge)
                # has Sharpe<=0 and is still sentineled.
                sharpe = -1e6  # sentinel — hard gate kills UNPROFITABLE clock exits
                sortino = -1e6

    # Entry fire rate: fraction of eligible bars where entry signal fires.
    _eligible_mask = evaluator._within_day_pos >= warmup_bars
    _n_eligible = int(_eligible_mask.sum())
    _n_entry_fires = int((entry_signals[_eligible_mask] > 0.5).sum()) if _n_eligible > 0 else 0
    _fire_rate = _n_entry_fires / max(_n_eligible, 1)

    # #6 fix (2026-06-02): FLAT-bar fire rate — among eligible bars where the strategy
    # is FLAT (no position open, hence actually able to enter), how often does the
    # entry signal fire? This is the true "how unconditional is the entry DECISION"
    # measure. A DAY-SELECTIVE hold strategy is flat-and-idle on the days it (correctly)
    # skips, so its flat-bar fire rate is LOW even when its raw bar-level fire rate is
    # high — on the days it trades, the signal is true all day but it enters ONCE and
    # holds. The old tautology gate used the raw bar-level rate and so hard-sentineled
    # day-selective winners (the dominant profitable 0DTE pattern; surfaced by the
    # positive control, where a Sharpe ~+12.8 planted winner fired 0.42 > 0.35 and was
    # killed). Bars (entry_bar, exit_bar] are in-position; the entry_bar itself was flat
    # (it fired from flat -> counts as a flat-fire). A pure always-enter churner fires
    # on ~every flat bar -> flat rate ~1.0 (still caught); a day-selective hold has many
    # flat-idle bars on skipped days -> low flat rate (now passes).
    _in_pos_mask = np.zeros(n_bars, dtype=bool)
    for _t in trades:
        _in_pos_mask[_t.entry_bar + 1:min(_t.exit_bar + 1, n_bars)] = True
    _flat_elig_mask = _eligible_mask & ~_in_pos_mask
    _n_flat_elig = int(_flat_elig_mask.sum())
    _n_flat_fire = int((entry_signals[_flat_elig_mask] > 0.5).sum()) if _n_flat_elig > 0 else 0
    _fire_rate_flat = _n_flat_fire / max(_n_flat_elig, 1)

    # Conditional Sharpe gap: Sharpe(entry days) - Sharpe(all days).
    # Measures whether the entry condition selects BETTER days to trade,
    # not just more days. Tautological entries produce gap ~0.
    _cond_sharpe_gap = 0.0
    if date_arr is not None and len(daily_returns) > 5:
        # Build per-day entry flag: did any entry signal fire on ELIGIBLE bars
        # (exclude warmup bars 0-29 where entry signals can't execute).
        _day_had_entry = []
        _ds = 0
        _wdp = evaluator._within_day_pos
        for _di in range(1, n_bars):
            if str(date_arr[_di]) != str(date_arr[_di - 1]):
                _eligible = (_wdp[_ds:_di] >= warmup_bars)
                _day_had_entry.append(bool(
                    np.any(entry_signals[_ds:_di][_eligible] > 0.5)
                ) if _eligible.any() else False)
                _ds = _di
        _eligible = (_wdp[_ds:] >= warmup_bars)
        _day_had_entry.append(bool(
            np.any(entry_signals[_ds:][_eligible] > 0.5)
        ) if _eligible.any() else False)
        _day_flags = np.array(_day_had_entry, dtype=bool)

        if 3 < _day_flags.sum() < len(_day_flags) - 3:
            _entry_day_returns = daily_arr[_day_flags]
            _ann = math.sqrt(252.0)
            _entry_std = float(np.std(_entry_day_returns))
            _entry_sharpe = float(
                np.mean(_entry_day_returns) / max(_entry_std, 1e-9) * _ann
            ) if _entry_std > 1e-9 else 0.0
            # Use pre-penalty Sharpe so gap measures entry selectivity
            # independent of exit quality penalties (M1 review fix).
            # Supplementary: also require entry-day MEAN return >= all-day mean.
            # Prevents variance manipulation (Red team Exploit 7) where
            # selecting low-variance days inflates Sharpe without higher returns.
            _entry_mean = float(np.mean(_entry_day_returns))
            _all_mean = float(np.mean(daily_arr))
            if _entry_mean < _all_mean:
                _entry_sharpe = min(_entry_sharpe, _raw_sharpe)  # cap gap at 0
            _cond_sharpe_gap = _entry_sharpe - _raw_sharpe

    return BacktestResult(
        returns=bar_returns,
        trades=trades,
        equity_curve=cum_returns,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        total_trades=len(trades),
        win_rate=win_rate,
        exit_utilization=exit_util,
        n_days=_n_days,
        entry_fire_rate=_fire_rate,
        entry_fire_rate_flat=_fire_rate_flat,
        conditional_sharpe_gap=_cond_sharpe_gap,
        avg_position_size=(
            # L3 fix (2026-06-02): off-by-one. The size actually USED for a trade
            # is `pending_size = size_signals[signal_bar]` captured the bar BEFORE
            # the fill (1-bar execution delay at ~line 1679), i.e. the SIGNAL bar
            # `entry_bar - 1`. Reading `size_signals[entry_bar]` reported the fill
            # bar's size, which is not what was used. Average the signal-bar size,
            # guarding the day-boundary case where a fill on the day's first
            # eligible bar has its signal on the prior day (index >= 0 always holds
            # since entry_bar >= 1 for any executed trade, but guard defensively).
            float(np.mean([
                size_signals[max(t.entry_bar - 1, 0)] for t in trades
            ]))
            if trades else 0.5
        ),
        return_skew=_return_skew,
        return_kurtosis=_return_kurtosis,
        max_drawdown_uncapped=max_dd_uncapped,
        profit_factor=profit_factor,
    )


# ---------------------------------------------------------------------------
# Helper functions for option PnL computation
# (avoid method calls on backtester instances for speed)
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

# ---------------------------------------------------------------------------
# Grid IV column names and delta grid for empirical IV surface interpolation
# ---------------------------------------------------------------------------

# 11-point delta grid: sorted in increasing delta order for np.interp.
# Put deltas are negative (convention: delta_put = delta_call - 1),
# call deltas are positive. ATMp and ATMc/ATM overlap at |delta|=0.50.
GRID_IV_COLUMNS: tuple = (
    "IV_5Dp", "IV_10Dp", "IV_25Dp", "IV_40Dp", "IV_ATMp",
    "IV_ATM", "IV_ATMc",
    "IV_40Dc", "IV_25Dc", "IV_10Dc", "IV_5Dc",
)

# Delta values corresponding to each grid column, sorted ascending.
# Put side: negative deltas. Call side: positive deltas.
# ATMp = -0.50, ATM/ATMc = +0.50 (put-call parity at ATM).
_GRID_DELTAS = np.array([
    -0.50, -0.40, -0.25, -0.10, -0.05,
     0.05,  0.10,  0.25,  0.40,  0.50,
], dtype=np.float64)

def _empirical_iv(grid_ivs: np.ndarray, delta_target: float,
                  atm_iv: float) -> float:
    """Interpolate IV from the 11-point empirical grid at a target delta.

    Uses np.interp (linear interpolation) on the 10-point delta grid
    (ATM and ATMc are averaged into a single +0.50 point).

    Args:
        grid_ivs: (11,) array of IVs from the current bar, ordered as
            GRID_IV_COLUMNS (5Dp, 10Dp, 25Dp, 40Dp, ATMp, ATM, ATMc,
            40Dc, 25Dc, 10Dc, 5Dc).
        delta_target: target delta for interpolation. Negative for puts,
            positive for calls. Range approximately [-0.50, +0.50].
        atm_iv: fallback ATM IV if grid data is missing.

    Returns:
        Interpolated IV (annualized).
    """
    # Fast path: if all grid IVs are zero (data gap), fall back to ATM IV
    if grid_ivs is None or len(grid_ivs) != 11:
        return max(atm_iv, 0.01)

    # Check for all-zero grid (data gap for entire bar)
    _any_nonzero = False
    for _v in grid_ivs:
        if _v > 0.001:
            _any_nonzero = True
            break
    if not _any_nonzero:
        return max(atm_iv, 0.01)

    # Build 10-point IV array from 11 grid columns.
    # Order: ATMp(-0.50), 40Dp(-0.40), 25Dp(-0.25), 10Dp(-0.10), 5Dp(-0.05),
    # 5Dc(+0.05), 10Dc(+0.10), 25Dc(+0.25), 40Dc(+0.40), ATM/ATMc(+0.50)
    # Build 10-point IV list (plain Python — avoids np.empty allocation overhead)
    atm_val = grid_ivs[5]   # IV_ATM
    atmc_val = grid_ivs[6]  # IV_ATMc
    if atm_val > 0.001 and atmc_val > 0.001:
        atm_avg = 0.5 * (atm_val + atmc_val)
    elif atm_val > 0.001:
        atm_avg = atm_val
    elif atmc_val > 0.001:
        atm_avg = atmc_val
    else:
        atm_avg = atm_iv
    ivs_10 = [
        grid_ivs[4],   # IV_ATMp -> delta -0.50
        grid_ivs[3],   # IV_40Dp -> delta -0.40
        grid_ivs[2],   # IV_25Dp -> delta -0.25
        grid_ivs[1],   # IV_10Dp -> delta -0.10
        grid_ivs[0],   # IV_5Dp -> delta -0.05
        grid_ivs[10],  # IV_5Dc -> delta +0.05
        grid_ivs[9],   # IV_10Dc -> delta +0.10
        grid_ivs[8],   # IV_25Dc -> delta +0.25
        grid_ivs[7],   # IV_40Dc -> delta +0.40
        atm_avg,       # ATM/ATMc -> delta +0.50
    ]
    # Replace any zero entries with ATM IV (partial data gaps)
    for _k in range(10):
        if ivs_10[_k] < 0.001:
            ivs_10[_k] = atm_iv

    # Replace stale grid IVs that are suspiciously low relative to ATM IV.
    # During vol spikes, far-OTM grid points can retain pre-spike quotes
    # (e.g., IV_10Dc=6% when ATM=84%), causing the protective wings to be
    # priced as nearly worthless and producing losses exceeding max wing width.
    # Threshold: any grid IV < 30% of ATM IV is considered stale.
    _stale_floor = 0.30 * atm_iv
    for _k in range(10):
        if ivs_10[_k] < _stale_floor:
            ivs_10[_k] = atm_iv

    # UPPER sanity bound (BLOCKER B2-fin, 2026-06-01 holistic review): symmetric to
    # the stale-FLOOR. Corrupt-data bars (8.6% have ATM pinned at the 0.01 data-gap
    # floor) carry 300-500% IV cells; used raw they mark defined-risk spreads at up to
    # 34x their structural max loss (verified: 1.83% IC / 0.57% BPC / 0.36% BCC bars
    # violated net < -width), corrupting drawdown, stop-loss firing, ruin, and the
    # proxy↔QC calibration.
    #
    # HIGH (2026-06-01 holistic re-audit): the median-based `3*central` ceiling does
    # NOT protect the SINGLE-corrupt-cell case. When one cell is 5.0 and the other
    # nine are clean-low (e.g. 0.15), the median stays 0.15, so the ceiling is
    # 3*0.15 = 0.45 and the lone high cell is clamped only to 0.45 — still ~3x its
    # true 0.15 neighbours. A 25Δ short leg then marks at IV 0.45 instead of 0.15 and
    # the spread nets BEYOND its width (verified: 20-wide IC -> -20.69 < -20). Add a
    # third anchor relative to the robust LOW end: cap each cell at k * min_nonzero
    # (k=2.5). 2.5 is the empirically-verified value that (a) pulls the lone corrupt
    # cell back under the spread width on the single-corrupt cases AND (b) does NOT
    # clip a genuine SPX skew — a real 0DTE put skew sits <=~2x ATM-low, well under
    # 2.5x the min cell (verified: genuine [0.30..0.12] grid IC mark -4.37 unchanged;
    # crisis [0.70..0.30] not clipped). The absolute 1.0 still catches the 500% case;
    # max(..., 0.30) keeps the central anchor from over-clamping a genuinely low-IV bar.
    _sorted_iv = sorted(_v for _v in ivs_10 if _v > 0.001)
    if _sorted_iv:
        _central_iv = _sorted_iv[len(_sorted_iv) // 2]
        _min_iv = _sorted_iv[0]
        _iv_ceiling = min(1.0, max(3.0 * _central_iv, 0.30), 2.5 * _min_iv)
        for _k in range(10):
            if ivs_10[_k] > _iv_ceiling:
                ivs_10[_k] = _iv_ceiling

    # Clamp delta_target to grid range
    delta_clamped = max(-0.50, min(0.50, delta_target))

    # Linear interpolation
    result = float(np.interp(delta_clamped, _GRID_DELTAS, ivs_10))
    return max(result, 0.01)


def _phi(x: float) -> float:
    """Standard normal PDF using math functions (no scipy dependency)."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _estimate_surface_params(grid_ivs: np.ndarray,
                             atm_iv: float) -> tuple:
    """Estimate Edgeworth surface parameters rho_t and beta_t from grid IVs.

    rho_t (leverage / skew): central difference of IV at ATM.
        rho_t = (IV_ATMc - IV_ATMp) / (2 * ATM_IV)

    beta_t (vol-of-vol / kurtosis): curvature at ATM using 25-delta points.
        beta_t = (IV_25Dc + IV_25Dp - 2 * IV_ATM) / (0.25^2 * ATM_IV)

    Args:
        grid_ivs: (11,) array in GRID_IV_COLUMNS order.
        atm_iv: ATM IV for normalization.

    Returns:
        (rho_t, beta_t) tuple. Returns (0.0, 0.0) if data is missing.
    """
    if grid_ivs is None or len(grid_ivs) != 11 or atm_iv < 0.001:
        return 0.0, 0.0

    # IV_ATMc = grid_ivs[6], IV_ATMp = grid_ivs[4]
    iv_atmc = grid_ivs[6]
    iv_atmp = grid_ivs[4]
    if iv_atmc > 0.001 and iv_atmp > 0.001:
        rho_t = (iv_atmc - iv_atmp) / (2.0 * atm_iv)
    else:
        rho_t = 0.0

    # IV_25Dc = grid_ivs[8], IV_25Dp = grid_ivs[2], IV_ATM = grid_ivs[5]
    iv_25dc = grid_ivs[8]
    iv_25dp = grid_ivs[2]
    iv_atm = grid_ivs[5]
    if iv_25dc > 0.001 and iv_25dp > 0.001 and iv_atm > 0.001:
        beta_t = (iv_25dc + iv_25dp - 2.0 * iv_atm) / (0.0625 * atm_iv)
    else:
        beta_t = 0.0

    return rho_t, beta_t


def _edgeworth_option_value(spot: float, strike: float, mtc: float,
                            iv: float, is_call: bool,
                            rho_t: float = 0.0,
                            beta_t: float = 0.0) -> float:
    """Edgeworth-corrected option pricing (Bandi, Fusari & Reno 2024).

    Applies a multiplicative correction to Black-Scholes based on the
    Edgeworth expansion of the risk-neutral density:

        price = BS_price * (1 + rho_t * skew_term + beta_t * kurt_term)

    where:
        d = (K - S) / (S * sigma * sqrt(tau))   [standardized moneyness]
        skew_term = -(d^2 - 1) * phi(d) / 6     [He_2 Hermite]
        kurt_term = (d^3 - 3d) * phi(d) / 24    [He_3 Hermite]

    rho_t and beta_t are dimensionless IV surface shape parameters
    (skew asymmetry and smile curvature). No additional 1/√τ or 1/τ
    scaling is applied — the φ(d) Gaussian density naturally damps
    corrections in the wings.

    Put prices derived via put-call parity (single derivation, no
    independent put clamping) to guarantee parity holds exactly.

    For rho_t=beta_t=0, this reduces exactly to Black-Scholes.

    Args:
        spot: current underlying price (SPX)
        strike: option strike price
        mtc: minutes to close (0 = expiry)
        iv: implied volatility (annualized)
        is_call: True for call, False for put
        rho_t: leverage parameter (skew of risk-neutral density)
        beta_t: vol-of-vol parameter (excess kurtosis)

    Returns:
        Option value in $/share.
    """
    # If no Edgeworth parameters or at expiry, return plain BS
    if (abs(rho_t) < 1e-10 and abs(beta_t) < 1e-10) or mtc <= 0:
        return _option_value(spot, strike, mtc, iv, is_call)

    # Always price the CALL with Edgeworth, then derive put via parity.
    # This ensures put-call parity: Put = Call - (S - K).
    # The Edgeworth expansion applies a multiplicative correction to BS,
    # and applying it independently to calls and puts would break parity.
    bs_call = _option_value(spot, strike, mtc, iv, True)
    bs_price = bs_call  # correction applied to call

    # Guard against pathological inputs. If the BS call is near-zero (deep OTM
    # call = deep ITM put), fall back to plain BS for the requested side —
    # NOT bs_call, which would return $0 for a put worth $150+ of intrinsic.
    if spot <= 0 or strike <= 0 or iv < 0.001 or bs_price < 1e-10:
        return _option_value(spot, strike, mtc, iv, is_call)

    tau = mtc / (252.0 * 390.0)
    sqrt_tau = math.sqrt(tau)
    sigma_sqrt_tau = iv * sqrt_tau

    if sigma_sqrt_tau < 1e-12:
        return bs_price

    # Standardized moneyness
    d = (strike - spot) / (spot * sigma_sqrt_tau)
    phi_d = _phi(d)

    # ρ_t and β_t from _estimate_surface_params are dimensionless IV surface
    # shape ratios (skew asymmetry and smile curvature). They do NOT need
    # additional 1/√τ or 1/τ scaling — those factors caused the correction
    # to explode at 0DTE timescales (τ≈0.002), pushing every correction to
    # the clamp boundaries and producing a constant 0.80x/1.25x bias.
    #
    # Cap to physically plausible ranges. Typical SPX values:
    # ρ ∈ [-0.07, 0.02] (puts more expensive than calls)
    # β ∈ [1, 5] (smile curvature from wing IVs)
    rho_capped = max(-0.20, min(0.20, rho_t))
    beta_capped = max(-10.0, min(10.0, beta_t))

    # Hermite polynomial corrections (NO τ-scaling — params already dimensionless):
    # He₂(d) = d² - 1 (skew: asymmetry of density around mode)
    # He₃(d) = d³ - 3d (kurtosis: fat tails beyond Gaussian)
    # The φ(d) factor naturally damps corrections in the wings.
    skew_correction = -(d * d - 1.0) * phi_d / 6.0
    kurt_correction = (d * d * d - 3.0 * d) * phi_d / 24.0

    # Multiplicative correction factor
    correction = 1.0 + rho_capped * skew_correction + beta_capped * kurt_correction

    # Safety clamp: prevent negative prices or extreme adjustments.
    # With correct scaling, corrections are typically 0.5-5% (not 20-25%).
    # The φ(d) decay naturally bounds wing corrections, so this clamp
    # should rarely bind.
    correction = max(0.75, min(1.30, correction))

    # AUDIT FIX (2026-06-01): apply the Edgeworth correction to the shared TIME VALUE
    # (extrinsic) only — NOT the intrinsic. Put-call parity makes the time value
    # identical for both sides, while intrinsic is deterministic at expiry and carries
    # no risk premium to correct. The prior code multiplied the correction into the
    # call's FULL price then subtracted intrinsic via parity, so a small (~5%)
    # correction on a deep-ITM-call / OTM-put's LARGE intrinsic swamped the put's
    # SMALL time value — zeroing or halving 21.6% of OTM put wings (IC/BPC/RPB long
    # legs). Correcting the time value preserves parity exactly AND each side's
    # intrinsic. (Calls change only for ITM calls, and only by correction×intrinsic.)
    _call_intrinsic = max(spot - strike, 0.0)
    _corrected_tv = max(bs_price - _call_intrinsic, 0.0) * correction  # shared time value
    if is_call:
        return _call_intrinsic + _corrected_tv
    return max(strike - spot, 0.0) + _corrected_tv  # put = put-intrinsic + shared tv


def _option_value(spot: float, strike: float, mtc: float,
                  iv: float, is_call: bool) -> float:
    """Black-Scholes (1973) closed-form option pricing using math.erfc.

    Replaces the Brenner-Subrahmanyam ATM approximation which compressed
    spread credits 10-45x (moneyness adjustment near-unity for SPX-scale
    strikes). The CDF-based nonlinearity in BS produces realistic OTM decay
    and therefore realistic vertical-spread credits.

    Uses math.erfc (C-level complementary error function) for Phi(x),
    which is faster than scipy.stats.norm.cdf and exact to double precision.

    Args:
        spot: current underlying price (SPX)
        strike: option strike price
        mtc: minutes to close (0 = expiry)
        iv: implied volatility (annualized, ATM)
        is_call: True for call, False for put

    Returns:
        Option value in $/share (same scale as spot/strike).
    """
    # E10 fix: guard against zero/negative spot or strike (data gaps)
    if spot <= 0 or strike <= 0:
        return 0.0
    if mtc <= 0:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    tau = mtc / (252.0 * 390.0)
    sigma_sqrt_tau = iv * math.sqrt(tau)
    if sigma_sqrt_tau < 1e-12:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + 0.5 * sigma_sqrt_tau * sigma_sqrt_tau) / sigma_sqrt_tau
    d2 = d1 - sigma_sqrt_tau
    # Phi(x) = 0.5 * erfc(-x / sqrt(2))
    if is_call:
        return spot * 0.5 * math.erfc(-d1 / _SQRT2) - strike * 0.5 * math.erfc(-d2 / _SQRT2)
    else:
        return strike * 0.5 * math.erfc(d2 / _SQRT2) - spot * 0.5 * math.erfc(d1 / _SQRT2)


def _skew_iv(atm_iv: float, spot: float, strike: float,
             mtc: float = 60.0, skew_slope: float = -0.15) -> float:
    """Parametric vol skew using standardized moneyness with intraday dynamics.

    iv(K) = atm_iv × (1 + effective_slope × d_std)
    where d_std = (K - S) / (S × σ × √τ)  [standardized moneyness]

    Intraday skew dynamics (Todorov 2019; Dim, Eraker & Vilkov 2023):
    - Morning (mtc > 240): skew is relatively flat as dealers hedge
      overnight positions. effective_slope = 0.7 × base_slope.
    - Midday (120 < mtc ≤ 240): standard skew. effective_slope = base_slope.
    - Afternoon (mtc ≤ 120): skew steepens as gamma exposure concentrates
      near expiry. effective_slope = 1.3 × base_slope.

    OTM call floor: SPX OTM call IV ≈ 1.0× ATM (no deep discount).
    The linear skew model produces a "smirk" for puts but OTM calls
    are floored at ATM IV (Gatheral 2004: SPX vol surface is a smirk,
    not a symmetric smile — the discount is on the put side, not the call).

    Floor at 50% ATM, cap at 3× ATM to prevent pathological values.
    """
    if spot <= 0 or strike <= 0 or atm_iv < 0.001:
        return max(atm_iv, 0.01)
    tau = max(mtc, 1.0) / (252.0 * 390.0)
    sigma_sqrt_tau = atm_iv * math.sqrt(tau)
    if sigma_sqrt_tau < 1e-9:
        return atm_iv
    # Time-varying slope: flatter in morning, steeper in afternoon
    if mtc > 240:
        effective_slope = skew_slope * 0.7   # morning: flatter
    elif mtc > 120:
        effective_slope = skew_slope          # midday: standard
    else:
        effective_slope = skew_slope * 1.3   # afternoon: steeper
    # E2 fix: term-structure-aware damping. As tau→0, d_std explodes for any
    # finite moneyness offset, pushing all OTM strikes to the 3x IV cap.
    # Real 0DTE skew flattens toward expiry (Todorov 2019). Dampen the skew
    # contribution by sqrt(tau/tau_ref) where tau_ref = 60 min normalized.
    tau_ref = 60.0 / (252.0 * 390.0)
    tau_damping = min(1.0, math.sqrt(tau / tau_ref))
    d_std = (strike - spot) / (spot * sigma_sqrt_tau)
    d_std *= tau_damping  # flatten skew contribution at short tau
    iv = atm_iv * (1.0 + effective_slope * d_std)
    # Fix #4: SPX vol surface has a modest call-side discount (OTM calls
    # trade ~3-8% below ATM IV). The old floor at 1.0× ATM prevented any
    # call discount, overpricing short OTM calls in bull call debit spreads.
    # New floor at 0.92× ATM allows the empirical discount while preventing
    # the linear model from pushing calls to unrealistic 0.50× ATM.
    # Reference: Gatheral (2004), Todorov (2019) — SPX smile is a smirk,
    # not symmetric; call-side discount is modest but real.
    if strike > spot:
        iv = max(iv, atm_iv * 0.92)
    return min(max(iv, atm_iv * 0.5, 0.01), atm_iv * 3.0)


def _net_pos_value(legs, strikes, spot, mtc, iv,
                   grid_ivs=None, rho_t: float = 0.0,
                   beta_t: float = 0.0) -> float:
    """Net per-contract value: Σ qty_sign × ratio × leg_value.

    Uses empirical IV surface + Edgeworth correction when grid_ivs is
    available; falls back to parametric skew + plain BS otherwise.
    rho_t/beta_t should be pre-computed by caller via _estimate_surface_params
    to avoid redundant computation across _net/_gross/_entry_costs.

    MEDIUM finding (2026-06-01 holistic review) — DEFERRED, needs approval:
    IV is sampled from the empirical grid at the leg's TARGET delta
    (`leg.delta_target`), but BS then prices the $5-ROUNDED strike. When the
    empirical grid's put skew is STEEPER than the strike-finder's parametric
    `_skew_iv(skew_slope=-0.15)` (`MultiLegOptionsBacktester._delta_to_strike`),
    the chosen strike's REALIZED delta drifts away from target (measured: target
    -0.10 -> realized -0.20..-0.225 at mtc=30-180 on a steep grid; -0.25 ->
    -0.30+), so the leg is priced at a too-shallow-delta IV. The realized-delta
    IV resample (a 2-4 step fixed point on delta<->grid-IV) was prototyped and
    corrects entry/exit marks by ~$1-3/share on steep-skew bars — but that is
    NOT applied here, deliberately, for two reasons:
      (1) It shifts ENTRY CREDIT materially, which would INVALIDATE the per-
          template credit-correction factors (BPC=0.80/BCC=0.73/IC=0.78/IB=0.81)
          that were calibrated 2026-05-19 against 630 matched QC trades under the
          CURRENT target-delta sampling. Changing it requires re-running the
          proxy<->QC reconciliation + re-approval (no unilateral calibration
          change — (internal doc)).
      (2) The cleaner root-cause fix is to align the strike-finder's skew_slope
          with the empirical grid (so target == realized by construction), not
          to resample IV downstream — also an approval-gated calibration change.
    Tracked for the next calibration cycle. The drift is a strike-finder/pricer
    IV-MODEL mismatch, bounded by the per-leg no-arb cap below and the
    _empirical_iv ceiling; it does not breach structural max loss.
    """

    total = 0.0
    _leg_vals = []  # (leg, strike, clamped per-share val) for the M2 vertical clamp
    for leg, strike in zip(legs, strikes):
        if grid_ivs is not None:
            leg_iv = _empirical_iv(grid_ivs, leg.delta_target, iv)
        else:
            leg_iv = _skew_iv(iv, spot, strike, mtc)
        if grid_ivs is not None:
            val = _edgeworth_option_value(spot, strike, mtc, leg_iv,
                                          leg.option_type == "call",
                                          rho_t, beta_t)
        else:
            val = _option_value(spot, strike, mtc, leg_iv,
                                leg.option_type == "call")
        # Single-option no-arbitrage bound (B2-fin defense-in-depth, 2026-06-01
        # holistic review): a European call is worth ≤ spot, a put ≤ strike, both ≥ 0.
        # A self-contained backstop against any residual mispriced leg (the IV upper
        # clamp in _empirical_iv is the primary fix; this guarantees no single leg can
        # ever exceed its arbitrage bound regardless of IV-surface corruption). The
        # legs share the (clamped) surface, so the spread nets within its structural
        # width — verified: a 500%-IV corrupt bar marks a 50-wide IC at −4.83 (was −169).
        _leg_cap = spot if leg.option_type == "call" else strike
        val = min(max(val, 0.0), _leg_cap)
        _leg_vals.append((leg, strike, val))
        total += leg.qty_sign * leg.ratio * val
    # M2 fix (2026-06-02, CLAMP-ONLY — not the full arbitrage-free surface): the
    # empirical-IV grid can be NON-MONOTONE, so the two legs of a defined-risk
    # vertical pick up wildly different IVs and the spread marks BEYOND its
    # structural width (audit repro: worst ~2.4× width on ~0.05% of bars, plus
    # credit>width "riskless-arb" marks and a debit-netted credit spread). The
    # per-LEG no-arb cap above bounds each leg individually but NOT the spread
    # net. A same-right vertical's net value is a STRUCTURAL invariant independent
    # of IV: a long (debit) vertical lies in [0, width·r]; a short (credit)
    # vertical lies in [-width·r, 0] (this code's sign convention: credit < 0).
    # Clamp each matched same-right opposite-sign equal-ratio vertical pair's
    # combined contribution into that structural band and fold the correction
    # back into `total`. Only clean defined-risk vertical pairs are clamped —
    # ratio legs (unequal ratios, e.g. RPB 1:2) and unpaired/naked legs are left
    # to the per-leg cap so RPB/BWB sizing and the _BASE_CREDIT_FACTORS are
    # untouched. This enforces BOTH requirements at once: |net pair| ≤ width AND
    # credit-pair stays a credit / debit-pair stays a debit.
    _used = [False] * len(_leg_vals)
    for _a in range(len(_leg_vals)):
        if _used[_a]:
            continue
        leg_a, strike_a, val_a = _leg_vals[_a]
        for _b in range(_a + 1, len(_leg_vals)):
            if _used[_b]:
                continue
            leg_b, strike_b, val_b = _leg_vals[_b]
            # Same right, opposite long/short, equal ratio = a defined-risk vertical.
            if (leg_a.option_type == leg_b.option_type
                    and leg_a.qty_sign == -leg_b.qty_sign
                    and leg_a.ratio == leg_b.ratio
                    and abs(strike_a - strike_b) > 1e-9):
                r = leg_a.ratio
                width = abs(strike_a - strike_b)
                pair_contrib = (leg_a.qty_sign * r * val_a
                                + leg_b.qty_sign * r * val_b)
                # Structural direction (IV-independent): identify the LONG leg
                # (qty_sign +1) and its strike. Call: long lower strike = debit;
                # long higher = credit. Put: long higher strike = debit; long
                # lower = credit. (Standard vertical taxonomy.)
                if leg_a.qty_sign == 1:
                    long_strike, short_strike = strike_a, strike_b
                else:
                    long_strike, short_strike = strike_b, strike_a
                if leg_a.option_type == "call":
                    is_debit_vertical = long_strike < short_strike
                else:  # put
                    is_debit_vertical = long_strike > short_strike
                bound = width * r
                if is_debit_vertical:
                    lo, hi = 0.0, bound      # long vertical: a (bounded) debit
                else:
                    lo, hi = -bound, 0.0     # short vertical: a (bounded) credit
                clamped = min(max(pair_contrib, lo), hi)
                total += clamped - pair_contrib  # fold the correction into the net
                _used[_a] = True
                _used[_b] = True
                break
    return total


def _gross_pos_value(legs, strikes, spot, mtc, iv,
                     grid_ivs=None, rho_t: float = 0.0,
                     beta_t: float = 0.0) -> float:
    """Gross per-contract value: Σ |qty_sign × ratio × leg_value|."""

    total = 0.0
    for leg, strike in zip(legs, strikes):
        if grid_ivs is not None:
            leg_iv = _empirical_iv(grid_ivs, leg.delta_target, iv)
        else:
            leg_iv = _skew_iv(iv, spot, strike, mtc)
        if grid_ivs is not None:
            val = _edgeworth_option_value(spot, strike, mtc, leg_iv,
                                          leg.option_type == "call",
                                          rho_t, beta_t)
        else:
            val = _option_value(spot, strike, mtc, leg_iv,
                                leg.option_type == "call")
        total += abs(leg.qty_sign * leg.ratio * val)
    return total


# Calibrated spread multiplier table (|delta| → multiplier relative to ATM spread).
# Derived from SPX options market microstructure literature:
# - Muravyev & Pearson (2020) "Options Trading Costs Are Lower Than You Think"
# - CBOE SPX market data empirical analysis
# ATM SPX option spread ≈ $0.15-0.20 per share ($15-20 per contract).
# OTM spreads widen proportionally to distance from ATM.
_SPREAD_MULT_TABLE = [
    # Recalibrated for 0DTE SPX: tick-floor dominance at low deltas.
    # SPX options have $0.05 min tick (below $3) or $0.10 (above $3).
    # The dollar bid-ask spread for OTM options is often the SAME as ATM
    # ($0.05-$0.10) because the tick floor constrains it. Previous 10x
    # multiplier at 5-delta double-counted: the cost formula already
    # multiplies by option mid-value (which drops for OTM), so applying
    # a large multiplier ON TOP overstated dollar costs 2-3x.
    #
    # Sources: Muravyev & Pearson (2020, RFS) effective spreads;
    # CBOE SPX tick rules; practitioner fill reports (tastytrade, IBKR).
    (0.50, 1.0),   # ATM: $0.10-0.20 on $5-8 option
    (0.40, 1.2),   # near-ATM: similar dollar spread
    (0.35, 1.4),   # 35-delta
    (0.25, 1.8),   # 25-delta: tick floor kicks in ($0.05-0.10 on $1.50-3)
    (0.20, 2.0),   # 20-delta
    (0.15, 2.5),   # 15-delta (iron condor short legs)
    (0.10, 3.0),   # 10-delta: $0.05 tick on $0.20-0.80
    (0.05, 4.0),   # 5-delta: tick ≈ spread ($0.05 on $0.05-0.30)
]


def _spread_multiplier(abs_delta: float) -> float:
    """Interpolate spread multiplier from the empirical table.

    Args:
        abs_delta: absolute delta of the leg (0.05 to 0.50)
    Returns:
        multiplier relative to ATM spread
    """
    if abs_delta >= 0.50:
        return 1.0
    if abs_delta <= 0.05:
        return 4.0  # tick-floor constrained; was 10.0 (see table comment)
    # Linear interpolation between table entries
    for i in range(len(_SPREAD_MULT_TABLE) - 1):
        d_hi, m_hi = _SPREAD_MULT_TABLE[i]
        d_lo, m_lo = _SPREAD_MULT_TABLE[i + 1]
        if d_lo <= abs_delta <= d_hi:
            frac = (abs_delta - d_lo) / (d_hi - d_lo) if d_hi > d_lo else 0.0
            return m_lo + frac * (m_hi - m_lo)
    return 1.0


def _entry_costs(legs, strikes, spot, mtc, iv,
                 fee_per_leg,
                 atm_spread: float = 0.015,
                 grid_ivs=None, rho_t: float = 0.0,
                 beta_t: float = 0.0,
                 skip_spread_crossing: bool = False) -> float:
    """Calibrated entry transaction cost per contract set, in $/share units.

    Returns cost in the same $/share scale as _option_value() and
    _net_pos_value(), so costs can be directly subtracted from PnL without
    a multiplier mismatch.

    The spread model: relative_spread × delta_multiplier × option_mid_value
    converts the dimensionless (ask-bid)/mid ratio to a $/share crossing cost.
    OTM legs pay wider relative spreads (multiplier) but have lower mid-values,
    so the dollar cost is moderated.

    Args:
        atm_spread: ATM option relative spread (ask-bid)/mid, dimensionless.
            Median from training split ~= 0.015.
        fee_per_leg: Per-leg exchange/broker fee in absolute dollars (e.g. $2.50).
            Converted to $/share internally (÷100 for the SPX contract multiplier).
        grid_ivs: Optional (11,) array of grid IVs for empirical surface pricing.
    """
    cost = 0.0
    # Forward-fill zero spread (no quote ≠ free to trade)
    if atm_spread <= 0.001:
        atm_spread = 0.015  # fallback to training median
    # Time-of-day spread adjustment: 0DTE spreads follow a U-shape
    # (Muravyev & Pearson 2020; SEC DERA 2025). Tight midday, wide
    # at open (dealer hedging) and close (illiquidity).
    if mtc > 360:       # first 30 min: wider spreads (1.5×)
        tod_mult = 1.5
    elif mtc > 300:     # first hour: somewhat wider (1.2×)
        tod_mult = 1.2
    elif mtc <= 15:     # last 15 min: market makers widen 2-3x (audit: 1.5x optimistic)
        tod_mult = 2.5
    elif mtc <= 30:     # last 30 min: wider spreads (2.0×)
        tod_mult = 2.0
    elif mtc <= 60:     # last hour: somewhat wider (1.3×)
        tod_mult = 1.3
    else:               # midday: tightest (1.0×)
        tod_mult = 1.0

    for leg, strike in zip(legs, strikes):
        # Fee in $/share (fee_per_leg is absolute $ per contract; /100 for SPX multiplier)
        cost += (fee_per_leg / 100.0) * leg.ratio
        # Delta-dependent spread cost in $/share:
        # relative_spread × multiplier × option_mid_value (F1 fix: was missing × value)
        abs_delta = abs(leg.delta_target)
        rel_spread = atm_spread * _spread_multiplier(abs_delta) * tod_mult
        # Compute the leg's option mid-value for the dollar conversion
        if grid_ivs is not None:
            leg_iv = _empirical_iv(grid_ivs, leg.delta_target, iv)
            leg_value = _edgeworth_option_value(spot, strike, mtc, leg_iv,
                                                leg.option_type == "call",
                                                rho_t, beta_t)
        else:
            leg_iv = _skew_iv(iv, spot, strike, mtc)
            leg_value = _option_value(spot, strike, mtc, leg_iv,
                                      leg.option_type == "call")
        # Spread-crossing cost: fraction of relative spread × EXTRINSIC value.
        # Short legs fill closer to bid (Muravyev & Pearson 2020).
        # Call-side wider than put-side (dealer inventory structurally short calls).
        # Calibrated 2026-05-19: BCC credit ratio 0.73 vs BPC 0.80.
        if leg.qty_sign < 0:
            _spread_frac = 0.75 if leg.option_type == "call" else 0.65
        else:
            _spread_frac = 0.50
        # F-fix (2026-06-01): cross the bid-ask on EXTRINSIC (time value) only,
        # never intrinsic. A deep-ITM option settles at parity — you do not pay a
        # spread on its intrinsic. Charging spread × FULL mid fabricated phantom
        # exit costs on big-move days (deep-ITM exits): e.g. an RPB FOMC gap-down
        # was charged ~$1136/share (vs the correct ~$0.14) because the $150 mid was
        # ~all intrinsic. This over-stated proxy losses 8x on RPB and biased every
        # template's big-move exits. Verified in isolation vs _settlement_exit_cost.
        _intrinsic = (max(spot - strike, 0.0) if leg.option_type == "call"
                      else max(strike - spot, 0.0))
        _extrinsic = max(leg_value - _intrinsic, 0.0)
        # Fixed-$ FLOOR (2026-06-01 audit): the extrinsic-only term alone wrongly
        # collapsed deep-ITM exits to ~0 (the fee-floor) — OPTIMISTIC on the losing
        # tail. Real SPX option spreads are a roughly fixed DOLLAR width (~$0.10-0.50),
        # not a fraction of mid, so an active close still crosses the bid-ask even when
        # extrinsic≈0. Floor the per-leg spread-crossing at a modeled minimum. (Also
        # lightly over-charges cheap far-OTM wings — the SAFE/conservative direction.)
        # tod_mult-scaled (review 2026-06-01): late-day deep-ITM exits (when the floor
        # BINDS) must not pay the midday minimum — real spreads widen 2.5x near close.
        # DOUBLE-COUNT FIX (2026-06-03): skip_spread_crossing=True (credit ENTRY only)
        # omits this per-leg bid-ask crossing because the credit haircut already
        # captures it (calibrated to QC MarketOrder fills). Exits + debit entry keep it.
        if not skip_spread_crossing:
            cost += max(_spread_frac * rel_spread * _extrinsic,
                        _MIN_SPREAD_COST_PER_LEG * tod_mult) * leg.ratio
        # Fix #5: Flat uncertainty charge per leg
        cost += 0.02 * leg.ratio
    return cost


def _settlement_exit_cost(legs, strikes, spot, iv,
                          fee_per_leg: float = 2.50) -> float:
    """Reduced exit cost for positions held to settlement (mtc → 0).

    SPX options are European-style, cash-settled. OTM legs expire worthless
    automatically — no closing transaction, no commission, no spread crossing.
    ITM legs settle at intrinsic value automatically with only a minimal
    clearing fee.

    This function charges only the OCC clearing fee ($0.02/leg) plus
    exchange fee for legs that are ITM at expiry. No spread-crossing cost.
    """
    cost = 0.0
    for leg, strike in zip(legs, strikes):
        is_call = leg.option_type == "call"
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        if intrinsic > 0.01:
            # ITM at settlement: exercise/assignment fee (no spread crossing)
            cost += (fee_per_leg / 100.0) * leg.ratio
        # OCC clearing fee per leg regardless
        cost += 0.02 * leg.ratio
    return cost
