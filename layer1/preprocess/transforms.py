"""Stateless per-day transforms for L1-preprocess.

Ports the deterministic portions of pipeline.preprocess_day() and
pipeline.apply_vix_forwardfill() so they can be reused by:

  - the training pipeline (for SSL + probe training)
  - Ablation 3 (L1-preprocess + derived/ + GP + LLM, no encoder)
  - any downstream eval harness that needs the same preprocessing

TZ WARNING
----------
The staleness-ramp and options-tail forward-fill logic here assume the
bar index within a day is in a consistent timezone and that the
"options cutoff" is a real structural gap. D74 showed that cutoff may
be a phantom caused by a query-window TZ confusion. Until the TZ audit
(diagnostic_tz_audit.py) resolves the question, treat `compute_staleness_ramp`
and `forwardfill_options_tail_inplace` as POTENTIALLY PHANTOM and do not
remove them from pipeline.py yet — use them only behind a feature flag.
See project_experiment_state.md D74/D75 status.
"""

from __future__ import annotations

import numpy as np

# Fixed indices in the 141-variate raw layout. Kept here so the pipeline
# and the preprocess module stay in sync. If the 141-variate layout ever
# changes (e.g., re-collection), update this module in the same PR.
TOTAL_VARIATES = 141
OPTIONS_GRID_END = 88       # variates 0..87 (11 grid points x 8 feats)
RELIABILITY_END = 99        # variates 88..98 (grid reliability)
STRIKE_AGG_SLICE = slice(99, 105)
ORDER_FLOW_SLICE = slice(131, 140)
STALENESS_INDEX = 140

# Gamma indices in the moneyness grid (feature 3 of 8 per grid point)
GAMMA_INDICES = [i * 8 + 3 for i in range(11)]

# Net premium liquidity and other skewed order-flow / strike-agg variates
# that need signed log1p compression before z-scoring
SIGNED_LOG1P_INDICES = [101, 102, 104, 132, 139]

# VIX term-structure variates used in forward-fill and derived recomputation
VIX_SPOT_INDEX = 119
VIX_9D_INDEX = 120
VIX_3M_INDEX = 121
VIX_6M_INDEX = 122
VIX_F1_INDEX = 123
VIX_F2_INDEX = 124
VIX_FUTURES_SPREAD_INDEX = 128   # f1 - f2 (arithmetic duplicate — drop candidate)
VIX_F1_BASIS_INDEX = 129         # (f1 - vix) / vix (arithmetic duplicate — drop candidate)
VIX_TERM_SLOPE_INDEX = 130       # linear slope (arithmetic duplicate — drop candidate)


def gamma_clamp(X: np.ndarray) -> np.ndarray:
    """Clamp option gamma variates to [0, 100] in-place.

    Raw Black-Scholes gamma for deep-ITM 0DTE contracts can explode
    arbitrarily at T→0. The clamp protects the z-score statistics from
    being dominated by a handful of extreme bars.
    """
    for gi in GAMMA_INDICES:
        X[:, gi] = np.clip(X[:, gi], 0.0, 100.0)
    return X


def signed_log1p_inplace(X: np.ndarray) -> np.ndarray:
    """Apply sign(x) * log1p(|x|) to skewed variates in-place.

    Targets variates whose raw distribution is heavy-tailed and signed
    (net premium, directional imbalances). log1p compression stabilizes
    z-score statistics without losing sign information.
    """
    for vi in SIGNED_LOG1P_INDICES:
        col = X[:, vi]
        X[:, vi] = np.sign(col) * np.log1p(np.abs(col))
    return X


def vix_forwardfill_inplace(X: np.ndarray) -> np.ndarray:
    """Forward-fill VIX front/second futures (variates 123, 124) when zero,
    then recompute the three derived VIX variates (128, 129, 130).

    The CBOE VIX futures feed is minute-resolution but can have sporadic
    zero minutes. Forward-fill maintains term-structure continuity for
    the downstream encoder. The derived variates (128/129/130) are
    currently in the 141-layout but are arithmetic duplicates — per D75
    they are candidates for removal at re-collection time. Until then
    recomputing them here keeps them consistent with the forward-filled
    f1/f2.
    """
    n_bars = X.shape[0]

    for vi in (VIX_F1_INDEX, VIX_F2_INDEX):
        col = X[:, vi].copy()
        last_good = 0.0
        for b in range(n_bars):
            if col[b] != 0.0:
                last_good = col[b]
            elif last_good != 0.0:
                col[b] = last_good
        X[:, vi] = col

    # Derived VIX variates — arithmetic duplicates, kept for layout stability
    vix = X[:, VIX_SPOT_INDEX]
    f1 = X[:, VIX_F1_INDEX]
    f2 = X[:, VIX_F2_INDEX]
    X[:, VIX_FUTURES_SPREAD_INDEX] = f1 - f2
    safe_vix = np.where(vix > 0, vix, 1.0)
    X[:, VIX_F1_BASIS_INDEX] = np.where(vix > 0, (f1 - vix) / safe_vix, 0.0)

    # Term-structure slope: linear regression on [9, 30, 90, 180] day tenors
    tenors = np.array([9.0, 30.0, 90.0, 180.0])
    x_c = tenors - tenors.mean()
    denom = (x_c ** 2).sum()
    if denom > 0:
        for b in range(n_bars):
            levels = np.array([
                X[b, VIX_9D_INDEX],
                X[b, VIX_SPOT_INDEX],
                X[b, VIX_3M_INDEX],
                X[b, VIX_6M_INDEX],
            ])
            if levels.sum() > 0:
                y_c = levels - levels.mean()
                X[b, VIX_TERM_SLOPE_INDEX] = float(np.dot(x_c, y_c) / denom)

    return X


def detect_options_cutoff(X: np.ndarray) -> int:
    """Return the last bar index with any options-grid variate non-zero.

    Used by the tail forward-fill logic. Note that D74 suggests this
    cutoff may be an artifact of TZ/window handling rather than a real
    platform-level gap — this function still returns the empirical
    boundary in whatever index the data has.
    """
    grid_data = X[:, :OPTIONS_GRID_END]
    any_nonzero = np.any(grid_data != 0, axis=1)
    nonzero_bars = np.where(any_nonzero)[0]
    return int(nonzero_bars[-1]) if len(nonzero_bars) > 0 else 0


def forwardfill_options_tail_inplace(X: np.ndarray, cutoff_int: int) -> np.ndarray:
    """Forward-fill options/strike-agg/order-flow tails in-place past the cutoff.

    HONESTY NOTE (D75, pending D74 TZ resolution):
    Today this forward-fills theta, gamma, spread, order flow, and strike
    aggregates with their last-live values, which creates a constant
    signal in the "dead zone" that the encoder may latch onto as a
    spurious regime cue. If D74 confirms the dead zone is phantom, this
    function should be REMOVED entirely. If D74 confirms a real gap, this
    should be replaced with zero-fill (or NaN-fill + mask) so the encoder
    can distinguish live vs forward-filled bars.
    """
    n_bars = X.shape[0]
    if cutoff_int < n_bars - 1:
        X[cutoff_int + 1:, :RELIABILITY_END] = X[cutoff_int, :RELIABILITY_END]
        X[cutoff_int + 1:, STRIKE_AGG_SLICE] = X[cutoff_int, STRIKE_AGG_SLICE]
        X[cutoff_int + 1:, ORDER_FLOW_SLICE] = X[cutoff_int, ORDER_FLOW_SLICE]
    return X


def compute_staleness_ramp(n_bars: int, cutoff_int: int) -> np.ndarray:
    """Compute the 0->1 staleness ramp across the dead zone.

    PHANTOM CANDIDATE (D75, pending D74 TZ resolution):
    If the options "cutoff" is real, this ramp is a useful signal that
    tells the encoder how stale the forward-filled data is. If the
    cutoff is a phantom, this ramp is pure noise and must be removed.
    """
    staleness = np.zeros(n_bars, dtype=np.float32)
    tail_len = max(1, n_bars - 1 - cutoff_int)
    for b in range(cutoff_int, n_bars):
        staleness[b] = min(1.0, (b - cutoff_int) / tail_len)
    return staleness


def liveness_mask(n_bars: int, cutoff_int: int) -> np.ndarray:
    """Return a (n_bars, TOTAL_VARIATES) liveness mask.

    1.0 = bar is a real measurement for this variate.
    0.0 = bar was forward-filled (or otherwise synthesized).
    Used downstream by:
      - SSL masking (we don't pretext-reconstruct forward-filled bars)
      - probe training (exclude dead regions from supervision)
    """
    mask = np.ones((n_bars, TOTAL_VARIATES), dtype=np.float32)
    if cutoff_int < n_bars - 1:
        mask[cutoff_int + 1:, :RELIABILITY_END] = 0.0
        mask[cutoff_int + 1:, STRIKE_AGG_SLICE] = 0.0
        mask[cutoff_int + 1:, ORDER_FLOW_SLICE] = 0.0
    return mask
