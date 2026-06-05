"""Expert-curated derived features, v1.

Implements the ~40 named scalar features specified in
layer1_data_transformer_spec.md §11.8.3. Each feature is:

  - SCALE-STABLE       — bounded or compressed so GP trees don't need
                         to rediscover normalization per run
  - SEMANTICALLY-NAMED — GP trees can be inspected and reasoned about
  - ARITHMETICALLY COMPOSABLE — ratios, diffs, and products of these
                         produce meaningful new signals
  - FIXED HASH         — `derivation_spec_hash()` pins the exact set and
                         formula set so L2 runs are reproducible

Scaffolding status (D75, 2026-04-10)
------------------------------------
All 40 features are declared in DERIVED_FEATURE_REGISTRY with their
semantic name, category, raw-141 input dependencies, and a brief
formula description. The compute function bodies are implemented for
features that can be computed from the current 141-variate layout
WITHOUT depending on:

  - calendar events (FOMC, CPI, earnings week) — deferred until a
    calendar source is plumbed into the collector
  - opening range / session phase with correct timezone — deferred
    until D74 TZ audit resolves the ET vs CT question
  - macro regime context (SKEW index, VVIX) — deferred until those
    variates are added at re-collection time

Deferred features raise NotImplementedError with a clear reason in
the docstring. The feature_registry is the source of truth — the
pipeline should read it when deciding which derived columns exist.

No feature here mutates its inputs. All functions take (X, meta) where
X is (n_bars, 141) float32 already run through L1-preprocess, and meta
is a dict with optional keys like 'date', 'session_open_bar',
'is_fomc_day', etc. Functions return (n_bars,) float32 arrays.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Raw-141 index aliases — keep in sync with preprocess/transforms.py
# ============================================================

# IV surface — wing and ATM IV grid points
IV_10DP = 9        # 10-delta put IV
IV_25DP = 17       # 25-delta put IV
IV_ATM  = 41       # ATM put IV (used as the "ATM IV" reference)
IV_25DC = 65       # 25-delta call IV
IV_10DC = 73       # 10-delta call IV

# ATM microstructure (grid point 5 of 11, features 2/3/6/7)
ATM_THETA       = 42
ATM_GAMMA       = 43
ATM_SPREAD      = 46
ATM_SPREAD_CHG  = 47

# Reliability — grid point 5 reliability flag (0.0 / 0.5 / 1.0)
RELY_ATM = 93

# Strike aggregates (99-104)
STRIKE_MAX_IMB_OFFSET = 99
STRIKE_MAX_LIQ_OFFSET = 100
STRIKE_N_QUOTED       = 101
STRIKE_SET_WIDTH      = 102
STRIKE_PC_IMB_RATIO   = 103
STRIKE_LIQ_WT_OFFSET  = 104

# SPX derived (105-118)
SPX_CLOSE       = 105
SPX_HL_RANGE    = 106
SPX_RET_1M      = 107
SPX_RET_5M      = 108
SPX_RET_15M     = 109
SPX_RET_30M     = 110
SPX_PRV_15M     = 111
SPX_PRV_30M     = 112
SPX_PRV_60M     = 113
SPX_MINS_CLOSE  = 114
SPX_RANGE_COMP  = 115
SPX_VWAP_DIST   = 116
SPX_CUM_RET     = 117
SPX_VOL_SURGE   = 118

# VIX complex (119-130)
VIX_SPOT  = 119
VIX_9D    = 120
VIX_3M    = 121
VIX_6M    = 122
VIX_F1    = 123
VIX_F2    = 124
VIX_9D_RATIO   = 125    # vix9d / vix
VIX_3M_RATIO   = 126    # vix / vix3m
VIX_6M_RATIO   = 127    # vix / vix6m
VIX_F1_F2_DIFF = 128    # arithmetic dupe of VIX_F1 - VIX_F2
VIX_F1_BASIS   = 129    # arithmetic dupe of (f1-vix)/vix
VIX_TERM_SLOPE = 130    # arithmetic dupe of linear fit slope

# Order flow (131-139)
OF_PC_LIQ_RATIO  = 131
OF_NET_PREM      = 132
OF_SPREAD_ASYM   = 133
OF_LIQ_CONC      = 134
OF_Q_IMB_ATM5    = 135
OF_SPREAD_SKEW   = 136
OF_NET_GAMMA     = 137
OF_TICK_MOMENTUM = 138
OF_IMB_CHG       = 139


# ============================================================
# Feature registry
# ============================================================

@dataclass
class DerivedFeatureSpec:
    name: str
    category: str
    inputs: Tuple[int, ...]
    formula: str
    implemented: bool = True


def _stub_not_implemented(reason: str) -> Callable[..., np.ndarray]:
    def _raise(X: np.ndarray, meta: Dict) -> np.ndarray:
        raise NotImplementedError(reason)
    return _raise


# ---- Category 1: IV surface shape (wing skew, term structure) ----

def feat_wing_skew_25d(X: np.ndarray, meta: Dict) -> np.ndarray:
    """25-delta put IV minus 25-delta call IV. Positive = put skew (fear)."""
    return (X[:, IV_25DP] - X[:, IV_25DC]).astype(np.float32)


def feat_wing_skew_10d(X: np.ndarray, meta: Dict) -> np.ndarray:
    """10-delta wing skew — the deep tails. More sensitive to crash pricing."""
    return (X[:, IV_10DP] - X[:, IV_10DC]).astype(np.float32)


def feat_iv_smile_curvature(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Butterfly: (10Dp + 10Dc) / 2 - ATM. Positive = convex smile."""
    wings = 0.5 * (X[:, IV_10DP] + X[:, IV_10DC])
    return (wings - X[:, IV_ATM]).astype(np.float32)


def feat_iv_rv_spread(X: np.ndarray, meta: Dict) -> np.ndarray:
    """ATM IV minus annualized PRV15 — implied/realized premium."""
    prv_ann = X[:, SPX_PRV_15M] * np.sqrt(252.0 * 390.0 / 15.0)  # 15-min PRV -> annualized
    return (X[:, IV_ATM] - prv_ann).astype(np.float32)


# ---- Category 2: Term structure (VIX + IV) ----

def feat_vix_term_ratio(X: np.ndarray, meta: Dict) -> np.ndarray:
    """VIX / VIX3M. <1 = contango (calm), >1 = backwardation (stress)."""
    vix3m = X[:, VIX_3M]
    safe = np.where(vix3m > 0, vix3m, 1.0)
    return np.where(vix3m > 0, X[:, VIX_SPOT] / safe, 0.0).astype(np.float32)


def feat_vix_9d_ratio(X: np.ndarray, meta: Dict) -> np.ndarray:
    """VIX9D / VIX. >1 = short-term stress exceeds 30-day — event pricing."""
    vix = X[:, VIX_SPOT]
    safe = np.where(vix > 0, vix, 1.0)
    return np.where(vix > 0, X[:, VIX_9D] / safe, 0.0).astype(np.float32)


def feat_f1_vix_basis(X: np.ndarray, meta: Dict) -> np.ndarray:
    """VIX front futures vs spot: (f1-vix)/vix. Contango proxy."""
    vix = X[:, VIX_SPOT]
    safe = np.where(vix > 0, vix, 1.0)
    return np.where(vix > 0, (X[:, VIX_F1] - vix) / safe, 0.0).astype(np.float32)


def feat_vix_futures_spread(X: np.ndarray, meta: Dict) -> np.ndarray:
    """f1 - f2. Positive = front premium."""
    return (X[:, VIX_F1] - X[:, VIX_F2]).astype(np.float32)


# ---- Category 3: ATM microstructure and gamma exposure ----

def feat_atm_gamma_concentration(X: np.ndarray, meta: Dict) -> np.ndarray:
    """ATM gamma scaled by reliability. Reflects dealer hedging pressure."""
    return (X[:, ATM_GAMMA] * X[:, RELY_ATM]).astype(np.float32)


def feat_atm_theta_per_min(X: np.ndarray, meta: Dict) -> np.ndarray:
    """ATM theta (already per-day in pipeline) per minute to close.

    Requires SPX_MINS_CLOSE to be correct — flagged below as TZ-dependent.
    """
    mins = np.maximum(X[:, SPX_MINS_CLOSE], 1.0)
    return (X[:, ATM_THETA] / mins).astype(np.float32)


def feat_atm_spread_zsc(X: np.ndarray, meta: Dict) -> np.ndarray:
    """ATM spread (already z-scored by L1-preprocess). Pass-through for GP."""
    return X[:, ATM_SPREAD].astype(np.float32)


def feat_atm_spread_momentum(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Minute-over-minute ATM spread change. Widening = liquidity withdrawal."""
    return X[:, ATM_SPREAD_CHG].astype(np.float32)


# ---- Category 4: Order flow / dealer positioning ----

def feat_pc_liquidity_ratio(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Put/call quoted-liquidity ratio. >1 = put-side dominance."""
    return X[:, OF_PC_LIQ_RATIO].astype(np.float32)


def feat_net_premium_flow(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Net premium-weighted liquidity (signed log1p applied by L1-preprocess)."""
    return X[:, OF_NET_PREM].astype(np.float32)


def feat_liquidity_concentration(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Fraction of total quoted volume within 5 strikes of ATM."""
    return X[:, OF_LIQ_CONC].astype(np.float32)


def feat_quote_imbalance_atm(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Bid-ask size imbalance near ATM. + = bid-side pressure (dip buyers)."""
    return X[:, OF_Q_IMB_ATM5].astype(np.float32)


def feat_spread_skew_pc(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Put spread / call spread near ATM. >1 = fear premium in put liquidity."""
    return X[:, OF_SPREAD_SKEW].astype(np.float32)


def feat_net_gamma_side(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Call-side minus put-side gamma*liquidity. Signed gamma exposure proxy."""
    return X[:, OF_NET_GAMMA].astype(np.float32)


# ---- Category 5: SPX realized dynamics ----

def feat_ret_accel(X: np.ndarray, meta: Dict) -> np.ndarray:
    """ret_5m - ret_1m * 5 — second-derivative of price vs linear extrapolation."""
    return (X[:, SPX_RET_5M] - 5.0 * X[:, SPX_RET_1M]).astype(np.float32)


def feat_prv_term_ratio(X: np.ndarray, meta: Dict) -> np.ndarray:
    """PRV15m / PRV60m — short vs long realized vol. >1 = vol spike."""
    denom = np.maximum(X[:, SPX_PRV_60M], 1e-6)
    return (X[:, SPX_PRV_15M] / denom).astype(np.float32)


def feat_range_compression(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Rolling std of bar ranges — low = compression pre-breakout."""
    return X[:, SPX_RANGE_COMP].astype(np.float32)


def feat_vwap_distance(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Current price vs session VWAP. Trend strength proxy."""
    return X[:, SPX_VWAP_DIST].astype(np.float32)


def feat_cumulative_return(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Session return from open. Directional bias of the day so far."""
    return X[:, SPX_CUM_RET].astype(np.float32)


def feat_volume_surge(X: np.ndarray, meta: Dict) -> np.ndarray:
    """Current bar volume / rolling mean. Attention spike detector."""
    return X[:, SPX_VOL_SURGE].astype(np.float32)


# ---- Category 6: Cross-asset vol consistency ----

def feat_iv_vix_spread(X: np.ndarray, meta: Dict) -> np.ndarray:
    """ATM IV (already in vol points) minus VIX spot. Dislocation detector."""
    return (X[:, IV_ATM] * 100.0 - X[:, VIX_SPOT]).astype(np.float32)


# ---- Category 7: Session phase (TZ-BLOCKED) ----

feat_session_phase_cos = _stub_not_implemented(
    "session_phase_cos requires SPX_MINS_CLOSE (variate 114) to be in the "
    "correct timezone. Blocked on D74 TZ audit resolution."
)

feat_session_phase_sin = _stub_not_implemented(
    "session_phase_sin — same TZ blocker as session_phase_cos."
)

feat_is_opening_30min = _stub_not_implemented(
    "is_opening_30min — needs session_open anchor with correct TZ. "
    "Blocked on D74."
)

feat_is_closing_30min = _stub_not_implemented(
    "is_closing_30min — same TZ blocker."
)

feat_distance_to_or_high = _stub_not_implemented(
    "distance_to_opening_range_high — needs correct session-open bar index. "
    "Blocked on D74 TZ audit."
)

feat_distance_to_or_low = _stub_not_implemented(
    "distance_to_opening_range_low — same blocker."
)


# ---- Category 8: Macro / calendar flags (CALENDAR-BLOCKED) ----

feat_is_fomc_day = _stub_not_implemented(
    "is_fomc_day requires a plumbed FOMC calendar source. Not yet in the "
    "collector. Deferred to post-D75 re-collection."
)

feat_is_cpi_day = _stub_not_implemented(
    "is_cpi_day — same calendar-source dependency as is_fomc_day."
)

feat_is_nfp_day = _stub_not_implemented(
    "is_nfp_day — Non-farm payrolls release day. Same dependency."
)

feat_is_opex_week = _stub_not_implemented(
    "is_opex_week — monthly SPX options expiration week. Calendar dependency."
)

feat_is_earnings_week = _stub_not_implemented(
    "is_earnings_week — aggregated large-cap earnings density. External feed."
)


# ---- Category 9: Macro regime indicators (VARIATE-BLOCKED) ----

feat_skew_index_level = _stub_not_implemented(
    "skew_index_level — CBOE SKEW index not in current 141-variate layout. "
    "Adding at D75 re-collection as part of Tier 2 variates."
)

feat_vvix_level = _stub_not_implemented(
    "vvix_level — VIX-of-VIX not in current 141 layout. Same Tier 2 addition."
)

feat_vix_dislocation = _stub_not_implemented(
    "vix_dislocation — requires intraday VIX realized dispersion. Tier 2."
)


# ============================================================
# Registry declaration
# ============================================================

DERIVED_FEATURE_REGISTRY: Dict[str, DerivedFeatureSpec] = {
    # Category 1: IV surface shape
    "wing_skew_25d": DerivedFeatureSpec(
        "wing_skew_25d", "iv_surface", (IV_25DP, IV_25DC),
        "IV(25Dp) - IV(25Dc)"),
    "wing_skew_10d": DerivedFeatureSpec(
        "wing_skew_10d", "iv_surface", (IV_10DP, IV_10DC),
        "IV(10Dp) - IV(10Dc)"),
    "iv_smile_curvature": DerivedFeatureSpec(
        "iv_smile_curvature", "iv_surface", (IV_10DP, IV_10DC, IV_ATM),
        "mean(wings) - ATM"),
    "iv_rv_spread": DerivedFeatureSpec(
        "iv_rv_spread", "iv_surface", (IV_ATM, SPX_PRV_15M),
        "ATM IV - annualized PRV15"),

    # Category 2: Term structure
    "vix_term_ratio": DerivedFeatureSpec(
        "vix_term_ratio", "term_structure", (VIX_SPOT, VIX_3M),
        "VIX / VIX3M"),
    "vix_9d_ratio": DerivedFeatureSpec(
        "vix_9d_ratio", "term_structure", (VIX_9D, VIX_SPOT),
        "VIX9D / VIX"),
    "f1_vix_basis": DerivedFeatureSpec(
        "f1_vix_basis", "term_structure", (VIX_F1, VIX_SPOT),
        "(f1 - vix) / vix"),
    "vix_futures_spread": DerivedFeatureSpec(
        "vix_futures_spread", "term_structure", (VIX_F1, VIX_F2),
        "f1 - f2"),

    # Category 3: ATM microstructure
    "atm_gamma_concentration": DerivedFeatureSpec(
        "atm_gamma_concentration", "microstructure", (ATM_GAMMA, RELY_ATM),
        "ATM gamma * reliability"),
    "atm_theta_per_min": DerivedFeatureSpec(
        "atm_theta_per_min", "microstructure", (ATM_THETA, SPX_MINS_CLOSE),
        "ATM theta / minutes-to-close (TZ-dependent)"),
    "atm_spread_zsc": DerivedFeatureSpec(
        "atm_spread_zsc", "microstructure", (ATM_SPREAD,),
        "ATM bid-ask spread (pass-through)"),
    "atm_spread_momentum": DerivedFeatureSpec(
        "atm_spread_momentum", "microstructure", (ATM_SPREAD_CHG,),
        "delta(ATM spread) minute-over-minute"),

    # Category 4: Order flow
    "pc_liquidity_ratio": DerivedFeatureSpec(
        "pc_liquidity_ratio", "order_flow", (OF_PC_LIQ_RATIO,),
        "put vol / call vol (quoted liquidity)"),
    "net_premium_flow": DerivedFeatureSpec(
        "net_premium_flow", "order_flow", (OF_NET_PREM,),
        "sign(net_prem) * log1p(|net_prem|)"),
    "liquidity_concentration": DerivedFeatureSpec(
        "liquidity_concentration", "order_flow", (OF_LIQ_CONC,),
        "fraction of volume within ATM ± 5 strikes"),
    "quote_imbalance_atm": DerivedFeatureSpec(
        "quote_imbalance_atm", "order_flow", (OF_Q_IMB_ATM5,),
        "(bid_size - ask_size) / total near ATM"),
    "spread_skew_pc": DerivedFeatureSpec(
        "spread_skew_pc", "order_flow", (OF_SPREAD_SKEW,),
        "mean put spread / mean call spread near ATM"),
    "net_gamma_side": DerivedFeatureSpec(
        "net_gamma_side", "order_flow", (OF_NET_GAMMA,),
        "sum(call gamma*vol) - sum(put gamma*vol) near ATM"),

    # Category 5: SPX dynamics
    "ret_accel": DerivedFeatureSpec(
        "ret_accel", "spx_dynamics", (SPX_RET_1M, SPX_RET_5M),
        "ret_5m - 5*ret_1m (accel vs linear)"),
    "prv_term_ratio": DerivedFeatureSpec(
        "prv_term_ratio", "spx_dynamics", (SPX_PRV_15M, SPX_PRV_60M),
        "PRV15 / PRV60"),
    "range_compression": DerivedFeatureSpec(
        "range_compression", "spx_dynamics", (SPX_RANGE_COMP,),
        "rolling std of bar ranges"),
    "vwap_distance": DerivedFeatureSpec(
        "vwap_distance", "spx_dynamics", (SPX_VWAP_DIST,),
        "(close - session VWAP) / VWAP"),
    "cumulative_return": DerivedFeatureSpec(
        "cumulative_return", "spx_dynamics", (SPX_CUM_RET,),
        "(close - session open) / session open"),
    "volume_surge": DerivedFeatureSpec(
        "volume_surge", "spx_dynamics", (SPX_VOL_SURGE,),
        "current volume / rolling mean volume"),

    # Category 6: Cross-asset
    "iv_vix_spread": DerivedFeatureSpec(
        "iv_vix_spread", "cross_asset", (IV_ATM, VIX_SPOT),
        "100 * ATM IV - VIX spot"),

    # Category 7: Session phase (TZ-BLOCKED)
    "session_phase_cos": DerivedFeatureSpec(
        "session_phase_cos", "session_phase", (SPX_MINS_CLOSE,),
        "cos(2*pi * (mins_elapsed / session_len))", implemented=False),
    "session_phase_sin": DerivedFeatureSpec(
        "session_phase_sin", "session_phase", (SPX_MINS_CLOSE,),
        "sin(2*pi * (mins_elapsed / session_len))", implemented=False),
    "is_opening_30min": DerivedFeatureSpec(
        "is_opening_30min", "session_phase", (SPX_MINS_CLOSE,),
        "1.0 if bar in first 30 min of session else 0.0", implemented=False),
    "is_closing_30min": DerivedFeatureSpec(
        "is_closing_30min", "session_phase", (SPX_MINS_CLOSE,),
        "1.0 if bar in last 30 min else 0.0", implemented=False),
    "distance_to_or_high": DerivedFeatureSpec(
        "distance_to_or_high", "session_phase", (SPX_CLOSE,),
        "(close - opening-range high) / opening-range high", implemented=False),
    "distance_to_or_low": DerivedFeatureSpec(
        "distance_to_or_low", "session_phase", (SPX_CLOSE,),
        "(close - opening-range low) / opening-range low", implemented=False),

    # Category 8: Calendar flags (CALENDAR-BLOCKED)
    "is_fomc_day": DerivedFeatureSpec(
        "is_fomc_day", "calendar", (),
        "FOMC announcement day flag", implemented=False),
    "is_cpi_day": DerivedFeatureSpec(
        "is_cpi_day", "calendar", (),
        "CPI release day flag", implemented=False),
    "is_nfp_day": DerivedFeatureSpec(
        "is_nfp_day", "calendar", (),
        "Non-farm payrolls release flag", implemented=False),
    "is_opex_week": DerivedFeatureSpec(
        "is_opex_week", "calendar", (),
        "Monthly SPX options expiration week flag", implemented=False),
    "is_earnings_week": DerivedFeatureSpec(
        "is_earnings_week", "calendar", (),
        "Large-cap earnings density flag", implemented=False),

    # Category 9: Macro regime indicators (VARIATE-BLOCKED, Tier 2 re-collection)
    "skew_index_level": DerivedFeatureSpec(
        "skew_index_level", "macro_regime", (),
        "CBOE SKEW index level", implemented=False),
    "vvix_level": DerivedFeatureSpec(
        "vvix_level", "macro_regime", (),
        "VIX-of-VIX (VVIX) level", implemented=False),
    "vix_dislocation": DerivedFeatureSpec(
        "vix_dislocation", "macro_regime", (),
        "VIX vs 30-day realized vol of SPX", implemented=False),
}


# ============================================================
# Computation dispatch
# ============================================================

# Map feature name -> computation function. Keep the source-of-truth names
# aligned with DERIVED_FEATURE_REGISTRY above.
_DISPATCH: Dict[str, Callable[[np.ndarray, Dict], np.ndarray]] = {
    "wing_skew_25d": feat_wing_skew_25d,
    "wing_skew_10d": feat_wing_skew_10d,
    "iv_smile_curvature": feat_iv_smile_curvature,
    "iv_rv_spread": feat_iv_rv_spread,
    "vix_term_ratio": feat_vix_term_ratio,
    "vix_9d_ratio": feat_vix_9d_ratio,
    "f1_vix_basis": feat_f1_vix_basis,
    "vix_futures_spread": feat_vix_futures_spread,
    "atm_gamma_concentration": feat_atm_gamma_concentration,
    "atm_theta_per_min": feat_atm_theta_per_min,
    "atm_spread_zsc": feat_atm_spread_zsc,
    "atm_spread_momentum": feat_atm_spread_momentum,
    "pc_liquidity_ratio": feat_pc_liquidity_ratio,
    "net_premium_flow": feat_net_premium_flow,
    "liquidity_concentration": feat_liquidity_concentration,
    "quote_imbalance_atm": feat_quote_imbalance_atm,
    "spread_skew_pc": feat_spread_skew_pc,
    "net_gamma_side": feat_net_gamma_side,
    "ret_accel": feat_ret_accel,
    "prv_term_ratio": feat_prv_term_ratio,
    "range_compression": feat_range_compression,
    "vwap_distance": feat_vwap_distance,
    "cumulative_return": feat_cumulative_return,
    "volume_surge": feat_volume_surge,
    "iv_vix_spread": feat_iv_vix_spread,
    "session_phase_cos": feat_session_phase_cos,
    "session_phase_sin": feat_session_phase_sin,
    "is_opening_30min": feat_is_opening_30min,
    "is_closing_30min": feat_is_closing_30min,
    "distance_to_or_high": feat_distance_to_or_high,
    "distance_to_or_low": feat_distance_to_or_low,
    "is_fomc_day": feat_is_fomc_day,
    "is_cpi_day": feat_is_cpi_day,
    "is_nfp_day": feat_is_nfp_day,
    "is_opex_week": feat_is_opex_week,
    "is_earnings_week": feat_is_earnings_week,
    "skew_index_level": feat_skew_index_level,
    "vvix_level": feat_vvix_level,
    "vix_dislocation": feat_vix_dislocation,
}


def compute_derived_features(
    X: np.ndarray,
    meta: Optional[Dict] = None,
    include_unimplemented: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute all registered derived features for one day.

    Args:
        X: (n_bars, 141) float32 — a single day's raw tensor already run
            through L1-preprocess (gamma clamp, log1p, vix forward-fill,
            z-score + clip). derived/ does NOT re-preprocess.
        meta: optional metadata dict with keys like 'date',
            'session_open_bar', 'is_fomc_day', etc. Currently unused by
            implemented features; required for session-phase and
            calendar stubs once they are implemented.
        include_unimplemented: if True, calling this function will raise
            NotImplementedError on the first blocked feature (useful as
            a TODO detector during development). If False (default),
            blocked features are silently skipped.

    Returns:
        dict mapping feature name -> (n_bars,) float32 array. Only
        contains features that were successfully computed. Callers can
        cross-check the return dict against DERIVED_FEATURE_REGISTRY to
        see what was skipped.
    """
    if meta is None:
        meta = {}

    out: Dict[str, np.ndarray] = {}
    for name, spec in DERIVED_FEATURE_REGISTRY.items():
        fn = _DISPATCH.get(name)
        if fn is None:
            continue
        if not spec.implemented and not include_unimplemented:
            continue
        out[name] = fn(X, meta)
    return out


def derivation_spec_hash() -> str:
    """Return the stable 16-hex hash of the full derivation spec.

    Covers:
      - every registered feature name
      - its category, input indices, and formula string
      - whether it is currently implemented

    Anything that mutates this hash is a BREAKING change for downstream
    L2 runs — the L1->L2 manifest pins (encoder, preprocessor,
    derivation) and L2 reproducibility depends on all three being
    stable across a run.
    """
    payload = []
    for name in sorted(DERIVED_FEATURE_REGISTRY.keys()):
        spec = DERIVED_FEATURE_REGISTRY[name]
        payload.append({
            "name": spec.name,
            "category": spec.category,
            "inputs": list(spec.inputs),
            "formula": spec.formula,
            "implemented": spec.implemented,
        })
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
