"""SIGNAL-BEARING POSITIVE CONTROL for the GP (task #34).

Every other fixture in the L2 test suite injects NOISE by construction
(``tests/test_experiment.py::_make_l1_parquet`` — constant scalars + random
typed vectors), so a losing GP run there is *expected* and a passing run proves
only plumbing, never DISCOVERY. Without a planted-edge control, a losing
production GP run is observationally identical whether (a) the data has no edge
or (b) the GP/evaluator is broken.

This module plants a LEARNABLE, observable-at-entry edge that a short credit
spread can exploit, then proves three things with REAL numbers:

  1. (build_planted_edge_fixture) the fixture matches the evaluator's minute
     column contract (layer2.io) and carries the planted signal.
  2. (test_planted_edge_registers_in_evaluator) the edge REGISTERS in
     ``vectorized_backtest``: a calm-only entry is clearly profitable while a
     trade-every-bar entry is ~0 / negative. This is the load-bearing check —
     if it fails, the planted edge is not real.
  3. (test_gp_discovers_planted_edge) a small seeded GP run via
     ``run_experiment`` DISCOVERS an entry that beats the random-entry baseline
     by a clear margin.

THE PLANTED EDGE
----------------
Days come in two latent types, deterministic by date (seeded coin flip, ~50/50):

  * CALM    : after entry the SPX path stays RANGE-BOUND (tiny steps, <0.2%/day),
              so a bull-put / iron-condor credit spread expires OTM and KEEPS the
              credit -> PROFIT. Observable signal: RealizedVol30m LOW, VIXSpot=12.
  * VOLATILE: after entry the SPX path makes a BIG DOWNWARD move (~-5%), breaching
              the short put strike, so the spread LOSES big. Observable signal:
              RealizedVol30m HIGH, VIXSpot=30.

ATM_IV is held ~constant across day types, so the credit received is type-
INDEPENDENT — the edge is purely WHEN to trade (calm), not credit size. The
signal terminals (RealizedVol30m primary, VIXSpot secondary) are NOT used in
option pricing (ATM_IV is), so they are clean, learnable predictors. A GP with
discovery power evolves an entry like ``LT(RealizedVol30m, ~0)`` to trade ONLY
calm days; the trade-every-day baseline trades both types and bleeds on the
volatile losers.

Why RealizedVol30m (not VIXSpot) is the PRIMARY planted signal: ``run_experiment``
calls ``prepare_terminal_data(lag_daily_vix=True)``, which lags the daily
VIX-family terminals one session. RealizedVol30m is an intraday terminal and is
NOT lagged, so the planted signal survives the production path unchanged. Both
terminals are in the scalar-only sampling set and neither is filtered by
``grammar._QC_NONTRANSFERABLE_SCALARS``.

A KEY EVALUATOR INTERACTION the fixture relies on (documented, not worked around)
-----------------------------------------------------------------------------
``MinutesToClose`` descends to ~31 and NEVER reaches ``EOD_FORCE_CLOSE_MTC`` (25).
The position therefore is NOT force-closed mid-session (which would mark the
short put at its residual mtc=25 extrinsic — the calibrated credit-haircut + entry
cost cannot cover that residual, so a *flat* day would book a small LOSS). Instead
the position rides to the NEXT-day session boundary, which settles it at mtc=0
(European cash settlement: OTM legs expire worthless, ~$0.04/share clearing fee).
That captures the full credit on a calm day. This mirrors a real 0DTE position
held to expiry and is the regime in which the planted edge is cleanly positive.
"""
from __future__ import annotations

import contextlib
import io as _io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layer2.io import PRICE_COLUMN, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS
from layer2.grammar import from_sexpr
from layer2.evaluator_vectorized import (
    EOD_FORCE_CLOSE_MTC,
    ENTRY_CUTOFF_MTC,
    prepare_terminal_data,
    vectorized_backtest,
)
from layer2.terminal_stats import compute_norm_stats_from_data
from layer2.templates import bull_put_credit_standard, iron_condor_standard


# Verified planted-edge parameters (see module docstring). These are the values
# at which step 2 produces calm-only Sharpe ~+11.5 and always-enter ~0/negative.
_BASE_IV = 0.20          # regime credit-mult 1.0; ATM_IV constant across day types
_CALM_STEP = 0.00004     # calm intraday step as fraction of spot (~0.04%/bar std)
_VOL_MOVE = -0.05        # volatile EOD drift (-5%): breaches the -0.25 short put
_CALM_VIX, _VOL_VIX = 12.0, 30.0
_CALM_RV, _VOL_RV = 2.0e-4, 8.0e-4   # RealizedVol30m: low (calm) vs high (volatile)

# GridReliability empirical distribution — MEASURED from the real parquet
# raw_data/local_store/l1_minute_scalars.parquet (mean 0.9367, median 1.0, p5 0.455:
# mostly reliable with an occasional sparse-chain dip to 0). Stored as 51 quantiles
# (every 2%) and inverse-CDF sampled per bar so the fixture's GridReliability MATCHES
# the real marginal distribution rather than a guessed constant. It is drawn
# INDEPENDENTLY of day type, so it is a fair, signal-free sampled terminal (the only
# exploitable edge remains RealizedVol30m). Data-driven, no guesswork.
_GRIDREL_QUANTILES = np.array([
    0.0000, 0.2727, 0.3636, 0.5455, 0.6364, 0.7273, 0.8182, 0.9091, 0.9091, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000,
])
_GRIDREL_CDF_X = np.linspace(0.0, 1.0, len(_GRIDREL_QUANTILES))


def build_planted_edge_fixture(
    n_days: int = 140,
    bars_per_date: int = 120,
    seed: int = 7,
    base_iv: float = _BASE_IV,
    calm_step: float = _CALM_STEP,
    vol_move: float = _VOL_MOVE,
):
    """Signal-bearing 1-minute parquet matching the evaluator column contract.

    Returns (DataFrame, is_volatile) where ``is_volatile[d]`` is the day-type of
    the d-th distinct date (ascending). Columns: the full L1 minute contract —
    ``date``, ``SPXClose``, all bypass + probe scalars, regime probs, and noise
    typed-vector columns (unused by the scalar-only condition).
    """
    rng = np.random.default_rng(seed)
    day_rng = np.random.default_rng(seed + 1)
    is_volatile = day_rng.random(n_days) < 0.5

    start = pd.Timestamp("2024-01-02")
    # mtc descends to 31, never reaching EOD_FORCE_CLOSE_MTC (25). See docstring:
    # the position settles at the next-day session boundary (mtc=0), capturing the
    # full credit on calm days. Entry stays eligible (mtc > ENTRY_CUTOFF_MTC = 30).
    mtc_bottom = ENTRY_CUTOFF_MTC + 1.0          # 31.0
    mtc_top = mtc_bottom + (bars_per_date - 1)

    rows = []
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        volatile = bool(is_volatile[d])
        spot = 5000.0
        path_rng = np.random.default_rng(seed * 100003 + d)
        drift_per_bar = (vol_move * 5000.0) / bars_per_date if volatile else 0.0
        vix = _VOL_VIX if volatile else _CALM_VIX
        rv = _VOL_RV if volatile else _CALM_RV
        for w in range(bars_per_date):
            if volatile:
                step = drift_per_bar + path_rng.normal(0.0, 0.5)
            else:
                step = path_rng.normal(0.0, calm_step * 5000.0)
            spot += step
            row = {
                "date": date, "window_idx": w, "bar_position": w,
                PRICE_COLUMN: float(spot),
                "ATM_IV": base_iv,                      # constant across day types
                "MinutesToClose": float(mtc_top - w),   # mtc_top .. 31 (never <=25)
                "VIXSpot": float(vix),                  # secondary signal
                "VIXTermSlope": -0.5,
                "RawSpread": 0.20,
                "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
                "RealizedVol30m": float(rv),            # PRIMARY planted signal
                # GridReliability: inverse-CDF sampled from the MEASURED real-parquet
                # distribution (_GRIDREL_QUANTILES), INDEPENDENT of day type -> a fair,
                # realistic, signal-free sampled terminal (the only exploitable edge
                # stays RealizedVol30m). Without this column it is constant-0, so every
                # tree referencing it is dead-on-arrival, unfairly suppressing the GP's
                # winner-generation rate. Data-driven (not a guessed constant).
                "GridReliability": float(np.interp(rng.random(), _GRIDREL_CDF_X,
                                                   _GRIDREL_QUANTILES)),
                "PredRV15": 0.30, "PredRV30": 0.30,
                "PredRegime": 0, "PredSpread": 0.0,
            }
            for col in TYPED_VECTOR_COLUMNS:            # pure noise (scalar-only ignores)
                row[col] = rng.standard_normal(384).astype(np.float32).tolist()
            for col in REGIME_PROB_COLUMNS:
                row[col] = float(rng.uniform())
            rows.append(row)
    return pd.DataFrame(rows), is_volatile


def _run_entry(df: pd.DataFrame, entry_sexpr: str, template):
    """Parse a hardcoded entry tree, build terminal_data per-fold, run the
    vectorized backtest holding to EOD (exit signal never fires)."""
    norm = compute_norm_stats_from_data(df)
    # lag_daily_vix=False here only matters for VIXSpot; the primary signal is the
    # un-lagged RealizedVol30m, so the GP path (lag=True) sees the same edge.
    td = prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)
    entry = from_sexpr(entry_sexpr)
    exit_t = from_sexpr("LT(MinutesToClose, EphReal(-5.0))")  # never fires -> hold
    size_t = from_sexpr("EphReal(0.5)")
    res = vectorized_backtest(entry, exit_t, size_t, df, template,
                              terminal_data=td, warmup_bars=15)
    return res, td


def _pnl_by_daytype(res, df, is_volatile):
    date_list = sorted(df["date"].unique())
    vol_dates = {date_list[i] for i in range(len(date_list)) if is_volatile[i]}
    dates_arr = df["date"].values
    calm_pnl = vol_pnl = 0.0
    for t in res.trades:
        if dates_arr[t.entry_bar] in vol_dates:
            vol_pnl += t.pnl
        else:
            calm_pnl += t.pnl
    return calm_pnl, vol_pnl


# ---------------------------------------------------------------------------
# Fixture-contract sanity (cheap; runs always)
# ---------------------------------------------------------------------------

def test_fixture_matches_minute_contract_and_carries_signal():
    """The planted fixture loads through the production minute-parquet validator
    and carries a real, separable signal in RealizedVol30m + the SPX path."""
    from layer2.io import load_minute_parquet, BYPASS_SCALAR_COLUMNS
    import tempfile, os

    df, is_vol = build_planted_edge_fixture(n_days=20, bars_per_date=40)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "planted.parquet")
        df.to_parquet(p, engine="pyarrow")
        loaded, schema = load_minute_parquet(p)   # raises if contract violated
    assert schema.n_dates == 20
    for col in BYPASS_SCALAR_COLUMNS:
        assert col in df.columns, f"fixture missing bypass scalar {col}"
    assert PRICE_COLUMN in df.columns and not df[PRICE_COLUMN].isna().any()

    # mtc never reaches the force-close band -> settlement-exit regime (docstring).
    assert df["MinutesToClose"].min() > EOD_FORCE_CLOSE_MTC

    # Signal separability: RealizedVol30m and the SPX EOD move are cleanly split
    # by day type, while ATM_IV is constant (credit is type-independent).
    date_list = sorted(df["date"].unique())
    moves, rv_vals = [], []
    for i, (date, g) in enumerate(df.groupby("date")):
        o, c = g[PRICE_COLUMN].iloc[0], g[PRICE_COLUMN].iloc[-1]
        moves.append((c - o) / o)
        rv_vals.append(g["RealizedVol30m"].iloc[0])
    moves, rv_vals = np.array(moves), np.array(rv_vals)
    assert rv_vals[is_vol].min() > rv_vals[~is_vol].max(), "RV must separate types"
    assert moves[~is_vol].mean() > -0.005, "calm days ~flat"
    assert moves[is_vol].mean() < -0.03, "volatile days move down hard"
    assert df["ATM_IV"].nunique() == 1, "ATM_IV must be type-independent"


# ---------------------------------------------------------------------------
# Step 2 — THE LOAD-BEARING CHECK: the planted edge registers in the evaluator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template_fn", [bull_put_credit_standard, iron_condor_standard])
def test_planted_edge_registers_in_evaluator(template_fn):
    """Before any GP: a calm-only entry must be clearly PROFITABLE in
    ``vectorized_backtest`` while trade-every-bar is ~0 / NEGATIVE. If this
    fails, the planted edge is not real and the discovery test is meaningless.

    Verified numbers (n_days=80, base_iv=0.20):
      bull_put_credit  calm-only Sharpe~+11.6 pnl~+250 wr=1.00 ; always-enter pnl~-995
      iron_condor      calm-only Sharpe~+11.5 pnl~+604 wr=1.00 ; always-enter Sharpe~-1.9
    """
    template = template_fn()
    df, is_vol = build_planted_edge_fixture(n_days=80, bars_per_date=120)

    # Calm-only: RealizedVol30m is LOW on calm days. Normalized (~N(0,1) over the
    # 50/50 mix) the calm cluster sits below 0, so LT(RealizedVol30m, 0) selects it.
    calm_res, _ = _run_entry(df, "LT(RealizedVol30m, EphReal(0.0))", template)
    # Always-enter: GT(MinutesToClose, -99) is unconditionally true.
    all_res, _ = _run_entry(df, "GT(MinutesToClose, EphReal(-99.0))", template)

    calm_total = sum(t.pnl for t in calm_res.trades)
    all_total = sum(t.pnl for t in all_res.trades)
    calm_pnl, calm_vol_pnl = _pnl_by_daytype(calm_res, df, is_vol)

    # (a) calm-only is clearly profitable
    assert calm_res.total_trades >= 20, (
        f"calm-only must produce enough trades (got {calm_res.total_trades})")
    assert calm_res.sharpe > 2.0, (
        f"calm-only Sharpe must be clearly positive; got {calm_res.sharpe:.3f}")
    assert calm_total > 0.0, (
        f"calm-only total PnL must be positive; got {calm_total:.1f}")
    assert calm_res.win_rate > 0.8, (
        f"calm-only win rate must be high; got {calm_res.win_rate:.2f}")

    # (b) the edge is "trade only calm": calm-only essentially never trades the
    # volatile losers, and almost all of its PnL comes from calm days.
    assert calm_vol_pnl >= -1.0, (
        f"calm-only should avoid volatile days; volatile PnL={calm_vol_pnl:.1f}")
    assert calm_pnl > 0.0

    # (c) trade-every-bar is clearly WORSE: it bleeds on the volatile losers, so
    # it is either ~0/negative Sharpe or a hard-gated sentinel (-1e6). Either way
    # its realized PnL is deeply negative and far below calm-only.
    assert all_total < 0.0, f"always-enter total PnL must be negative; got {all_total:.1f}"
    assert all_total < calm_total - 100.0, (
        f"always-enter ({all_total:.1f}) must be far below calm-only ({calm_total:.1f})")
    assert (all_res.sharpe <= 0.0 or all_res.sharpe <= -1e5), (
        f"always-enter Sharpe must be <= 0 (or hard-gated sentinel); "
        f"got {all_res.sharpe:.3f}")


# ---------------------------------------------------------------------------
# Step 4 helper — gate interaction probe (shared by the discovery test)
# ---------------------------------------------------------------------------

def _planted_strategy_clears_production_gates(df, template, min_trades):
    """Run the canonical calm-only strategy through the FULL VectorizedFitness
    gate cascade (fire-rate, random-entry null, exit-utilization, regime) exactly
    as production does. Returns (cleared: bool, detail: dict). Used by step 4 to
    report whether the production gates ADMIT or BLOCK the planted signal."""
    from layer2.fitness import (
        VectorizedFitnessEvaluator, FAILED_FITNESS_SENTINEL, DEFAULT_OBJECTIVES,
    )
    norm = compute_norm_stats_from_data(df)
    td = prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)
    fe = VectorizedFitnessEvaluator(
        template=template, data=df, terminal_data=td,
        warmup_bars=15, min_trades=min_trades,
        random_entry_margin=0.30, regime_gate_enabled=True,
    )
    entry = from_sexpr("LT(RealizedVol30m, EphReal(0.0))")
    exit_t = from_sexpr("LT(MinutesToClose, EphReal(-5.0))")
    size_t = from_sexpr("EphReal(0.5)")
    fit = fe.evaluate(entry, exit_t, size_t)
    # Infeasible fitness is the 1e6 sentinel base + a small graded violation in
    # [0,1) (finding #4). A strategy is "cleared" iff it is NOT at the sentinel on
    # every objective (i.e. at least one objective is a real, sub-sentinel value).
    cleared = not bool(np.all(fit >= FAILED_FITNESS_SENTINEL))
    detail = {
        "fitness": fit.tolist(),
        "random_entry_sharpe": fe._random_entry_sharpe,
        "objectives": list(DEFAULT_OBJECTIVES),
    }
    return cleared, detail


# ---------------------------------------------------------------------------
# Step 3 — PROVE DISCOVERY (slow: a real mini-GP run)
# ---------------------------------------------------------------------------

def test_production_gates_admit_day_selective_winner():
    """#6 + #7 FIX (surfaced by this positive control): the canonical planted winner —
    ``LT(RealizedVol30m, 0)``, raw Sharpe ~ +12.8 on 50 trades, a SELECTIVE-entry +
    hold-to-close 0DTE credit spread — is now ADMITTED by the production gate cascade.

    Three gates used to kill this dominant profitable 0DTE pattern: max-hold (#2), the
    fire-rate *tautology* gate (#6 — measured signal-true among ALL bars, so a strategy
    selective at the DAY level but firing every bar of the days it trades scored ~0.42 >
    0.35), and exit-utilization (#7 — hold-to-close has ~0% signal-driven exits). Now #6
    gates on the FLAT-bar fire rate (entries among bars where the strategy is actually
    able to enter — low for a day-selective hold) and #7 only gates UNPROFITABLE
    vestigial-exit strategies. The winner passes; an always-enter churner is still caught.
    """
    from layer2.fitness import VectorizedFitnessEvaluator, FAILED_FITNESS_SENTINEL

    df, _ = build_planted_edge_fixture(n_days=120, bars_per_date=48, seed=7)
    norm = compute_norm_stats_from_data(df)
    td = prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)
    tmpl = bull_put_credit_standard()
    entry = from_sexpr("LT(RealizedVol30m, EphReal(0.0))")
    exit_t = from_sexpr("LT(MinutesToClose, EphReal(-5.0))")
    size_t = from_sexpr("EphReal(0.5)")

    # raw: a huge, real winner that only enters ~once per calm day.
    raw = vectorized_backtest(entry, exit_t, size_t, df, tmpl,
                              terminal_data=td, warmup_bars=15)
    assert raw.sharpe > 2.0 and raw.total_trades >= 20, "planted winner must be real"
    # ...yet its raw BAR-level fire rate trips the old 0.35 tautology ceiling, and its
    # exit_util ~0 trips the old #7 gate — both old traps for a day-selective hold.
    assert raw.entry_fire_rate > 0.35, "raw bar-level fire rate must exceed 0.35 (the #6 trap)"
    assert raw.exit_utilization < 0.20, "hold-to-close has ~0 signal exits (the #7 trap)"
    assert raw.entry_fire_rate_flat <= 0.35, "FLAT-bar fire rate must be low (the #6 fix)"

    # through the FULL production gate cascade -> now ADMITTED (feasible, real fitness).
    fe = VectorizedFitnessEvaluator(template=tmpl, data=df, terminal_data=td,
                                    warmup_bars=15, min_trades=8,
                                    random_entry_margin=0.30, regime_gate_enabled=True)
    fit = fe.evaluate(entry, exit_t, size_t)
    assert not bool(np.all(fit >= FAILED_FITNESS_SENTINEL)), (
        f"#6+#7: the day-selective hold winner (raw Sharpe ~+12.8) must now be ADMITTED, "
        f"not sentineled; got fitness={fit.tolist()}")
    assert fit[0] < 0, "admitted winner's neg_sharpe must reflect its real positive Sharpe"

    # An always-enter churner must STILL be caught (fires on ~every FLAT bar).
    churn = from_sexpr("GT(MinutesToClose, EphReal(-99.0))")
    rawc = vectorized_backtest(churn, exit_t, size_t, df, tmpl, terminal_data=td, warmup_bars=15)
    assert rawc.entry_fire_rate_flat > 0.35, "churner must have a high FLAT-bar fire rate"
    fit_c = fe.evaluate(churn, exit_t, size_t)
    assert bool(np.all(fit_c >= FAILED_FITNESS_SENTINEL)), "always-enter churner must stay gated"


@pytest.mark.slow
def test_gp_discovers_planted_edge(tmp_path):
    """A small seeded GP run via ``run_experiment`` must DISCOVER an entry that
    beats the random-entry baseline by a clear margin.

    Validates DISCOVERY POWER end-to-end now that the hold-to-close gate cascade is
    fixed (#2 max-hold demotion, #6 flat-bar fire rate, #7 profit-conditional exit-util):
    the planted day-selective winner is admitted, so the GP can evolve and select it.
    The fast ``test_production_gates_admit_day_selective_winner`` above pins the gate
    behaviour deterministically; this slow test confirms the full search finds the edge.

    Config is DATA-DRIVEN, not guessed: the winner-generation rate was MEASURED at
    ~2.0% (random grow() trees on the completed fixture that pass the gates with
    positive Sharpe). Required pop for P(>=1 winner in the gen-0 population) >= 0.99 is
    ceil(ln(0.01)/ln(1-0.02)) = 228; we use the PRODUCTION pop_size=256 (P=0.9943).
    seed_fraction=0 (pure random — the strongest discovery proof; no seed head-start).
    With a winner present at gen 0 ~99.4% of the time and NSGA-III elitism preserving
    it, a short run suffices, so n_generations=10. Assertions are conservative (best
    solidly positive AND clearly above a REAL random-entry baseline) — the printed
    numbers are the real evolved Sharpe vs the real baseline.
    """
    from layer2.experiment import ExperimentConfig, run_experiment

    # bars_per_date=120 is LOAD-BEARING, not arbitrary (root-caused 2026-06-02):
    # the fixture carries the 10 EMB_*-prefixed TYPED_VECTOR_COLUMNS, so
    # run_experiment's `_effective_warmup = max(warmup, 60 if any EMB_ cols)`
    # (experiment.py #197) forces warmup=60 EVEN in scalar-only — overriding the
    # backtester_warmup_bars=15 below. The fitness fire-rate / conditional-Sharpe
    # gates use PER-DAY eligibility (`_within_day_pos >= warmup_bars`,
    # evaluator_vectorized.py ~L2249/L2287). At 48 bars/day < 60 EVERY day has 0
    # eligible bars → all individuals sentineled → gen-0 feasibility 0/256 →
    # NSGA-III selects nothing → the run collapses to churners (val=-1e6). At 120
    # bars/day there are 60 post-warmup eligible bars/day → feasibility restored
    # (MEASURED gen-0 11/256, 3 generalizers; run_experiment discovers val +16.08).
    # Production minute data has ~390 bars/day so warmup=60 is harmless there; this
    # is purely a fixture-density requirement. 120 matches the direct-eval tests.
    df, is_vol = build_planted_edge_fixture(n_days=120, bars_per_date=120, seed=7)
    parquet = tmp_path / "planted.parquet"
    df.to_parquet(parquet, engine="pyarrow")

    template = bull_put_credit_standard()
    config = ExperimentConfig(
        l1_parquet_path=str(parquet),
        condition="scalar-only",         # exposes RealizedVol30m + VIXSpot as terminals
        template_names=("bull_put_credit",),
        pop_size=256,                    # DERIVED: >=228 for P(>=1 winner)>=0.99 @ 2% gen rate
        n_generations=10,                # winner present at gen 0 ~99.4%; elitism preserves it
        seed=123,
        seed_fraction=0.0,               # pure-random discovery (strongest proof; synth seed=0 trades)
        use_vectorized=True,             # PRODUCTION path (matches --minute). Without it the
                                         # config defaults to the non-vectorized evaluator:
                                         # stale multi-leg pricing AND ~100x slower (13.9M
                                         # per-row pandas lookups). Discovery + speed both
                                         # require the vectorized evaluator (verified +16.08).
        n_partitions=4,
        train_end="2024-03-25",          # ~84 train days -> ~40 calm -> plenty of trades
        val_end="2024-04-25",            # ~31 val days
        embargo_days=0,
        output_dir=str(tmp_path / "out"),
        backtester_warmup_bars=15,       # NOTE: overridden to 60 by EMB_-col auto-warmup
                                         # (#197); fixture uses 120 bars/day to compensate
        min_trades=8,                    # fast; floor=ceil(0.03*84)=3 < 8
        dsr_gate_enabled=False,          # annotate-only anyway; off for a clean front
        regime_gate_enabled=True,
    )

    # Sanity: VIXSpot AND RealizedVol30m are in the scalar-only sampled terminal
    # set (and unfiltered by _QC_NONTRANSFERABLE_SCALARS) — the GP CAN reach the
    # planted signal.
    from layer2.grammar import build_scalar_only_terminal_set, _QC_NONTRANSFERABLE_SCALARS
    sampled = {t.name for t in build_scalar_only_terminal_set()}
    assert "RealizedVol30m" in sampled and "RealizedVol30m" not in _QC_NONTRANSFERABLE_SCALARS
    assert "VIXSpot" in sampled and "VIXSpot" not in _QC_NONTRANSFERABLE_SCALARS

    err = _io.StringIO()
    with contextlib.redirect_stderr(err):
        result = run_experiment(config)
    err_text = err.getvalue()

    tr = result.template_results[0]
    front = tr.pareto_front
    assert len(front) >= 1, "GP produced an empty Pareto front on signal-bearing data"

    # Random-entry baseline (the gate's null) — parsed from the evaluator's log.
    rand_sharpes = []
    for line in err_text.splitlines():
        if "random-entry baseline" in line and "sharpe=" in line:
            try:
                rand_sharpes.append(float(line.split("sharpe=")[1].split()[0]))
            except (IndexError, ValueError):
                pass
    random_baseline = rand_sharpes[0] if rand_sharpes else 0.0

    # Best evolved individual by held-out val Sharpe.
    scored = [ind for ind in front if ind.get("val_sharpe") is not None]
    assert scored, "no Pareto member carries a val_sharpe"
    best = max(scored, key=lambda d: d["val_sharpe"])

    print("\n[positive-control] random-entry baseline Sharpe = "
          f"{random_baseline:.3f}")
    print(f"[positive-control] best evolved val Sharpe      = {best['val_sharpe']:.3f}")
    print(f"[positive-control] best entry_tree = {best['entry_tree']!r}")
    print(f"[positive-control] best exit_tree  = {best['exit_tree']!r}")
    print(f"[positive-control] best size_tree  = {best['size_tree']!r}")
    print(f"[positive-control] best total_trades = {best.get('total_trades')}")

    # DISCOVERY: the evolved champion is solidly positive on held-out val. The planted
    # winner's val Sharpe is MEASURED at ~+15.6 (the calm-only entry on the val split),
    # so a conservative >1.0 floor still leaves a wide margin while tolerating RNG drift.
    assert best["val_sharpe"] > 1.0, (
        f"best evolved val Sharpe must be solidly positive on planted-edge data "
        f"(planted winner measures ~+15.6); got {best['val_sharpe']:.3f}")
    # The random-entry NULL is itself a hard-gate sentinel on a clean binary signal
    # (random timing has no edge), so the "beats null by a margin" check is only
    # meaningful when the baseline is a REAL Sharpe — guard it (otherwise vacuous).
    if random_baseline > -1e5:
        assert best["val_sharpe"] > random_baseline + 1.0, (
            f"best evolved val Sharpe ({best['val_sharpe']:.3f}) must clearly beat the "
            f"random-entry baseline ({random_baseline:.3f})")

    # Step 4 — GATE INTERACTION: do the FULL production gates ADMIT the planted
    # winner? Run the canonical calm-only strategy through the whole cascade.
    cleared, detail = _planted_strategy_clears_production_gates(
        df, template, min_trades=config.min_trades)
    print(f"[positive-control][gate-interaction] production gates admit planted "
          f"calm-only strategy = {cleared}; detail={detail}")
    # The planted winner is genuinely profitable AND selective (fire rate ~0.42 on
    # the calm half, well under the 0.35 tautology ceiling is NOT guaranteed — see
    # report), so we ASSERT the GP-discovered champion (above) rather than force a
    # pass/fail on this single hand-written strategy. The boolean + detail are
    # printed so a blocked gate is surfaced explicitly, not hidden.
