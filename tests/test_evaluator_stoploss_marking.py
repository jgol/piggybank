"""Stop-loss adverse-fill markup must reach the fitness Sharpe basis (bar_returns).

BLOCKER regression (2026-06-01 holistic review of layer2/evaluator_vectorized.py):

  On a stop_loss exit the proxy applies STOP_LOSS_FILL_MARKUP to the REALIZED
  PnL (a stop fills during the fast adverse move at a price worse than the BS-mid
  close the proxy marks). That markup multiplied `pnl` — which feeds `equity` and
  `trade.pnl` — but the per-bar Sharpe basis `bar_returns[i]` only booked the
  exit_cost. Sharpe/Sortino are computed ENTIRELY from `bar_returns`, so the extra
  adverse-fill loss (pnl * (markup-1)) was INVISIBLE to fitness. Every stop-loss-
  heavy strategy's selected Sharpe was therefore optimistic by the markup.

  The pre-existing reconciliation guard
  (test_vectorized_wiring.py::test_a1_haircut_reconciliation_and_sortino_cap) uses
  a fixture that exits only via session / end_of_data — it NEVER stop-losses, so it
  could not catch this. This file runs a strategy that DOES stop-loss and pins:

    (1) sum(bar_returns) == sum(trade.pnl)/notional  to ~machine precision WITH a
        stop_loss exit present (the invariant that was broken: pre-fix the markup
        was in trade.pnl but NOT bar_returns, so the two diverged once a stop fired);

    (2) the markup actually moves the Sharpe BASIS: with the markup ON, the booked
        mean daily return (and sum(bar_returns)) is strictly MORE NEGATIVE than with
        markup=1.0, and the gap equals the booked markup delta. (We assert the gap on
        the *mean booked loss*, not the raw Sharpe ratio: for a nearly-all-loss
        series Sharpe = mean/std is NOT monotonic in a uniform loss scale-up — std
        can grow faster than |mean| — so "markup => lower Sharpe" is not a sound
        invariant. "markup => more-negative booked return, now visible in the Sharpe
        basis" is the economically meaningful and robust one.)
"""
import numpy as np
import pandas as pd
import pytest

import layer2.evaluator_vectorized as ev
from layer2.evaluator_vectorized import vectorized_backtest, prepare_terminal_data
from layer2.templates import template_by_name
from layer2.grammar import from_sexpr

NOTIONAL = 1000.0


def _adverse_bpc_df(n_dates: int = 8, bars: int = 60) -> pd.DataFrame:
    """A bull-put-credit-friendly tape where every session gaps DOWN mid-day.

    BPC sells a put spread; a ~3% intraday drop drives the short put deep ITM and
    trips the time-dependent credit stop-loss. Deterministic (no RNG) so the test
    is exactly reproducible.
    """
    rows = []
    for d in range(n_dates):
        date = f"2024-{d + 1:02d}-15"
        base = 5000.0
        for b in range(bars):
            if b >= 15:
                # Adverse: fall ~3% over ~20 bars -> short put deep ITM -> stop.
                px = base * (1.0 - 0.030 * min((b - 15) / 20.0, 1.0))
            else:
                px = base
            rows.append(dict(
                date=date, bar_position=b,
                ATM_IV=0.20, RawSpread=0.02, DeltaSpread1=0.0, DeltaSpread5=0.0,
                MinutesToClose=max(0, 390 - b * 6), VIXSpot=18.0,
                VIXTermSlope=-0.5, RealizedVol30m=0.02, BarOfDay=b, SPXClose=px,
            ))
    return pd.DataFrame(rows)


def _run(df, td, markup: float):
    """Run the always-enter / never-voluntarily-exit BPC strategy under a given
    STOP_LOSS_FILL_MARKUP, restoring the module constant afterwards."""
    tpl = template_by_name("bull_put_credit_standard")  # credit; short put
    entry = from_sexpr("GT(EphReal(1.0), EphReal(0.0))")    # always true
    exit_ = from_sexpr("GT(EphReal(-10.0), EphReal(0.0))")  # always false -> stop is the exit
    size = from_sexpr("EphReal(0.5)")
    _orig = ev.STOP_LOSS_FILL_MARKUP
    ev.STOP_LOSS_FILL_MARKUP = markup
    try:
        return vectorized_backtest(
            entry, exit_, size, df, tpl, terminal_data=td,
            notional=NOTIONAL, warmup_bars=2, min_bars_in_trade=1,
        )
    finally:
        ev.STOP_LOSS_FILL_MARKUP = _orig


def _daily_mean(returns: np.ndarray, dates) -> float:
    """Mean of per-DAY summed bar_returns (the same aggregation the Sharpe uses)."""
    s = pd.Series(returns)
    return float(s.groupby(np.asarray(dates)).sum().mean())


def test_stoploss_markup_reconciles_to_realized_pnl():
    """(1) WITH a stop_loss exit present, sum(bar_returns) reconciles to
    sum(trade.pnl)/notional to ~machine precision. Pre-fix this DIVERGED by the
    markup the moment a stop fired (markup in trade.pnl, absent from bar_returns).
    """
    df = _adverse_bpc_df()
    td = prepare_terminal_data(df)
    r = _run(df, td, markup=ev.STOP_LOSS_FILL_MARKUP)  # production markup (>1.0)

    n_stop = sum(1 for t in r.trades if t.exit_reason == "stop_loss")
    assert n_stop >= 1, (
        f"fixture must produce >=1 stop_loss exit for this guard to bite; "
        f"got exits {[t.exit_reason for t in r.trades]}"
    )
    assert ev.STOP_LOSS_FILL_MARKUP > 1.0, (
        "production markup must be >1.0 or this test proves nothing")

    sum_ret = float(np.sum(r.returns))
    sum_pnl = float(sum(t.pnl for t in r.trades))
    assert abs(sum_ret - sum_pnl / NOTIONAL) < 1e-6, (
        f"reconciliation FAILED with a stop_loss present: "
        f"sum(bar_returns)={sum_ret:.10f} vs sum(pnl)/notional={sum_pnl / NOTIONAL:.10f} "
        f"(diff={sum_ret - sum_pnl / NOTIONAL:.2e}) — the stop-loss markup is not in the Sharpe basis"
    )


def test_stoploss_markup_worsens_the_booked_sharpe_basis():
    """(2) The markup must move the SHARPE BASIS, not just trade.pnl. With the markup
    ON, the booked sum(bar_returns) AND the mean daily booked return are strictly MORE
    NEGATIVE than with markup=1.0. Pre-fix, sum(bar_returns) was INDEPENDENT of the
    markup (the extra loss lived ONLY in trade.pnl/equity) — so this gap would have
    been exactly 0. This pins that the fix routed the markup loss into the fitness
    basis.

    We assert the DIRECTION of the gap, not its exact magnitude: the markup feeds
    `equity`, and entry sizing is capital-dependent, so a later trade can be sized
    slightly differently between the two runs (a legitimate second-order effect).
    The EXACT booked amount is proven separately by the reconciliation test — the
    only way sum(bar_returns) can equal sum(trade.pnl)/notional WITH the markup in
    trade.pnl is if the identical markup delta was booked into bar_returns.
    """
    df = _adverse_bpc_df()
    td = prepare_terminal_data(df)

    r_markup = _run(df, td, markup=1.10)
    r_flat = _run(df, td, markup=1.0)

    stops_markup = [t for t in r_markup.trades if t.exit_reason == "stop_loss"]
    assert len(stops_markup) >= 1, "fixture must stop-loss"

    sum_markup = float(np.sum(r_markup.returns))
    sum_flat = float(np.sum(r_flat.returns))

    # Headline fix: the Sharpe basis RESPONDS to the markup (pre-fix gap == 0 exactly).
    assert sum_markup < sum_flat - 1e-9, (
        f"sum(bar_returns) did NOT get worse with the stop-loss markup: "
        f"markup={sum_markup:.8f} vs flat={sum_flat:.8f} — markup not reaching the Sharpe basis"
    )

    # Mean DAILY booked return (the Sharpe numerator basis) is strictly worse.
    mean_markup = _daily_mean(r_markup.returns, df["date"].values)
    mean_flat = _daily_mean(r_flat.returns, df["date"].values)
    assert mean_markup < mean_flat - 1e-12, (
        f"mean daily booked return must be more negative with markup: "
        f"{mean_markup:.10f} vs {mean_flat:.10f}")

    # Mechanism: the EARLIEST stop-loss trade is identical across runs (it precedes
    # any equity divergence, so its size cannot differ), and its realized pnl in the
    # markup run is EXACTLY the markup multiple of the flat run. This proves the
    # markup is applied to the realized loss. (Later trades may be sized differently
    # once equity diverges — a legitimate capital-dependent effect — so we only pin
    # the first, divergence-free one.)
    stops_flat = {(t.entry_bar, t.exit_bar): t.pnl
                  for t in r_flat.trades if t.exit_reason == "stop_loss"}
    first = min(stops_markup, key=lambda t: t.entry_bar)
    pf = stops_flat.get((first.entry_bar, first.exit_bar))
    assert pf is not None and pf < 0.0, (
        "earliest markup stop must have a matching losing flat-run stop")
    assert first.pnl == pytest.approx(pf * 1.10, rel=1e-9), (
        f"earliest stop pnl not marked up by 1.10 (markup applied to realized loss?): "
        f"markup={first.pnl:.6f} flat={pf:.6f} ratio={first.pnl / pf:.6f}")

    # Both runs reconcile exactly (this is what proves the EXACT booked amount).
    for r in (r_markup, r_flat):
        s_ret = float(np.sum(r.returns))
        s_pnl = float(sum(t.pnl for t in r.trades)) / NOTIONAL
        assert abs(s_ret - s_pnl) < 1e-6, (
            f"reconciliation broke at a markup run: {s_ret:.10f} vs {s_pnl:.10f}")


def test_no_stoploss_path_is_unaffected_by_markup():
    """Control: a benign tape that NEVER stop-losses must be byte-identical under
    any markup (the markup only touches stop_loss exits with pnl<0). Guards against
    the booking leaking into non-stop exits."""
    rows = []
    for d in range(6):
        date = f"2024-{d + 1:02d}-15"
        base = 5000.0
        for b in range(50):
            px = base * (1.0 + 0.0002 * b)  # mild rally -> BPC wins, no stop
            rows.append(dict(
                date=date, bar_position=b, ATM_IV=0.15, RawSpread=0.02,
                DeltaSpread1=0.0, DeltaSpread5=0.0, MinutesToClose=max(0, 390 - b * 8),
                VIXSpot=14.0, VIXTermSlope=-0.5, RealizedVol30m=0.01, BarOfDay=b,
                SPXClose=px,
            ))
    df = pd.DataFrame(rows)
    td = prepare_terminal_data(df)
    r_markup = _run(df, td, markup=1.10)
    r_flat = _run(df, td, markup=1.0)
    assert all(t.exit_reason != "stop_loss" for t in r_markup.trades), (
        "control tape must not stop-loss")
    np.testing.assert_allclose(r_markup.returns, r_flat.returns, atol=1e-12,
                               err_msg="markup leaked into a non-stop-loss path")
