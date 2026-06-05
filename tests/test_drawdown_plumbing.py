"""max_drawdown_uncapped + profit_factor plumbing on BacktestResult.

The legacy `max_drawdown` is censored at 1.0, which makes a -1x and a -8x
drawdown indistinguishable — useless as a drawdown objective. `max_drawdown_
uncapped` exposes the true peak-to-trough depth; `profit_factor` = gross_profit/
gross_loss. (Both remain PROXY statistics — the equity floor de-levers, so they
under-report QC ruin; ruin is gated at the QC stage. These feed within-template
rank pressure only.)
"""
import pathlib
import warnings

import pytest

warnings.filterwarnings("ignore")
_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"


def test_fields_exist_with_defaults():
    from layer2.evaluator import BacktestResult
    import dataclasses
    names = {f.name for f in dataclasses.fields(BacktestResult)}
    assert "max_drawdown_uncapped" in names
    assert "profit_factor" in names


@pytest.mark.skipif(not pathlib.Path(_PARQUET).exists(), reason="minute parquet absent")
def test_uncapped_exceeds_cap_and_profit_factor_sane():
    import json
    from layer2.io import load_minute_parquet
    from layer2.evaluator_vectorized import prepare_terminal_data, vectorized_backtest
    from layer2.terminal_stats import compute_norm_stats_from_data
    from layer2.templates import base_template_by_name
    from layer2.grammar import from_sexpr

    df, _ = load_minute_parquet(_PARQUET)
    norm = compute_norm_stats_from_data(df[df["date"] <= "2024-03-29"])
    sl = df[(df["date"] > "2024-09-27") & (df["date"] <= "2025-03-31")].reset_index(drop=True)
    td = prepare_terminal_data(sl, norm_stats_override=norm)
    # An always-on, max-size iron condor is a GENUINE deep loser post-B2-fin. (We no
    # longer use the RPB "loser" here: the 2026-06-01 IV upper-clamp removed corrupt-
    # bar phantom losses and flipped its proxy Sharpe -0.22 → +0.29 — confirming prior
    # strategy evaluations were partly driven by data-corruption artifacts.)
    r = vectorized_backtest(
        from_sexpr("GT(MinutesToClose, EphReal(-100.0))"),   # always enter
        from_sexpr("LT(MinutesToClose, EphReal(-100.0))"),   # never voluntarily exit
        from_sexpr("EphReal(0.9)"), sl, base_template_by_name("iron_condor"),
        delta_tree=None, terminal_data=td)
    # legacy field stays censored at 1.0; uncapped exposes the true (deeper) depth
    assert r.max_drawdown <= 1.0 + 1e-9
    assert r.max_drawdown_uncapped >= r.max_drawdown
    assert r.max_drawdown_uncapped > 1.0, "deep-drawdown loser must exceed the 1.0 cap"
    # profit_factor: a net loser (negative Sharpe) has gross_profit < gross_loss
    assert 0.0 <= r.profit_factor < 1.0
    assert r.sharpe < 0  # sanity: an always-on max-size IC is a loser


@pytest.mark.skipif(not pathlib.Path(_PARQUET).exists(), reason="minute parquet absent")
def test_profit_factor_matches_trade_pnls():
    """profit_factor must equal gross_profit/gross_loss of the actual trades."""
    import json
    from layer2.io import load_minute_parquet
    from layer2.evaluator_vectorized import prepare_terminal_data, vectorized_backtest
    from layer2.terminal_stats import compute_norm_stats_from_data
    from layer2.templates import base_template_by_name
    from layer2.grammar import from_sexpr

    df, _ = load_minute_parquet(_PARQUET)
    norm = compute_norm_stats_from_data(df[df["date"] <= "2024-03-29"])
    sl = df[(df["date"] > "2024-09-27") & (df["date"] <= "2025-03-31")].reset_index(drop=True)
    td = prepare_terminal_data(sl, norm_stats_override=norm)
    p = [x for x in json.load(open("generated_qc/champ_panel/winners_scan.json"))
         if x["template"] == "bear_call_credit"][0]
    r = vectorized_backtest(from_sexpr(p["entry"]), from_sexpr(p["exit"]),
                            from_sexpr(p["size"]), sl, base_template_by_name("bear_call_credit"),
                            delta_tree=from_sexpr(p["delta"]), terminal_data=td)
    pnls = [t.pnl for t in r.trades]
    gp = sum(x for x in pnls if x > 0)
    gl = -sum(x for x in pnls if x < 0)
    expected = (gp / gl) if gl > 1e-9 else (10.0 if gp > 0 else 0.0)
    assert abs(r.profit_factor - expected) < 1e-6


@pytest.mark.skipif(not pathlib.Path(_PARQUET).exists(), reason="minute parquet absent")
def test_no_trades_edge_zeros():
    """A never-firing entry -> 0 trades -> profit_factor and uncapped drawdown
    both default to 0.0 (the min_trades gate rejects these first)."""
    from layer2.io import load_minute_parquet
    from layer2.evaluator_vectorized import prepare_terminal_data, vectorized_backtest
    from layer2.terminal_stats import compute_norm_stats_from_data
    from layer2.templates import base_template_by_name
    from layer2.grammar import from_sexpr

    df, _ = load_minute_parquet(_PARQUET)
    norm = compute_norm_stats_from_data(df[df["date"] <= "2024-03-29"])
    sl = df[(df["date"] > "2024-09-27") & (df["date"] <= "2024-10-15")].reset_index(drop=True)
    td = prepare_terminal_data(sl, norm_stats_override=norm)
    r = vectorized_backtest(from_sexpr("LT(EphReal(1.0), EphReal(0.0))"),  # never true
                            from_sexpr("LT(EphReal(1.0), EphReal(0.0))"),
                            from_sexpr("EphReal(0.5)"), sl, base_template_by_name("bear_call_credit"),
                            delta_tree=None, terminal_data=td)
    assert r.total_trades == 0
    assert r.profit_factor == 0.0
    assert r.max_drawdown_uncapped == 0.0


@pytest.mark.skipif(not pathlib.Path(_PARQUET).exists(), reason="minute parquet absent")
def test_equity_floor_liquidates_rather_than_amplifies():
    """BLOCKER-3 regression (2026-06-01 audit): the equity floor is 0.0, NOT 0.10.

    The old `available_capital = notional * max(1+equity, 0.10)` floor let a ruined
    book keep trading, AMPLIFYING equity to -800%/-1450% (adj_dd ~29). A cash account
    cannot lose >100%: with a 0.0 floor, available_capital -> 0 as equity -> -100%,
    the margin gate halts new entries, and the drawdown bottoms near liquidation.

    An always-on, max-size iron condor is a forced blowup. Under the fix its
    adj_dd must sit in a REALISTIC ruin band (a few x position size), NOT the old
    ~29 amplification, and the trade count must be bounded (entries halted at ruin).
    """
    from layer2.io import load_minute_parquet
    from layer2.evaluator_vectorized import prepare_terminal_data, vectorized_backtest
    from layer2.terminal_stats import compute_norm_stats_from_data
    from layer2.templates import base_template_by_name
    from layer2.grammar import from_sexpr

    df, _ = load_minute_parquet(_PARQUET)
    norm = compute_norm_stats_from_data(df[df["date"] <= "2024-03-29"])
    sl = df[(df["date"] > "2024-09-27") & (df["date"] <= "2025-03-31")].reset_index(drop=True)
    td = prepare_terminal_data(sl, norm_stats_override=norm)
    r = vectorized_backtest(
        from_sexpr("GT(MinutesToClose, EphReal(-100.0))"),   # always enter
        from_sexpr("LT(MinutesToClose, EphReal(-100.0))"),   # never voluntarily exit
        from_sexpr("EphReal(0.9)"), sl, base_template_by_name("iron_condor"),
        delta_tree=None, terminal_data=td)
    assert r.sharpe < 0, "always-on max-size IC must be a loser"
    assert r.max_drawdown_uncapped > 1.0, "a blowup must exceed the 1.0 legacy cap"
    adj_dd = r.max_drawdown_uncapped / max(r.avg_position_size, 0.05)
    # Realistic liquidation: a few x position size. The old 0.10 floor amplified the
    # SAME strategy to adj_dd ~29 — this bound (well below that) is the regression.
    assert adj_dd < 8.0, (
        f"adj_dd={adj_dd:.2f} too deep — the equity floor is amplifying ruin "
        f"(old 0.10-floor artifact), not liquidating near -100%")
    # The margin gate must halt entries once the book is ruined (not enter every day
    # of the ~125-day window).
    n_days = sl["date"].nunique()
    assert r.total_trades < n_days, (
        f"trades={r.total_trades} vs {n_days} days — ruined book should stop entering")
