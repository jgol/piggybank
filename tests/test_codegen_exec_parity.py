"""Codegen <-> proxy execution-constant parity.

The proxy evaluator (``layer2.evaluator_vectorized.vectorized_backtest``) is the
training / selection environment in which strategies are evolved and ranked; the
generated QuantConnect algorithm (``layer3.codegen.generate_qc_algorithm``) is the
serving environment in which they are deployed. Any execution constant that
differs between the two is *training-serving skew* and makes a proxy-good
strategy realize different PnL in QC, breaking the proxy's ability to predict QC
performance (Sculley et al., 2015, *Hidden Technical Debt in Machine Learning
Systems*, NeurIPS).

These tests pin the shared constants to a single source of truth (the proxy
defaults) so they cannot silently drift again — e.g. codegen formerly hard-coded
MAX_BARS_IN_TRADE=180 with a false "proxy=180" comment while the proxy used 330.
"""
import inspect

from layer2.evaluator_vectorized import vectorized_backtest
from layer3.codegen import generate_qc_algorithm

_PROXY = inspect.signature(vectorized_backtest).parameters


def _generate_sample_algorithm() -> str:
    """Generate a representative V1 (scalar-only, no delta_tree) QC algorithm."""
    return generate_qc_algorithm(
        strategy_id="exec_parity_probe",
        template_name="bull_put_credit",          # a V1 defined-risk template
        entry_sexpr="GT(ATM_IV, EphReal(0.18))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)",
        start_date="2025-01-02",
        end_date="2025-01-31",
        delta_sexpr=None,                          # V1: no Level-B delta tree
    )


def test_max_bars_in_trade_matches_proxy_default():
    """Generated QC MAX_BARS_IN_TRADE must equal the proxy's max_bars_in_trade.

    Holding policy is part of the strategy's realized PnL; serving a shorter cap
    than the proxy used during evolution changes captured theta on the SAME
    strategy (training-serving skew).
    """
    proxy_max = _PROXY["max_bars_in_trade"].default
    code = _generate_sample_algorithm()
    assert f"MAX_BARS_IN_TRADE = {proxy_max}" in code, (
        f"codegen MAX_BARS_IN_TRADE must equal proxy default {proxy_max}; "
        f"generated code does not contain 'MAX_BARS_IN_TRADE = {proxy_max}'"
    )


def test_min_bars_in_trade_matches_proxy_default():
    """Generated QC MIN_BARS_IN_TRADE must equal the proxy's min_bars_in_trade."""
    proxy_min = _PROXY["min_bars_in_trade"].default
    code = _generate_sample_algorithm()
    assert f"MIN_BARS_IN_TRADE = {proxy_min}" in code, (
        f"codegen MIN_BARS_IN_TRADE must equal proxy default {proxy_min}"
    )


def test_v1_stop_loss_base_matches_proxy():
    """P0-3: a V1 (no dynamic delta_tree) strategy's credit stop-loss base must
    equal the proxy V1 branch (evaluator_vectorized.py:1200):
        stop_loss_credit_multiple * STOP_LOSS_EXECUTION_DISCOUNT.
    Previously codegen always emitted the Level-B formula (1.5 + 0.25*2)*0.85 = 1.70
    for V1 strategies, vs the proxy's 2.5*0.85 = 2.125 — a systematic skew that let
    proxy losers run further than QC.
    """
    from layer2.evaluator_vectorized import STOP_LOSS_EXECUTION_DISCOUNT
    proxy_mult = _PROXY["stop_loss_credit_multiple"].default
    code = _generate_sample_algorithm()  # delta_sexpr=None -> V1
    expected = f"_stop_loss_multiple = {proxy_mult} * {STOP_LOSS_EXECUTION_DISCOUNT}"
    assert expected in code, f"expected V1 stop base '{expected}' in generated code"
    # V1 must NOT use the Level-B delta-dependent formula
    assert "(1.5 + self._entry_short_delta * 2.0)" not in code, (
        "V1 strategy must not emit the Level-B delta-dependent stop base"
    )


def test_level_b_stop_loss_base_uses_delta_formula():
    """P0-3: a dynamic-delta_tree (Level B) strategy must use the proxy Level-B
    branch (1.5 + short_delta*2.0) * STOP_LOSS_EXECUTION_DISCOUNT
    (evaluator_vectorized.py:1198), with _entry_short_delta set at entry."""
    from layer2.evaluator_vectorized import STOP_LOSS_EXECUTION_DISCOUNT
    code = generate_qc_algorithm(
        strategy_id="exec_parity_levelb",
        template_name="bull_put_credit",   # member of BASE_TEMPLATE_DELTA_RANGES
        entry_sexpr="GT(ATM_IV, EphReal(0.18))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)",
        start_date="2025-01-02",
        end_date="2025-01-31",
        delta_sexpr="EphReal(0.0)",        # dynamic delta tree -> Level B
    )
    expected = (
        f"_stop_loss_multiple = (1.5 + self._entry_short_delta * 2.0) "
        f"* {STOP_LOSS_EXECUTION_DISCOUNT}"
    )
    assert expected in code, f"expected Level-B stop base '{expected}' in generated code"


def test_iron_butterfly_level_b_stop_keys_on_atm_short():
    """P0-3 (IB): iron_butterfly Level-B shorts are ATM (0.50) and the delta_tree
    controls the WING, not the short. The stop-loss base keys on the SHORT delta,
    so generated IB code must pin _entry_short_delta to 0.50 — matching the proxy's
    _compute_dynamic_legs (which returns the 0.50 short). Keying on the ~0.20 wing
    would make the served stop ~24% too tight."""
    code = generate_qc_algorithm(
        strategy_id="exec_parity_ib",
        template_name="iron_butterfly",    # member of BASE_TEMPLATE_DELTA_RANGES
        entry_sexpr="GT(ATM_IV, EphReal(0.18))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)",
        start_date="2025-01-02",
        end_date="2025-01-31",
        delta_sexpr="EphReal(0.0)",        # dynamic delta tree -> Level B
    )
    assert "self._entry_short_delta = 0.50" in code, (
        "IB Level-B must pin _entry_short_delta to the ATM 0.50 short for the stop base"
    )


def test_vix_terminals_use_real_cboe_source():
    """P0-5: VIXSpot / VIXTermSlope / VIXChange / VIXMean5d must come from the real
    CBOE VIX + VIX9D (the L1 collector's source, proven available in QC backtest via
    gp_condA_iron_condor.py:53-54), not the ATM_IV*100 proxy that mis-scaled VIXSpot
    (~10 vs ~17) and froze VIXTermSlope to a dead constant."""
    from layer3.codegen import validate_generated_code
    code = _generate_sample_algorithm()
    ok, err = validate_generated_code(code)
    assert ok, f"generated code must be valid Python: {err}"
    # Real CBOE subscriptions present
    assert 'self.AddData(CBOE, "VIX", Resolution.Daily)' in code
    assert 'self.AddData(CBOE, "VIX9D", Resolution.Daily)' in code
    # VIXSpot terminal no longer fabricated as ATM_IV*100
    assert "vix_spot = atm_iv * 100.0" not in code, (
        "VIXSpot must not be the ATM_IV*100 proxy"
    )
    # VIXTermSlope has a live VIX9D - VIX path (not just the frozen constant)
    assert "vix_term_slope = _vix9d - vix_spot" in code, (
        "VIXTermSlope must compute VIX9D - VIX from real CBOE indices"
    )
    # VIXChange differences the real VIX day-over-day, not ATM_IV*100
    assert "(atm_iv - getattr(self, '_prev_day_atm_iv', atm_iv)) * 100.0" not in code, (
        "VIXChange must not difference the ATM_IV*100 proxy"
    )
    assert "vix_spot - getattr(self, '_prev_day_vix'" in code, (
        "VIXChange must difference real VIX day-over-day"
    )


def test_norm_stats_override_threads_into_generated_code():
    """P0-6: normalization must be identical at train (proxy) and serve (QC). When
    the caller passes the per-fold MINUTE stats the strategy was evolved under, the
    generated _NORM_STATS must use them (not the frozen daily TERMINAL_NORM_STATS),
    so the same raw terminal value normalizes identically in QC and the proxy.
    Absent an override, the frozen constants are used (backward-compatible)."""
    from layer2.terminal_stats import TERMINAL_NORM_STATS
    fc, fs, _ = TERMINAL_NORM_STATS["ATM_IV"]

    # Default: frozen constants embedded.
    default_code = _generate_sample_algorithm()
    assert f'"ATM_IV": ({fc}, {fs})' in default_code

    # Override: per-fold stats replace the frozen ones for the given terminals.
    custom = {
        "ATM_IV": (0.1234, 0.0456, "robust"),
        "MinutesToClose": (202.0, 117.0, "standard"),
    }
    code = generate_qc_algorithm(
        strategy_id="norm_override",
        template_name="bull_put_credit",
        entry_sexpr="GT(ATM_IV, EphReal(0.18))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)",
        start_date="2025-01-02",
        end_date="2025-01-31",
        norm_stats=custom,
    )
    assert '"ATM_IV": (0.1234, 0.0456)' in code
    assert '"MinutesToClose": (202.0, 117.0)' in code
    assert f'"ATM_IV": ({fc}, {fs})' not in code, "frozen ATM_IV stats must be replaced"
    # A terminal NOT in the override keeps its frozen stats.
    vc, vs, _ = TERMINAL_NORM_STATS["VIXSpot"]
    assert f'"VIXSpot": ({vc}, {vs})' in code

    # Persisted fold stats arrive from JSON as LISTS (JSON has no tuples), so a
    # list value must work identically to a tuple (positional [0]/[1] access).
    code_list = generate_qc_algorithm(
        strategy_id="norm_override_list",
        template_name="bull_put_credit",
        entry_sexpr="GT(ATM_IV, EphReal(0.18))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)",
        start_date="2025-01-02",
        end_date="2025-01-31",
        norm_stats={"ATM_IV": [0.222, 0.033, "robust"]},
    )
    assert '"ATM_IV": (0.222, 0.033)' in code_list


def test_fold_stats_json_roundtrip_into_codegen():
    """P0-6 (A): the per-fold stats a GP run persists (compute_norm_stats_from_data,
    serialized to JSON as lists) must round-trip cleanly into the codegen override,
    so a deployed strategy normalizes with the exact stats it was evolved under."""
    import json
    import numpy as np
    import pandas as pd
    from layer2.terminal_stats import compute_norm_stats_from_data

    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "ATM_IV": rng.uniform(0.08, 0.20, 500),
        "MinutesToClose": rng.uniform(0.0, 405.0, 500),
    })
    stats = compute_norm_stats_from_data(df)            # {name: (center, scale, method)}
    reloaded = json.loads(json.dumps(stats))            # tuples -> lists (as persisted)
    code = generate_qc_algorithm(
        strategy_id="fold_roundtrip",
        template_name="bull_put_credit",
        entry_sexpr="GT(ATM_IV, EphReal(0.18))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)",
        start_date="2025-01-02",
        end_date="2025-01-31",
        norm_stats=reloaded,
    )
    ac, asc = reloaded["ATM_IV"][0], reloaded["ATM_IV"][1]
    mc, msc = reloaded["MinutesToClose"][0], reloaded["MinutesToClose"][1]
    assert f'"ATM_IV": ({ac}, {asc})' in code
    assert f'"MinutesToClose": ({mc}, {msc})' in code


# ---------------------------------------------------------------------------
# R1/R2 — EOD-close / entry-cutoff time-base parity (2026-05-29)
# ---------------------------------------------------------------------------
# The proxy gates entries/exits on MinutesToClose; codegen gates on wall-clock ET.
# MinutesToClose is measured to the 16:15 ET SPXW PM-settlement (collector), so an
# mtc threshold maps to a clock time as (16:15 - mtc). These tests pin the two
# representations to ONE shared trading window so training (proxy) and serving
# (QC) enter and force-close at the same instant. Previously the proxy gated at
# mtc<=10/<=15 (== 16:05/16:00 ET) while codegen gated at 15:50/15:45 — a ~15-min
# asymmetric, proxy-optimistic skew (the proxy banked EOD theta + late entries QC
# never realized behind its 16:00 hard-guard).

_SETTLE_MINUTES = 16 * 60 + 15  # 16:15 ET SPXW PM-settlement reference


def _mtc_to_walltime(mtc):
    """(hour, minute) ET for a MinutesToClose threshold, via 16:15 - mtc."""
    return divmod(_SETTLE_MINUTES - int(mtc), 60)


def test_eod_gate_constants_map_to_codegen_walltimes():
    """The proxy mtc gate constants must map to codegen's wall-clock gates:
    EOD_FORCE_CLOSE_MTC -> 15:50 force-close, ENTRY_CUTOFF_MTC -> 15:45 cutoff.
    A drifting constant lands on the wrong clock minute and fails here."""
    from layer2.evaluator_vectorized import (
        EOD_FORCE_CLOSE_MTC, ENTRY_CUTOFF_MTC,
    )
    assert _mtc_to_walltime(EOD_FORCE_CLOSE_MTC) == (15, 50), (
        f"EOD_FORCE_CLOSE_MTC={EOD_FORCE_CLOSE_MTC} must == 15:50 ET force-close"
    )
    assert _mtc_to_walltime(ENTRY_CUTOFF_MTC) == (15, 45), (
        f"ENTRY_CUTOFF_MTC={ENTRY_CUTOFF_MTC} must == 15:45 ET entry cutoff"
    )
    # You must stop entering before you force out (cutoff is earlier -> larger mtc).
    assert ENTRY_CUTOFF_MTC > EOD_FORCE_CLOSE_MTC


def test_codegen_emits_walltimes_derived_from_proxy_constants():
    """Generated QC code must force-close and cut entries at the wall-clock times
    DERIVED from the proxy mtc constants — locking codegen to the proxy (single
    source of truth). Drift on either side breaks this."""
    from layer2.evaluator_vectorized import (
        EOD_FORCE_CLOSE_MTC, ENTRY_CUTOFF_MTC,
    )
    code = _generate_sample_algorithm()
    ch, cm = _mtc_to_walltime(EOD_FORCE_CLOSE_MTC)   # (15, 50)
    eh, em = _mtc_to_walltime(ENTRY_CUTOFF_MTC)      # (15, 45)
    assert f"self.Time.hour == {ch} and self.Time.minute >= {cm}" in code, (
        f"codegen force-close must gate at {ch:02d}:{cm:02d} "
        f"(== 16:15 - EOD_FORCE_CLOSE_MTC={EOD_FORCE_CLOSE_MTC})"
    )
    assert f"self.Time.hour == {eh} and self.Time.minute >= {em}" in code, (
        f"codegen entry cutoff must gate at {eh:02d}:{em:02d} "
        f"(== 16:15 - ENTRY_CUTOFF_MTC={ENTRY_CUTOFF_MTC})"
    )


def test_proxy_gates_use_the_named_constants():
    """The proxy must reference EOD_FORCE_CLOSE_MTC / ENTRY_CUTOFF_MTC at its gates
    (not a re-hardcoded literal), so the single source of truth is genuinely wired."""
    src = inspect.getsource(vectorized_backtest)
    assert "mtc <= EOD_FORCE_CLOSE_MTC" in src, (
        "proxy EOD force-close must use the EOD_FORCE_CLOSE_MTC constant"
    )
    assert "mtc <= ENTRY_CUTOFF_MTC" in src, (
        "proxy entry cutoff must use the ENTRY_CUTOFF_MTC constant"
    )


# ---------------------------------------------------------------------------
# P0-5b — CBOE VIX source-failure must be loud, not silent
# ---------------------------------------------------------------------------
# P0-5 restored real CBOE VIX, but if the AddData(CBOE,"VIX") subscription never
# delivers data the VIX terminals degrade silently — VIXSpot reverts to the
# ATM_IV*100 proxy, the others to frozen constants — re-injecting the skew P0-5
# removed, with no error. This defends the P0-5 parity guarantee at serve time by
# forcing a dead subscription to FAIL LOUD.


def test_vix_source_failure_is_loud_not_silent():
    """A dead CBOE VIX subscription must not produce a silently-accepted result:
    the generated code must detect 'real VIX never populated' and emit a
    machine-detectable runtime statistic + an Error, in a guaranteed
    (OnEndOfAlgorithm) path — not quietly fall back to ATM_IV*100 forever."""
    from layer3.codegen import validate_generated_code
    code = _generate_sample_algorithm()
    ok, err = validate_generated_code(code)
    assert ok, f"generated code must be valid Python: {err}"

    # Health flags initialized.
    for init in (
        "self._vix_ever_populated = False",
        "self._vix_source_failed = False",
        "self._vix_fallback_bars = 0",
        "self._day_count = 0",
    ):
        assert init in code, f"missing P0-5b init: {init}"

    # The proxy fallback is COUNTED wherever VIXSpot reverts — no longer silent.
    assert "vix_spot = self._derived_vix" in code
    assert "self._vix_fallback_bars += 1" in code, (
        "VIXSpot proxy fallback must be counted, not silent"
    )

    # Real-VIX availability latches _vix_ever_populated from the live security price.
    assert "self.Securities[self.vix].Price > 0" in code
    assert "self._vix_ever_populated = True" in code

    # In-loop early warning fires ONCE (latched) and only after >=3 distinct days
    # with no real VIX (warmup-robust — a working daily subscription delivers by
    # day 2, so day-1 fallback never false-alarms).
    assert "self._day_count >= 3" in code, (
        "dead-VIX gate must require >=3 days so warmup never false-alarms"
    )
    assert "not self._vix_source_failed" in code, (
        "the early-warning Error must be latched to fire once"
    )

    # Machine-detectable invalid marker emitted in BOTH the in-loop gate and the
    # guaranteed end-of-algorithm backstop (the latter fires for any run length).
    assert code.count('self.SetRuntimeStatistic("vix_source_failed", "1")') >= 2, (
        "vix_source_failed runtime stat must be set in both the in-loop gate and "
        "the OnEndOfAlgorithm backstop"
    )
    assert "VIX SOURCE FAILURE (final)" in code, (
        "OnEndOfAlgorithm must emit a guaranteed final failure check"
    )
    assert "if not getattr(self, '_vix_ever_populated', False):" in code, (
        "the end-of-algorithm backstop must be unconditional on run length"
    )
