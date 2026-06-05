"""Codegen eval-SCHEDULE parity with the proxy (2026-05-31 over-firing fix).

ROOT CAUSE this guards against
------------------------------
The proxy evaluator computes entry / exit / size / delta signals over ALL bars
(``evaluator_vectorized.py`` evaluates each tree once over the whole array, lines
1017-1051). The generated QC algorithm formerly evaluated the entry tree only
when flat and the exit tree only when in-position. For trees with NESTED Lag/Delta
(which buffer their intermediate value via ``_lag_expr``/``_delta_expr`` *during*
evaluation), skipping evaluation on a bar starved the buffer, so a later ``Lag^k``
reached across the gap to a STALE value. The bcc_f2 entry
``GT(VIXChange, Lag^3(VIXChange))`` consequently re-fired ~6.4x/active-day in QC
vs the proxy's ~1.5 (QC Sharpe -4.0). Proven offline in
``layer3/diagnostics/operator_parity.py``: 51 vs 12 entries → IDENTICAL once all
trees are evaluated every bar.

THE FIX (codegen.py:_on_bar): evaluate ALL trees every bar (continuous buffers),
store on ``self._sig_*``, and gate only the ACTION by position state.

Two layers of defence:
  * STRUCTURAL — the generated source must evaluate every tree up-front and the
    action gates must read the cached signals (catches a schedule revert).
  * OFFLINE PARITY — the codegen's real operators, under the deployed schedule,
    must reproduce the proxy's trades on the bcc_f2 regression case (catches an
    operator/expression regression). Skipped if the minute parquet is absent.
"""
import os

import pytest

from layer3.codegen import generate_qc_algorithm, validate_generated_code

_BCC_F2 = dict(
    strategy_id="sched_parity_bcc_f2",
    template_name="bear_call_credit_standard",
    entry_sexpr="GT(VIXChange, Lag(Lag(Lag(VIXChange, EphInt(3)), EphInt(30)), EphInt(30)))",
    exit_sexpr="CrossBelow(RealizedVol30m_5m, Lag(Sub(Lag(EphReal(-0.7563), EphInt(3)), Delta(DeltaSpread5, EphInt(30))), EphInt(3)))",
    size_sexpr="Delta(ATM_IV_5m, EphInt(1))",
    start_date="2025-01-02",
    end_date="2025-01-31",
)

_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"


# ---------------------------------------------------------------------------
# STRUCTURAL — always runs (no data dependency)
# ---------------------------------------------------------------------------
def test_all_trees_evaluated_up_front_every_bar():
    """_on_bar must evaluate entry/exit/size/delta up-front EVERY bar so the
    nested Lag/Delta expr-buffers stay continuous (proxy parity)."""
    code = generate_qc_algorithm(**_BCC_F2)
    for stmt in (
        "self._sig_entry = bool(self._eval_entry(s))",
        "self._sig_exit = bool(self._eval_exit(s))",
        "self._sig_size = max(0.0, min(1.0, self._eval_size(s)))",
        "self._sig_delta = self._eval_delta(s)",
    ):
        assert stmt in code, f"missing up-front per-bar evaluation: {stmt!r}"


def test_action_gates_read_cached_signals_not_inline_eval():
    """The position-state branches must consume the cached signals, and the
    pre-fix inline evaluations must be GONE (else the buffers gap again)."""
    code = generate_qc_algorithm(**_BCC_F2)
    assert "entry_signal = self._sig_entry" in code
    assert "if self._sig_exit:" in code
    assert "self._pending_size = self._sig_size" in code
    # pre-fix patterns must not return
    assert "entry_signal = self._eval_entry(s)" not in code, (
        "entry must not be re-evaluated inline in the flat branch"
    )
    assert "if self._eval_exit(s):" not in code, (
        "exit must not be re-evaluated inline in the in-position branch"
    )


def test_each_signal_tree_evaluated_exactly_once_per_bar():
    """Each tree is evaluated exactly once per bar (the up-front cache). A second
    real call would double-append its Lag/Delta buffer and re-break parity.
    Comment lines that merely mention the methods are ignored."""
    code = generate_qc_algorithm(**_BCC_F2)
    real = [ln.strip() for ln in code.splitlines() if not ln.strip().startswith("#")]
    for meth in ("_eval_entry(s)", "_eval_exit(s)", "_eval_size(s)", "_eval_delta(s)"):
        hits = sum(ln.count(f"self.{meth}") for ln in real)
        assert hits == 1, f"self.{meth} must be called exactly once (found {hits})"


def test_level_b_open_uses_cached_delta_not_reeval():
    """Level-B dynamic open must read the cached _sig_delta, not re-evaluate the
    delta tree on the fill bar (which double-appends its buffer)."""
    code = generate_qc_algorithm(
        strategy_id="sched_parity_lb", template_name="bull_put_credit",
        entry_sexpr="GT(ATM_IV, EphReal(0.18))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)", delta_sexpr="EphReal(0.0)",
        start_date="2025-01-02", end_date="2025-01-31",
    )
    ok, err = validate_generated_code(code)
    assert ok, f"Level-B generated code must be valid: {err}"
    assert "_delta_val = self._sig_delta" in code
    assert "_delta_val = self._eval_delta(s)" not in code
    assert code.count("self._eval_delta(s)") == 1  # only the up-front eval


def test_delta_within_day_mask_present():
    """Delta (terminal and computed-expression) must mask within_day_pos<lag -> 0,
    matching the proxy (evaluator_vectorized.py:306). Without it, early-day Delta
    returns (current - 0) = current instead of 0 — a serving-vs-objective skew."""
    code = generate_qc_algorithm(**_BCC_F2)
    assert "if not buf or len(buf) <= k:" in code, (
        "_delta must return 0.0 until the buffer holds > k entries this day"
    )
    assert "if len(self._buffers[key]) <= k:" in code, (
        "_delta_expr must return 0.0 until the buffer holds > k entries this day"
    )


def test_diagnostic_log_off_by_default():
    """diagnostic_log defaults off — no QCDIAG line, behaviour unchanged."""
    code = generate_qc_algorithm(**_BCC_F2)
    assert "QCDIAG" not in code


def test_diagnostic_log_emits_full_terminal_dict():
    """diagnostic_log=True emits a valid QCDIAG line dumping the FULL normalized
    terminal dict `s` once/day (G11 proxy<->QC calibration, codegen.py 2026-06-01).

    Tracks the COMMITTED behaviour: the prior per-named-terminal `.4f` format was
    replaced by a generic `sorted(s.items())` dump so ONE diagnostic run yields
    EVERY terminal for the MAE/offset re-fit (not just this strategy's tree
    terminals). This test was previously asserting the retired `.4f` shape and was
    stale-red against the committed generic format; updated 2026-06-03.
    """
    code = generate_qc_algorithm(diagnostic_log=True, **_BCC_F2)
    ok, err = validate_generated_code(code)
    assert ok, f"diagnostic code must be valid Python: {err}"
    assert "QCDIAG" in code
    # Generic dump of the WHOLE terminal dict (every terminal in `s`, so VIXChange,
    # ATM_IV_5m, DeltaSpread5, … are all logged at runtime), excluding internal
    # _-prefixed keys except _raw_RawSpread.
    assert ("_diag_vals = ' '.join(f'{_k}={float(_v):.5f}' for _k, _v in "
            "sorted(s.items()) if (not _k.startswith('_') or _k == '_raw_RawSpread'))") in code
    assert 'QCDIAG {self.Time:%Y-%m-%d %H:%M} {_diag_vals}' in code
    # Gated to the first bar at/after 10:00 ET, once per day (copy-paste-able).
    assert ("if self.Time.hour >= 10 and getattr(self, '_qcdiag_day', None) "
            "!= self.Time.date():") in code
    # Session-open gap-timing line (round-2a) is emitted alongside.
    assert "QCGAP" in code


# ---------------------------------------------------------------------------
# OFFLINE PARITY — the regression case (needs the minute parquet)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(_PARQUET),
                    reason="minute parquet not present (data-dependent test)")
def test_bcc_f2_scheduled_parity_offline():
    """The deployed schedule, on the proxy's own terminal values, must reproduce
    the proxy's trades EXACTLY for bcc_f2 — the over-firing regression case."""
    from layer3.diagnostics.operator_parity import diagnose, StrategySpec
    spec = StrategySpec(
        strategy_id="bcc_f2",
        template_name=_BCC_F2["template_name"],
        entry_sexpr=_BCC_F2["entry_sexpr"],
        exit_sexpr=_BCC_F2["exit_sexpr"],
        size_sexpr=_BCC_F2["size_sexpr"],
        is_credit=True,
    )
    r = diagnose(spec, _PARQUET, window=("2025-01-02", "2025-01-31"),
                 train_end="2024-09-27", warmup_start="2024-12-15")
    # operators reproduce the proxy signals exactly when eval'd continuously
    assert r["continuous_entry_match_rate"] == 1.0
    assert r["continuous_exit_match_rate"] == 1.0
    # and the deployed schedule reproduces the proxy's trade COUNT (was 51 vs 12)
    assert r["scheduled_codegen_entries"] == r["scheduled_proxy_entries"], (
        f"over-firing regressed: codegen {r['scheduled_codegen_entries']} vs "
        f"proxy {r['scheduled_proxy_entries']} entries"
    )
    # every active day shared (no day-selection drift under identical terminals)
    assert r["scheduled_shared_days"] == r["scheduled_proxy_active_days"]


@pytest.mark.skipif(not os.path.exists(_PARQUET),
                    reason="minute parquet not present (data-dependent test)")
def test_delta_operator_parity_offline():
    """A strategy whose entry uses Delta(<expression>) and exit uses Delta(<terminal>)
    must match the proxy's continuous signal EXACTLY (1.000) — the within-day Delta
    mask fix. Pre-fix, Delta(<expression>) matched only ~0.9935 (divergence at the
    first bar of each day)."""
    from layer3.diagnostics.operator_parity import diagnose, StrategySpec
    spec = StrategySpec(
        strategy_id="delta_mask", template_name="bull_put_credit",
        entry_sexpr="GT(Delta(Lag(ATM_IV, EphInt(3)), EphInt(5)), EphReal(0.0))",
        exit_sexpr="LT(Delta(RealizedVol30m, EphInt(10)), EphReal(0.0))",
        size_sexpr="EphReal(0.5)", is_credit=True,
    )
    r = diagnose(spec, _PARQUET, window=("2025-01-02", "2025-01-31"),
                 train_end="2024-09-27", warmup_start="2024-12-15")
    assert r["continuous_entry_match_rate"] == 1.0, (
        f"Delta(<expression>) entry diverges: {r['continuous_entry_match_rate']}"
    )
    assert r["continuous_exit_match_rate"] == 1.0, (
        f"Delta(<terminal>) exit diverges: {r['continuous_exit_match_rate']}"
    )
