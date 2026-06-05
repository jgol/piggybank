"""Runtime smoke tests for generated codegen methods.

`validate_generated_code` only checks that the code PARSES — it does not catch
AttributeErrors / NameErrors that fire at runtime. A T2 inter-day buffer
(`_daily_atm_iv`) was created lazily in one path but read in another, so a
syntactically-valid backtest crashed at the first day boundary with 0 trades.
These tests exec the generated class and actually CALL the pure methods to catch
that class of "validates but doesn't run" failure offline (memory: "it runs" !=
"it works"; run a runtime smoke test after codegen changes)."""
from collections import deque

from layer3.diagnostics.operator_parity import build_codegen_instance, StrategySpec

_SPEC = StrategySpec(
    strategy_id="rt_bcc", template_name="bear_call_credit_standard",
    entry_sexpr="GT(VIXChange, EphReal(0.0))",
    exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
    size_sexpr="Delta(ATM_IV_5m, EphInt(1))", is_credit=True)


def test_interday_buffers_initialized_before_use():
    """Every inter-day rolling buffer must be created in Initialize, BEFORE any
    read — lazy creation in one path missed the other and crashed at runtime."""
    _, code = build_codegen_instance(_SPEC, ("2025-01-02", "2025-01-31"), norm_serial={})
    for buf in ("_daily_closes", "_daily_ivs", "_daily_atm_iv"):
        init = f"self.{buf} = deque(maxlen=10)"
        assert init in code, f"{buf} not initialized in Initialize"
        # initialized before the first append/read
        assert code.index(init) < code.index(f"self.{buf}.append"), (
            f"{buf} read before it is initialized"
        )


def test_get_interday_runs_for_all_terminals():
    """_get_interday must execute (no AttributeError) for every inter-day terminal,
    with the buffers an Initialize-equivalent setup provides."""
    inst, _ = build_codegen_instance(_SPEC, ("2025-01-02", "2025-01-31"), norm_serial={})
    inst._daily_closes = deque([6000, 6010, 6020, 6030, 6040, 6050, 6060], maxlen=10)
    inst._daily_ivs = deque([16, 16.5, 17, 16.8, 17.2], maxlen=10)
    inst._daily_atm_iv = deque([0.13, 0.14, 0.135, 0.138, 0.142], maxlen=10)
    inst._last_valid_iv = 0.14
    vals = {n: inst._get_interday(n)
            for n in ("SPXReturn3d", "VIXMean5d", "RV5d", "IVRVGap5d")}
    assert all(isinstance(v, float) for v in vals.values())
    # T2: IVRVGap5d must use ATM_IV (~0.13), NOT VIX/100 (~0.17)
    assert vals["IVRVGap5d"] == (sum([0.13, 0.14, 0.135, 0.138, 0.142]) / 5
                                 - vals["RV5d"]), "IVRVGap5d must be mean(ATM_IV)-RV5d"
