"""Codegen→QC transfer parity for this session's proxy-only changes (Phase B,
2026-06-01 discovery sweep). The generated QC algorithm must mirror the proxy's
B-2 (debit time-decay bars), B-3 (debit risk basis = max(gross, net)), and B3
(ruin halt) so champions transfer. We assert on the generated source (the QC
algorithm cannot be run without the QC cloud)."""
import re

import pytest

from layer3.codegen import generate_qc_algorithm, validate_generated_code


def _gen(template, delta=None):
    return generate_qc_algorithm(
        strategy_id="t", template_name=template,
        entry_sexpr="GT(ATM_IV, EphReal(0.5))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)", delta_sexpr=delta,
        start_date="2024-01-02", end_date="2024-02-01")


# B-2: debit time-decay gate — 240 for ratio/3-leg, 40 otherwise (matches proxy).
@pytest.mark.parametrize("template,delta,want", [
    ("ratio_put_backspread", "EphReal(0.5)", 240),  # 2-leg debit WITH a ratio leg
    ("bull_call_debit", None, 40),                   # 2-leg debit, no ratio
    ("bear_put_debit", None, 40),
])
def test_b2_debit_underwater_bars(template, delta, want):
    code = _gen(template, delta)
    m = re.search(r"_bars_in_trade >= (\d+)", code)
    assert m and int(m.group(1)) == want, f"{template}: want {want} underwater bars"


# B-3: debit max-loss risk basis = max(entry_gross, net debit).
def test_b3_debit_risk_basis_uses_gross():
    code = _gen("ratio_put_backspread", "EphReal(0.5)")
    assert "self._entry_gross = sum(abs(h.HoldingsValue)" in code, "entry_gross tracked"
    assert "_risk_basis = max(self._entry_gross, debit_paid)" in code, "risk basis = max(gross,net)"
    assert "-0.80 * _risk_basis" in code, "debit gate uses the risk basis, not net-only"


# B3: ruin halt — sticky _ruined flag set at total value ≤ 0, gates entries.
@pytest.mark.parametrize("template,delta", [
    ("iron_condor", None), ("bull_put_credit", None),
    ("ratio_put_backspread", "EphReal(0.5)"),
])
def test_b3_ruin_halt_present(template, delta):
    code = _gen(template, delta)
    assert "self._ruined = False" in code, "ruin flag initialized"
    assert "self.Portfolio.TotalPortfolioValue <= 0.0" in code, "ruin detected at blown-up account"
    assert "if self._ruined:" in code, "entry gate checks the ruin flag"


def test_all_templates_still_valid():
    for tpl, delta in [("iron_condor", None), ("iron_butterfly", None),
                       ("bull_put_credit", None), ("bear_call_credit", None),
                       ("bull_call_debit", None), ("bear_put_debit", None),
                       ("ratio_put_backspread", "EphReal(0.5)")]:
        ok, err = validate_generated_code(_gen(tpl, delta))
        assert ok, f"{tpl} codegen invalid: {err}"
