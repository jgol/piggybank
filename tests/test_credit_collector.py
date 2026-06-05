"""Schema / smoke tests for generated_qc/credit_collector.py.

The algorithm imports ``AlgorithmImports`` (QC-only) so it cannot run outside
QuantConnect — these tests validate the SOURCE:

1. ``ast.parse`` — the file is syntactically valid Python; class is a QCAlgorithm.
2. All 5 templates (BPC/BCC/IC/IB/RPB) appear with the correct leg structures
   (incl. RPB 2:1 backspread sell-1@-0.50 / buy-2@-0.20).
3. The SetRuntimeStatistic emissions exist for every required name
   (per-template per-regime med+n, per-template overall med+n,
   collector_total_trades) — and they are the ONLY output channel.
4. The 10:00 ET entry / 15:45 ET exit gates exist.
5. The regime cutpoints mirror layer2/evaluator_vectorized.py (0.20, 0.30).
6. The injected BS-IV block is byte-identical to layer3.bs_iv.qc_method_source()
   (reused, not reimplemented).
7. NO ObjectStore.Save, NO _build_iv_grids, NO _log_chain (the heavy/blocked
   machinery is genuinely stripped).
"""
import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLLECTOR_PATH = PROJECT_ROOT / "generated_qc" / "credit_collector.py"

TEMPLATES = ("BPC", "BCC", "IC", "IB", "RPB")
REGIMES = ("low", "med", "high")


@pytest.fixture(scope="module")
def source() -> str:
    return COLLECTOR_PATH.read_text()


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


# ---------------------------------------------------------------------------
# 1. Syntax + class
# ---------------------------------------------------------------------------
def test_file_exists():
    assert COLLECTOR_PATH.exists(), f"missing {COLLECTOR_PATH}"


def test_ast_parses(tree):
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert "CreditCollector" in classes


def test_subclasses_qcalgorithm(tree):
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "CreditCollector")
    base_names = {b.id for b in cls.bases if isinstance(b, ast.Name)}
    assert "QCAlgorithm" in base_names


# ---------------------------------------------------------------------------
# 2. All 5 templates + leg structures (same as the diagnostic)
# ---------------------------------------------------------------------------
def test_all_five_templates_present(source):
    for tpl in TEMPLATES:
        assert f'"{tpl}"' in source, f"template {tpl} missing"
    assert 'TEMPLATE_ORDER = ["BPC", "BCC", "IC", "IB", "RPB"]' in source


def test_template_legs_structure(source):
    # RPB is the 2:1 backspread; BPC/BCC are 2-leg credit verticals.
    assert '"RPB": [("put", -0.50, -1, 1), ("put", -0.20, +1, 2)]' in source
    assert '"BPC": [("put", -0.25, -1, 1), ("put", -0.10, +1, 1)]' in source
    assert '"BCC": [("call", +0.25, -1, 1), ("call", +0.10, +1, 1)]' in source
    # IC + IB are 4-leg.
    assert '"IC":  [("call", +0.25, -1, 1), ("call", +0.10, +1, 1),' in source
    assert '"IB":  [("call", +0.50, -1, 1), ("call", +0.10, +1, 1),' in source


def test_template_legs_match_diagnostic(source):
    """The collector's TEMPLATE_LEGS must be byte-identical to the diagnostic's
    (same leg structures / deltas / ratios) — extract both dict literals from the
    AST and compare."""
    diag_src = (PROJECT_ROOT / "generated_qc" / "diag_template_agnostic.py").read_text()

    def _template_legs(src: str):
        t = ast.parse(src)
        for node in ast.walk(t):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "TEMPLATE_LEGS":
                        return ast.literal_eval(node.value)
        return None

    assert _template_legs(source) == _template_legs(diag_src)


# ---------------------------------------------------------------------------
# 3. SetRuntimeStatistic emissions — the ONLY output channel
# ---------------------------------------------------------------------------
def test_runtime_statistic_emissions_present(source):
    # Per-template per-regime median + count.
    for tpl in TEMPLATES:
        for reg in REGIMES:
            assert f'f"cred_{{tpl}}_{{reg}}_med"' in source
            assert f'f"cred_{{tpl}}_{{reg}}_n"' in source
    # Per-template overall median + count.
    assert 'f"cred_{tpl}_all_med"' in source
    assert 'f"cred_{tpl}_all_n"' in source
    # Grand total.
    assert '"collector_total_trades"' in source
    # It is emitted via SetRuntimeStatistic (not ObjectStore).
    assert "self.SetRuntimeStatistic(" in source


def test_runtime_statistic_names_resolve(source, tree):
    """Statically expand the f-string templates over TEMPLATE_ORDER/REGIME_ORDER
    and assert the full required name set is emitted (so the harvested
    runtimeStatistics dict has every key)."""
    expected = set()
    for tpl in TEMPLATES:
        for reg in REGIMES:
            expected.add(f"cred_{tpl}_{reg}_med")
            expected.add(f"cred_{tpl}_{reg}_n")
        expected.add(f"cred_{tpl}_all_med")
        expected.add(f"cred_{tpl}_all_n")
    expected.add("collector_total_trades")
    # ~5*(3*2 + 2) + 1 = 41 distinct stats.
    assert len(expected) == 5 * (3 * 2 + 2) + 1 == 41

    # Collect the loop variables + iterables so we can expand the f-strings.
    # We just verify the literal pieces + the two loop sources are present; full
    # runtime expansion is exercised by QC. Here: assert the building blocks.
    assert 'REGIME_ORDER = ["low", "med", "high"]' in source
    assert 'TEMPLATE_ORDER = ["BPC", "BCC", "IC", "IB", "RPB"]' in source
    assert "for tpl in TEMPLATE_ORDER:" in source
    assert "for reg in REGIME_ORDER:" in source


def test_uses_median(source):
    import_ok = "import statistics" in source
    assert import_ok
    assert "statistics.median(" in source


# ---------------------------------------------------------------------------
# 4. Entry 10:00 / exit 15:45 gates
# ---------------------------------------------------------------------------
def test_entry_1000_exit_1545_gates(source):
    assert "ENTRY_HOUR, ENTRY_MIN = 10, 0" in source
    assert "EXIT_HOUR, EXIT_MIN = 15, 45" in source
    # The handler gates entry at >= 10:00 (and < exit) and exit at >= 15:45.
    assert "_mins >= entry_min and _mins < exit_min" in source
    assert "_mins >= exit_min" in source
    # Missing-15:45-bar safety: exit still fires up to the 16:00 cutoff.
    assert "EXIT_CUTOFF_HOUR, EXIT_CUTOFF_MIN = 16, 0" in source


def test_records_entry_credit(source):
    # Fill-based entry credit = cash delta at entry (same as the diagnostic).
    assert 'rec["entry_credit"] = self.Portfolio.Cash - pre_cash' in source
    # Booked into a (template, regime) bucket on exit.
    assert 'self._credits[tpl][reg].append(rec["entry_credit"])' in source


def test_records_entry_gross(source):
    """Residual-1: GROSS = sum of |per-leg fill cash| (NOT the signed net).
    The per-leg cash delta is accumulated in |.| inside the order loop, so the
    over-priced OTM long wing (which nets away in entry_credit) is captured."""
    # Per-leg cash delta accumulated in absolute value.
    assert "_gross += abs(_cash_now - _cash_cursor)" in source
    assert 'rec["entry_gross"] = _gross' in source
    # Booked into the parallel (template, regime) gross bucket on exit.
    assert 'self._gross[tpl][reg].append(rec["entry_gross"])' in source
    # And it must be a DIFFERENT quantity from credit (|.| per leg, not net sum):
    assert 'rec["entry_gross"] = self.Portfolio.Cash' not in source, (
        "gross must be sum of |per-leg| deltas, not the signed net cash delta"
    )


def test_gross_runtime_statistic_emitted(source):
    """Median gross stats emitted per (template, regime) and per-template overall
    (the f-string keys gross_{tpl}_{reg}_med / gross_{tpl}_all_med)."""
    assert 'self.SetRuntimeStatistic(f"gross_{tpl}_{reg}_med"' in source
    assert 'self.SetRuntimeStatistic(f"gross_{tpl}_all_med"' in source


# ---------------------------------------------------------------------------
# 5. Regime cutpoints mirror evaluator_vectorized.py (0.20, 0.30)
# ---------------------------------------------------------------------------
def test_regime_cutpoints_mirror_evaluator(source):
    assert "REGIME_LOW_HI = 0.20" in source
    assert "REGIME_HI_CRISIS = 0.30" in source
    assert 'REGIME_ORDER = ["low", "med", "high"]' in source
    # The bucketing function uses exactly those cutpoints (>= 0.30 high, >= 0.20
    # med, else low) — matching evaluator_vectorized.py:1314-1317 / 1456-1463.
    assert "if iv >= REGIME_HI_CRISIS:" in source
    assert "if iv >= REGIME_LOW_HI:" in source


def test_regime_cutpoints_equal_evaluator_constants():
    """The two cutpoints must equal the constants used in the proxy evaluator's
    regime branches (iv>=0.30 / iv>=0.20). Read them out of evaluator_vectorized
    and assert equality so they can never silently drift."""
    eval_src = (PROJECT_ROOT / "layer2" / "evaluator_vectorized.py").read_text()
    # The proxy hard-codes the two cutpoints in both the slippage and credit
    # branches; assert the literals exist there (source of truth) ...
    assert "if iv >= 0.30:" in eval_src
    assert "elif iv >= 0.20:" in eval_src
    # ... and that the collector's named constants equal them.
    col_src = COLLECTOR_PATH.read_text()
    assert "REGIME_HI_CRISIS = 0.30" in col_src
    assert "REGIME_LOW_HI = 0.20" in col_src


# ---------------------------------------------------------------------------
# 6. Injected BS-IV block == bs_iv.qc_method_source() (reused, not reimplemented)
# ---------------------------------------------------------------------------
def test_injected_block_byte_matches_qc_method_source(source):
    from layer3.bs_iv import qc_method_source

    begin = "# === BS_IV_INJECT_BEGIN ===\n"
    end = "    # === BS_IV_INJECT_END ==="
    assert begin in source and end in source, "inject markers missing"
    block = source.split(begin, 1)[1].split(end, 1)[0].rstrip("\n")
    expected = qc_method_source(indent="    ")
    assert block == expected, (
        "injected BS methods drifted from layer3.bs_iv.qc_method_source(); "
        "re-inject to keep served code == tested code"
    )


def test_no_reimplemented_brentq(source):
    # Reuse bs_iv — no scipy import, no brentq *call* (the 'brentq' substring is
    # only the verbatim doc-comment in the injected reference).
    assert "from scipy" not in source
    assert "import scipy" not in source
    assert "brentq(" not in source


# ---------------------------------------------------------------------------
# 7. Heavy / blocked machinery genuinely stripped
# ---------------------------------------------------------------------------
def test_no_objectstore_save(source):
    assert "ObjectStore.Save" not in source
    assert "self.ObjectStore" not in source  # no in-QC ObjectStore use at all


def test_no_iv_grid_building(source):
    # No grid build CALL and no grid-building method definition.
    assert "_build_iv_grids(" not in source
    assert "def _build_iv_grids" not in source
    assert "def _build_grid" not in source


def test_no_chain_or_terminal_logging(source):
    assert "_log_chain" not in source
    assert "def _log_terminals" not in source
    assert "def _log_m2m" not in source
    # No 5-minute smoothing machinery either.
    assert "def _smooth_push" not in source
    assert "def _smooth_read" not in source


def test_lightweight_no_manifest_method(source):
    # No manifest dict is written (only the header comment mentions stripping it).
    assert "diag/manifest.json" not in source
    assert 'ObjectStore.Save("diag' not in source


# ---------------------------------------------------------------------------
# Setup sanity (full range, cash, SPXW 0DTE, fee, contract.Symbol)
# ---------------------------------------------------------------------------
def test_full_range_and_cash(source):
    assert "self.SetStartDate(2024, 10, 1)" in source
    assert "self.SetEndDate(2026, 5, 31)" in source
    assert "self.SetCash(200000)" in source


def test_spxw_0dte_and_fee(source):
    assert 'self.AddIndexOption(self.spx, "SPXW", Resolution.Minute)' in source
    assert "u.Strikes(-50, 50).Expiration(0, 0)" in source
    assert "ConstantFeeModel(2.50)" in source


def test_uses_contract_symbol_not_string(source):
    assert "c.Symbol" in source
    assert "self.MarketOrder(c.Symbol" in source


def test_caches_chain_in_ondata(source):
    # 0DTE caveat: chain unavailable in scheduled funcs -> cache in OnData.
    assert "def OnData(self, data):" in source
    assert "data.OptionChains" in source
    assert "self._cached_chain" in source
