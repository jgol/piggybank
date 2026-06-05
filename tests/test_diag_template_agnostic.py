"""Schema / smoke tests for generated_qc/diag_template_agnostic.py.

The algorithm imports ``AlgorithmImports`` (QC-only) so it cannot run outside
QuantConnect — these tests validate the SOURCE:

1. ``ast.parse`` — the file is syntactically valid Python.
2. The injected ``self._bs_iv`` / ``self._bs_delta`` methods are byte-identical
   to ``layer3.bs_iv.qc_method_source()`` (single source of truth; the served
   QC code cannot silently diverge from the unit-tested reference). The solver
   itself is covered by tests/test_codegen_bs_iv.py.
3. All 5 templates (BPC/BCC/IC/IB/RPB) appear.
4. The 3 IV-path column suffixes (_qcnative/_qccorrected/_bsself) appear on the
   6 IV-derived terminals (spec §3b).
5. The chain header has the 14 spec'd columns (spec §3a).
6. ObjectStore chunking (<=500 rows/key, distinct prefixes), OnEndOfAlgorithm
   flush + manifest, and the entry-10:00 / exit-15:45 gates exist (spec §4, §6).
"""
import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAG_PATH = PROJECT_ROOT / "generated_qc" / "diag_template_agnostic.py"


@pytest.fixture(scope="module")
def source() -> str:
    return DIAG_PATH.read_text()


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


# ---------------------------------------------------------------------------
# 1. Syntax
# ---------------------------------------------------------------------------
def test_file_exists():
    assert DIAG_PATH.exists(), f"missing {DIAG_PATH}"


def test_ast_parses(tree):
    # The fixture already parsed; assert the module has the algorithm class.
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert "DiagTemplateAgnostic" in classes


def test_subclasses_qcalgorithm(tree):
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "DiagTemplateAgnostic")
    base_names = {b.id for b in cls.bases if isinstance(b, ast.Name)}
    assert "QCAlgorithm" in base_names


# ---------------------------------------------------------------------------
# 2. Injected BS-IV solver matches bs_iv.qc_method_source() exactly
# ---------------------------------------------------------------------------
def test_injected_bs_methods_present(source):
    # The instance methods must be present (these are the fallback + the point).
    assert "self._bs_iv(" in source
    assert "self._bs_delta(" in source
    assert "def _bs_iv(self, mid, S, K, T, r, is_call):" in source
    assert "def _bs_delta(self, S, K, T, r, sigma, is_call):" in source
    assert "def _norm_cdf(self, x):" in source
    assert "def _bs_price(self, S, K, T, r, sigma, is_call):" in source


def test_injected_block_byte_matches_qc_method_source(source):
    """The block between the inject markers == bs_iv.qc_method_source() verbatim.
    This guarantees the served solver and the unit-tested reference are one."""
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
    """We must REUSE bs_iv, not write our own solver: no scipy import and no
    brentq *call*. (The string 'brentq' legitimately appears in the injected
    bs_iv reference as a comment documenting the matched bracket — that is the
    verbatim, unit-tested source we reuse, not a reimplementation.)"""
    assert "from scipy" not in source
    assert "import scipy" not in source
    assert "brentq(" not in source  # no brentq invocation, only the comment


# ---------------------------------------------------------------------------
# 3. All 5 templates appear
# ---------------------------------------------------------------------------
def test_all_five_templates_present(source):
    for tpl in ("BPC", "BCC", "IC", "IB", "RPB"):
        assert f'"{tpl}"' in source, f"template {tpl} missing"
    # And the canonical ordered list.
    assert 'TEMPLATE_ORDER = ["BPC", "BCC", "IC", "IB", "RPB"]' in source


def test_template_legs_structure(source):
    """RPB must be the 2:1 backspread (sell 1 @ -0.50, buy 2 @ -0.20)."""
    assert '"RPB": [("put", -0.50, -1, 1), ("put", -0.20, +1, 2)]' in source
    # IC + IB are 4-leg; BPC/BCC are 2-leg credit verticals.
    assert '"BPC": [("put", -0.25, -1, 1), ("put", -0.10, +1, 1)]' in source
    assert '"BCC": [("call", +0.25, -1, 1), ("call", +0.10, +1, 1)]' in source


# ---------------------------------------------------------------------------
# 4. Three IV-path column suffixes on the 6 IV-derived terminals
# ---------------------------------------------------------------------------
def test_three_iv_path_suffixes_present(source):
    for suffix in ("_qcnative", "_qccorrected", "_bsself"):
        assert suffix in source, f"IV-path suffix {suffix} missing"


def test_six_iv_derived_terminals_three_way(source):
    """Each of the 6 IV-derived terminals is logged 3 ways. Build the algorithm's
    terminal header the same way the class does and check every suffix column."""
    iv_derived = ["ATM_IV", "RawSpread", "PutCallSkew",
                  "GridReliability", "ATM_IV_5m", "RawSpread_5m"]
    # Confirm the class declares exactly these as IV-derived.
    assert ('_IV_DERIVED = ["ATM_IV", "RawSpread", "PutCallSkew",\n'
            '                   "GridReliability", "ATM_IV_5m", "RawSpread_5m"]') in source
    # The header builder emits qcnative/qccorrected/bsself per IV-derived name.
    for name in iv_derived:
        assert f'f"{{name}}_qcnative"' in source  # the builder line (once)
    # And the qccorrection constants are the stale single-month patch.
    assert "QC_CORR_SLOPE = 0.3869" in source
    assert "QC_CORR_INTERCEPT = 0.0292" in source


# ---------------------------------------------------------------------------
# 5. Chain header has the 14 spec'd columns
# ---------------------------------------------------------------------------
def test_chain_header_14_columns(source):
    expected_cols = [
        "ts", "symbol", "strike", "right", "expiry", "spot", "mtc",
        "bid", "ask", "mid", "qc_iv", "qc_delta", "bs_iv", "bs_delta",
    ]
    assert len(expected_cols) == 14
    # Extract the _CHAIN_HEADER literal from the parsed AST and check it.
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "DiagTemplateAgnostic")
    header_val = None
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_CHAIN_HEADER":
                    header_val = ast.literal_eval(node.value)
    assert header_val is not None, "_CHAIN_HEADER not found"
    assert header_val.split(",") == expected_cols


# ---------------------------------------------------------------------------
# 6. ObjectStore chunking + flush + manifest + entry/exit gates
# ---------------------------------------------------------------------------
def test_objectstore_chunking(source):
    assert "CHUNK_ROWS = 500" in source
    # Chunk guard fires at the cap on every stream.
    assert ">= self.CHUNK_ROWS" in source
    assert "self.ObjectStore.Save(" in source
    # Distinct prefixes per stream.
    for prefix in ('"diag/chain/"', '"diag/terminals/"',
                   '"diag/trades/"', '"diag/m2m/"'):
        assert prefix in source, f"prefix {prefix} missing"


def test_chunk_key_le_500_rows_per_file(source):
    """The flush guard must trigger at <=500 buffered rows (spec §6)."""
    # Each stream appends one row then checks the buffer length against the cap.
    assert "if len(self._chain_rows) >= self.CHUNK_ROWS:" in source
    assert "if len(self._terminal_rows) >= self.CHUNK_ROWS:" in source
    assert "if len(self._trade_rows) >= self.CHUNK_ROWS:" in source
    assert "if len(self._m2m_rows) >= self.CHUNK_ROWS:" in source


def test_on_end_of_algorithm_flush_and_manifest(source):
    assert "def OnEndOfAlgorithm(self):" in source
    # All four streams flushed at end.
    assert 'for stream in ("chain", "terminals", "trades", "m2m"):' in source
    assert 'self.ObjectStore.Save("diag/manifest.json"' in source


def test_entry_1000_exit_1545_gates(source):
    # Fixed 10:00 ET entry, 15:45 ET mechanical exit (spec §4).
    assert "ENTRY_HOUR, ENTRY_MIN = 10, 0" in source
    assert "EXIT_HOUR, EXIT_MIN = 15, 45" in source
    # The mechanical handler gates entry at >= 10:00 and exit at >= 15:45.
    assert "_mins >= entry_min" in source
    assert "_mins >= exit_min" in source


def test_rth_gate(source):
    # Per-bar handler gated to RTH 9:31 .. 15:45 ET (spec §3).
    assert "RTH_START_HOUR, RTH_START_MIN = 9, 31" in source
    assert "RTH_END_HOUR, RTH_END_MIN = 15, 45" in source


# ---------------------------------------------------------------------------
# Component-completeness sanity (chain snapshot, M2M path, full range)
# ---------------------------------------------------------------------------
def test_full_range_and_cash(source):
    assert "self.SetStartDate(2024, 10, 1)" in source
    assert "self.SetEndDate(2026, 5, 31)" in source
    assert "self.SetCash(200000)" in source


def test_spxw_0dte_and_cboe_vix(source):
    assert 'self.AddIndexOption(self.spx, "SPXW", Resolution.Minute)' in source
    assert "u.Strikes(-50, 50).Expiration(0, 0)" in source
    assert 'self.AddData(CBOE, "VIX", Resolution.Daily)' in source
    assert 'self.AddData(CBOE, "VIX9D", Resolution.Daily)' in source
    assert "ConstantFeeModel(2.50)" in source


def test_chain_snapshot_window_and_cadence(source):
    assert "CHAIN_EVERY = 15" in source           # per 15th bar
    assert "TERMINAL_EVERY = 5" in source          # per 5th bar
    assert "ATM_STRIKE_WINDOW = 12" in source      # ATM +/- 12 strikes
    assert "if self._bar_count % self.CHAIN_EVERY == 0:" in source
    assert "if self._bar_count % self.TERMINAL_EVERY == 0:" in source


def test_m2m_path_logging(source):
    # M2M every 10 bars + at max-adverse-excursion + at EOD (spec §4).
    assert "M2M_EVERY = 10" in source
    assert 'self._log_m2m(tpl, rec, "sample"' in source
    assert 'self._log_m2m(tpl, rec, "mae"' in source
    assert 'self._log_m2m(tpl, rec, "eod"' in source
    assert "max_adverse" in source


def test_per_leg_qc_and_bs_delta_logged(source):
    # Every selected leg logs both qc_delta and bs_delta (spec §4).
    assert '"qc_delta": qc_delta' in source
    assert '"bs_delta": bs_delta' in source


def test_uses_contract_symbol_not_string(source):
    # 0DTE caveat: always use contract.Symbol.
    assert "c.Symbol" in source
    assert "self.MarketOrder(c.Symbol" in source


# ---------------------------------------------------------------------------
# Issue-fix regression tests (4 confirmed fixes, 2026-05-29)
# ---------------------------------------------------------------------------
def _method_body(source: str, name: str) -> str:
    """Source text of one method body (via AST line span)."""
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines()
    return "\n".join(lines[fn.lineno - 1:fn.end_lineno])


def test_issue1_smoothing_advances_every_bar(source):
    """Issue 1: the 5-minute smoothing buffers must be PUSHED once per RTH bar
    (in _on_bar / _update_iv_buffers), and the 5th-bar terminal logger must only
    READ them — so the maxlen=5 deque spans 5 minutes, not 25. Guard against a
    regression to the old push-only-every-5th-bar behaviour."""
    # The push/read split exists (the old combined _smooth(...) is gone).
    assert "def _smooth_push(self, base, val):" in source
    assert "def _smooth_read(self, base):" in source
    # The per-bar handler builds grids + RV and pushes EVERY bar (before the
    # %TERMINAL_EVERY gate), and the logger receives the prebuilt grids.
    on_bar = _method_body(source, "_on_bar")
    assert "self._update_iv_buffers(qc_grid, bs_grid, rv30m)" in on_bar
    assert "self._log_terminals(spx_price, mtc, qc_grid, bs_grid, rv30m)" in on_bar
    # The push happens strictly before the every-5th-bar terminal gate.
    assert (on_bar.index("_update_iv_buffers")
            < on_bar.index("% self.TERMINAL_EVERY")), \
        "smoothing must advance before (and independently of) the 5th-bar gate"
    # _update_iv_buffers pushes all five 5-minute series exactly once.
    upd = _method_body(source, "_update_iv_buffers")
    for series in ("ATM_IV", "RawSpread", "ATM_IV_qc",
                   "RawSpread_qc", "RealizedVol30m"):
        assert upd.count(f'self._smooth_push("{series}"') == 1
    # The terminal logger must NOT push (read-only there).
    logt = _method_body(source, "_log_terminals")
    assert "_smooth_push" not in logt, "logger must read, not push (no double-count)"
    assert "_smooth_read(" in logt
    # maxlen=5 buffer == 5 one-per-minute samples == 5 minutes.
    assert "deque(maxlen=5)" in source


def test_issue2_exit_runs_when_1545_bar_missing(source):
    """Issue 2: a missing 15:45 bar must not leave positions open overnight. The
    mechanical EXIT must still run at the first bar at/after 15:45 (up to a
    16:00 cutoff), while new entries / logging stop at 15:45."""
    assert "EXIT_CUTOFF_HOUR, EXIT_CUTOFF_MIN = 16, 0" in source
    on_bar = _method_body(source, "_on_bar")
    # Past RTH end, exit-only handling still fires within the cutoff window.
    assert "if _mins > rth_end:" in on_bar
    assert "_mins <= exit_cutoff" in on_bar
    assert "self._mechanical_trades(spx_price, mtc)" in on_bar
    # Entry remains blocked after 15:45 (entry gate keeps the < exit_min bound).
    assert "_mins >= entry_min and _mins < exit_min" in source


def test_issue3_realized_pnl_is_fill_not_mark(source):
    """Issue 3: realized_pnl must be the fill-based booked PnL (cash delta across
    liquidation), and the old pre-liquidation mark is retained as eod_mark."""
    close = _method_body(source, "_close_mechanical")
    # Cash captured immediately before and after the liquidation loop.
    assert "cash_before = self.Portfolio.Cash" in close
    assert "cash_after = self.Portfolio.Cash" in close
    # realized = entry_credit + (cash_after - cash_before) — true booked PnL.
    assert "realized = rec[\"entry_credit\"] + (cash_after - cash_before)" in close
    # The old mark survives as eod_mark (offline: slippage = eod_mark - fill).
    assert "eod_mark = self._position_unrealized(rec)" in close
    # The trade header carries realized_pnl AND eod_mark (extract via AST).
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "DiagTemplateAgnostic")
    builder = next(n for n in cls.body
                   if isinstance(n, ast.FunctionDef) and n.name == "_build_trade_header")
    header = next(ast.literal_eval(node.value)
                  for node in ast.walk(builder)
                  if isinstance(node, ast.Return)
                  and isinstance(node.value, ast.Constant))
    cols = header.split(",")
    assert "realized_pnl" in cols and "eod_mark" in cols
    # eod_mark immediately follows realized_pnl (matches the row write order).
    assert cols.index("eod_mark") == cols.index("realized_pnl") + 1


def test_issue4_mae_row_logs_real_position_value(source):
    """Issue 4: the MAE M2M row must log the position value AT the adverse bar,
    not a 0.0 placeholder."""
    assert '"max_adverse_posval"' in source
    mech = _method_body(source, "_mechanical_trades")
    # Captured alongside max_adverse, and passed to the MAE row (not 0.0).
    assert 'rec["max_adverse_posval"] = pos_val' in mech
    assert 'self._log_m2m(tpl, rec, "mae",\n' in mech
    assert "rec[\"max_adverse_posval\"], rec[\"max_adverse\"]" in mech
    # The old hard-coded 0.0 placeholder for the MAE row is gone.
    assert 'self._log_m2m(tpl, rec, "mae", 0.0,' not in source
