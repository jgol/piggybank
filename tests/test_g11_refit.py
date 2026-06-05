"""Validates the G11 terminal-noise re-fit (scripts/g11_refit_from_qcdiag.py).

We synthesize a QCDIAG paste FROM the proxy's own 10:00-ET terminals (so the
"QC" side is the proxy plus a KNOWN per-terminal offset + small noise), then
assert the re-fit (a) aligns to the right proxy bar (recovered MAE≈injected
noise, not inflated by mis-alignment), (b) recovers the injected directional
offset, (c) EXCLUDES grid-IV pricing inputs, and (d) drops cold-start days for
multi-day interday terminals. This pins the alignment + math without needing a
live QC backtest.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore")

_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"
pytestmark = pytest.mark.skipif(not Path(_PARQUET).exists(),
                                reason="minute parquet not present")


def _synth_qcdiag(tmp_path, offsets, noise=0.0, seed=0):
    """Build a QCDIAG paste from the proxy 10:00-ET bars + known offsets/noise."""
    import scripts.g11_refit_from_qcdiag as rf
    sl, td = rf.proxy_terminals()
    dates = sl["date"].values
    mtc = sl["MinutesToClose"].values.astype(float)
    rng = np.random.RandomState(seed)
    # one row per distinct Jan trading day, at the bar closest to 10:00 (mtc=375)
    jan_days = [d for d in sorted(set(dates)) if "2025-01" in str(d)]
    lines, truth_terms = [], [c for c in td if td[c].ndim == 1 and not c.startswith("_")]
    for d in jan_days:
        idx = np.where(dates == d)[0]
        j = idx[np.argmin(np.abs(mtc[idx] - 375))]
        parts = [f"QCDIAG {d} 10:00"]
        for t in truth_terms:
            v = float(td[t][j]) + offsets.get(t, 0.0) + (rng.normal(0, noise) if noise else 0.0)
            parts.append(f"{t}={v:.5f}")
        lines.append(" ".join(parts))
    p = tmp_path / "qcdiag_paste.txt"
    p.write_text("\n".join(lines))
    return p, sl, td


def test_alignment_zero_divergence(tmp_path, capsys, monkeypatch):
    """QCDIAG == proxy (no offset/noise) -> recovered MAE≈0 for every terminal.
    A non-trivial MAE here would mean the proxy-bar alignment is wrong."""
    import scripts.g11_refit_from_qcdiag as rf
    paste, _, _ = _synth_qcdiag(tmp_path, offsets={}, noise=0.0)
    monkeypatch.setattr("sys.argv", ["x", str(paste)])
    rc = rf.main()
    assert rc == 0
    res = __import__("json").loads(
        Path("generated_qc/phase_c/g11_refit_result.json").read_text())
    # intraday terminals should align to MAE ~ 0 (float round only)
    for t in ("ATM_IV", "VIXChange", "MinutesToClose", "RawSpread"):
        if t in res["sigma"]:
            assert res["sigma"][t] < 0.02, f"{t} MAE {res['sigma'][t]} -> misalignment"


def test_recovers_injected_offset(tmp_path, monkeypatch):
    """A known +0.30 ATM_IV / -0.20 VIXChange bias is recovered in the offset dict."""
    import scripts.g11_refit_from_qcdiag as rf
    paste, _, _ = _synth_qcdiag(tmp_path, offsets={"ATM_IV": 0.30, "VIXChange": -0.20}, noise=0.0)
    monkeypatch.setattr("sys.argv", ["x", str(paste)])
    rf.main()
    res = __import__("json").loads(
        Path("generated_qc/phase_c/g11_refit_result.json").read_text())
    assert abs(res["offset"]["ATM_IV"] - 0.30) < 0.02
    assert abs(res["offset"]["VIXChange"] + 0.20) < 0.02
    # the offset's magnitude shows up as MAE too (|bias| dominates with no noise)
    assert abs(res["sigma"]["ATM_IV"] - 0.30) < 0.02


def test_grid_iv_excluded(tmp_path, monkeypatch):
    """Grid-IV pricing inputs never receive a sigma/offset (they are not GP terminals)."""
    import scripts.g11_refit_from_qcdiag as rf
    paste, _, _ = _synth_qcdiag(tmp_path, offsets={"IV_25Dp": 0.5, "IV_ATM": 0.5}, noise=0.0)
    monkeypatch.setattr("sys.argv", ["x", str(paste)])
    rf.main()
    res = __import__("json").loads(
        Path("generated_qc/phase_c/g11_refit_result.json").read_text())
    for g in ("IV_25Dp", "IV_ATM", "IV_5Dc"):
        assert g not in res["sigma"], f"{g} is a pricing input, must be excluded"
        assert g not in res["offset"]


def test_internal_and_unknown_tokens_skipped(tmp_path, monkeypatch):
    """Real QC logs carry an internal `_raw_RawSpread=...` token (and could carry
    other non-terminal keys). The refit must silently skip anything not in the
    proxy terminal_data, never crash or emit a sigma for it."""
    import scripts.g11_refit_from_qcdiag as rf
    paste, _, _ = _synth_qcdiag(tmp_path, offsets={}, noise=0.0)
    # append a leaked-internal token and a junk token to every line
    lines = [ln + " _raw_RawSpread=9.99999 BogusTerm=42.0" for ln in
             paste.read_text().splitlines()]
    paste.write_text("\n".join(lines))
    monkeypatch.setattr("sys.argv", ["x", str(paste)])
    assert rf.main() == 0
    res = __import__("json").loads(
        Path("generated_qc/phase_c/g11_refit_result.json").read_text())
    assert "_raw_RawSpread" not in res["sigma"] and "BogusTerm" not in res["sigma"]
    # real terminals still calibrated despite the junk tokens
    assert "ATM_IV" in res["sigma"]


def test_paste_order_independence(tmp_path, monkeypatch):
    """Warm-day gate must be robust to a REVERSED paste — sorting by date inside
    the refit makes the result identical regardless of copy/paste order."""
    import scripts.g11_refit_from_qcdiag as rf
    paste, _, _ = _synth_qcdiag(tmp_path, offsets={}, noise=0.0)
    fwd = paste.read_text().splitlines()
    monkeypatch.setattr("sys.argv", ["x", str(paste)])
    rf.main()
    res_fwd = __import__("json").loads(
        Path("generated_qc/phase_c/g11_refit_result.json").read_text())["report"]
    paste.write_text("\n".join(reversed(fwd)))   # reversed paste
    rf.main()
    res_rev = __import__("json").loads(
        Path("generated_qc/phase_c/g11_refit_result.json").read_text())["report"]
    nf = {r["terminal"]: r["n"] for r in res_fwd if r["mae"] is not None}
    nr = {r["terminal"]: r["n"] for r in res_rev if r["mae"] is not None}
    assert nf == nr, "warm-day sample counts must not depend on paste order"


def test_multiday_warm_day_gate(tmp_path, monkeypatch):
    """Multi-day terminals use fewer samples than intraday (cold-start days dropped)."""
    import scripts.g11_refit_from_qcdiag as rf
    paste, _, _ = _synth_qcdiag(tmp_path, offsets={}, noise=0.01)
    monkeypatch.setattr("sys.argv", ["x", str(paste)])
    rf.main()
    res = __import__("json").loads(
        Path("generated_qc/phase_c/g11_refit_result.json").read_text())
    by_term = {r["terminal"]: r for r in res["report"] if r["mae"] is not None}
    if "ATM_IV" in by_term and "VIXMean5d" in by_term:
        # VIXMean5d drops its leading cold days -> strictly fewer pairs than ATM_IV
        assert by_term["VIXMean5d"]["n"] == by_term["ATM_IV"]["n"] - rf._WARM_SKIP["VIXMean5d"]
    if "ATM_IV" in by_term and "RV5d" in by_term:
        # RV5d needs 7 closes -> drops 7 (more than VIXMean5d's 5)
        assert by_term["RV5d"]["n"] == by_term["ATM_IV"]["n"] - rf._WARM_SKIP["RV5d"]
