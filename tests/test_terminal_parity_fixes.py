"""Proxy↔QC terminal parity fixes (2026-06-01). The G11 diagnostic exposed ~9
terminals that did not transfer to QC; these tests pin each fix so they cannot
silently regress. Two layers: (A) generated-QC source assertions, (B) regenerated
parquet property assertions.
"""
from pathlib import Path

import pytest

from layer3.codegen import generate_qc_algorithm, validate_generated_code

_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"


def _gen():
    return generate_qc_algorithm(
        strategy_id="p", template_name="iron_condor",
        entry_sexpr="GT(ATM_IV, EphReal(0.0))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.3)", start_date="2025-01-02", end_date="2025-01-31")


# ---- A. codegen source assertions -----------------------------------------

def test_codegen_session_open_captured_from_first_price():
    """OvernightGap/SessionReturn/SessionPosition anchor on the first fresh index
    Price, NOT the stale Securities.Open (which collapsed the gap to ~0)."""
    code = _gen()
    assert "_sec.Open" not in code, "stale Securities.Open must be gone"
    assert "if self._session_open_price is None and spx_price > 0:" in code, \
        "session-open must be captured from the first fresh Price"
    assert "self._overnight_gap = math.log(spx_price / _pc)" in code
    assert "self._prev_session_close = getattr(self, '_last_spx_price', None)" in code


def test_codegen_rv_session_reset_and_partial_window():
    """RV buffer clears at the day boundary (0DTE: no overnight carry) and returns
    a partial-window estimate during warmup (no 0.014 floor)."""
    code = _gen()
    assert "self._rv_log_hl_buf.clear()" in code, "RV buffer must session-reset"
    assert "return 0.014" not in code, "the 0.014 morning floor must be gone"
    assert "if not self._rv_log_hl_buf:" in code, "partial-window guard present"
    # scale collapsed from 60x to ~1.0 after the proxy double-sqrt removal
    assert "raw_rv * 60.0" not in code and "raw_rv * 1.0" in code


def test_codegen_still_valid_all_templates():
    for tpl in ("iron_condor", "iron_butterfly", "bull_put_credit", "bear_call_credit"):
        ok, err = validate_generated_code(generate_qc_algorithm(
            strategy_id="t", template_name=tpl,
            entry_sexpr="GT(ATM_IV, EphReal(0.0))",
            exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
            size_sexpr="EphReal(0.3)", start_date="2025-01-02", end_date="2025-01-31"))
        assert ok, f"{tpl}: {err}"


# ---- B. regenerated parquet property assertions ---------------------------

@pytest.mark.skipif(not Path(_PARQUET).exists(), reason="minute parquet absent")
def test_parquet_rv_no_morning_floor_and_continuous():
    import pandas as pd
    df = pd.read_parquet(_PARQUET)
    # warmup bars (5..28) must be ABOVE zero on most days (floor removed)
    warm = df[(df.bar_position >= 5) & (df.bar_position <= 28)]["RealizedVol30m"]
    assert (warm > 0).mean() > 0.95, "RealizedVol30m still floored in the morning"
    # continuity across the partial->full seam (bar 28 ~ bar 29)
    b28 = df[df.bar_position == 28]["RealizedVol30m"].median()
    b29 = df[df.bar_position == 29]["RealizedVol30m"].median()
    assert abs(b28 - b29) / max(b29, 1e-9) < 0.25, "discontinuity at the RV seam"
    # double-sqrt removed: single-Parkinson scale ~0.0003, not ~0.02
    assert df["RealizedVol30m"].median() < 0.002, "double-sqrt scale not removed"


@pytest.mark.skipif(not Path(_PARQUET).exists(), reason="minute parquet absent")
def test_parquet_spread_spikes_removed():
    import pandas as pd
    df = pd.read_parquet(_PARQUET)
    # the +5-clamp spikes (raw >0.035) are forward-filled away (rare first-bar
    # spikes may remain; require <0.1% of bars)
    assert (df["RawSpread"] > 0.035).mean() < 0.001, "spread spikes not cleaned"
    # DeltaSpread1 no longer carries the ±spike-pair pollution
    assert df["DeltaSpread1"].abs().max() < 0.5, "DeltaSpread1 still spike-polluted"


def test_norm_stats_updated_for_changed_terminals():
    from layer2.terminal_stats import TERMINAL_NORM_STATS
    # RealizedVol30m single-Parkinson scale (was 0.0152 double-sqrt)
    assert TERMINAL_NORM_STATS["RealizedVol30m"][0] < 0.001
    assert TERMINAL_NORM_STATS["RealizedVol30m_5m"][0] < 0.001
    # DeltaSpread scales tightened after spike-cleaning (were ~0.067/0.078)
    assert TERMINAL_NORM_STATS["DeltaSpread1"][1] < 0.02
    assert TERMINAL_NORM_STATS["DeltaSpread5"][1] < 0.02


# ---- C. G11 round-2: RV5d daily-close alignment ----------------------------
# Root cause: the proxy defined each day's "close" as the LAST bar (MTC=0,
# ~16:14); QC's market-hours guard freezes _last_spx_price at the 15:59 bar.
# Fix: BOTH sides anchor the daily close to the 16:00 bar (MinutesToClose==15).

def _synthetic_minute_df(day_closes_1600, late_tick_offset):
    """Build a minute DataFrame for len(day_closes_1600) days where the 16:00 bar
    (MTC==15) carries `day_closes_1600[d]` and a LATE post-16:00 tick (MTC<15)
    drifts the LAST bar away by `late_tick_offset`. This is exactly the case the
    old proxy mishandled (it read the last bar, not the 16:00 close)."""
    import numpy as np
    import pandas as pd
    rows = []
    for d, close1600 in enumerate(day_closes_1600):
        date = f"2025-03-{d + 3:02d}"
        # 30 intraday bars MTC 44..15 ramping to the 16:00 close, then a few
        # post-16:00 bars (MTC 14..0) that drift away (the late-tick artifact).
        for k, mtc in enumerate(range(44, 14, -1)):  # 44..15 (30 bars)
            frac = k / 29.0
            px = close1600 - (1.0 - frac) * 3.0  # ramps up to close1600 at MTC=15
            rows.append({"date": date, "bar_position": k,
                         "MinutesToClose": float(mtc), "SPXClose": px,
                         "ATM_IV": 0.13, "VIXSpot": 16.0})
        for j, mtc in enumerate(range(14, -1, -1)):  # 14..0 (15 post-close bars)
            rows.append({"date": date, "bar_position": 30 + j,
                         "MinutesToClose": float(mtc),
                         "SPXClose": close1600 + late_tick_offset,
                         "ATM_IV": 0.13, "VIXSpot": 16.0})
    return pd.DataFrame(rows)


def test_proxy_rv5d_uses_1600_close_not_last_bar():
    """The proxy's RV5d must be computed from the 16:00 bar (MTC==15), so a late
    post-16:00 tick on the LAST bar does NOT change RV5d. Pre-fix, the proxy read
    the last bar and the late tick leaked into the daily returns."""
    import numpy as np
    from layer2.evaluator_vectorized import prepare_terminal_data
    closes = [5800.0, 5815.0, 5790.0, 5825.0, 5810.0, 5840.0, 5830.0, 5855.0,
              5845.0, 5860.0]  # 10 days -> RV5d valid from day 6
    no_tick = prepare_terminal_data(_synthetic_minute_df(closes, 0.0),
                                    normalize_terminals=False)["RV5d"]
    with_tick = prepare_terminal_data(_synthetic_minute_df(closes, 12.0),
                                      normalize_terminals=False)["RV5d"]
    # Identical: the 16:00 close is unchanged, so the late tick is ignored.
    assert np.allclose(no_tick, with_tick), (
        "RV5d changed when only the post-16:00 last bar moved -> proxy is still "
        "reading the last bar instead of the 16:00 close")


def test_proxy_rv5d_matches_simulated_qc_1600_capture():
    """Proxy RV5d (from prepare_terminal_data) must EQUAL an independent RV5d built
    from a simulated-QC daily-close series that captures the 16:00 bar — proving
    the daily-close DEFINITION now agrees across the two sides."""
    import numpy as np
    import pandas as pd
    from layer2.evaluator_vectorized import prepare_terminal_data
    closes = [5800.0, 5815.0, 5790.0, 5825.0, 5810.0, 5840.0, 5830.0, 5855.0,
              5845.0, 5860.0]
    df = _synthetic_minute_df(closes, 9.0)  # late tick present (would diverge pre-fix)

    proxy_rv = prepare_terminal_data(df, normalize_terminals=False)["RV5d"]

    # Simulated QC: append the 16:00 (MTC==15) bar's SPXClose per day, then apply
    # the codegen _get_interday("RV5d") formula (ddof=0, last 5 of 6 returns).
    qc_daily = []
    for date in sorted(df["date"].unique()):
        g = df[df["date"] == date]
        bar1600 = g.iloc[int(np.argmin(np.abs(g["MinutesToClose"].values - 15.0)))]
        qc_daily.append(float(bar1600["SPXClose"]))

    def qc_rv5(closes_buf):
        # Mirrors codegen _get_interday("RV5d") AFTER the G11 fix: gate len>=6,
        # wrap-free window range(L-5, L) over closes[i]/closes[i-1], ddof=0.
        if len(closes_buf) < 6:
            return 0.15
        rets = [np.log(closes_buf[i] / max(closes_buf[i - 1], 1.0))
                for i in range(len(closes_buf) - 5, len(closes_buf))]
        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / len(rets)
        return float(np.sqrt(var) * np.sqrt(252))

    # Per day d, QC has closes for days 0..d-1 buffered at that day's open.
    df = df.reset_index(drop=True)
    for d, date in enumerate(sorted(df["date"].unique())):
        buf = qc_daily[:d]            # prior-day closes only (no look-ahead)
        qc_val = qc_rv5(buf)
        first_bar = int(df.index[df["date"] == date][0])
        proxy_val = float(proxy_rv[first_bar])
        assert abs(qc_val - proxy_val) < 1e-9, (
            f"day {date}: proxy RV5d {proxy_val} != simulated-QC {qc_val}")


def test_codegen_rv5d_gate_matches_proxy_first_eval_day():
    """The codegen RV5d gate must be `len(closes) >= 6`, NOT `>= 7`. With `>= 7`
    QC went live one day after the proxy (proxy fires at d>=6) and lagged its
    value by a full day. Verified behaviorally against the REAL _get_interday."""
    import math
    from collections import deque
    import numpy as np
    from layer3.diagnostics.operator_parity import build_codegen_instance, StrategySpec
    code = _gen()
    assert "if len(closes) >= 6:" in code, "RV5d gate must be >=6 (was off-by-one >=7)"
    assert "if len(closes) >= 7:" not in code, "stale >=7 RV5d gate must be gone"
    # wrap-free window: range(len-5, len), NOT the old range(len-6, len-1) which
    # hit closes[-1] (newest) at the len==6 boundary.
    assert "for i in range(len(closes)-5, len(closes))" in code, \
        "RV5d window must be the wrap-free range(len-5, len)"

    spec = StrategySpec(
        strategy_id="rv5", template_name="iron_condor",
        entry_sexpr="GT(RV5d, EphReal(0.0))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))", size_sexpr="EphReal(0.3)")
    inst, _ = build_codegen_instance(spec, ("2025-01-02", "2025-01-31"), norm_serial={})

    closes = [5800., 5815., 5790., 5825., 5810., 5840., 5830., 5855., 5845., 5860.]
    rets_full = np.diff(np.log(np.array(closes)))

    def proxy_rv5(d):  # evaluator_vectorized: std(_daily_rets[d-6:d-1])*sqrt(252)
        if d >= 6:
            return float(np.std(rets_full[d - 6:d - 1]) * math.sqrt(252))
        return 0.15

    # _get_interday reads _daily_ivs unconditionally; provide empty deques the
    # __new__-bypassed instance lacks (RV5d itself only consumes _daily_closes).
    inst._daily_ivs = deque(maxlen=10)
    inst._daily_atm_iv = deque(maxlen=10)
    # At day d's open the QC buffer holds closes[0..d-1] (= d entries). The first
    # real (non-default) eval must occur at d=6, matching the proxy — not d=7.
    for d in range(len(closes)):
        inst._daily_closes = deque(closes[:d], maxlen=10)
        assert abs(inst._get_interday("RV5d") - proxy_rv5(d)) < 1e-9, (
            f"day {d}: codegen RV5d lags the proxy (off-by-one gate)")


def test_codegen_rv5d_captures_1600_close():
    """The generated QC algorithm captures the 16:00 cash close (MTC==15) BEFORE
    the market-hours guard returns, and rolls THAT into the daily-close buffer
    (not the 15:59 _last_spx_price)."""
    code = _gen()
    assert "if t.hour == 16 and t.minute == 0:" in code, "16:00 capture missing"
    assert "self._close_1600_px = _close_px" in code, "16:00 price not stored"
    assert "_day_close_px = getattr(self, '_close_1600_px', 0.0)" in code, \
        "daily-close roll must prefer the 16:00 capture"
    # _close_1600_px is initialized in Initialize, before any read.
    assert "self._close_1600_px = 0.0" in code
    assert code.index("self._close_1600_px = 0.0") < \
        code.index("_day_close_px = getattr(self, '_close_1600_px', 0.0)")


# ---- D. G11 round-2: PutCallSkew 0.08-delta reliability gate ----------------
# Root cause: the collector zeroed grid points whose nearest observed delta was
# >0.08 from target, so the parquet recorded PutCallSkew=0 when a 25Δ leg was
# unreliable; QC extrapolated the spline regardless (+0.44 offset). Fix: gate
# the skew to 0.0 unless BOTH 25Δ legs have a live contract within 0.08 of 0.25.

def test_codegen_putcallskew_has_reliability_gate():
    code = _gen()
    assert "_put_25d_reliable = (len(_puts) >= 2 and" in code
    assert "_call_25d_reliable = (len(_calls) >= 2 and" in code
    assert "min(abs(0.25 - p[0]) for p in _puts) < 0.08" in code, \
        "put-25Δ reliability must mirror the collector's n<0.08 rule"
    assert "min(abs(0.25 - c[0]) for c in _calls) < 0.08" in code, \
        "call-25Δ reliability must mirror the collector's n<0.08 rule"
    assert "and _put_25d_reliable and _call_25d_reliable):" in code, \
        "skew assignment must be gated on BOTH legs being reliable"


def _extract_skew_gate(code):
    """Pull the exact emitted PutCallSkew gate block out of the generated source
    (zero re-implementation drift) so we can exec it against controlled fixtures.
    Spans the _put_25d_reliable line through the gated put_call_skew assignment."""
    import textwrap
    lines = code.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if "_put_25d_reliable = (len(_puts) >= 2 and" in ln)
    end = next(i for i, ln in enumerate(lines)
               if "put_call_skew = _put_25d_iv - _call_25d_iv" in ln)
    return textwrap.dedent("\n".join(lines[start:end + 1]))


def _run_skew_gate(puts, calls, put_25d_iv, call_25d_iv):
    block = _extract_skew_gate(_gen())
    ns = {"_puts": puts, "_calls": calls,
          "_put_25d_iv": put_25d_iv, "_call_25d_iv": call_25d_iv,
          "put_call_skew": 0.0, "abs": abs, "min": min, "len": len}
    exec(block, ns)
    return ns["put_call_skew"]


def test_putcallskew_reliable_both_legs_emits_skew():
    """Both 25Δ legs have a live contract within 0.08 of 0.25 -> skew is emitted."""
    puts = [(0.10, 0.20, 0.01), (0.26, 0.18, 0.01), (0.45, 0.15, 0.01)]   # nearest |0.25-0.26|=0.01
    calls = [(0.12, 0.14, 0.01), (0.24, 0.13, 0.01), (0.50, 0.12, 0.01)]  # nearest |0.25-0.24|=0.01
    skew = _run_skew_gate(puts, calls, put_25d_iv=0.18, call_25d_iv=0.13)
    assert abs(skew - (0.18 - 0.13)) < 1e-12, "reliable legs must emit the real skew"


def test_putcallskew_unreliable_put_leg_yields_zero():
    """No live put contract within 0.08 of 0.25 (nearest is 0.10, dist 0.15) ->
    skew gated to 0.0, MATCHING the parquet the collector zeroed. QC would
    otherwise extrapolate a non-zero put wing (the +0.44 offset)."""
    puts = [(0.05, 0.30, 0.01), (0.10, 0.25, 0.01), (0.55, 0.15, 0.01)]   # nearest |0.25-0.10|=0.15 > 0.08
    calls = [(0.12, 0.14, 0.01), (0.24, 0.13, 0.01), (0.50, 0.12, 0.01)]  # call leg reliable
    skew = _run_skew_gate(puts, calls, put_25d_iv=0.26, call_25d_iv=0.13)
    assert skew == 0.0, "unreliable put-25Δ leg must zero the skew (parquet parity)"


def test_putcallskew_unreliable_call_leg_yields_zero():
    """Symmetric: an unreliable call-25Δ leg also zeroes the skew."""
    puts = [(0.10, 0.20, 0.01), (0.26, 0.18, 0.01), (0.45, 0.15, 0.01)]   # put leg reliable
    calls = [(0.05, 0.16, 0.01), (0.10, 0.13, 0.01), (0.55, 0.10, 0.01)]  # nearest |0.25-0.10|=0.15 > 0.08
    skew = _run_skew_gate(puts, calls, put_25d_iv=0.18, call_25d_iv=0.12)
    assert skew == 0.0, "unreliable call-25Δ leg must zero the skew (parquet parity)"


# ---- E. G11 round-3: PutCallSkew code-path is CONVENTION-IDENTICAL ----------
# The G11 diagnostic showed PutCallSkew has a +0.46-normalized mean proxy↔QC gap
# (QC 25Δ skew > parquet) at 10:00 over 20 Jan-2025 days. Investigation
# (2026-06-01) traced this to live-chain-vs-collected-surface PROVENANCE, NOT a
# convention bug: ATM_IV (the same spline machinery sampled at delta 0.50)
# transfers at corr 0.997 with ~0 offset, the per-day skew-gap std (0.91 norm) is
# ~2x its mean (0.46 norm), and the gap is two-signed (15/20 QC>proxy, 5/20
# proxy>QC). A deterministic convention bug (cf. the RV double-sqrt) would offset
# ATM_IV too and have near-zero residual variance. These tests PIN the conclusion:
# given IDENTICAL contract mids, the collector's BS+spline path (the parquet) and
# codegen's BS+spline path (QC serve) must produce the SAME 25Δ IVs and skew. Any
# future convention regression — a different r, a put-delta sign flip, a T or
# spline-boundary-condition change on ONE side — breaks this and is caught here.
# (If this ever fails, the gap has a fixable convention root cause; do NOT paper
# over it with a hardcoded skew offset — that is a prohibited calibration crutch.)

import ast as _ast
import math as _math
import textwrap as _textwrap

import numpy as _np
import pytest as _pytest


def _collector_bs_funcs():
    """Import _bs_price/_bs_iv/_bs_greeks from research_collector WITHOUT executing
    its module-level QuantBook() (QC-only). Extract the three pure functions by AST
    and exec them in an isolated namespace."""
    src = Path("layer1/data/research_collector.py").read_text()
    tree = _ast.parse(src)
    norm = _pytest.importorskip("scipy.stats").norm
    brentq = _pytest.importorskip("scipy.optimize").brentq
    ns = {"math": _math, "norm": norm, "brentq": brentq, "np": _np}
    wanted = {"_bs_price", "_bs_iv", "_bs_greeks"}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name in wanted:
            exec(_ast.get_source_segment(src, node), ns)  # noqa: S102 — trusted repo source
    return ns["_bs_price"], ns["_bs_iv"], ns["_bs_greeks"]


def _codegen_cubic_spline():
    """Extract codegen's emitted natural-cubic-spline _cubic_spline_eval (it lives
    inside the QC template STRING, so it is not an importable symbol) and exec it as
    a standalone function — byte-identical to what QC runs."""
    code = _gen()
    lines = code.split("\n")
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip().startswith("def _cubic_spline_eval("))
    # body runs until the next sibling @staticmethod / def at the same indent
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent and \
           (ln.lstrip().startswith("def ") or ln.lstrip().startswith("@")):
            end = i
            break
        end = i + 1
    block = _textwrap.dedent("\n".join(lines[start:end]))
    ns = {"math": _math}
    exec(block, ns)  # noqa: S102 — extracted from in-repo generated source
    return ns["_cubic_spline_eval"]


def _skew_both_paths(strikes, S=6000.0, r=0.05, mins_left=375):
    """Build the SAME observed option mids on a chain, then compute IV(25Δp)−IV(25Δc)
    via (a) the collector path and (b) the codegen path. Returns (coll_skew, qc_skew,
    coll_put, coll_call, qc_put, qc_call)."""
    import layer3.bs_iv as qc
    from scipy.interpolate import CubicSpline
    c_bs_price, c_bs_iv, c_bs_greeks = _collector_bs_funcs()
    qc_cubic = _codegen_cubic_spline()

    # collector T-convention: mins_left over a 6.5h session; codegen: over 390min.
    # 6.5*60 == 390, so these are identical — asserted explicitly below.
    T_coll = max(1e-6, (0 + mins_left / (6.5 * 60)) / 365.0)
    T_qc = max(1e-6, (0 + mins_left / 390.0) / 365.0)
    assert T_coll == T_qc, "collector/codegen time-to-expiry conventions must match"

    def true_iv(K):
        m = _math.log(K / S)
        return max(0.03, 0.12 - 0.9 * m + 2.5 * m * m)  # negative-skew smile

    contracts = []
    for K in strikes:
        for is_call in (False, True):
            mid = c_bs_price(S, float(K), T_qc, r, true_iv(K), is_call)
            if mid > 0.05:
                contracts.append((float(K), is_call, mid))

    # collector path: _bs_iv + _bs_greeks delta; puts keyed on abs(delta); scipy not-a-knot
    cP, cC = [], []
    for K, is_call, mid in contracts:
        iv = c_bs_iv(mid, S, K, T_coll, r, is_call)
        if iv is None:
            continue
        d = c_bs_greeks(S, K, T_coll, r, iv, is_call)[0]
        if d is None:
            continue
        (cC if is_call else cP).append((abs(d), iv))

    def coll_at(pts):
        de = _np.array([p[0] for p in pts]); iv = _np.array([p[1] for p in pts])
        si = _np.argsort(de); de, iv = de[si], iv[si]
        u = _np.concatenate([[True], _np.diff(de) > 1e-6]); de, iv = de[u], iv[u]
        return float(_np.clip(CubicSpline(de, iv, extrapolate=True)(0.25), 0, 5))

    # codegen path: bs_iv.py _bs_iv + _bs_delta; puts keyed on abs(delta); natural spline
    qP, qC = [], []
    for K, is_call, mid in contracts:
        iv = qc._bs_iv(mid, S, K, T_qc, r, is_call)
        if iv is None or iv <= 0.001:
            continue
        d = qc._bs_delta(S, K, T_qc, r, iv, is_call)
        if d is None or abs(d) <= 0.001:
            continue
        (qC if is_call else qP).append((abs(d), iv))

    def qc_at(pts):
        pts = sorted(pts, key=lambda x: x[0])
        out = [pts[0]]
        for i in range(1, len(pts)):
            if abs(pts[i][0] - pts[i - 1][0]) > 1e-6:
                out.append(pts[i])
        return qc_cubic([p[0] for p in out], [p[1] for p in out], 0.25)

    cput, ccall = coll_at(cP), coll_at(cC)
    qput, qcall = qc_at(qP), qc_at(qC)
    return cput - ccall, qput - qcall, cput, ccall, qput, qcall


def test_putcallskew_codepaths_identical_dense_chain():
    """Dense symmetric SPX 0DTE chain (5-pt strikes, ±8%): collector and codegen
    must agree on 25Δp, 25Δc, AND the skew to < 1e-3 vol-points. This is the proof
    that the measured +0.46-norm production gap is INPUT provenance, not the math."""
    strikes = _np.arange(5520, 6485, 5)
    cskew, qskew, cput, ccall, qput, qcall = _skew_both_paths(strikes)
    assert abs(qput - cput) < 1e-3, f"25Δ put IV diverged: coll={cput} qc={qput}"
    assert abs(qcall - ccall) < 1e-3, f"25Δ call IV diverged: coll={ccall} qc={qcall}"
    assert abs(qskew - cskew) < 1e-3, (
        f"skew diverged with identical inputs: coll={cskew} qc={qskew} "
        "(a >1e-3 gap means a CONVENTION bug, not provenance — fix the math)")
    # And the skew is the right SIGN (negative-skew smile -> puts richer)
    assert cskew > 0 and qskew > 0, "negative-skew smile must yield put-over-call skew"


def test_putcallskew_codepaths_identical_sparse_chain():
    """Sparser 15-pt grid (fewer wing knots): still convention-identical. Guards the
    boundary-condition difference (collector scipy not-a-knot vs codegen natural
    cubic) from biasing the 25Δ point as coverage thins."""
    strikes = _np.arange(5550, 6505, 15)
    cskew, qskew, *_ = _skew_both_paths(strikes)
    assert abs(qskew - cskew) < 2e-3, (
        f"sparse-chain skew diverged: coll={cskew} qc={qskew} — spline boundary "
        "conditions must not bias the 25Δ wing")


# ---- D. QC-non-transferable scalar removal (final calibration) -------------
# After rounds 1-2 fixed every bug, 8 raw scalars still don't transfer proxy->QC
# (corr<0.5 or wing-provenance scatter). They are excluded from the GP SAMPLING
# pool but kept PARSEABLE. This pins both invariants so they can't silently drift.

def test_nontransferable_scalars_excluded_from_sampling_but_parseable():
    from layer2.grammar import (Grammar, from_sexpr, build_scalar_only_terminal_set,
                                build_probes_only_terminal_set, build_terminal_set,
                                _QC_NONTRANSFERABLE_SCALARS)
    expected = {"SessionReturn", "OvernightGap", "RawSpread", "RawSpread_5m",
                "DeltaSpread1", "DeltaSpread5", "PutCallSkew", "SessionPosition"}
    assert _QC_NONTRANSFERABLE_SCALARS == expected, "removed-set drifted"
    # (1) none leak into ANY condition's sampling pool
    for ts in (None, build_scalar_only_terminal_set(), build_probes_only_terminal_set()):
        pool = {t.name for t in Grammar(terminals=ts).terminals}
        assert not (pool & _QC_NONTRANSFERABLE_SCALARS), "removed terminal leaked into sampling"
    # (2) the parser still accepts all of them (historical strategies must parse)
    for nm in _QC_NONTRANSFERABLE_SCALARS:
        assert from_sexpr(f"GT({nm}, EphReal(0.0))") is not None
    # (3) build_terminal_set (the parser source) is left intact — still lists them
    assert _QC_NONTRANSFERABLE_SCALARS <= {t.name for t in build_terminal_set()}


# ---- BLOCKER-14: scalar-only SEED path strips non-transferable scalars ------ #

def test_scalar_only_seed_substitution_strips_nontransferable_leaves():
    """Every Level-B template's scalar-only-substituted entry+exit+size+delta seed
    must contain ZERO QC-non-transferable terminals. Before the fix, the seed path
    passed plain scalars through unchanged, so SessionReturn (in all 5 Level-B
    seeds) leaked into the seeded ~15% of the population."""
    from layer2.grammar import (
        substitute_stripped_terminals_for_scalar_only as _sub,
        _leaf_terminal_names, _QC_NONTRANSFERABLE_SCALARS,
    )
    from layer2.templates import BASE_TEMPLATE_FACTORIES
    templates = [f() for f in BASE_TEMPLATE_FACTORIES]
    assert len(templates) == 5, "expected 5 Level-B base templates"
    for tpl in templates:
        for slot in ("entry_seed", "exit_seed", "size_seed", "delta_seed"):
            seed = getattr(tpl, slot, None)
            if seed is None:
                continue
            substituted = _sub(seed)
            leaks = _leaf_terminal_names(substituted) & _QC_NONTRANSFERABLE_SCALARS
            assert leaks == set(), (
                f"{tpl.name}.{slot} substituted seed still references "
                f"non-transferable terminals: {leaks}")


def test_scalar_only_seed_at_least_one_template_actually_substituted():
    """Guard against a vacuous pass: confirm the RAW Level-B seeds DO contain a
    non-transferable terminal (so the fix is load-bearing, not testing nothing)."""
    from layer2.grammar import _leaf_terminal_names, _QC_NONTRANSFERABLE_SCALARS
    from layer2.templates import BASE_TEMPLATE_FACTORIES
    raw_leaks = set()
    for f in BASE_TEMPLATE_FACTORIES:
        tpl = f()
        for slot in ("entry_seed", "exit_seed", "size_seed", "delta_seed"):
            seed = getattr(tpl, slot, None)
            if seed is not None:
                raw_leaks |= (_leaf_terminal_names(seed) & _QC_NONTRANSFERABLE_SCALARS)
    assert "SessionReturn" in raw_leaks, (
        "expected raw Level-B seeds to gate on SessionReturn (non-transferable) — "
        "if this fails the substitution test above is vacuous")


def test_scalar_only_substitution_targets_are_in_sampling_pool():
    """Each substitution TARGET must be a TRANSFERABLE terminal that is actually in
    the scalar-only SAMPLING pool (so the replacement is one the GP could itself
    have generated, and is itself transferable)."""
    from layer2.grammar import (
        _SCALAR_ONLY_NONTRANSFERABLE_SUBSTITUTIONS as _SUBS,
        _SCALAR_ONLY_TERM_SUBSTITUTIONS as _PROBE_SUBS,
        _QC_NONTRANSFERABLE_SCALARS, Grammar, build_scalar_only_terminal_set,
    )
    # Every non-transferable scalar has a mapping
    assert set(_SUBS) == set(_QC_NONTRANSFERABLE_SCALARS), \
        "every non-transferable scalar must have a seed substitution target"
    pool = {t.name for t in Grammar(terminals=build_scalar_only_terminal_set()).terminals}
    for src, tgt in _SUBS.items():
        assert tgt in pool, f"target {tgt!r} for {src!r} not in scalar-only sampling pool"
        assert tgt not in _QC_NONTRANSFERABLE_SCALARS, \
            f"target {tgt!r} is itself non-transferable"
    # The probe substitution PredSpread must NOT route to a non-transferable target
    for src, tgt in _PROBE_SUBS.items():
        assert tgt not in _QC_NONTRANSFERABLE_SCALARS, \
            f"probe sub {src!r}->{tgt!r} re-introduces a non-transferable leaf"


def test_scalar_only_substitution_type_safe():
    """Substitution is REAL→REAL (type-safe) — a replaced leaf has the same GType as
    the original, so the seed tree stays type-valid."""
    from layer2.grammar import (
        substitute_stripped_terminals_for_scalar_only as _sub, from_sexpr, GType,
    )
    # SessionReturn is REAL; its substitute must be REAL too.
    out = _sub(from_sexpr("GT(SessionReturn, EphReal(0.0))"))
    # walk to the left child leaf
    left = out.children[0]
    assert left.defn.ret_type == GType.REAL, "substituted leaf must remain REAL-typed"
    assert left.defn.name != "SessionReturn", "SessionReturn must have been replaced"
