"""Numerical-correctness tests for the scipy-free Black-Scholes IV/delta used to
keep the proxy (L1 collector) and QC-served codegen on ONE IV definition (P0-4).

Covers:
1. Round-trip: bs_iv(bs_price(sigma)) recovers sigma to <= 1e-3.
2. scipy parity: the scipy-free bisection agrees with scipy.optimize.brentq on the
   same price function and bracket (proving the QC-runtime solver matches what the
   collector's brentq would return).
3. delta: bs_delta matches N(d1) / N(d1)-1.
4. Transform equivalence: the methods emitted by qc_method_source() (injected into
   generated QC code) produce identical output to the importable reference
   functions, so served code and tested code cannot diverge.
5. Collector parity (opportunistic): if research_collector imports locally, the
   bisection matches its brentq _bs_iv.
"""
import math

import pytest

from layer3.bs_iv import bs_iv, bs_price, bs_delta, qc_method_source

# SPX-like grid, 0DTE-ish small T (years). r matches RISK_FREE_RATE=0.05.
_S = 5000.0
_R = 0.05
_KS = [4800.0, 4900.0, 4950.0, 5000.0, 5050.0, 5100.0, 5200.0]
_TS = [0.0007, 0.002, 0.005, 0.02]          # ~6.5h .. ~5 trading days
_SIGMAS = [0.05, 0.10, 0.15, 0.25, 0.40, 0.80]
_RIGHTS = [True, False]


def _grid():
    for K in _KS:
        for T in _TS:
            for sig in _SIGMAS:
                for is_call in _RIGHTS:
                    yield K, T, sig, is_call


def _well_posed(K, T, sig, is_call):
    """IV inversion is identifiable only where vega is non-negligible. Deep-ITM
    tiny-T options have d1 huge -> pdf(d1)~0 -> vega~0, so BS price is flat in sigma
    and IV is non-unique: brentq, this bisection, and the collector each land on a
    different valid root. Those points sit at |delta|~1, far from the 0.05-0.50 grid
    the spline samples, so they never reach the ATM/skew terminals. Gate on
    vega = S*pdf(d1)*sqrt(T) (price change per unit sigma); vega>1 => the root is
    well-separated and the inversion recovers sigma to well under 1e-3. Return
    (price, is_well_posed)."""
    price = bs_price(_S, K, T, _R, sig, is_call)
    sqrtT = math.sqrt(T)
    d1 = (math.log(_S / K) + (_R + 0.5 * sig * sig) * T) / (sig * sqrtT)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    vega = _S * pdf_d1 * sqrtT
    return price, (price > 0.05 and vega > 1.0)


def test_round_trip_recovers_sigma():
    """Inverting a BS price recovers the sigma that produced it (<= 1e-3)."""
    checked = 0
    for K, T, sig, is_call in _grid():
        price, ok = _well_posed(K, T, sig, is_call)
        if not ok:
            continue
        iv = bs_iv(price, _S, K, T, _R, is_call)
        assert iv is not None, f"no IV for K={K} T={T} sig={sig} call={is_call} price={price}"
        assert abs(iv - sig) < 1e-3, (
            f"round-trip off: K={K} T={T} sig={sig} call={is_call} -> iv={iv}"
        )
        checked += 1
    assert checked > 100, f"too few grid points exercised ({checked})"


def test_matches_scipy_brentq():
    """The scipy-free bisection agrees with scipy.optimize.brentq on the same
    price function + bracket (what research_collector._bs_iv uses)."""
    brentq = pytest.importorskip("scipy.optimize").brentq
    checked = 0
    for K, T, sig, is_call in _grid():
        price, ok = _well_posed(K, T, sig, is_call)
        if not ok:
            continue
        f = lambda s: bs_price(_S, K, T, _R, s, is_call) - price
        if f(0.01) > 0 or f(5.0) < 0:
            continue
        ref = brentq(f, 0.01, 5.0, xtol=1e-4, maxiter=50)
        ours = bs_iv(price, _S, K, T, _R, is_call)
        assert ours is not None
        assert abs(ours - ref) < 1e-3, (
            f"bisection vs brentq mismatch K={K} T={T} sig={sig}: {ours} vs {ref}"
        )
        checked += 1
    assert checked > 100


def test_bs_delta_matches_closed_form():
    """delta == N(d1) for calls, N(d1)-1 for puts."""
    ncdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    for K, T, sig, is_call in _grid():
        d1 = (math.log(_S / K) + (_R + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
        expected = ncdf(d1) if is_call else ncdf(d1) - 1.0
        got = bs_delta(_S, K, T, _R, sig, is_call)
        assert got is not None and abs(got - expected) < 1e-12


def test_bad_inputs_return_none():
    assert bs_iv(1.0, _S, 5000.0, 0.0, _R, True) is None      # T<=1e-6
    assert bs_iv(0.0, _S, 5000.0, 0.01, _R, True) is None     # mid<=0
    assert bs_iv(1e9, _S, 5000.0, 0.01, _R, True) is None     # above sigma=5 cap
    assert bs_delta(_S, 5000.0, 0.01, _R, 0.0, True) is None  # sigma<=0


def _exec_qc_methods():
    """Build a dummy class from qc_method_source() and instantiate it."""
    ns = {"math": math}
    exec("class _M:\n" + qc_method_source(indent="    "), ns)
    return ns["_M"]()


def test_qc_method_source_matches_reference():
    """Methods injected into generated QC code == importable reference functions
    (single source of truth; served code cannot silently diverge from tested code)."""
    m = _exec_qc_methods()
    for K, T, sig, is_call in _grid():
        price = bs_price(_S, K, T, _R, sig, is_call)
        assert m._bs_price(_S, K, T, _R, sig, is_call) == price
        assert m._bs_delta(_S, K, T, _R, sig, is_call) == bs_delta(_S, K, T, _R, sig, is_call)
        if price > 0.05:
            assert m._bs_iv(price, _S, K, T, _R, is_call) == bs_iv(price, _S, K, T, _R, is_call)


def test_collector_parity_if_importable():
    """If research_collector imports locally (scipy + no hard QC import), the
    bisection matches its brentq _bs_iv. Skipped if the module can't import."""
    try:
        from layer1.data.research_collector import _bs_iv as collector_bs_iv
    except Exception as e:  # QC-coupled imports may be unavailable locally
        pytest.skip(f"research_collector not importable locally: {e}")
    checked = 0
    for K, T, sig, is_call in _grid():
        price, ok = _well_posed(K, T, sig, is_call)
        if not ok:
            continue
        ref = collector_bs_iv(price, _S, K, T, _R, is_call)
        ours = bs_iv(price, _S, K, T, _R, is_call)
        if ref is None:
            continue
        assert ours is not None and abs(ours - ref) < 1e-3, (
            f"collector parity mismatch K={K} T={T} sig={sig}: {ours} vs {ref}"
        )
        checked += 1
    assert checked > 50
