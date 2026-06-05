"""Scipy-free Black-Scholes IV / delta — single source of truth for proxy<->QC parity.

The L1 collector self-computes implied volatility by inverting the Black-Scholes
mid-price with ``scipy.optimize.brentq`` (``research_collector._bs_iv``,
RISK_FREE_RATE=0.05, expiry 16:15 ET) and then computes greeks from that IV
(``_bs_greeks``: delta = N(d1) for calls, N(d1)-1 for puts). The L1 Parquet that
the GP proxy trains/selects on therefore carries *this* IV definition, and the
frozen terminal normalization stats are computed from it.

QuantConnect's native ``OptionContract.ImpliedVolatility`` uses a different solver
and conventions and runs ~1.9x higher on 0DTE SPX (empirically ~0.28 vs the
collector's ~0.10-0.15 on a VIX~16 day). Feeding QC-native IV into the spline at
serve time while the GP was evolved on collector-IV is *training-serving skew*
(Sculley et al., 2015, "Hidden Technical Debt in Machine Learning Systems",
NeurIPS) — the documented cause of "good offline metrics, bad online results."
A single empirically-fit linear correction (0.3869*x + 0.0292, calibrated on one
month) does not generalize across regimes.

This module is the fix: the generated QC algorithm self-computes IV + delta with
the SAME logic as the collector, so train and serve share one IV definition and
no correction is needed.

Design:
- ``_BS_SRC`` is the single textual source of the math. It is exec'd here to
  define the importable reference functions (``bs_price``/``bs_iv``/``bs_delta``/
  ``norm_cdf``) used by tests, AND transformed by :func:`qc_method_source` into
  instance methods (``self._bs_*``) injected verbatim into the generated QC class.
  One source => the served code and the tested code cannot diverge.
- QuantConnect's Python runtime does not guarantee scipy, so root finding uses a
  bisection (Black-Scholes price is monotone increasing in sigma; vega>0). The
  bracket [0.01, 5.0] and xtol 1e-4 match ``research_collector._bs_iv`` exactly,
  and the bisection is unit-tested to agree with ``scipy.optimize.brentq`` and to
  round-trip a known sigma to <= 1e-3.

References:
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate
  Liabilities. Journal of Political Economy, 81(3), 637-654.
- Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems.
  Advances in Neural Information Processing Systems, 28.
"""
from __future__ import annotations

import math
import re

# The single source of truth for the BS math. Bare module-style functions that
# call each other by name; :func:`qc_method_source` rewrites them to ``self._*``
# instance methods for the generated QCAlgorithm. Uses only ``math`` (no numpy /
# no scipy) so it runs unchanged in QuantConnect's runtime.
_BS_SRC = r'''
def _norm_cdf(x):
    # Standard normal CDF via erf (matches scipy.stats.norm.cdf to ~1e-15).
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

def _bs_price(S, K, T, r, sigma, is_call):
    # Black-Scholes (1973) European price, no dividend (matches collector _bs_price).
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def _bs_delta(S, K, T, r, sigma, is_call):
    # delta = N(d1) (call) / N(d1)-1 (put), matching collector _bs_greeks.
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return None
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0

def _bs_iv(mid, S, K, T, r, is_call):
    # Invert _bs_price for sigma. Scipy-free bisection on [0.01, 5.0] (price is
    # monotone increasing in sigma); bracket + tol match research_collector._bs_iv
    # (brentq, 0.01, 5.0, xtol=1e-4). Returns None when out of bracket / bad input.
    # At vega~0 (deep ITM, tiny T) price is flat in sigma so the root is not
    # unique; callers sample the spline only at |delta|<=0.50, away from that
    # region, and the L1 collector shares this property -> train/serve stay
    # consistent there even though a single root is not identifiable.
    if T <= 1e-6 or mid <= 0.0 or S <= 0.0 or K <= 0.0:
        return None
    lo = 0.01
    hi = 5.0
    f_lo = _bs_price(S, K, T, r, lo, is_call) - mid
    f_hi = _bs_price(S, K, T, r, hi, is_call) - mid
    if f_lo > 0.0 or f_hi < 0.0:
        return None  # mid not bracketed by sigma in [0.01, 5.0]
    for _ in range(100):
        m = 0.5 * (lo + hi)
        f_m = _bs_price(S, K, T, r, m, is_call) - mid
        if abs(f_m) < 1e-6 or (hi - lo) < 1e-4:
            return m if 0.01 <= m <= 5.0 else None
        if f_m > 0.0:
            hi = m
        else:
            lo = m
    iv = 0.5 * (lo + hi)
    return iv if 0.01 <= iv <= 5.0 else None
'''

# Define the importable reference functions (used by tests and any local tooling)
# directly in this module's namespace so they resolve `math` and each other.
exec(_BS_SRC, globals())  # noqa: S102 — trusted in-repo source, not user input

_FN_NAMES = ("_norm_cdf", "_bs_price", "_bs_delta", "_bs_iv")
# Public aliases without the underscore for ergonomic imports in tests.
norm_cdf = _norm_cdf          # type: ignore[name-defined] # noqa: F821
bs_price = _bs_price          # type: ignore[name-defined] # noqa: F821
bs_delta = _bs_delta          # type: ignore[name-defined] # noqa: F821
bs_iv = _bs_iv                # type: ignore[name-defined] # noqa: F821

_DEF_RE = re.compile(r"^def (_\w+)\((.*)\):$")


def qc_method_source(indent: str = "    ") -> str:
    """Return ``_BS_SRC`` rewritten as QCAlgorithm instance methods.

    Each ``def _fn(args):`` becomes ``def _fn(self, args):`` and each inter-function
    call ``_fn(`` becomes ``self._fn(`` so the methods resolve via the instance.
    Lines are prefixed with ``indent`` (class-body indentation). The generated
    methods are byte-faithful to the reference functions exec'd above.
    """
    out = []
    for line in _BS_SRC.strip("\n").split("\n"):
        m = _DEF_RE.match(line)
        if m:
            name, args = m.group(1), m.group(2)
            line = f"def {name}(self, {args}):" if args else f"def {name}(self):"
        else:
            for fn in _FN_NAMES:
                # Only rewrite genuine calls, not attribute access (self._fn) or
                # substrings of longer identifiers.
                line = re.sub(rf"(?<![\w.]){fn}\(", f"self.{fn}(", line)
        out.append(indent + line if line else line)
    return "\n".join(out)
