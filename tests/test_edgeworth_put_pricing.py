"""Edgeworth option pricing must correct the SHARED TIME VALUE only, never the
intrinsic — so put-call parity holds exactly and OTM put wings are not zeroed.

Regression for the 2026-06-01 audit bug: the prior `_edgeworth_option_value`
applied the multiplicative Edgeworth correction to the BS CALL's FULL price, then
derived the put via parity `max(corrected_call - (spot-strike), 0)`. A ~5%
correction on a deep-ITM-call / OTM-put's LARGE intrinsic swamped the put's SMALL
time value: it ZEROED deep OTM put wings (-100%) and DOUBLED near ones (+105%).
Since IC / BPC / RPB long legs ARE OTM puts, this badly mispriced their entry
credit and exit value (proxy credit nearly 2x too large), breaking proxy->QC
transfer. The fix corrects only the shared time value:
    call = max(S-K,0) + tv*correction ;  put = max(K-S,0) + tv*correction
which is parity-exact AND preserves each side's intrinsic.
"""
import math
import pytest

from layer2.evaluator_vectorized import _edgeworth_option_value, _option_value

SPOT = 5000.0
# Realistic SPX 0DTE skew/kurt so the correction is non-trivial (not ~1.0).
RHO, BETA = -0.07, 3.0


@pytest.mark.parametrize("iv,mtc", [(0.22, 90.0), (0.30, 60.0), (0.18, 200.0)])
def test_put_call_parity_holds_exactly(iv, mtc):
    """C - P == (S - K) for every strike, even with the Edgeworth correction on."""
    for k in [4800, 4900, 4950, 4975, 5000, 5025, 5050, 5100, 5200]:
        c = _edgeworth_option_value(SPOT, float(k), mtc, iv, True, RHO, BETA)
        p = _edgeworth_option_value(SPOT, float(k), mtc, iv, False, RHO, BETA)
        assert abs((c - p) - (SPOT - k)) < 1e-6, (
            f"parity broken at K={k}: C-P={c-p:.6f} vs S-K={SPOT-k:.1f}")


def test_otm_put_time_value_not_zeroed():
    """The headline bug: moderately-OTM puts (K<spot) must keep positive time value
    close to their BS value — NOT zeroed/halved by a correction on the mirror call's
    intrinsic. Allow the genuine Edgeworth correction (a few %), reject collapse."""
    iv, mtc = 0.22, 90.0
    for k in [4980, 4960, 4940, 4920]:           # 20-80 pt OTM puts
        p = _edgeworth_option_value(SPOT, float(k), mtc, iv, False, RHO, BETA)
        bs = _option_value(SPOT, float(k), mtc, iv, False)
        assert bs > 0.0, "test setup: OTM put has nonzero BS value"
        assert p > 0.5 * bs, (
            f"OTM put K={k} collapsed: edgeworth={p:.4f} vs BS={bs:.4f} "
            f"(<50% = the zeroing bug)")
        # And it must not be wildly inflated either (the +105% old-bug direction).
        assert p < 1.5 * bs, f"OTM put K={k} inflated: {p:.4f} vs BS {bs:.4f}"


def test_reduces_to_black_scholes_without_skew():
    """rho=beta=0 -> exactly Black-Scholes for both sides (the documented contract)."""
    iv, mtc = 0.20, 120.0
    for k in [4900, 4975, 5000, 5025, 5100]:
        for is_call in (True, False):
            e = _edgeworth_option_value(SPOT, float(k), mtc, iv, is_call, 0.0, 0.0)
            b = _option_value(SPOT, float(k), mtc, iv, is_call)
            assert abs(e - b) < 1e-9, f"K={k} call={is_call}: {e} != BS {b}"


def test_itm_call_preserves_intrinsic():
    """An ITM call must equal intrinsic + corrected_time_value, so its value never
    dips BELOW intrinsic (the old full-price correction could shave intrinsic)."""
    iv, mtc = 0.22, 90.0
    for k in [4900, 4950, 4975]:                 # ITM calls (K < spot)
        c = _edgeworth_option_value(SPOT, float(k), mtc, iv, True, RHO, BETA)
        intrinsic = SPOT - k
        assert c >= intrinsic - 1e-9, (
            f"ITM call K={k} below intrinsic: {c:.4f} < {intrinsic:.1f}")


def test_deep_otm_put_decays_to_zero():
    """Genuinely worthless deep-OTM puts still decay to ~0 (the fix doesn't floor
    them at a spurious positive value)."""
    p = _edgeworth_option_value(SPOT, 4700.0, 60.0, 0.20, False, RHO, BETA)
    assert 0.0 <= p < 0.05, f"deep-OTM put should be ~0, got {p:.4f}"
