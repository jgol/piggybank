"""Tests for layer2.terminal_stats normalization constants.

Focus: ThetaUrgency clamp-saturation fix (2026-06-01). Raw ThetaUrgency =
1/sqrt(max(MinutesToClose, 1)) is a one-sided heavy UPPER tail. The prior
robust stat (median-center, IQR/1.35 scale) mapped the close region to
z >> +5, so the evaluator's OOD clamp [-5, +5] saturated >5% of all bars and
100% of bars with MTC < 10 — erasing end-of-day discrimination. These tests
pin the re-derived (mean-center, max->+4.5) standard stat so the regression
cannot return:

  * clamp rate (|z| >= 5) is < 1% of bars on the train split (target was
    < 0.5%; we got 0.0%);
  * the final-10-minute gradient is preserved (each minute maps to a distinct,
    strictly-increasing, EphReal-reachable normalized value).
"""
import numpy as np
import pandas as pd
import pytest

from layer2.terminal_stats import (
    TERMINAL_NORM_STATS, normalize,
)

# Same train-split convention used to derive the 2026-06-01 minute stats.
_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"
_TRAIN_END = "2024-09-27"
_CLAMP = 5.0  # evaluator OOD clamp is [-5, +5]


def _theta_urgency_train():
    """Raw ThetaUrgency over the train split (date <= 2024-09-27)."""
    df = pd.read_parquet(_PARQUET, columns=["date", "MinutesToClose"])
    train = df[df["date"] <= _TRAIN_END]
    mtc = train["MinutesToClose"].values.astype(np.float64)
    mtc = mtc[np.isfinite(mtc)]
    return 1.0 / np.sqrt(np.maximum(mtc, 1.0))


def test_theta_urgency_is_standard_method():
    """The evaluator applies a purely affine (raw-center)/scale and ignores
    the method field, so the stat must be 'standard' (not 'robust') — the
    fix lives entirely in center/scale, but we pin the method to document it."""
    center, scale, method = TERMINAL_NORM_STATS["ThetaUrgency"]
    assert method == "standard"
    assert scale > 0
    # Mean-center, scale so raw max (1.0 at MTC=1) maps to +4.5.
    # 2026-06-02: re-derived after the H3 bar-0 MTC repair removed the 882 bar-0
    # ThetaUrgency=1.0 artifacts from the raw distribution (mean 0.100529→0.098183).
    assert abs(center - 0.098183) < 1e-4
    assert abs(scale - 0.200404) < 1e-4


@pytest.mark.skipif(
    not __import__("os").path.exists(_PARQUET),
    reason="minute scalar parquet not present",
)
def test_theta_urgency_clamp_rate_below_one_percent():
    """REGRESSION PIN: < 1% of train bars saturate the [-5,+5] OOD clamp.

    Before the fix this was 5.185% (100% of MTC<10 bars). After: 0.0%.
    """
    tu = _theta_urgency_train()
    z = np.array([normalize("ThetaUrgency", v) for v in tu])

    frac_clamped = float(np.mean(np.abs(z) >= _CLAMP))
    assert frac_clamped < 0.01, (
        f"ThetaUrgency clamp rate {frac_clamped:.4%} exceeds 1% — the "
        f"close-region tail is saturating again (target < 0.5%)."
    )
    # Tighter target the fix actually achieves.
    assert frac_clamped < 0.005

    # The whole normalized range must sit strictly inside the clamp (no bar
    # touches +/-5) on the train split.
    assert z.max() < _CLAMP, f"max normalized ThetaUrgency {z.max():.3f} >= +5"
    assert z.min() > -_CLAMP, f"min normalized ThetaUrgency {z.min():.3f} <= -5"


def test_theta_urgency_final_minutes_gradient_preserved():
    """Each of the final 10 minutes maps to a DISTINCT, strictly-increasing,
    EphReal-reachable normalized value — the property the prior stat destroyed
    (all close-region bars pinned to +5)."""
    minutes = list(range(1, 12))  # MTC = 1..11
    zs = [normalize("ThetaUrgency", 1.0 / np.sqrt(max(m, 1.0))) for m in minutes]

    # Strictly decreasing in MTC (i.e. urgency rises as the close approaches).
    for a, b in zip(zs, zs[1:]):
        assert a > b, f"gradient not monotone: {a} !> {b}"

    # All distinct (no collapse onto the clamp).
    assert len(set(round(z, 6) for z in zs)) == len(zs)

    # Every value within the clamp range AND inside the EphReal-reachable band
    # well enough that a GT(ThetaUrgency, EphReal) threshold can separate them.
    assert all(abs(z) < _CLAMP for z in zs)
    # The within-final-10 spread is large (MTC=1 vs MTC=10) so thresholds bite.
    assert zs[0] - zs[9] > 1.0, "final-10-minute spread collapsed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
