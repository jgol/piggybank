"""G11 terminal-noise directional-bias mechanism (MH3, 2026-06-01 holistic review).

The proxy↔QC terminal gap is NOT zero-mean; a systematic offset crosses thresholds
that symmetric jitter cannot. _add_terminal_noise now applies a per-terminal mean
offset (_TERMINAL_NOISE_OFFSET) before the symmetric sigma noise. The dict is EMPTY
until recalibrated against a fresh QC trace, so today it reduces to the prior
zero-mean behavior.
"""
import numpy as np

import layer2.evaluator_vectorized as ev


def test_offset_dict_empty_by_default():
    """Until recalibrated, the offset table is empty (zero-mean noise preserved)."""
    assert ev._TERMINAL_NOISE_OFFSET == {}
    assert ev._DEFAULT_NOISE_OFFSET == 0.0


def test_zero_noise_scale_is_identity():
    data = {"ATM_IV": np.zeros(1000)}
    out = ev._add_terminal_noise(data, np.random.RandomState(0), noise_scale=0.0)
    assert out is data  # early return, untouched


def test_offset_shifts_terminal_mean():
    """A nonzero offset shifts the terminal's mean by offset*noise_scale (the
    directional bias), distinct from the zero-mean sigma noise."""
    saved = dict(ev._TERMINAL_NOISE_OFFSET)
    saved_sigma = dict(ev._TERMINAL_NOISE_SIGMA)
    try:
        ev._TERMINAL_NOISE_OFFSET["ATM_IV"] = 0.5      # inject a directional bias
        ev._TERMINAL_NOISE_SIGMA["ATM_IV"] = 0.10      # plus some symmetric noise
        data = {"ATM_IV": np.zeros(100000, dtype=np.float64)}
        out = ev._add_terminal_noise(data, np.random.RandomState(7), noise_scale=1.0)
        # mean should be ≈ the offset (0.5), not 0 — the bias is modeled
        assert abs(float(out["ATM_IV"].mean()) - 0.5) < 0.01, \
            "offset must shift the terminal mean (directional bias)"
        # and the spread should reflect the sigma (×_MAE_TO_SIGMA)
        assert float(out["ATM_IV"].std()) > 0.05, "symmetric sigma noise still applied"
    finally:
        ev._TERMINAL_NOISE_OFFSET.clear(); ev._TERMINAL_NOISE_OFFSET.update(saved)
        ev._TERMINAL_NOISE_SIGMA.clear(); ev._TERMINAL_NOISE_SIGMA.update(saved_sigma)


def test_offset_only_no_sigma_still_shifts():
    """A terminal with an offset but zero sigma is shifted by the offset (no noise)."""
    saved = dict(ev._TERMINAL_NOISE_OFFSET)
    try:
        ev._TERMINAL_NOISE_OFFSET["SPXClose"] = 0.3   # SPXClose has sigma 0
        data = {"SPXClose": np.zeros(1000, dtype=np.float64)}
        out = ev._add_terminal_noise(data, np.random.RandomState(1), noise_scale=1.0)
        assert np.allclose(out["SPXClose"], 0.3), "offset-only terminal shifts by offset"
    finally:
        ev._TERMINAL_NOISE_OFFSET.clear(); ev._TERMINAL_NOISE_OFFSET.update(saved)
