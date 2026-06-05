"""Layer 1 preprocess module.

Deterministic, parameter-free-except-for-train-stats transforms that sit
between raw 141-variate tensors from ObjectStore and the L1 encoder (SSL
iTransformer) + derived/ feature module. Produces the `preprocessor_hash`
component of the L1 provenance triple (encoder_hash, preprocessor_hash,
derivation_spec_hash).

Extracted from layer1/training/pipeline.py per D75 L1-split decision.
"""

from .transforms import (
    gamma_clamp,
    signed_log1p_inplace,
    vix_forwardfill_inplace,
    detect_options_cutoff,
    forwardfill_options_tail_inplace,
    compute_staleness_ramp,
)
from .zscore import ZScoreScaler
from .preprocessor import L1Preprocessor, PreprocessConfig

__all__ = [
    "gamma_clamp",
    "signed_log1p_inplace",
    "vix_forwardfill_inplace",
    "detect_options_cutoff",
    "forwardfill_options_tail_inplace",
    "compute_staleness_ramp",
    "ZScoreScaler",
    "L1Preprocessor",
    "PreprocessConfig",
]
