"""Layer 1 derived/ module.

Hand-curated, scale-stable, semantically-named scalar features that GP
trees in L2 can compose arithmetically. Sits alongside the L1 encoder
but is NOT produced by it — derived/ is pure deterministic feature
engineering versioned by `derivation_spec_hash` (the third entry in
the L1->L2 provenance triple).

Why derived/ exists (from spec §11.6):
    The L1 encoder emits a 64-d opaque embedding. GP trees cannot do
    arithmetic on opaque dims without starving on interpretability. The
    derived/ features give GP a ~30-60 scalar substrate to compose
    (ratios, products, lags, thresholds) while the encoder provides the
    high-capacity nonlinear representation on the side. Without derived/,
    H3 Ablation 3 ("L1-preprocess + derived/ + GP + LLM", no encoder)
    would be starved and would lose to the full stack for structural
    reasons rather than because the encoder adds real value.

Versioning:
    expert_v1 — initial ~40 feature list, 2026-04-10 scaffold
    expert_v2 — reserved for post-ablation refinement

See spec §11.8.3 for the rationale behind each feature category.
"""

from .expert_v1 import (
    DERIVED_FEATURE_REGISTRY,
    compute_derived_features,
    derivation_spec_hash,
)

__all__ = [
    "DERIVED_FEATURE_REGISTRY",
    "compute_derived_features",
    "derivation_spec_hash",
]
