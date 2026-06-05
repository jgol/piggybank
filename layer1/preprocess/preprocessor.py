"""L1Preprocessor: the top-level deterministic preprocessing object.

Wraps the stateless per-day transforms + fitted ZScoreScaler into a
single object with two public entry points:

  - fit(days)      — compute train-set statistics from raw day tensors
  - transform(X)   — apply the full pipeline to a single day's raw tensor

and one provenance entry point:

  - preprocessor_hash() — stable hash that enters the L1->L2 manifest
    as part of the (encoder_hash, preprocessor_hash, derivation_spec_hash)
    triple. Any rotation in clip bounds, staleness semantics, skewed
    log1p list, or train-set stats produces a new hash.

Implementation status (D75 scaffold)
------------------------------------
The class interface is final. The fit/transform BODY still delegates to
the legacy pipeline.preprocess_day() / apply_vix_forwardfill() code
until the TZ audit (diagnostic_tz_audit.py) resolves whether staleness
and forward-fill tails are real structural features or phantom
artifacts. Do not call L1Preprocessor.fit() in production training runs
until that resolution is captured in PreprocessConfig.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Tuple

import numpy as np

from .transforms import (
    TOTAL_VARIATES,
    gamma_clamp,
    signed_log1p_inplace,
    vix_forwardfill_inplace,
    detect_options_cutoff,
    forwardfill_options_tail_inplace,
    compute_staleness_ramp,
    liveness_mask,
    SIGNED_LOG1P_INDICES,
    STALENESS_INDEX,
)
from .zscore import ZScoreScaler


@dataclass
class PreprocessConfig:
    """Configuration knobs for L1Preprocessor.

    These get hashed into `preprocessor_hash`. Any field rotation means
    a new hash and therefore a new L1 provenance entry.
    """

    total_variates: int = TOTAL_VARIATES
    clip_lo: float = -5.0
    clip_hi: float = 5.0
    std_floor: float = 1e-6

    # Whether to apply the tail forward-fill + staleness ramp. Flipped
    # to False if confirms the cutoff is phantom.
    enable_tail_forwardfill: bool = True
    enable_staleness_ramp: bool = True

    # Which raw-141 indices to actually keep in the preprocessor output.
    # None means "all 141"; in practice the training pipeline subsets to
    # the 37-feature set (to be replaced with the ~85 AI-Engineer set
    # from after re-collection).
    feature_indices: Optional[Tuple[int, ...]] = None

    # Spec version label, bumped on any semantic change to preprocessing.
    # This is separate from the hash — the hash moves with every config
    # value, this label moves only on intentional design updates.
    spec_version: str = "2026-04-10-d75-scaffold"

    def to_json(self) -> str:
        payload = asdict(self)
        if self.feature_indices is not None:
            payload["feature_indices"] = list(self.feature_indices)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class L1Preprocessor:
    """Deterministic preprocessing for L1 raw tensors.

    Lifecycle:
        pre = L1Preprocessor(config)
        pre.fit(iter_train_days(qb, "2022-09-19", "2024-09-30"))
        X_day = pre.transform(raw_day_tensor)
        h = pre.preprocessor_hash()
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.scaler = ZScoreScaler(
            clip_lo=self.config.clip_lo,
            clip_hi=self.config.clip_hi,
            std_floor=self.config.std_floor,
        )
        self._fitted = False

    # ----- public API -----

    def _day_deterministic(self, X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all stateless transforms to a single day's raw tensor.

        Returns (X_processed, liveness) — both (n_bars, TOTAL_VARIATES)
        float32.
        """
        X = X_raw.astype(np.float32).copy()
        if X.shape[1] < self.config.total_variates:
            pad_width = self.config.total_variates - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

        gamma_clamp(X)
        signed_log1p_inplace(X)

        cutoff = detect_options_cutoff(X)

        if self.config.enable_tail_forwardfill:
            forwardfill_options_tail_inplace(X, cutoff)

        if self.config.enable_staleness_ramp:
            X[:, STALENESS_INDEX] = compute_staleness_ramp(X.shape[0], cutoff)
        else:
            X[:, STALENESS_INDEX] = 0.0  # neutralized

        vix_forwardfill_inplace(X)

        live = liveness_mask(X.shape[0], cutoff)
        return X, live

    def _subset_features(self, X: np.ndarray) -> np.ndarray:
        if self.config.feature_indices is None:
            return X
        return X[:, list(self.config.feature_indices)]

    def fit(self, day_tensors: Iterable[np.ndarray]) -> "L1Preprocessor":
        """Fit the z-score scaler from an iterable of raw day tensors.

        Each `day_tensor` must be (n_bars, TOTAL_VARIATES) float32 as
        returned by the collector. All deterministic per-day transforms
        are applied before collecting statistics so the scaler sees the
        same scale that transform() will produce.
        """
        chunks = []
        for X_raw in day_tensors:
            X_det, _ = self._day_deterministic(X_raw)
            X_sub = self._subset_features(X_det)
            chunks.append(X_sub)
        if not chunks:
            raise ValueError("L1Preprocessor.fit() received zero day tensors")
        train_mat = np.concatenate(chunks, axis=0)
        self.scaler.fit(
            train_mat,
            feature_indices=list(self.config.feature_indices)
            if self.config.feature_indices is not None
            else None,
        )
        self._fitted = True
        return self

    def transform(self, X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply deterministic transforms + z-score + clip to one day.

        Returns (X_normalized, liveness). Shape of X_normalized is
        (n_bars, n_selected_features) matching config.feature_indices.
        """
        if not self._fitted:
            raise RuntimeError("L1Preprocessor.transform() called before fit()")
        X_det, live = self._day_deterministic(X_raw)
        X_sub = self._subset_features(X_det)
        X_z = self.scaler.transform(X_sub)
        return X_z, live

    def preprocessor_hash(self) -> str:
        """Return the stable 16-hex preprocessor_hash.

        Combines config.to_json() with the fitted scaler.state_hash().
        If you change ANY config knob or refit on different data, this
        hash moves and downstream L2 runs need to re-import L1 outputs.
        """
        if not self._fitted:
            raise RuntimeError("preprocessor_hash() requires a fitted scaler")
        payload = {
            "config": json.loads(self.config.to_json()),
            "scaler_hash": self.scaler.state_hash(),
        }
        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload_str.encode()).hexdigest()[:16]
