"""Global z-score scaler with clip [-5, +5] for L1-preprocess.

Fits on training-window bars only (walk-forward-safe) and transforms
any array to the same scale. Mirrors the statistics currently computed
inline in pipeline.make_split_and_loaders() so we can reuse them
without touching the training script.

Hash identity
-------------
The ZScoreScaler contributes to `preprocessor_hash` via its
`state_hash()` method. Any change to fitted mean/std produces a new
hash, which invalidates downstream L2 runs. This is intentional — the
encoder and GP strategies must never compare embeddings across
incompatible normalization frames.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ZScoreScaler:
    """Per-variate mean/std with clip bounds.

    Attributes:
        mean: (n_features,) float64 — training-set per-variate mean
        std:  (n_features,) float64 — training-set per-variate std (lower-floored)
        clip_lo: float — lower clip bound after z-scoring
        clip_hi: float — upper clip bound after z-scoring
        std_floor: float — minimum std value to avoid division blow-up
        n_samples: int — number of training bars used to fit
    """

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    clip_lo: float = -5.0
    clip_hi: float = 5.0
    std_floor: float = 1e-6
    n_samples: int = 0
    feature_indices: Optional[list] = None  # Which raw-141 indices were fit
    _fitted: bool = field(default=False, repr=False)

    def fit(self, train_bars: np.ndarray, feature_indices: Optional[list] = None) -> "ZScoreScaler":
        """Fit per-variate mean/std from training-window bars.

        Args:
            train_bars: (N, F) float32 — flattened training bars (each row is
                one bar's feature vector, already subset to the training split)
            feature_indices: optional list of raw-141 indices that these F
                columns correspond to. Stored for downstream hash + audit.
        """
        if train_bars.ndim != 2:
            raise ValueError(f"train_bars must be 2D (N, F), got shape {train_bars.shape}")

        self.mean = train_bars.mean(axis=0).astype(np.float64)
        self.std = train_bars.std(axis=0).astype(np.float64)
        self.std = np.maximum(self.std, self.std_floor)
        self.n_samples = int(train_bars.shape[0])
        self.feature_indices = list(feature_indices) if feature_indices is not None else None
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Z-score then clip. Returns a new array, does not modify input."""
        if not self._fitted or self.mean is None or self.std is None:
            raise RuntimeError("ZScoreScaler must be fit() before transform()")
        z = (X.astype(np.float32) - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.clip(z, self.clip_lo, self.clip_hi)

    def fit_transform(self, X: np.ndarray, feature_indices: Optional[list] = None) -> np.ndarray:
        self.fit(X, feature_indices=feature_indices)
        return self.transform(X)

    def state_dict(self) -> dict:
        """Serializable state for checkpointing / hashing."""
        if not self._fitted:
            raise RuntimeError("Cannot export state_dict before fit()")
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "clip_lo": self.clip_lo,
            "clip_hi": self.clip_hi,
            "std_floor": self.std_floor,
            "n_samples": self.n_samples,
            "feature_indices": self.feature_indices,
        }

    def load_state_dict(self, state: dict) -> "ZScoreScaler":
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.std = np.array(state["std"], dtype=np.float64)
        self.clip_lo = float(state["clip_lo"])
        self.clip_hi = float(state["clip_hi"])
        self.std_floor = float(state["std_floor"])
        self.n_samples = int(state["n_samples"])
        self.feature_indices = state.get("feature_indices")
        self._fitted = True
        return self

    def state_hash(self) -> str:
        """Stable 16-hex hash of the fitted statistics.

        Used as part of preprocessor_hash. Changes if any fitted stat,
        clip bound, or feature index list changes.
        """
        if not self._fitted:
            raise RuntimeError("Cannot hash before fit()")
        payload = json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
