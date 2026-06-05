"""L1 Batch Inference: frozen encoder -> L1Output for GP consumption.

D84 typed-vector contract. The PCA-compressed embedding (EmbPC{0..19}) has
been retired — GP now consumes 7 typed vector objects + scalars. Flat PCA
over 3072-d emb_fine destroyed the cross-variate structure the encoder
learned; typed vectors preserve it.

Usage:
    forecaster = L1BatchForecaster(checkpoint_path)
    forecaster.fit_probes(train_windows, train_targets)
    outputs = forecaster.encode_dataset(X_windows, W_windows, window_meta)
    forecaster.to_parquet(outputs, "l1_output.parquet")
"""
from __future__ import annotations
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPX_DAILY_OHLC_CSV = _REPO_ROOT / "raw_data" / "spx_daily_ohlc.csv"


def _load_spx_daily_opens(csv_path: Path = _SPX_DAILY_OHLC_CSV) -> Dict[str, float]:
    """Load date -> open-price map from the QC-collected daily OHLC CSV.

    The v3 corpus replaced raw SPX close with `session_log_return` (D61).
    The GP backtester needs an actual dollar SPX level to set option strikes
    and compute option values. We reconstruct:

        spx_price_at_bar = open_today * exp(session_log_return_at_bar)

    because `session_log_return` is defined as log(price / open_today). The
    identity is exact. This helper returns the daily opens keyed by YYYY-MM-DD.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SPX daily OHLC not found at {csv_path}. Run "
            f"scripts/ingest_spx_daily_ohlc.py first (paste lives at "
            f"raw_data/spx_daily_ohlc_paste.txt)."
        )
    df = pd.read_csv(csv_path)
    return dict(zip(df["date"].astype(str), df["open"].astype(float)))


def _load_checkpoint_flexible(path_or_dict, map_location):
    """Load a checkpoint that may be either a raw torch file or a
    UTF-8/base64-wrapped torch payload (QC ObjectStore format, also used by
    the local shim at `raw_data/local_store/ssl_model_v3`). Caller passes a
    path, a bytes/str payload, or an already-loaded dict.
    """
    if isinstance(path_or_dict, dict):
        return path_or_dict
    if isinstance(path_or_dict, (str, Path)):
        path = Path(path_or_dict)
        # Try raw torch.load first (files written by plain `torch.save`).
        try:
            return torch.load(path, map_location=map_location,
                              weights_only=False)
        except Exception:
            # Fall back to base64-wrapped format: file is UTF-8 text whose
            # contents are base64-encoded bytes of a torch archive.
            raw = path.read_text(encoding="utf-8")
            buf = io.BytesIO(base64.b64decode(raw))
            return torch.load(buf, map_location=map_location,
                              weights_only=False)
    raise TypeError(
        f"checkpoint must be path, dict, or pre-loaded; got {type(path_or_dict)}"
    )

# ---------------------------------------------------------------------------
# L1Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class L1Output:
    """One row per window — everything the GP needs for fitness evaluation.

    Channel 1 (D84 typed vectors, each (D*n_layers,) float32 — 384-d on
    the default multi-layer iTransformer):
      emb_shared    — global mean over all variates
      emb_grid      — options moneyness grid (raw variates 0-87)
      emb_strike    — strike-level aggregates (99-104)
      emb_spx       — SPX-derived features (105-118)
      emb_vix       — VIX term structure (119-130)
      emb_flow_raw  — raw order flow (131-139)
      emb_flow_agg  — multi-scale flow aggregates (v3 only, 141-158);
                      zero vector on v2 corpus

    Channel 2 (probe forecasts, scalars): pred_rv_15, pred_rv_30,
      pred_regime (4-d probabilities), predicted_regime (argmax), pred_spread.

    Channel 3 (raw bypass scalars): raw_spread, delta_spread_1, delta_spread_5,
      atm_iv, minutes_to_close, vix_spot, vix_term_slope, spx_close, realized_vol_30m.
    """
    date: str
    window_idx: int
    bar_position: int
    # Channel 1: typed vectors (replaces retired emb_pca)
    emb_shared: np.ndarray
    emb_grid: np.ndarray
    emb_strike: np.ndarray
    emb_spx: np.ndarray
    emb_vix: np.ndarray
    emb_flow_raw: np.ndarray
    emb_flow_agg: np.ndarray
    # R2 per-variate embeddings (bypass mean-pool bottleneck)
    emb_var_atm_iv: np.ndarray       # v41: ATM call IV
    emb_var_atm_spread: np.ndarray   # v46: ATM bid-ask spread
    emb_var_spx_ret: np.ndarray      # v105: SPX session log return
    # Channel 2: probe forecasts
    pred_rv_15: float
    pred_rv_30: float
    pred_regime: np.ndarray      # (4,) class probabilities
    predicted_regime: int
    pred_spread: float
    # Channel 3: raw bypass scalars
    raw_spread: float
    delta_spread_1: float
    delta_spread_5: float
    atm_iv: float
    minutes_to_close: float
    vix_spot: float
    vix_term_slope: float
    spx_close: float
    realized_vol_30m: float
    # new probes (Optional for backward compat — None if not fitted)
    pred_gamma_accel: Optional[float] = None
    pred_smile_convexity: Optional[float] = None
    pred_jump: Optional[float] = None
    pred_flow_toxicity: Optional[float] = None

# ---------------------------------------------------------------------------
# Constants (must match pipeline.py exactly)
# ---------------------------------------------------------------------------

_RAW_ATM_SPREAD = 46
_RAW_ATM_SPREAD_CHG = 47
_RAW_ATM_IV = 41
_RAW_VIX_SPOT = 119          # VIX spot index
_RAW_VIX9D = 120             # VIX9D (9-day expected vol)
_RAW_MINS_TO_CLOSE = 114
_RAW_SESSION_LOG_RET = 105   # session log return (SPX price proxy)
_RAW_PARKINSON_RV_30M = 112  # Parkinson RV 30-minute

_SSL_FEATURES_V2 = sorted(
    list(range(0, 88)) + list(range(99, 105)) +
    list(range(105, 119)) + list(range(119, 131)) + list(range(131, 140))
)
_N_SSL_FEATURES = len(_SSL_FEATURES_V2)  # 129
# v3 adds 18 flow aggregates (raw variates 141-158) on top of v2
_SSL_FEATURES_V3 = sorted(_SSL_FEATURES_V2 + list(range(141, 159)))  # 147


def _ssl_pos(raw_idx: int, is_v3: bool) -> int:
    """Map a raw variate index (0-158) → SSL position in X_windows.

    X_windows is a sliced view: `data[:, ACTIVE_SSL_FEATURES]` where
    ACTIVE_SSL_FEATURES is a sorted list of raw indices kept for SSL.
    The SSL position of raw_idx is its rank in that sorted list.
    """
    features = _SSL_FEATURES_V3 if is_v3 else _SSL_FEATURES_V2
    return features.index(raw_idx)

_PROBE_FEATURE_GROUPS = {
    "options_grid": (0, 88), "strike_agg": (88, 94),
    "spx_derived": (94, 108), "vix_term": (108, 120), "order_flow": (120, 129),
}

# typed-vector position ranges in the SSL layout. Each tuple is
# (start, end) half-open, indexing into the per-variate token dimension of
# the encoder's (B, V, D*L) output. EMB_SHARED is computed separately
# (global mean over all variates).
#
# v3 layout adds flow_roll3 at SSL positions 129-137 (raw 141-149) and
# flow_roll15 at 138-146 (raw 150-158). On v2, EMB_FLOW_AGG positions
# don't exist and the emb_flow_agg vector is emitted as zeros.
_TYPED_VECTOR_SSL_RANGES_V2 = {
    "emb_grid":     (0, 88),
    "emb_strike":   (88, 94),
    "emb_spx":      (94, 108),
    "emb_vix":      (108, 120),
    "emb_flow_raw": (120, 129),
    # emb_flow_agg: N/A on v2 — zeros
}
_TYPED_VECTOR_SSL_RANGES_V3 = {
    **_TYPED_VECTOR_SSL_RANGES_V2,
    "emb_flow_agg": (129, 147),
}


# v9 Bug 7: Maps `batch_forecast` SSL group name → grammar `GType` name. The
# inverse-lookup keys are the `GType` enum names that
# `layer2.grammar.EMB_TYPE_TO_RAW_RANGE` is keyed on. Used by the consistency
# assertion below to verify the two maps stay aligned.
_BATCH_FORECAST_TO_GRAMMAR_GROUP = {
    "emb_grid":     "EMB_GRID",
    "emb_strike":   "EMB_STRIKE",
    "emb_spx":      "EMB_SPX",
    "emb_vix":      "EMB_VIX",
    "emb_flow_raw": "EMB_FLOW_RAW",
    "emb_flow_agg": "EMB_FLOW_AGG",
}


def assert_ssl_position_maps_consistent() -> None:
    """v9 Bug 7: assert the SSL-position map in this module is consistent
    with `layer2.grammar.EMB_TYPE_TO_RAW_RANGE`.

    Both maps describe the same typed-vector groups; if one is updated and
    the other isn't, the encoder pools different variates into a group than
    the grammar's EmbProj projection assumes — silently corrupting every
    EmbProj_* output. This assertion converts that drift surface into a
    loud import-time failure.

    For each group:
      * grammar.EMB_TYPE_TO_RAW_RANGE[group] gives a raw-variate range.
      * Each raw index → SSL position via `_ssl_pos(raw_idx, is_v3=True)`.
      * The resulting SSL positions must form a contiguous range.
      * That range must EQUAL `_TYPED_VECTOR_SSL_RANGES_V3[group_lower]`.

    Raises RuntimeError on any mismatch.
    """
    # Lazy import to avoid circular dep at module load: grammar.py is in
    # layer2 (no dep on layer1) but layer1.inference.batch_forecast.py
    # would not normally pull in layer2 — only the assertion does.
    try:
        from layer2.grammar import EMB_TYPE_TO_RAW_RANGE, GType
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot run SSL-position-map consistency assertion: layer2 "
            f"is not importable ({exc!r}). This usually indicates a "
            f"PYTHONPATH issue."
        ) from exc

    errors: List[str] = []
    for bf_name, ssl_range in _TYPED_VECTOR_SSL_RANGES_V3.items():
        gram_name = _BATCH_FORECAST_TO_GRAMMAR_GROUP.get(bf_name)
        if gram_name is None:
            errors.append(
                f"batch_forecast group {bf_name!r} has no grammar mapping "
                f"in _BATCH_FORECAST_TO_GRAMMAR_GROUP"
            )
            continue
        gtype = getattr(GType, gram_name, None)
        if gtype is None:
            errors.append(
                f"grammar.GType has no member {gram_name!r}"
            )
            continue
        raw_range = EMB_TYPE_TO_RAW_RANGE.get(gtype)
        if raw_range is None:
            errors.append(
                f"grammar.EMB_TYPE_TO_RAW_RANGE has no entry for {gtype}"
            )
            continue
        ssl_positions = [_ssl_pos(r, is_v3=True) for r in raw_range]
        if not ssl_positions:
            errors.append(f"raw range {raw_range} for {gram_name} is empty")
            continue
        # Contiguous?
        expected = list(range(ssl_positions[0], ssl_positions[-1] + 1))
        if ssl_positions != expected:
            errors.append(
                f"{gram_name}: raw range {raw_range} → SSL positions "
                f"{ssl_positions} are NOT contiguous"
            )
            continue
        derived = (ssl_positions[0], ssl_positions[-1] + 1)
        if derived != ssl_range:
            errors.append(
                f"{gram_name}: derived SSL range {derived} from raw "
                f"{raw_range} does not match _TYPED_VECTOR_SSL_RANGES_V3"
                f"[{bf_name!r}]={ssl_range}"
            )
    if errors:
        raise RuntimeError(
            "SSL-position map drift detected between "
            "layer1.inference.batch_forecast._TYPED_VECTOR_SSL_RANGES_V3 "
            "and layer2.grammar.EMB_TYPE_TO_RAW_RANGE:\n  " +
            "\n  ".join(errors)
        )

def _build_fine_group_indices() -> List[List[int]]:
    """8 semantic sub-groups for emb_fine pooling (mirrors pipeline.py)."""
    grid_iv = [i * 8 + 1 for i in range(11)]
    grid_spread = [i * 8 + 6 for i in range(11)] + [i * 8 + 7 for i in range(11)]
    grid_greeks = [i * 8 + 2 for i in range(11)] + [i * 8 + 3 for i in range(11)]
    grid_other = [i for i in range(88) if i not in grid_iv + grid_spread + grid_greeks]
    fine = [grid_iv, grid_spread, grid_greeks, grid_other]
    for key in ("strike_agg", "spx_derived", "vix_term", "order_flow"):
        gs, ge = _PROBE_FEATURE_GROUPS[key]
        fine.append(list(range(gs, ge)))
    return fine

_FINE_GROUP_INDICES = _build_fine_group_indices()

# ---------------------------------------------------------------------------
# Minimal encoder reconstruction (no QC/pipeline.py dependency)
# ---------------------------------------------------------------------------

def _reconstruct_encoder(hp: dict, n_variates: int, device: torch.device):
    """Reconstruct iTransformerEncoder from hyperparams."""
    class _Block(nn.Module):
        def __init__(self, d, nh, dff, drop):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(dff, d), nn.Dropout(drop))
            self.drop = nn.Dropout(drop)
        def forward(self, x):
            z2 = self.norm1(x)
            a, w = self.attn(z2, z2, z2)
            x = x + self.drop(a)
            x = x + self.ffn(self.norm2(x))
            return x, w

    class _PatchTokenizerV2(nn.Module):
        def __init__(self, seq_len, d_model, patch_size=12, dropout=0.2):
            super().__init__()
            self.patch_size = patch_size
            self.n_patches = seq_len // patch_size
            self.patch_proj = nn.Linear(patch_size, d_model)
            self.patch_norm = nn.LayerNorm(d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.1)
            self.concat_proj = nn.Linear(self.n_patches * d_model, d_model)
            self.concat_norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x_vt):
            B, V, T = x_vt.shape
            x = x_vt.reshape(B * V, self.n_patches, self.patch_size)
            z = self.patch_proj(x)
            z = self.patch_norm(z)
            z = torch.nn.functional.gelu(z)
            z = z + self.pos_embed
            z = z.reshape(B * V, self.n_patches * z.shape[-1])
            z = self.concat_proj(z)
            z = self.concat_norm(z)
            z = self.drop(z)
            return z.reshape(B, V, -1)

    class _TemporalAttentionPool(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.score = nn.Linear(d_model, 1)

        def forward(self, z):
            w = torch.softmax(self.score(z), dim=1)
            return (z * w).sum(dim=1)

    class _CNNTokenizer(nn.Module):
        def __init__(self, seq_len, d_model, dropout=0.2):
            super().__init__()
            self.seq_len = seq_len
            mid = d_model // 2
            self.conv = nn.Sequential(
                nn.Conv1d(1, mid, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(mid, d_model, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.pool = _TemporalAttentionPool(d_model)

        def forward(self, x_vt):
            B, V, T = x_vt.shape
            x = x_vt.reshape(B * V, 1, T)
            z = self.conv(x).transpose(1, 2)
            z = self.pool(z)
            return z.reshape(B, V, -1)

        def forward_embed(self, x_vt):
            B, V, T = x_vt.shape
            x = x_vt.reshape(B * V, 1, T)
            z = self.conv(x).transpose(1, 2)
            return z, (B, V)

    class _TemporalAttentionBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_ff, d_model),
                                     nn.Dropout(dropout))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            z2 = self.norm1(x)
            a, _ = self.attn(z2, z2, z2)
            x = x + self.drop(a)
            x = x + self.ffn(self.norm2(x))
            return x

    class _Encoder(nn.Module):
        def __init__(self, hp, V):
            super().__init__()
            T, D = hp["seq_len"], hp["d_model"]
            tok_type = hp.get("tokenizer", "linear")
            if tok_type == "patch":
                patch_size = hp.get("patch_size", 12)
                self.tok = _PatchTokenizerV2(T, D, patch_size, hp["dropout"])
            elif tok_type == "cnn":
                self.tok = _CNNTokenizer(T, D, hp["dropout"])
            else:
                self.tok = nn.Linear(T, D)
            self._tok_type = tok_type
            self.variate_embed = nn.Parameter(torch.randn(V, D) * 0.02)
            self.embed_drop = nn.Dropout(hp["dropout"])
            self.blocks = nn.ModuleList([
                _Block(D, hp["n_heads"], hp["d_ff"], hp["dropout"])
                for _ in range(hp["n_layers"])
            ])
            self.final_norm = nn.LayerNorm(D)
            self.use_mask_indicator = bool(hp.get("use_mask_indicator", False))
            if self.use_mask_indicator:
                self.mask_proj = nn.Linear(T, D)
            self.use_temporal_attn = bool(hp.get("use_temporal_attn", False))
            if self.use_temporal_attn and tok_type == "cnn":
                self._temporal_seq_len = T
                self.temporal_pos_embed = nn.Parameter(torch.randn(T, D) * 0.02)
                self.temporal_block = _TemporalAttentionBlock(
                    D, hp["n_heads"], hp["d_ff"], hp["dropout"])

        def encode_variates(self, x, multi_layer=False, mask=None):
            x_vt = x.transpose(1, 2)
            if self.use_temporal_attn and self._tok_type == "cnn":
                z_seq, (B, V) = self.tok.forward_embed(x_vt)
                z_seq = z_seq + self.temporal_pos_embed.unsqueeze(0)
                z_seq = self.temporal_block(z_seq)
                z = z_seq.mean(dim=1).reshape(B, V, -1)
            else:
                z = self.tok(x_vt)
            if self.use_mask_indicator:
                if mask is None:
                    B, T_in, V_in = x.shape
                    mask = torch.zeros(B, T_in, V_in, dtype=torch.float32, device=x.device)
                z = z + self.mask_proj(mask.transpose(1, 2).float())
            z = z + self.variate_embed.unsqueeze(0)
            z = self.embed_drop(z)
            if multi_layer:
                layers = []
                for block in self.blocks:
                    z, _ = block(z)
                    layers.append(self.final_norm(z))
                return torch.cat(layers, dim=-1)
            for block in self.blocks:
                z, _ = block(z)
            return self.final_norm(z)

    model = _Encoder(hp, n_variates)
    model.to(device)
    return model

# ---------------------------------------------------------------------------
# Standalone probe fitting / prediction (no encoder needed)
# ---------------------------------------------------------------------------

def fit_probes_standalone(
    emb_fine: np.ndarray,
    targets: Dict[str, np.ndarray],
) -> Dict[str, Tuple]:
    """Fit probe models on pre-computed emb_fine embeddings.

    Returns dict of (scaler, model) tuples keyed by probe name.
    Pure sklearn — no encoder checkpoint needed. Used by per-fold
    probe refit in experiment.py (L2.62 look-ahead fix).

    Args:
        emb_fine: (N, 3072) float32 — fine-group-pooled encoder embeddings.
        targets: dict with keys 'rv_15', 'rv_30', 'regime', 'spread'
                 and optionally 'gamma_accel', 'smile_convex', 'flow_tox', 'jump_30'.
    """
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    # Validate required core targets
    _REQUIRED = {"rv_15", "rv_30", "regime", "spread"}
    _missing = _REQUIRED - set(targets.keys())
    if _missing:
        raise ValueError(f"Core probe targets missing: {_missing}")

    probes: Dict[str, Tuple] = {}

    # Ridge probes (continuous targets)
    for key in ("rv_15", "rv_30", "spread"):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(emb_fine)
        mdl = RidgeCV(alphas=np.logspace(-3, 3, 10))
        mdl.fit(X_s, targets[key])
        probes[key] = (scaler, mdl)

    # Regime classifier (4-class)
    if "regime" in targets:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(emb_fine)
        mdl = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 10), cv=3,
            scoring="balanced_accuracy", max_iter=1000,
            solver="lbfgs", class_weight="balanced",
        )
        mdl.fit(X_s, targets["regime"])
        probes["regime"] = (scaler, mdl)

    # new probes — filter NaN per target independently
    for key in ("gamma_accel", "smile_convex", "flow_tox"):
        if key not in targets:
            continue
        y = targets[key]
        valid = ~np.isnan(y)
        if valid.sum() > 100:
            scaler = StandardScaler()
            X_s = scaler.fit_transform(emb_fine[valid])
            mdl = RidgeCV(alphas=np.logspace(-3, 3, 10))
            mdl.fit(X_s, y[valid])
            probes[key] = (scaler, mdl)

    # Jump classifier (binary)
    if "jump_30" in targets:
        y = targets["jump_30"]
        valid = ~np.isnan(y)
        if valid.sum() > 100:
            scaler = StandardScaler()
            X_s = scaler.fit_transform(emb_fine[valid])
            mdl = LogisticRegressionCV(
                Cs=np.logspace(-3, 3, 10), cv=3,
                scoring="balanced_accuracy", max_iter=1000,
                solver="lbfgs", class_weight="balanced",
            )
            mdl.fit(X_s, y[valid].astype(int))
            probes["jump_30"] = (scaler, mdl)

    return probes


def predict_probes_standalone(
    probes: Dict[str, Tuple],
    emb_fine: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run fitted probes on emb_fine. Returns dict of prediction arrays.

    Mirrors L1BatchForecaster._predict_probes() including v9 P0-A
    RV log-inversion (exp+sqrt for std-scale output).
    """
    out: Dict[str, np.ndarray] = {}

    # Continuous probes
    for key in ("rv_15", "rv_30", "spread"):
        if key in probes:
            scaler, mdl = probes[key]
            out[key] = mdl.predict(scaler.transform(emb_fine))

    # RV: invert log transform, then sqrt → std-scale (v9 P0-A)
    for key in ("rv_15", "rv_30"):
        if key in out:
            logv = np.clip(out[key], -20.0, 5.0)
            variance = np.exp(logv)
            out[key] = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)

    # Regime classifier
    if "regime" in probes:
        scaler, mdl = probes["regime"]
        X_s = scaler.transform(emb_fine)
        out["regime_proba"] = mdl.predict_proba(X_s)
        out["predicted_regime"] = np.argmax(out["regime_proba"], axis=1)

    # new continuous probes
    for key in ("gamma_accel", "smile_convex", "flow_tox"):
        if key in probes:
            scaler, mdl = probes[key]
            out[key] = mdl.predict(scaler.transform(emb_fine)).astype(np.float32)

    # Jump probability
    if "jump_30" in probes:
        scaler, mdl = probes["jump_30"]
        proba = mdl.predict_proba(scaler.transform(emb_fine))
        out["jump_proba"] = (
            proba[:, 1].astype(np.float32) if proba.shape[1] > 1
            else proba[:, 0].astype(np.float32)
        )

    return out


# ---------------------------------------------------------------------------
# L1BatchForecaster
# ---------------------------------------------------------------------------

class L1BatchForecaster:
    """Batch inference: frozen L1 encoder -> L1Output for GP consumption."""

    def __init__(self, checkpoint_path_or_dict,
                 device: Optional[torch.device] = None,
                 spx_daily_opens: Optional[Dict[str, float]] = None):
        self.device = device or torch.device("cpu")
        if spx_daily_opens is not None:
            self._spx_daily_opens = spx_daily_opens
        else:
            try:
                self._spx_daily_opens = _load_spx_daily_opens()
            except FileNotFoundError:
                # Non-fatal: only callers that invoke _extract_bypass (i.e.
                # encode_dataset) need SPX dollar prices. Probe-only callers
                # (h1_paired_tests.py, test_encoder_determinism.py) don't.
                self._spx_daily_opens = None

        ckpt = _load_checkpoint_flexible(
            checkpoint_path_or_dict, map_location=self.device
        )

        self.z_mean = ckpt["z_mean"]
        self.z_std = ckpt["z_std"]
        self.log1p_cols = ckpt.get("log1p_cols", [])
        hp = ckpt.get("ssl_hyperparams", {"seq_len": 60, "d_model": 128,
                       "n_heads": 4, "n_layers": 3, "d_ff": 512, "dropout": 0.2,
                       "tokenizer": "linear"})
        n_variates = ckpt.get("n_ssl_features", _N_SSL_FEATURES)
        self.is_v3 = (n_variates > 129)

        # v5 hardening (Code Reviewer finding Q1): `_ssl_pos` assumes the
        # sorted raw-index union stored in _SSL_FEATURES_V2/V3 matches the
        # pipeline's `ACTIVE_SSL_FEATURES` at checkpoint creation time. If
        # the pipeline ever changes the SSL feature selection, `_ssl_pos`
        # silently returns wrong indices and every bypass scalar is
        # mis-read. Assert invariance at load so drift fails loudly.
        ckpt_features = ckpt.get("ssl_features")
        if ckpt_features is not None:
            expected = _SSL_FEATURES_V3 if self.is_v3 else _SSL_FEATURES_V2
            if list(ckpt_features) != list(expected):
                raise RuntimeError(
                    f"Checkpoint ssl_features drift: `_ssl_pos` cannot be "
                    f"trusted. "
                    f"ckpt[:5]={list(ckpt_features)[:5]}..."
                    f"[{len(ckpt_features)} total], "
                    f"expected[:5]={list(expected)[:5]}..."
                    f"[{len(expected)} total]. "
                    f"Either (a) regenerate the checkpoint with the current "
                    f"pipeline, or (b) update _SSL_FEATURES_V2/_SSL_FEATURES_V3 "
                    f"in batch_forecast.py to match."
                )

        self.hp = hp
        self.d_model = hp["d_model"]
        self.n_layers = hp.get("n_layers", 3)
        self.n_variates = n_variates
        self.emb_dim = self.d_model * self.n_layers  # 384 per typed vector (multi-layer)
        self.n_fine_groups = len(_FINE_GROUP_INDICES)  # 8 — used for probe input
        self.emb_fine_dim = self.n_fine_groups * self.emb_dim  # 3,072 (probe input)
        self._typed_ranges = (
            _TYPED_VECTOR_SSL_RANGES_V3 if self.is_v3 else _TYPED_VECTOR_SSL_RANGES_V2
        )

        self.model = _reconstruct_encoder(hp, n_variates, self.device)
        load_result = self.model.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        if load_result.missing_keys:
            raise RuntimeError(f"Checkpoint missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            # SSL-012: temporal-attn pre-tokenizer modules are inference-
            # irrelevant for the linear tokenizer arm (their output is
            # collapsed by `tok` in the post-temporal pathway and they are not
            # used during multi_layer probe extraction here). Allow these to
            # be silently dropped, but ANY OTHER unexpected key indicates the
            # checkpoint and inference adapter have diverged — fail loud
            # rather than continuing with a quietly-broken encoder.
            allowed_drop_prefixes = (
                "cell_proj.", "temporal_pos_embed", "temporal_block.",
                "temporal_readout.",
                # Classifier-only heads — reconstruction-only inference path
                # ignores these by design.
                "pool_q", "pool_attn.", "regime_head.", "emb_head.",
            )
            unexpected = list(load_result.unexpected_keys)
            disallowed = [k for k in unexpected if not any(
                k.startswith(p) or k == p.rstrip(".") for p in allowed_drop_prefixes
            )]
            if disallowed:
                raise RuntimeError(
                    f"Checkpoint has unexpected keys NOT in the allowed-drop list: "
                    f"{disallowed}. The L2-inference encoder is not equipped to "
                    f"handle these — silently dropping them would produce a "
                    f"distribution shift between training and L2 inference."
                )
            ok_drops = [k for k in unexpected if k not in disallowed]
            if ok_drops:
                print(f"[L1BatchForecaster] Dropped inference-irrelevant ckpt keys: {ok_drops}")
        self.model.eval()

        # Probes fit on emb_fine internally (same 3072-d input they used
        # before PCA; the probe module never cared about the PCA compression
        # — PCA was added purely to shrink the GP-facing terminal set).
        self._probe_rv15 = None
        self._probe_rv30 = None
        self._probe_regime = None
        self._probe_spread = None

    def _preprocess(self, X_raw: np.ndarray) -> np.ndarray:
        """Apply log1p + z-score + clip (mirrors training pipeline)."""
        X = X_raw.copy()
        if self.log1p_cols:
            X[:, :, self.log1p_cols] = (np.sign(X[:, :, self.log1p_cols])
                                        * np.log1p(np.abs(X[:, :, self.log1p_cols])))
        X = ((X - self.z_mean) / self.z_std).astype(np.float32)
        return np.clip(X, -5.0, 5.0)

    def _encode_per_variate(self, X_z: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Encode z-scored windows → per-variate multi-layer tokens (N, V, D*L).

        The raw per-variate tokens are the source of both the typed vectors
        (D84 Channel 1 — group-pooled from this) and the emb_fine probe input
        (fine-group-pooled from this). Single forward pass, two pooling paths
        downstream.
        """
        N = len(X_z)
        V = self.n_variates
        per_variate = np.empty((N, V, self.emb_dim), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, N, batch_size):
                e = min(s + batch_size, N)
                x_batch = torch.from_numpy(X_z[s:e]).to(self.device)
                z_ml = self.model.encode_variates(x_batch, multi_layer=True)  # (B, V, D*L)
                per_variate[s:e] = z_ml.cpu().numpy()
        return per_variate

    # R2 per-variate positions: raw variate index → SSL position.
    # These variates bypass the mean-pool bottleneck — the GP gets direct
    # access to the individual encoder output for each.
    _PER_VARIATE_RAW_INDICES = {
        "EmbVar_v41":  41,   # ATM call IV
        "EmbVar_v46":  46,   # ATM bid-ask spread
        "EmbVar_v105": 105,  # SPX session log return
    }

    def _pool_typed_vectors(self, per_variate: np.ndarray) -> Dict[str, np.ndarray]:
        """Group-pool per-variate tokens into the 7 D84 typed vectors.

        Input:  per_variate (N, V, D*L)
        Output: dict with 7 keys (emb_shared, emb_grid, ..., emb_flow_agg),
                each (N, D*L). On v2, emb_flow_agg is zeros.
        """
        N = len(per_variate)
        out: Dict[str, np.ndarray] = {
            "emb_shared": per_variate.mean(axis=1),  # mean over all variates
        }
        for key, (gs, ge) in self._typed_ranges.items():
            out[key] = per_variate[:, gs:ge, :].mean(axis=1)
        # v2 fallback: emb_flow_agg doesn't exist; emit zeros so downstream code
        # can always index all 7 keys.
        if "emb_flow_agg" not in out:
            out["emb_flow_agg"] = np.zeros((N, self.emb_dim), dtype=np.float32)
        return out

    def _extract_per_variate_embeddings(self, per_variate: np.ndarray) -> Dict[str, np.ndarray]:
        """R2: Extract individual variate embeddings (no pooling).

        Input:  per_variate (N, V, D*L)
        Output: dict with 3 keys (EmbVar_v41, EmbVar_v46, EmbVar_v105),
                each (N, D*L) — the raw encoder output for that single variate.
        """
        out: Dict[str, np.ndarray] = {}
        for col_name, raw_idx in self._PER_VARIATE_RAW_INDICES.items():
            ssl_pos = _ssl_pos(raw_idx, self.is_v3)
            out[col_name] = per_variate[:, ssl_pos, :]  # (N, D*L)
        return out

    def _pool_emb_fine(self, per_variate: np.ndarray) -> np.ndarray:
        """Fine-group-pool per-variate tokens for probe input (3072-d on v2/v3)."""
        fine_parts = [per_variate[:, idx, :].mean(axis=1) for idx in _FINE_GROUP_INDICES]
        return np.concatenate(fine_parts, axis=1)

    def encode_to_emb_fine(self, X_windows: np.ndarray) -> np.ndarray:
        """Encode raw windows to emb_fine (3072-d) without fitting probes.

        Public interface for probe bundle generation. Avoids coupling
        external scripts to private _preprocess/_encode/_pool methods.

        Args:
            X_windows: raw windows (N, T, V) in natural units.
        Returns:
            emb_fine: (N, 3072) float32 — fine-group-pooled embeddings.
        """
        per_variate = self._encode_per_variate(self._preprocess(X_windows))
        emb_fine = self._pool_emb_fine(per_variate)
        del per_variate  # free immediately (can be ~2 GB for full corpus)
        return emb_fine

    # -- Probes -------------------------------------------------------------

    def fit_probes(self, X_windows: np.ndarray, targets: Dict[str, np.ndarray]):
        """Fit probe models on emb_fine extracted from training windows.

        X_windows: raw training windows (N, T, V); will be preprocessed and encoded.
        targets keys: 'rv_15', 'rv_30' (float), 'regime' (int 0-3), 'spread' (float)
        """
        per_variate = self._encode_per_variate(self._preprocess(X_windows))
        emb_fine = self._pool_emb_fine(per_variate)
        self.fit_probes_from_embeddings(emb_fine, targets)

    def fit_probes_from_embeddings(self, emb_fine: np.ndarray,
                                   targets: Dict[str, np.ndarray]):
        """Fit probe models on pre-computed emb_fine (no encoding needed).

        Used by per-fold probe refit where emb_fine is loaded from the
        probe fitting bundle rather than re-encoded from raw windows.
        """
        # Clear stale probe attributes from prior calls (e.g.,
        # per-fold refit where a target was present in fold N but absent
        # in fold N+1). Without this, _predict_probes() would use the
        # stale model from the prior call via hasattr().
        for _stale_attr in ("_probe_gamma_accel", "_probe_smile_convex",
                            "_probe_flow_tox", "_probe_jump"):
            if hasattr(self, _stale_attr):
                delattr(self, _stale_attr)

        fitted = fit_probes_standalone(emb_fine, targets)
        # Store on self for _predict_probes() compatibility
        _ATTR_MAP = {
            "rv_15": "_probe_rv15", "rv_30": "_probe_rv30",
            "spread": "_probe_spread", "regime": "_probe_regime",
            "gamma_accel": "_probe_gamma_accel",
            "smile_convex": "_probe_smile_convex",
            "flow_tox": "_probe_flow_tox",
            "jump_30": "_probe_jump",
        }
        for key, attr in _ATTR_MAP.items():
            if key in fitted:
                setattr(self, attr, fitted[key])

    def _predict_probes(self, emb_fine: np.ndarray) -> Dict[str, np.ndarray]:
        """Run fitted probes on emb_fine (3072-d per row).

        v9 P0-A scale alignment (2026-04-24): RV probes are fitted on
        `log_rv_15`/`log_rv_30` targets where the underlying signal is the
        Parkinson RV variance over 15/30-minute windows. The previous
        implementation `exp(logv)` recovered VARIANCE values (~1e-4 to 3e-3),
        but downstream templates (`PredRV15 < 0.40`) and grammar arithmetic
        compared those values to vol-decimal scalars — a unit mismatch that
        made `PredRV15 < 0.40` constant-True on 100% of corpus rows.

        Fix: emit STD-scale (sqrt of variance) so the published values share
        units with the bypass `RealizedVol30m` (which is itself a sqrt-based
        Parkinson RV). The sqrt is applied AFTER the exp-clip so the inverse
        log transform's numerical floor still protects against probe
        outliers (`exp(-20) ≈ 2e-9`, then sqrt → `4.5e-5`).

        H2 pre-reg v4 docstring (legacy): RV probes are fitted on
        `log_rv_15`/`log_rv_30` targets (pipeline.compute_probe_targets
        applies `log(max(rv, LOG_RV_FLOOR))`).
        """
        out = {}
        for attr, key in [("_probe_rv15", "rv_15"), ("_probe_rv30", "rv_30"),
                          ("_probe_spread", "spread")]:
            scaler, mdl = getattr(self, attr)
            out[key] = mdl.predict(scaler.transform(emb_fine))
        # RV: invert the log transform, then sqrt to convert variance →
        # std-scale (v9 P0-A). Clamp the log argument to a wide but finite
        # band so numerical glitches (large residuals far from training
        # support) can never produce inf.
        for key in ("rv_15", "rv_30"):
            logv = np.clip(out[key], -20.0, 5.0)  # exp(-20)≈2e-9, exp(5)≈148
            variance = np.exp(logv)
            # epsilon clamp: variance is non-negative by construction
            # (exp output) but guard floating-point flush-to-zero.
            std = np.sqrt(np.maximum(variance, 1e-12))
            out[key] = std.astype(np.float32)
        scaler, mdl = self._probe_regime
        out["regime_proba"] = mdl.predict_proba(scaler.transform(emb_fine))

        # new probes (backward-compatible: skip if not fitted)
        for attr, key in [("_probe_gamma_accel", "gamma_accel"),
                          ("_probe_smile_convex", "smile_convex"),
                          ("_probe_flow_tox", "flow_tox")]:
            if hasattr(self, attr):
                scaler, mdl = getattr(self, attr)
                out[key] = mdl.predict(scaler.transform(emb_fine)).astype(np.float32)

        if hasattr(self, "_probe_jump"):
            scaler, mdl = self._probe_jump
            proba = mdl.predict_proba(scaler.transform(emb_fine))
            # Probability of jump (class 1)
            out["jump_proba"] = proba[:, 1].astype(np.float32) if proba.shape[1] > 1 else proba[:, 0].astype(np.float32)

        return out

    # -- Bypass scalars -----------------------------------------------------

    def _extract_bypass(self, X_raw: np.ndarray,
                        window_meta: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract pass-through scalars from raw-value windows (last bar).

        X_raw is the PRE-preprocess SSL-sliced windows from
        `pipeline.load_probe_days`. Values are in natural units (log1p on
        v101/v102/v104/v139 only; VIX f/f applied to v123-v124 and their
        v128-v130 derivatives). Indexing uses SSL positions via `_ssl_pos`
        — raw variate indices are NOT valid direct indices into X_raw.

        `spx_close` is reconstructed as a dollar SPX level, not the stored
        `session_log_return`:
            spx_price = open_today * exp(session_log_return_at_last_bar)
        The daily opens come from the QC-collected CSV at
        raw_data/spx_daily_ohlc.csv (see `_load_spx_daily_opens`).
        """
        is_v3 = self.is_v3
        pos_spread = _ssl_pos(_RAW_ATM_SPREAD, is_v3)
        pos_spread_chg = _ssl_pos(_RAW_ATM_SPREAD_CHG, is_v3)
        pos_atm_iv = _ssl_pos(_RAW_ATM_IV, is_v3)
        pos_mins = _ssl_pos(_RAW_MINS_TO_CLOSE, is_v3)
        pos_vix_spot = _ssl_pos(_RAW_VIX_SPOT, is_v3)
        pos_vix9d = _ssl_pos(_RAW_VIX9D, is_v3)
        pos_log_ret = _ssl_pos(_RAW_SESSION_LOG_RET, is_v3)
        pos_rv30 = _ssl_pos(_RAW_PARKINSON_RV_30M, is_v3)

        last = X_raw[:, -1, :]
        prev5 = X_raw[:, -6, :] if X_raw.shape[1] >= 6 else X_raw[:, 0, :]
        session_log_ret = last[:, pos_log_ret].astype(np.float64)

        if self._spx_daily_opens is None:
            raise RuntimeError(
                "SPX daily opens not loaded. Run "
                "scripts/ingest_spx_daily_ohlc.py first, or pass "
                "spx_daily_opens=... to L1BatchForecaster()."
            )
        n = len(X_raw)
        assert len(window_meta) == n, (
            f"window_meta length ({len(window_meta)}) != n_windows ({n})"
        )
        open_arr = np.full(n, np.nan, dtype=np.float64)
        missing_dates: List[str] = []
        for i, meta in enumerate(window_meta):
            d = str(meta["date"])
            o = self._spx_daily_opens.get(d)
            if o is None:
                missing_dates.append(d)
            else:
                open_arr[i] = o
        if missing_dates:
            unique_missing = sorted(set(missing_dates))
            raise KeyError(
                f"SPX daily open missing for {len(unique_missing)} trading "
                f"days (first few: {unique_missing[:5]}). Recollect SPX "
                f"daily OHLC from QC — see layer1/data/collect_spx_daily_ohlc.py."
            )

        spx_price = open_arr * np.exp(session_log_ret)

        # v9 P0-A: collector stores `parkinson_rv(30)` which on inspection
        # carries minute-timescale Parkinson VARIANCE values (median ~2e-4,
        # max ~5e-3) — NOT the std-scale the formula's `sqrt(...)` term
        # would suggest. Templates compare RealizedVol30m to vol-decimal
        # thresholds; without sqrt every comparison is degenerate. Apply
        # sqrt + epsilon clamp so the published bypass scalar lives in the
        # SAME std-scale as the v9-fixed PredRV15/PredRV30 probe outputs.
        rv30_var = last[:, pos_rv30].astype(np.float64)
        rv30_std = np.sqrt(np.maximum(rv30_var, 1e-12)).astype(np.float32)

        result = {
            "raw_spread": last[:, pos_spread],
            "delta_spread_1": last[:, pos_spread_chg],
            "delta_spread_5": last[:, pos_spread] - prev5[:, pos_spread],
            "atm_iv": last[:, pos_atm_iv],
            "minutes_to_close": last[:, pos_mins],
            "vix_spot": last[:, pos_vix_spot],
            "vix_term_slope": last[:, pos_vix9d] - last[:, pos_vix_spot],
            "spx_close": spx_price.astype(np.float32),
            "realized_vol_30m": rv30_std,
        }
        # NaN guard on all bypass scalars
        for k in result:
            result[k] = np.nan_to_num(result[k], nan=0.0, posinf=0.0, neginf=0.0)
        return result

    # -- Main pipeline ------------------------------------------------------

    def encode_dataset(self, X_windows: np.ndarray, W_windows: np.ndarray,
                       window_meta: List[Dict]) -> List[L1Output]:
        """Encode all windows → L1Output rows.

        X_windows: (N, T, V) SSL-feature windows in PRE-preprocess natural
                   units — V=129 on v2, 147 on v3. Used both as the encoder
                   input (after `_preprocess`) and as the source for bypass
                   scalars (raw values at SSL-mapped positions via `_ssl_pos`).
        W_windows: (N, T, V) SSL-aligned C10 mask-weight tensor. Accepted for
                   API stability; currently unused by bypass extraction.
        window_meta: list of {'date', 'window_idx', 'bar_position'}
        """
        assert self._probe_rv15 is not None, "Call fit_probes() first"
        N = len(X_windows)
        assert len(W_windows) == N == len(window_meta)

        per_variate = self._encode_per_variate(self._preprocess(X_windows))
        typed = self._pool_typed_vectors(per_variate)            # dict of (N, D*L)
        per_var = self._extract_per_variate_embeddings(per_variate)  # R2 per-variate
        emb_fine = self._pool_emb_fine(per_variate)              # (N, 3072) for probes
        preds = self._predict_probes(emb_fine)
        bp = self._extract_bypass(X_windows, window_meta)

        return [L1Output(
            date=window_meta[i]["date"], window_idx=window_meta[i]["window_idx"],
            bar_position=window_meta[i]["bar_position"],
            # Channel 1: typed vectors
            emb_shared=typed["emb_shared"][i],
            emb_grid=typed["emb_grid"][i],
            emb_strike=typed["emb_strike"][i],
            emb_spx=typed["emb_spx"][i],
            emb_vix=typed["emb_vix"][i],
            emb_flow_raw=typed["emb_flow_raw"][i],
            emb_flow_agg=typed["emb_flow_agg"][i],
            # R2 per-variate embeddings
            emb_var_atm_iv=per_var["EmbVar_v41"][i],
            emb_var_atm_spread=per_var["EmbVar_v46"][i],
            emb_var_spx_ret=per_var["EmbVar_v105"][i],
            # Channel 2: probes
            pred_rv_15=float(preds["rv_15"][i]),
            pred_rv_30=float(preds["rv_30"][i]),
            pred_regime=preds["regime_proba"][i].astype(np.float32),
            predicted_regime=int(np.argmax(preds["regime_proba"][i])),
            pred_spread=float(preds["spread"][i]),
            # new probes (None if not fitted)
            pred_gamma_accel=float(preds["gamma_accel"][i]) if "gamma_accel" in preds else None,
            pred_smile_convexity=float(preds["smile_convex"][i]) if "smile_convex" in preds else None,
            pred_jump=float(preds["jump_proba"][i]) if "jump_proba" in preds else None,
            pred_flow_toxicity=float(preds["flow_tox"][i]) if "flow_tox" in preds else None,
            # Channel 3: raw bypass scalars
            raw_spread=float(bp["raw_spread"][i]),
            delta_spread_1=float(bp["delta_spread_1"][i]),
            delta_spread_5=float(bp["delta_spread_5"][i]),
            atm_iv=float(bp["atm_iv"][i]),
            minutes_to_close=float(bp["minutes_to_close"][i]),
            vix_spot=float(bp["vix_spot"][i]),
            vix_term_slope=float(bp["vix_term_slope"][i]),
            spx_close=float(bp["spx_close"][i]),
            realized_vol_30m=float(bp["realized_vol_30m"][i]),
        ) for i in range(N)]

    # -- Serialization ------------------------------------------------------

    # Column rename mapping: snake_case -> PascalCase grammar terminal names
    _COLUMN_RENAME = {
        "pred_rv_15": "PredRV15",
        "pred_rv_30": "PredRV30",
        "pred_spread": "PredSpread",
        "pred_gamma_accel": "PredGammaAccel",
        "pred_smile_convexity": "PredSmileConvexity",
        "pred_jump": "PredJump",
        "pred_flow_toxicity": "PredFlowToxicity",
        "raw_spread": "RawSpread",
        "delta_spread_1": "DeltaSpread1",
        "delta_spread_5": "DeltaSpread5",
        "atm_iv": "ATM_IV",
        "minutes_to_close": "MinutesToClose",
        "vix_spot": "VIXSpot",
        "vix_term_slope": "VIXTermSlope",
        "predicted_regime": "PredRegime",
        "spx_close": "SPXClose",
        "realized_vol_30m": "RealizedVol30m",
    }

    def to_parquet(self, outputs: List[L1Output], path: str) -> pd.DataFrame:
        """Save as Parquet for GP consumption.

        D84 typed vectors (each 384-d float32 on multi-layer iTransformer)
        are stored as Arrow list columns keyed by the grammar terminal name
        (EMB_SHARED, EMB_GRID, EMB_STRIKE, EMB_SPX, EMB_VIX, EMB_FLOW_RAW,
        EMB_FLOW_AGG). Scalar columns use PascalCase grammar terminal names
        (PredRV15, ATM_IV, etc.) so batch_forecast output drops directly
        into the evaluator's EvaluationContext.update() bar_data dict.
        """
        _scalar_keys = ["date", "window_idx", "bar_position", "pred_rv_15",
                        "pred_rv_30", "predicted_regime", "pred_spread",
                        "raw_spread", "delta_spread_1", "delta_spread_5",
                        "atm_iv", "minutes_to_close", "vix_spot",
                        "vix_term_slope", "spx_close", "realized_vol_30m"]
        _typed_vec_cols = [
            ("emb_shared",   "EMB_SHARED"),
            ("emb_grid",     "EMB_GRID"),
            ("emb_strike",   "EMB_STRIKE"),
            ("emb_spx",      "EMB_SPX"),
            ("emb_vix",      "EMB_VIX"),
            ("emb_flow_raw", "EMB_FLOW_RAW"),
            ("emb_flow_agg", "EMB_FLOW_AGG"),
            # R2 per-variate embeddings (bypass mean-pool bottleneck)
            ("emb_var_atm_iv",     "EMB_VAR_ATM_IV"),
            ("emb_var_atm_spread", "EMB_VAR_ATM_SPREAD"),
            ("emb_var_spx_ret",    "EMB_VAR_SPX_RET"),
        ]
        records = []
        for o in outputs:
            row = {k: getattr(o, k) for k in _scalar_keys}
            # Typed vectors as list columns — stored under grammar terminal names
            for attr, col in _typed_vec_cols:
                row[col] = getattr(o, attr).astype(np.float32).tolist()
            # new probes (include if non-None)
            for attr in ["pred_gamma_accel", "pred_smile_convexity", "pred_jump", "pred_flow_toxicity"]:
                val = getattr(o, attr, None)
                if val is not None:
                    row[attr] = float(val)
            # Regime probability vector still flattened for GP
            row.update({f"RegimeProb{k}": float(o.pred_regime[k])
                        for k in range(len(o.pred_regime))})
            records.append(row)
        df = pd.DataFrame(records)
        df.rename(columns=self._COLUMN_RENAME, inplace=True)

        # v8 (RF-4): override MinutesToClose with data-derived value. The
        # prior `session_length - minutes_since_open` formula assumed a
        # full 6.5-hour session, producing wrong values on NYSE half-days
        # (~3/year: Black Friday, day-before-July-4, Christmas Eve when
        # weekday). Compute from the ACTUAL last bar per date in QC's
        # data — self-validating vs any calendar, handles future/
        # unscheduled early closures automatically, zero external
        # dependencies.
        # BAR_MINUTES=1: corpus is at 1-minute resolution (verified:
        # 405 bars/day × 1 min ≈ 9:30 AM to 4:15 PM ET with small
        # pre/post padding).
        _BAR_MINUTES = 1
        _last_bar_by_date = df.groupby("date")["bar_position"].transform("max")
        df["MinutesToClose"] = (
            (_last_bar_by_date - df["bar_position"]) * float(_BAR_MINUTES)
        ).astype(np.float32)

        df.to_parquet(path, index=False, engine="pyarrow")
        print(f"Saved {len(df)} rows -> {path} "
              f"({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
        return df

    @staticmethod
    def from_parquet(path: str) -> pd.DataFrame:
        """Load L1Output Parquet as DataFrame."""
        return pd.read_parquet(path, engine="pyarrow")
