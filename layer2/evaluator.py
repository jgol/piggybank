"""L2 GP Tree Evaluator — executes expression trees against L1Output data.

Contains:
  - SimpleBacktester: scalar-only directional strategies (legacy path)
  - MultiLegOptionsBacktester: multi-leg options with BS pricing (legacy path,
    superseded by evaluator_vectorized.py for production runs)
  - _delta_to_strike: Newton-Raphson strike solver with skew (skew_slope=-0.15)
  - PCA bases / fold-recenter caching for EmbProj operators
  - BacktestResult / Trade data classes shared across all evaluator paths
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation).

    Accurate to ~1.5e-9 vs scipy.stats.norm.ppf. 270x faster because
    it avoids scipy's object-dispatch overhead.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < 0.5:
        # Rational approximation for lower region
        t = math.sqrt(-2.0 * math.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return -(t - (c0 + c1 * t + c2 * t * t) /
                 (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (
            1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

import numpy as np
import pandas as pd

from layer2.grammar import (
    EMB_TYPES, FuncNode, GType, Node, Regime, Side, TermNode, to_str,
)
from layer2.terminal_stats import NORMALIZED_TERMINALS, normalize

# Set of GType values that represent typed vector embeddings ( contract).
# Used by the terminal/function dispatch to distinguish vector-valued terms
# from scalar-valued terms.
_EMB_TYPE_SET = frozenset(EMB_TYPES)

# ---------------------------------------------------------------------------
# NaN/Inf protection
# ---------------------------------------------------------------------------

def _safe_real(v: Any) -> float:
    """Coerce to float; NaN/Inf -> 0.0."""
    if v is None:
        return 0.0
    try:
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_bool(v: Any) -> bool:
    """Coerce to bool; NaN/None -> False."""
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return bool(v)


# ---------------------------------------------------------------------------
# PCA basis cache (v8) — lazy-loaded on first EmbProj_* invocation
# ---------------------------------------------------------------------------

_PCA_BASES_CACHE: Optional[Dict[str, Dict[str, np.ndarray]]] = None
# v9: walk-forward fold-fixed re-centering stats. Lazy-loaded on first
# EmbProj_* invocation in production. Module-level cache so test fixtures
# can `_reset_*_cache()` between tests.
_FOLD_RECENTER_STATS_CACHE: Optional[Dict[str, Dict[str, Dict[str, np.ndarray]]]] = None
_FOLD_RECENTER_SHA_CACHE: Optional[str] = None
_PCA_BASES_SHA_CACHE: Optional[str] = None


def _resolve_fold_ids_for_data(data: "pd.DataFrame") -> Optional[List[str]]:
    """Compute per-row fold_id list for the passed DataFrame.

    Uses `layer2.inference.fold_recenter.assign_fold_id`. Returns None if
    the DataFrame has no `date` column (unit-test fixtures fall through).
    Loud failure on any row whose date is outside all fold definitions —
    silent drop would corrupt fold-recenter dispatch.
    """
    if "date" not in data.columns:
        return None
    try:
        from layer2.inference.fold_recenter import (
            assign_fold_id, DEFAULT_FOLD_DEFINITIONS,
        )
    except Exception:
        return None
    out: List[str] = []
    for d in data["date"].astype(str).tolist():
        fid = assign_fold_id(d, DEFAULT_FOLD_DEFINITIONS)
        if fid is None:
            raise RuntimeError(
                f"row date={d!r} is outside all fold definitions: "
                f"{DEFAULT_FOLD_DEFINITIONS}. Either widen fold_definitions "
                f"or amend the parquet."
            )
        out.append(fid)
    return out


def _dominant_fold_id(fold_ids: Optional[List[str]]) -> Optional[str]:
    """Return the modal fold_id from a per-row list, or None.

    Used as part of the v9 fitness cache key. A `score_on_data(...)` call
    on a single split (train/val/test) has rows belonging to exactly one
    fold; the modal value is that fold_id. Cross-fold mixing — which the
    pre-reg does not contemplate — would produce a heterogeneous list and
    we'd still hash to the dominant fold (a deliberate degraded but loud
    behavior).
    """
    if not fold_ids:
        return None
    # Modal fold via Counter-like O(N) walk.
    from collections import Counter as _Counter
    counts = _Counter(fold_ids)
    return counts.most_common(1)[0][0]


def _load_pca_bases_cached(raise_on_missing: bool = True
                            ) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Load layer2/pca_bases.npz (+ manifest verification) with module-level caching.

    v8 (Fix 1, Code Reviewer C3 / Reality Checker R2): the EmbProj_* grammar
    operators MUST have bases to evaluate meaningfully. Silent-None fallback
    produces 0.0 scalars — indistinguishable from "GP correctly ignores
    EmbProj" (the exact pathology v8 is meant to fix). To make the failure
    mode loud:

    * `raise_on_missing=True` (the v9 DEFAULT — Bug 5 fix): raises
      RuntimeError immediately if bases are missing / SHA-mismatched. Both
      production paths (run_experiment, evaluator EmbProj dispatch) get
      loud-failure-on-missing without having to remember to opt in. This
      fails the 73h pilot at T+0, not T+72h with silent null results.
    * `raise_on_missing=False` (legacy permissive path, retained for tests):
      returns None, the EmbProj handler's zero fallback engages. Used only
      where bases are legitimately optional (e.g. unit-test fixtures that
      never invoke EmbProj_*).

    Returns dict keyed by typed-vector name each with
    {"components": (K, 384), "mean_": (384,)}.
    """
    global _PCA_BASES_CACHE
    if _PCA_BASES_CACHE is not None and _PCA_BASES_CACHE != {}:
        return _PCA_BASES_CACHE
    try:
        from layer2.inference.pca_bases import load_bases
        bases = load_bases()
        # Drop manifest key; evaluator only needs group arrays.
        out = {k: v for k, v in bases.items() if not k.startswith("_")}
        _PCA_BASES_CACHE = out
        return out
    except Exception as exc:
        _PCA_BASES_CACHE = {}  # negative-caches
        if raise_on_missing:
            raise RuntimeError(
                f"PCA bases unavailable (layer2/pca_bases.npz): {exc}\n"
                f"EmbProj_* grammar operators require a fitted basis. Run\n"
                f"    python -m layer2.inference.pca_bases\n"
                f"to regenerate, then re-invoke the experiment."
            ) from exc
        return None


def _reset_pca_bases_cache() -> None:
    """Test helper: clear the module-level cache so a subsequent
    _load_pca_bases_cached call re-reads from disk. Intended for tests
    that mutate the on-disk state."""
    global _PCA_BASES_CACHE, _PCA_BASES_SHA_CACHE
    _PCA_BASES_CACHE = None
    _PCA_BASES_SHA_CACHE = None


def _load_fold_recenter_cached(raise_on_missing: bool = True
                                ) -> Optional[Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Load layer2/fold_recenter_stats.npz with module-level caching.

    v9: every EmbProj_X_k output for a row in fold F is post-multiplied by
    the (mean, std) tied to F. Stats are fit walk-forward (val uses train
    rows; test uses train+val rows; train uses identity). See
    `layer2/inference/fold_recenter.py`.

    If `raise_on_missing=True` (production default), raises RuntimeError
    when the stats file is absent or sha-mismatched. The evaluator's
    EmbProj branch catches the resulting None as a defensive fallback to
    apply identity recenter, but only after the cache has populated; a
    cold first call always sees `raise_on_missing=True` and the production
    path fails loudly.

    L4-10 cross-check: at load time, the manifest's `pca_bases_sha256`
    and `l1_parquet_sha256` are checked against the SHAs of the
    actually-loaded `pca_bases.npz` and `l1_pilot.parquet`. On
    mismatch: RuntimeError — NEVER silent fallback.
    """
    global _FOLD_RECENTER_STATS_CACHE, _FOLD_RECENTER_SHA_CACHE
    if _FOLD_RECENTER_STATS_CACHE is not None and _FOLD_RECENTER_STATS_CACHE != {}:
        return _FOLD_RECENTER_STATS_CACHE
    try:
        from layer2.inference.fold_recenter import (
            load_fold_recenter_stats, STATS_PATH, STATS_MANIFEST_PATH,
            _file_sha256 as _fr_file_sha256,
        )
        from layer2.inference.pca_bases import _BASES_PATH as _PCA_BASES_PATH
        import json as _json
        stats = load_fold_recenter_stats(STATS_PATH, STATS_MANIFEST_PATH)
        # L4-10 cross-checks against the actually-loaded artifacts.
        if STATS_MANIFEST_PATH.exists():
            try:
                _man = _json.loads(STATS_MANIFEST_PATH.read_text())
            except Exception as exc:
                raise RuntimeError(
                    f"fold_recenter manifest at {STATS_MANIFEST_PATH} is "
                    f"unreadable: {exc!r}"
                ) from exc
            _expected_pca_sha = _man.get("pca_bases_sha256", "")
            if _expected_pca_sha:
                if _PCA_BASES_PATH.exists():
                    _actual_pca_sha = _fr_file_sha256(_PCA_BASES_PATH)
                    if _actual_pca_sha != _expected_pca_sha:
                        raise RuntimeError(
                            f"L4-10 sha cross-check failed: "
                            f"fold_recenter_manifest.pca_bases_sha256 "
                            f"({_expected_pca_sha[:16]}...) does not match "
                            f"actually-loaded layer2/pca_bases.npz "
                            f"({_actual_pca_sha[:16]}...). Re-fit "
                            f"fold_recenter after a PCA refit:\n"
                            f"    python -m layer2.inference.fold_recenter"
                        )
                else:
                    raise RuntimeError(
                        f"L4-10 sha cross-check: pca_bases.npz missing at "
                        f"{_PCA_BASES_PATH}, but fold_recenter_manifest pins "
                        f"pca_bases_sha256={_expected_pca_sha[:16]}..."
                    )
            _expected_parquet_sha = _man.get("l1_parquet_sha256", "")
            _parquet_path_str = _man.get("l1_parquet_path", "")
            if _expected_parquet_sha and _parquet_path_str:
                from pathlib import Path as _Path
                _parquet_path = _Path(_parquet_path_str)
                if _parquet_path.exists():
                    _actual_parquet_sha = _fr_file_sha256(_parquet_path)
                    if _actual_parquet_sha != _expected_parquet_sha:
                        raise RuntimeError(
                            f"L4-10 sha cross-check failed: "
                            f"fold_recenter_manifest.l1_parquet_sha256 "
                            f"({_expected_parquet_sha[:16]}...) does not match "
                            f"actually-loaded l1_pilot.parquet "
                            f"({_actual_parquet_sha[:16]}...). Re-fit "
                            f"fold_recenter after a parquet regeneration:\n"
                            f"    python -m layer2.inference.fold_recenter"
                        )
            _FOLD_RECENTER_SHA_CACHE = _man.get(
                "fold_recenter_stats_sha256", ""
            )
        else:
            _FOLD_RECENTER_SHA_CACHE = ""
        _FOLD_RECENTER_STATS_CACHE = stats
        return stats
    except Exception as exc:
        _FOLD_RECENTER_STATS_CACHE = {}
        if raise_on_missing:
            raise RuntimeError(
                f"fold_recenter_stats unavailable "
                f"(layer2/fold_recenter_stats.npz): {exc}\n"
                f"v9 EmbProj_* dispatch requires fold-recenter stats. Run\n"
                f"    python -m layer2.inference.fold_recenter\n"
                f"to fit + persist."
            ) from exc
        return None


def _reset_fold_recenter_cache() -> None:
    """Test helper: clear fold-recenter cache."""
    global _FOLD_RECENTER_STATS_CACHE, _FOLD_RECENTER_SHA_CACHE
    _FOLD_RECENTER_STATS_CACHE = None
    _FOLD_RECENTER_SHA_CACHE = None


def _load_pca_bases_sha() -> str:
    """Return the cached PCA bases SHA256, loaded from manifest on first
    call. Used as part of the v9 evaluator cache key tuple."""
    global _PCA_BASES_SHA_CACHE
    if _PCA_BASES_SHA_CACHE is not None:
        return _PCA_BASES_SHA_CACHE
    try:
        from layer2.inference.pca_bases import _MANIFEST_PATH
        import json as _json
        man = _json.loads(_MANIFEST_PATH.read_text())
        _PCA_BASES_SHA_CACHE = man.get("pca_bases_sha256", "")
    except Exception:
        _PCA_BASES_SHA_CACHE = ""
    return _PCA_BASES_SHA_CACHE


def _load_fold_recenter_sha() -> str:
    """Return the cached fold-recenter stats SHA256."""
    global _FOLD_RECENTER_SHA_CACHE
    if _FOLD_RECENTER_SHA_CACHE is None:
        _load_fold_recenter_cached(raise_on_missing=False)
    return _FOLD_RECENTER_SHA_CACHE or ""


# ---------------------------------------------------------------------------
# EvaluationContext — rolling buffer for temporal operators
# ---------------------------------------------------------------------------

class EvaluationContext:
    """Rolling buffer of terminal values.

    Holds two parallel buffers:
      - `buffer` — scalar terminal values (REAL/INT/etc.), used by Lag/Delta
        /CrossAbove/CrossBelow and the scalar-valued terminals (Channel 2/3/4).
      - `vector_buffer` — typed-vector terminals (EMB_*, D84 Channel 1), used by
        EmbNorm / EmbCos / EmbSub / EmbLag. Keyed by terminal name (e.g.
        "EMB_GRID"). Access via `get_vec(name, lag=0)`.
    """

    def __init__(self, max_lag: int = 30, emb_dim: int = 384,
                 current_fold_id: Optional[str] = None):
        # emb_dim default = d_model × n_layers = 128 × 3 = 384, matching the
        # canonical SSL-010-LOCAL iTransformer multi-layer output. Pre-v4
        # the default was 128 (single-layer assumption); zero-vector fallbacks
        # in EmbSub/EmbCos/EmbNorm then shape-mismatched against real 384-d
        # typed vectors, silently sending otherwise-valid strategies to
        # FAILED_FITNESS_SENTINEL. Keep as explicit kwarg for future
        # architectures with different emb_dim.
        #
        # current_fold_id (v9): walk-forward fold the evaluator is currently
        # scoring against. The EmbProj_* dispatch consults this to pick the
        # right per-fold per-PC re-centering stats. Optional in tests;
        # required at production EmbProj evaluation time. Default None
        # disables the recenter step (legacy behavior, train-equivalent).
        #
        # IMPORTANT (L8-1 boundary): the fold_id is propagated INTO this
        # context exclusively via `set_current_fold_id(...)` from the
        # backtester loop — NOT through the per-bar `bar_data` dict. Routing
        # via `bar_data` would expose `fold_id` to grammar terminal
        # expressions, allowing trees to learn fold-specific behavior
        # (a forbidden form of leakage). The grammar contains no terminal
        # named `fold_id`; only the EmbProj_* dispatch reads
        # `ctx.current_fold_id`.
        self.max_lag = max_lag
        self.emb_dim = emb_dim
        self.current_fold_id = current_fold_id
        self.buffer: Dict[str, deque] = {}
        self.vector_buffer: Dict[str, deque] = {}  # deque of np.ndarray
        self.bar_idx: int = 0
        self.current_regime: Optional[Regime] = None
        self._prev_eval_cache: Dict[str, float] = {}
        self._curr_eval_cache: Dict[str, float] = {}
        self._zero_vec = np.zeros(self.emb_dim, dtype=np.float32)

    def set_current_fold_id(self, fold_id: Optional[str]) -> None:
        """Set the fold_id for subsequent EmbProj_* dispatches (L8-1 boundary).

        Production callers (the backtester loops in SimpleBacktester /
        OptionsBacktester / MultiLegOptionsBacktester) MUST call this
        BEFORE evaluating any tree on a bar in fold F. The fold_id MUST
        NOT be smuggled through `bar_data` keys — see class docstring.
        """
        self.current_fold_id = (str(fold_id) if fold_id is not None else None)

    # Backwards-compat shim: a few legacy tests / fixtures referenced
    # `ctx.fold_id` directly. Forward to current_fold_id without exposing
    # a writable backdoor that bypasses set_current_fold_id.
    @property
    def fold_id(self) -> Optional[str]:
        return self.current_fold_id

    @fold_id.setter
    def fold_id(self, value: Optional[str]) -> None:
        self.current_fold_id = (str(value) if value is not None else None)

    def update(self, bar_data: dict) -> None:
        """Push new bar values into rolling buffer. Call once per bar BEFORE tree eval.

        Values may be scalars (go to `buffer`) or numpy vectors (go to
        `vector_buffer`). Dispatched by isinstance so callers don't have to
        route explicitly.

        L8-1 boundary: this method REJECTS any reserved `__fold_id` key in
        bar_data. The backtester must call `set_current_fold_id(...)`
        BEFORE this update; routing fold_id through bar_data is a forbidden
        leakage path and will raise immediately.
        """
        if "__fold_id" in bar_data:
            raise RuntimeError(
                "L8-1 boundary violation: bar_data contains reserved key "
                "'__fold_id'. The fold_id MUST be propagated via "
                "EvaluationContext.set_current_fold_id(), not bar_data."
            )
        self._prev_eval_cache = dict(self._curr_eval_cache)
        self._curr_eval_cache.clear()
        for name, val in bar_data.items():
            if isinstance(val, np.ndarray):
                # Typed vector (Channel 1) — route to vector buffer.
                if name not in self.vector_buffer:
                    self.vector_buffer[name] = deque(maxlen=self.max_lag + 1)
                # Store as float32 for memory; reshape 1-D vectors.
                self.vector_buffer[name].append(val.astype(np.float32, copy=False).ravel())
            else:
                if name not in self.buffer:
                    self.buffer[name] = deque(maxlen=self.max_lag + 1)
                raw = _safe_real(val)
                # L2 grammar fix: normalize REAL terminals to ~N(0,1) before
                # buffer insertion. This makes EphReal[-1,1] meaningful for
                # all terminals and eliminates scale-dominance in arithmetic.
                # terminal_stats.py is the SINGLE SOURCE OF TRUTH.
                if name in NORMALIZED_TERMINALS:
                    raw = normalize(name, raw)
                self.buffer[name].append(raw)
        # Regime key: PascalCase from Parquet (PredRegime) or snake_case fallback
        _rk = next((k for k in ("PredRegime", "predicted_regime") if k in bar_data), None)
        if _rk is not None:
            self.current_regime = Regime(max(0, min(int(bar_data[_rk]), 3)))
        self.bar_idx += 1

    def reset_session(self) -> None:
        """Clear rolling buffers at session (day) boundary.

        For 0DTE options, each trading day is independent — different contracts,
        different session. Temporal operators (Lag, Delta, EmbLag) should not
        reach into the previous session. Clearing buffers ensures lagged values
        return 0.0/zero-vector on the first bars of each new session.
        """
        for buf in self.buffer.values():
            buf.clear()
        for buf in self.vector_buffer.values():
            buf.clear()
        self._prev_eval_cache.clear()
        self._curr_eval_cache.clear()

    def get(self, name: str, lag: int = 0) -> float:
        """Get scalar terminal value at current bar (lag=0) or lagged bar."""
        buf = self.buffer.get(name)
        if buf is None or len(buf) == 0:
            return 0.0
        idx = len(buf) - 1 - lag
        return buf[idx] if idx >= 0 else 0.0

    def get_vec(self, name: str, lag: int = 0) -> np.ndarray:
        """Get typed-vector terminal value at current bar (lag=0) or lagged bar.

        Returns a zero vector (not None) when the buffer is empty or the lag
        reaches past the rolling window start — keeps downstream ops total
        on early bars.
        """
        buf = self.vector_buffer.get(name)
        if buf is None or len(buf) == 0:
            return self._zero_vec
        idx = len(buf) - 1 - lag
        if idx < 0:
            return self._zero_vec
        return buf[idx]

    def get_prev(self, name: str) -> float:
        """Get previous bar value."""
        buf = self.buffer.get(name)
        return buf[-2] if buf and len(buf) >= 2 else 0.0

    def cache_eval(self, key: str, value: float) -> None:
        """Cache subtree result for CrossAbove/CrossBelow prev-bar lookback."""
        self._curr_eval_cache[key] = value

    def get_prev_eval(self, key: str) -> Optional[float]:
        """Get previous bar's cached evaluation for a subtree key, or None if missing."""
        return self._prev_eval_cache.get(key, None)


# ---------------------------------------------------------------------------
# TreeEvaluator — recursive tree walker
# ---------------------------------------------------------------------------

class TreeEvaluator:
    """Evaluate a GP expression tree against an EvaluationContext."""

    def evaluate(self, node: Node, ctx: EvaluationContext) -> Any:
        if isinstance(node, TermNode):
            return self._eval_terminal(node, ctx)
        return self._eval_function(node, ctx)

    def _eval_terminal(self, node: TermNode, ctx: EvaluationContext) -> Any:
        if node.ret_type in (GType.REGIME, GType.SIDE):
            return node.value
        if node.ret_type == GType.INT:
            return int(node.value) if node.value is not None else 1
        # Typed vector terminals ( Channel 1) — EMB_SHARED / EMB_GRID / etc.
        # Retrieve the vector from the context's vector_buffer by terminal name.
        if node.ret_type in _EMB_TYPE_SET:
            return ctx.get_vec(node.name)
        # Ephemeral / literal REAL constants: return stored value (not from context)
        if node.value is not None:
            return _safe_real(node.value)
        return _safe_real(ctx.get(node.name))

    def _eval_function(self, node: FuncNode, ctx: EvaluationContext) -> Any:  # noqa: C901
        name = node.name
        ch = node.children

        # -- Temporal (v1: Lag/Delta only on TermNode first arg) --
        if name == "Lag":
            lag = max(0, min(int(self.evaluate(ch[1], ctx)), ctx.max_lag))
            if isinstance(ch[0], TermNode) and ch[0].ret_type == GType.REAL:
                return _safe_real(ctx.get(ch[0].name, lag))
            # FuncNode: compute the expression and store in a synthetic buffer
            # so lagging works for computed expressions (matches vectorized path
            # and codegen's _lag_expr).
            val = _safe_real(self.evaluate(ch[0], ctx))
            buf_key = f"_expr_{id(node) & 0xFFFFFFFF:08x}"
            if buf_key not in ctx.buffer:
                ctx.buffer[buf_key] = deque(maxlen=ctx.max_lag + 1)
            ctx.buffer[buf_key].append(val)
            return _safe_real(ctx.get(buf_key, lag))

        if name == "Delta":
            lag = max(1, min(int(self.evaluate(ch[1], ctx)), ctx.max_lag))
            if isinstance(ch[0], TermNode) and ch[0].ret_type == GType.REAL:
                return _safe_real(ctx.get(ch[0].name, 0) - ctx.get(ch[0].name, lag))
            # FuncNode: compute, buffer, and delta (matches vectorized + codegen)
            val = _safe_real(self.evaluate(ch[0], ctx))
            buf_key = f"_expr_{id(node) & 0xFFFFFFFF:08x}"
            if buf_key not in ctx.buffer:
                ctx.buffer[buf_key] = deque(maxlen=ctx.max_lag + 1)
            ctx.buffer[buf_key].append(val)
            return _safe_real(ctx.get(buf_key, 0) - ctx.get(buf_key, lag))

        # -- typed-vector operators (EmbNorm / EmbCos / EmbSub / EmbLag) --
        # Prefix-dispatched: the operator name is "Emb{Op}_{TYPE}" (e.g.
        # "EmbCos_EMB_GRID"). Vector values are numpy float32 arrays from
        # ctx.vector_buffer. Zero-vector fallback on lag-past-start is handled
        # inside ctx.get_vec, so these ops are total.
        if name.startswith("EmbNorm_"):
            v = self.evaluate(ch[0], ctx)
            v = v if isinstance(v, np.ndarray) else ctx._zero_vec
            return float(np.linalg.norm(v))

        if name.startswith("EmbCos_"):
            va = self.evaluate(ch[0], ctx)
            vb = self.evaluate(ch[1], ctx)
            va = va if isinstance(va, np.ndarray) else ctx._zero_vec
            vb = vb if isinstance(vb, np.ndarray) else ctx._zero_vec
            na = float(np.linalg.norm(va)); nb = float(np.linalg.norm(vb))
            if na < 1e-8 or nb < 1e-8:
                return 0.0
            return float(np.dot(va, vb) / (na * nb))

        if name.startswith("EmbSub_"):
            va = self.evaluate(ch[0], ctx)
            vb = self.evaluate(ch[1], ctx)
            va = va if isinstance(va, np.ndarray) else ctx._zero_vec
            vb = vb if isinstance(vb, np.ndarray) else ctx._zero_vec
            return va - vb

        # v8: EmbProj_{EMB_X}_{k} — project typed vector onto k-th PC of
        # pre-fit PCA basis. Prefix-dispatched; basis loaded lazily + cached
        # on first call. Returns scalar REAL. Zero-vector fallback inside
        # project() returns 0.0 (centered zero vector × any basis = zero).
        if name.startswith("EmbProj_"):
            v = self.evaluate(ch[0], ctx)
            v = v if isinstance(v, np.ndarray) else ctx._zero_vec
            # v9 Bug 4 fix: explicitly detect uninitialized buffer
            # (early-bar / lag-past-start) and return literal 0.0 rather
            # than projecting through `(0 - mean_) · components[k]`, which
            # produces a non-zero structural constant (e.g. EmbProj_EMB_VIX_1
            # ≈ +7.37 on a zero vector). The structural constant
            # contaminated the first ~max_lag bars of every dataset by
            # injecting a constant predictor into trees that hadn't yet
            # accumulated buffer. Behavior change is bounded — only fires
            # on early bars before ctx.update() has filled the buffer.
            if v is ctx._zero_vec:
                return 0.0
            # Defensive: also handle the all-zero ndarray case (e.g. v2
            # corpus EMB_FLOW_AGG which is emitted as zeros).
            if not np.any(v):
                return 0.0
            # Parse operator name: EmbProj_EMB_{GROUP}_{K}
            # (group may itself contain "_", e.g. EMB_FLOW_RAW)
            # → split on "_", take last field as K (int), rest minus prefix
            # as group name.
            suffix = name[len("EmbProj_"):]  # e.g. "EMB_FLOW_RAW_3"
            _dot = suffix.rfind("_")
            try:
                k = int(suffix[_dot + 1:])
                group = suffix[:_dot]  # e.g. "EMB_FLOW_RAW"
            except (ValueError, IndexError):
                return 0.0
            bases = _load_pca_bases_cached()
            if bases is None or group not in bases:
                return 0.0
            g = bases[group]
            components = g["components"]  # (K_group, 384)
            if k >= components.shape[0]:
                return 0.0
            centered = v - g["mean_"]
            # v9 P0-B: Mahalanobis normalization. Divide raw projection by
            # sqrt(explained_variance_[k]) so every EmbProj_X_k operator
            # has unit variance on the train slice. This collapses the
            # 5.8x-64.8x scale gap between EmbProj outputs and bypass
            # scalars that previously made `GT/LT(EmbProj_*, scalar)`
            # one-sided saturated. Pre-computed `pc_std_` is loaded from
            # the v9 pca_bases.npz; missing key raises in load_bases
            # rather than silently re-introducing the pathology.
            raw = float(np.dot(centered, components[k]))
            mahal = raw / float(g["pc_std_"][k])
            # v9 walk-forward fold-fixed re-centering. For the row's fold
            # F, subtract recenter_mean[F, group, k] and divide by
            # recenter_std[F, group, k]. Stats fit on strictly-preceding
            # folds (train fold uses identity). See
            # layer2/inference/fold_recenter.py for the no-leak guarantee.
            stats = _load_fold_recenter_cached(raise_on_missing=False)
            if stats is None or not stats:
                # No stats present (legacy / unit test) — return Mahalanobis-
                # only value (still valid, just not fold-corrected).
                return mahal
            fold_id = ctx.current_fold_id
            if fold_id is None:
                # Tests that don't pass a fold_id fall through to the
                # train-equivalent identity recenter. Production paths
                # MUST set ctx.current_fold_id (loud failure path below).
                return mahal
            if fold_id not in stats:
                raise RuntimeError(
                    f"fold_id={fold_id!r} not in fold_recenter_stats "
                    f"(known: {sorted(stats.keys())}). Either widen the "
                    f"fold_definitions in layer2/inference/fold_recenter.py "
                    f"or check the L1 parquet date coverage."
                )
            grp_stats = stats[fold_id].get(group)
            if grp_stats is None:
                raise RuntimeError(
                    f"fold_id={fold_id!r} stats missing group={group!r}. "
                    f"Re-fit fold_recenter stats."
                )
            mean_arr = grp_stats["mean"]
            std_arr = grp_stats["std"]
            if k >= len(mean_arr):
                # K mismatch — pca_bases / fold_recenter desync.
                raise RuntimeError(
                    f"fold_recenter stats group={group!r} has K={len(mean_arr)} "
                    f"but operator requested PC k={k}. Re-fit fold_recenter "
                    f"after a PCA refit."
                )
            return (mahal - float(mean_arr[k])) / float(std_arr[k])

        if name.startswith("EmbLag_"):
            # Child 0: typed-vector expression. For TermNode (the common case)
            # we route directly through the vector buffer with lag. For nested
            # expressions (EmbSub(EmbLag(EMB_GRID, 5), EMB_GRID)), the nested
            # EmbLag already produced a specific vector — we can't "relag" it
            # meaningfully, so fall back to evaluating ch[0] with no lag.
            lag = max(0, min(int(self.evaluate(ch[1], ctx)), ctx.max_lag))
            if isinstance(ch[0], TermNode) and ch[0].ret_type in _EMB_TYPE_SET:
                return ctx.get_vec(ch[0].name, lag)
            v = self.evaluate(ch[0], ctx)
            return v if isinstance(v, np.ndarray) else ctx._zero_vec

        # -- CrossAbove / CrossBelow --
        if name in ("CrossAbove", "CrossBelow"):
            a = _safe_real(self.evaluate(ch[0], ctx))
            b = _safe_real(self.evaluate(ch[1], ctx))
            ka, kb = to_str(ch[0]), to_str(ch[1])
            # Cache current bar's eval results BEFORE reading previous bar
            ctx.cache_eval(ka, a)
            ctx.cache_eval(kb, b)
            pa, pb = ctx.get_prev_eval(ka), ctx.get_prev_eval(kb)
            # No prior bar cached — can't determine crossing
            if pa is None or pb is None:
                return False
            if name == "CrossAbove":
                return a > b and pa <= pb
            return a < b and pa >= pb

        # -- Comparison / Boolean --
        if name == "GT":
            return _safe_real(self.evaluate(ch[0], ctx)) > _safe_real(self.evaluate(ch[1], ctx))
        if name == "LT":
            return _safe_real(self.evaluate(ch[0], ctx)) < _safe_real(self.evaluate(ch[1], ctx))
        if name == "AND":
            return _safe_bool(self.evaluate(ch[0], ctx)) and _safe_bool(self.evaluate(ch[1], ctx))
        if name == "OR":
            return _safe_bool(self.evaluate(ch[0], ctx)) or _safe_bool(self.evaluate(ch[1], ctx))
        if name == "NOT":
            return not _safe_bool(self.evaluate(ch[0], ctx))

        # -- Arithmetic --
        if name == "Add":
            return _safe_real(self.evaluate(ch[0], ctx)) + _safe_real(self.evaluate(ch[1], ctx))
        if name == "Sub":
            return _safe_real(self.evaluate(ch[0], ctx)) - _safe_real(self.evaluate(ch[1], ctx))
        if name == "Mul":
            return _safe_real(self.evaluate(ch[0], ctx)) * _safe_real(self.evaluate(ch[1], ctx))
        if name == "Div":  # analytic quotient: a / sqrt(1 + b^2)
            a = _safe_real(self.evaluate(ch[0], ctx))
            b = _safe_real(self.evaluate(ch[1], ctx))
            return a / math.sqrt(1.0 + b * b)
        if name == "Sqrt":  # protected: sqrt(abs(a))
            return math.sqrt(abs(_safe_real(self.evaluate(ch[0], ctx))))

        # -- Conditional --
        if name in ("IfThenElse", "IfSide"):
            cond = _safe_bool(self.evaluate(ch[0], ctx))
            return self.evaluate(ch[1] if cond else ch[2], ctx)

        # -- Regime --
        if name == "InRegime":
            return ctx.current_regime == self.evaluate(ch[0], ctx)
        if name == "RegimeIs":
            # NOTE: RegimeIs(A, B) comparing two literal Regime terminals is always
            # constant. Consider adding a CurrentRegime terminal to make this useful.
            return self.evaluate(ch[0], ctx) == self.evaluate(ch[1], ctx)

        # Unknown function fallback
        if node.ret_type == GType.BOOL:
            return False
        return Side.NEUTRAL if node.ret_type == GType.SIDE else 0.0


# ---------------------------------------------------------------------------
# BacktestResult & Trade
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    side: Side
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int
    exit_reason: str = "unknown"  # "signal", "max_hold", "eod", "session", "end_of_data"


@dataclass
class BacktestResult:
    returns: np.ndarray
    trades: List[Trade]
    equity_curve: np.ndarray
    max_drawdown: float
    sharpe: float
    sortino: float = 0.0  # annualized Sortino ratio (downside deviation only)
    total_trades: int = 0
    win_rate: float = 0.0
    n_days: int = 1  # number of trading days in the evaluation window
    exit_utilization: float = 0.0  # fraction of trades closed by exit signal (vs max_hold/eod)
    entry_fire_rate: float = 0.0  # fraction of eligible bars where entry signal fires
    # #6: fraction of FLAT eligible bars (no position open) where entry signal fires —
    # the unconditionality measure the tautology gate uses (day-selective holds score
    # low here even when entry_fire_rate is high). Defaults to entry_fire_rate's role
    # for any legacy path that doesn't set it.
    entry_fire_rate_flat: float = 0.0
    conditional_sharpe_gap: float = 0.0  # Sharpe(entry days) - Sharpe(all days)
    avg_position_size: float = 0.5  # mean size_signal at entry bars (for drawdown normalization)
    return_skew: float = 0.0  # skewness of daily returns (scipy.stats.skew convention)
    return_kurtosis: float = 0.0  # excess kurtosis of daily returns (scipy.stats.kurtosis convention)
    max_drawdown_uncapped: float = 0.0  # peak-to-trough of cum returns, NOT capped at 1.0
    # (max_drawdown is censored at 1.0; the objective/diagnostics need the true
    # depth to distinguish a -1x from a -8x drawdown). Still a PROXY statistic —
    # the equity floor de-levers, so this under-reports QC ruin; ruin is gated at
    # the QC deployment stage, not here. Use ONLY as within-template rank pressure.
    profit_factor: float = 0.0  # gross_profit / gross_loss over trades (>=1.0 = net edge)

    @property
    def psr(self) -> float:
        """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2014).

        P(SR > 0) accounting for skewness and kurtosis of returns.
        Returns probability [0, 1] that the true Sharpe is positive.
        """
        from scipy.stats import norm as _norm_dist, skew as _skew, kurtosis as _kurt
        if self.n_days < 5 or len(self.returns) == 0:
            return 0.0
        sr = self.sharpe
        n = self.n_days
        # H2 fix: compute moments on DAILY returns, not per-bar.
        # Per-bar returns have massive zero-padding (non-position bars) that
        # distorts skew/kurtosis. Daily aggregation matches the iid assumption.
        # If daily returns unavailable, fall back to per-bar with warning.
        daily = self.returns  # fallback
        if hasattr(self, '_daily_returns') and self._daily_returns is not None:
            daily = self._daily_returns
        elif len(self.returns) > n * 2:
            # Heuristic: if returns array is much longer than n_days, it's per-bar.
            # Aggregate by splitting into n_days chunks (approximate).
            chunk = len(self.returns) // max(n, 1)
            if chunk > 1:
                trimmed = self.returns[:chunk * n]
                daily = trimmed.reshape(n, chunk).sum(axis=1)
        s = float(_skew(daily)) if len(daily) > 2 else 0.0
        k = float(_kurt(daily)) if len(daily) > 3 else 0.0
        # SE(SR) adjusted for non-normality (Lo 2002, eq. 4):
        # Var(SR) = (1/n)[1 - γ₃·SR + (γ₄-1)/4·SR²]
        # scipy.stats.kurtosis returns excess kurtosis k = γ₄ - 3
        # so (γ₄-1)/4 = (k+3-1)/4 = (k+2)/4
        se_sr = math.sqrt((1.0 - s * sr + ((k + 2.0) / 4.0) * sr ** 2) / max(n, 1))
        if se_sr < 1e-10:
            return 1.0 if sr > 0 else 0.0
        return float(_norm_dist.cdf(sr / se_sr))


# ---------------------------------------------------------------------------
# SimpleBacktester
# ---------------------------------------------------------------------------

class SimpleBacktester:
    """Single-leg directional backtester for GP tree triplets.

    PnL = directional movement of underlying x notional, minus costs.
    Multi-leg spreads are deferred to templates.py.
    """

    def __init__(self, fee_per_leg: float = 1.30, spread_cost_bps: float = 5.0,
                 notional: float = 1000.0, max_bars_in_trade: int = 180,
                 warmup_bars: int = 15):
        self.fee_per_leg = fee_per_leg  # cost per leg; round-trip = 2x
        self.spread_cost_bps = spread_cost_bps
        self.notional = notional
        self.max_bars_in_trade = max_bars_in_trade
        self.warmup_bars = warmup_bars

    def run(self, entry_tree: Node, exit_tree: Node, side_tree: Node,
            data: pd.DataFrame, terminal_columns: Optional[List[str]] = None) -> BacktestResult:
        """Iterate bars, evaluate entry/exit/side trees, track PnL."""
        evaluator = TreeEvaluator()
        # v9: pre-compute per-row fold_id list so each ctx.update() can
        # inject the row's fold into the EvaluationContext for EmbProj
        # re-centering dispatch.
        _fold_ids = _resolve_fold_ids_for_data(data)
        ctx = EvaluationContext(max_lag=30, emb_dim=384)
        cols = terminal_columns or list(data.columns)
        price_col = next(
            (c for c in ("SPXClose", "spx_close", "close") if c in data.columns),
            "SPXClose"  # default — will produce warning if missing
        )
        n_bars = len(data)
        bar_returns = np.zeros(n_bars)
        trades: List[Trade] = []

        # Position state
        in_position = False
        pos_side: Side = Side.NEUTRAL
        entry_bar = 0
        entry_price = 0.0

        has_date = "date" in data.columns

        _prev_date = None
        for i in range(n_bars):
            row = data.iloc[i]
            bar_data = {c: row[c] for c in cols if c in row.index}
            # Session boundary reset: clear rolling buffers so Lag/EmbLag
            # cannot reach into the previous day (0DTE sessions are independent).
            if has_date:
                _cur_date = str(row["date"])
                if _prev_date is not None and _cur_date != _prev_date:
                    ctx.reset_session()
                _prev_date = _cur_date
            # L8-1: fold_id is set on the context BEFORE update/eval, NOT
            # routed through bar_data.
            if _fold_ids is not None:
                ctx.set_current_fold_id(_fold_ids[i])
            ctx.update(bar_data)
            price = float(row[price_col]) if price_col in row.index else 0.0

            if i < self.warmup_bars:
                continue  # warm-up: Lag/Delta need history before signals are valid

            # Day boundary — force close (0DTE cannot hold overnight)
            if in_position and has_date and i > 0:
                if str(data.iloc[i]["date"]) != str(data.iloc[i - 1]["date"]):
                    prev_price = float(data.iloc[i - 1][price_col])
                    full_pnl = self._trade_pnl(pos_side, entry_price, prev_price)
                    cost = self._round_trip_cost()
                    bar_returns[i - 1] += -cost / self.notional  # deduct cost on close bar
                    trades.append(Trade(entry_bar, i - 1, pos_side, entry_price, prev_price,
                                        full_pnl, i - 1 - entry_bar))
                    in_position = False

            if in_position:
                bars_held = i - entry_bar
                should_exit = (
                    _safe_bool(evaluator.evaluate(exit_tree, ctx))
                    or bars_held >= self.max_bars_in_trade
                )
                prev_price = float(data.iloc[i - 1][price_col])
                if should_exit:
                    # Record final bar's price movement (intermediate bars already captured)
                    # then deduct round-trip costs on this exit bar
                    final_bar_pnl = self._bar_pnl(pos_side, prev_price, price)
                    cost = self._round_trip_cost()
                    bar_returns[i] = (final_bar_pnl - cost) / self.notional
                    # Full trade PnL for the Trade record
                    full_pnl = self._trade_pnl(pos_side, entry_price, price)
                    trades.append(Trade(entry_bar, i, pos_side, entry_price, price, full_pnl, bars_held))
                    in_position = False
                else:
                    bar_returns[i] = self._bar_pnl(pos_side, prev_price, price) / self.notional
            else:
                if _safe_bool(evaluator.evaluate(entry_tree, ctx)):
                    side_val = evaluator.evaluate(side_tree, ctx)
                    if isinstance(side_val, Side) and side_val != Side.NEUTRAL:
                        in_position, pos_side = True, side_val
                        entry_bar, entry_price = i, price

        # Force-close open position at end of data
        if in_position:
            fp = float(data.iloc[-1][price_col])
            prev_price = float(data.iloc[-2][price_col]) if n_bars >= 2 else entry_price
            final_bar_pnl = self._bar_pnl(pos_side, prev_price, fp)
            cost = self._round_trip_cost()
            bar_returns[-1] = (final_bar_pnl - cost) / self.notional
            full_pnl = self._trade_pnl(pos_side, entry_price, fp)
            trades.append(Trade(entry_bar, n_bars - 1, pos_side, entry_price, fp, full_pnl, n_bars - 1 - entry_bar))

        eq = np.cumsum(bar_returns)
        bpd = self._derive_bars_per_day(data)
        _n_days = data["date"].nunique() if "date" in data.columns else max(1, len(data) // bpd)
        return BacktestResult(
            returns=bar_returns, trades=trades, equity_curve=eq,
            max_drawdown=self._max_drawdown(eq),
            sharpe=self._sharpe(bar_returns, bars_per_day=bpd),
            sortino=self._sortino(bar_returns, bars_per_day=bpd),
            total_trades=len(trades),
            win_rate=sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0.0,
            n_days=_n_days,
        )

    def compute_fitness(self, result: BacktestResult) -> Dict[str, float]:
        """Multi-objective fitness for NSGA-III (all maximized)."""
        from layer2.fitness import trade_count_score
        return {
            "sharpe": result.sharpe,
            "neg_max_drawdown": -result.max_drawdown,
            "trade_count_score": trade_count_score(result.total_trades, result.n_days),
            "win_rate": result.win_rate,
        }

    # -- helpers --

    def _round_trip_cost(self) -> float:
        """Total cost for one round-trip trade (2 legs + spread on entry & exit)."""
        return 2 * self.fee_per_leg + self.notional * self.spread_cost_bps / 10000.0 * 2

    def _trade_pnl(self, side: Side, entry_log_ret: float, exit_log_ret: float) -> float:
        """PnL for completed trade using additive log-return differences.

        SPXClose contains the reconstructed dollar price (open * exp(cumulative_log_return)).
        Used by MultiLegOptionsBacktester for spot pricing. Bar-over-bar change =
        log_ret[exit] - log_ret[entry].
        """
        direction = 1.0 if side == Side.CALL else -1.0
        raw = direction * (exit_log_ret - entry_log_ret) * self.notional
        return raw - self._round_trip_cost()

    def _bar_pnl(self, side: Side, prev_log_ret: float, curr_log_ret: float) -> float:
        """Unrealized bar-over-bar PnL using additive log-return differences.

        Since SPXClose is session_log_return (values near 0), the fractional
        change between bars is simply (curr - prev), already a log return.
        """
        direction = 1.0 if side == Side.CALL else -1.0
        return direction * (curr_log_ret - prev_log_ret) * self.notional

    @staticmethod
    def _max_drawdown(equity_curve: np.ndarray) -> float:
        """Max drawdown from peak as fraction of peak equity.

        Edge case: if peak equity <= 0 (strategy never went positive), returns
        the absolute max drawdown (not normalized) since fractional drawdown
        is undefined when the denominator is non-positive.
        """
        if len(equity_curve) == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        dd = running_max - equity_curve
        max_dd = float(np.max(dd))
        peak = float(np.max(running_max))
        if peak <= 0:
            # Fractional drawdown undefined; return absolute drawdown
            return max_dd
        return max_dd / peak

    @staticmethod
    def _sharpe(returns: np.ndarray, bars_per_day: int) -> float:
        """Annualized Sharpe from per-bar returns (252 trading days × bars_per_day).

        v9 P0-C / Bug 1 fix: `bars_per_day` is now a REQUIRED argument (no
        default). Pre-fix code hardcoded `bars_per_day=78` (5-minute-bar
        assumption); the L1 corpus actually has 12 bars/day at the v8/RF-4
        sampling cadence, producing a sqrt(78/12) ≈ 2.55x overstatement on
        every reported Sharpe. Callers MUST derive this from the input
        DataFrame (typically via `_derive_bars_per_day`) so a future bar-grid
        change can't silently re-introduce the artifact. Loud failure if
        omitted: missing-positional TypeError beats silent sqrt(78/12).
        """
        if len(returns) == 0 or np.std(returns) < 1e-10:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * math.sqrt(252 * bars_per_day))

    @staticmethod
    def _sortino(returns: np.ndarray, bars_per_day: int) -> float:
        """Annualized Sortino ratio: mean / downside_deviation × sqrt(252 × bpd).

        Downside deviation uses only negative returns (below zero target).
        More appropriate than Sharpe for asymmetric return distributions
        typical of options strategies (capped upside, tail downside).

        Reference: Sortino & van der Meer (1991). Downside risk.
        """
        if len(returns) == 0:
            return 0.0
        mean_ret = float(np.mean(returns))
        downside = returns[returns < 0]
        if len(downside) == 0:
            # All non-negative returns — Sortino is theoretically infinite;
            # cap at 10.0 to avoid numerical issues in GP fitness.
            return 10.0 if mean_ret > 1e-10 else 0.0
        downside_dev = float(np.sqrt(np.mean(downside ** 2)))
        if downside_dev < 1e-10:
            return 0.0
        return float(mean_ret / downside_dev * math.sqrt(252 * bars_per_day))

    @staticmethod
    def _derive_bars_per_day(data: pd.DataFrame) -> int:
        """Derive the number of bars per trading day from the input frame.

        Counts the modal `rows_per_date` value across the input. Asserts the
        derived value lands in [10, 405] — the canonical 30-min cadence is 12
        bars/day; the 5-min cadence (legacy) is 78; full 1-min sampling is
        390-405 (D74 CT/ET phantom corrected). Outside this range we suspect
        a data-shape bug and fail loudly rather than ship a silently wrong
        Sharpe annualization.

        Falls back to 12 (the v8/RF-4 canonical cadence) if the frame lacks
        a `date` column — but ONLY for ad-hoc backtests that wouldn't be
        reflected in the registered pipeline.
        """
        if "date" not in data.columns:
            # Ad-hoc path (e.g. unit tests). Use canonical 30-min cadence.
            return 12
        per_date = data.groupby("date").size()
        if len(per_date) == 0:
            return 12
        # Modal rows-per-date is the cadence. Use median for stability against
        # short days (e.g. half-trading-day before holiday).
        bpd = int(per_date.median())
        if not (10 <= bpd <= 405):
            raise RuntimeError(
                f"Derived bars_per_day={bpd} is outside the canonical "
                f"[10, 405] band. Distribution head: "
                f"{per_date.value_counts().head(3).to_dict()}. "
                f"Refusing to compute Sharpe with an unexpected bar grid; "
                f"investigate the L1 Parquet generation."
            )
        return bpd


# ---------------------------------------------------------------------------
# OptionsBacktester — simplified 0DTE options payoff model
# ---------------------------------------------------------------------------

class OptionsBacktester:
    """Container for option pricing static methods used by MultiLegOptionsBacktester.

    The `run()` and `compute_fitness()` methods were removed (dead code —
    production uses vectorized_backtest exclusively). Only the two static
    pricing methods remain, called by MultiLegOptionsBacktester._leg_value()
    and ._entry_costs().
    """

    BS_ATM_APPROX_FACTOR = 0.4  # Brenner & Subrahmanyam (1988)

    @staticmethod
    def _option_value(spot: float, strike: float, minutes_to_expiry: float,
                      iv: float, is_call: bool) -> float:
        """Simplified 0DTE option value: intrinsic + decaying time value.

        Uses the Brenner-Subrahmanyam (1988) ATM approximation with a linear
        moneyness decay. See BS_ATM_APPROX_FACTOR class attribute for details.
        """
        if minutes_to_expiry <= 0:
            return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        tau = minutes_to_expiry / (252 * 390)  # annualized
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        moneyness = abs(spot - strike) / spot if spot > 0 else 0.0
        time_value = spot * iv * (tau ** 0.5) * OptionsBacktester.BS_ATM_APPROX_FACTOR * max(0.0, 1.0 - 2.0 * moneyness)
        return intrinsic + time_value

    @staticmethod
    def _spread_cost(moneyness_pct: float) -> float:
        """Spread cost increases with distance from ATM (in dollars)."""
        base = 0.15  # ATM spread ~ $0.15
        return base * (1.0 + 3.0 * min(moneyness_pct, 0.05) / 0.05)



# ---------------------------------------------------------------------------
# MultiLegOptionsBacktester — templates (2-4 leg defined-risk structures)
# ---------------------------------------------------------------------------

class MultiLegOptionsBacktester:
    """Multi-leg 0DTE backtester for D88 template-based strategies.

    Consumes a `Template` (from `layer2.templates`) plus a DataFrame of L1
    outputs. Computes per-leg option values via the same simplified-BS
    approximation as `OptionsBacktester`, then aggregates net position P&L
    across all legs of the template.

    Key differences from single-leg `OptionsBacktester`:
      - Leg strikes derived from delta-targets (not from a single offset)
      - Net P&L = Σ over legs of (qty_sign × ratio × leg_value)
      - Fees and spread costs scale per-leg (4-leg iron condor = 4× both)
      - Side determined by template direction (no separate side_tree)
    """

    # L2 grammar fix: minimum strike separation for SPX $5 increments.
    MIN_STRIKE_SEPARATION = 5.0

    def __init__(self, template, notional: float = 1000.0,
                 max_bars_in_trade: int = 180, warmup_bars: int = 15,
                 fee_per_leg: float = 2.50, default_iv: float = 0.20,
                 default_minutes_to_expiry: float = 60.0,
                 min_bars_in_trade: int = 15,
                 slippage_pct: float = 0.003,
                 forbidden_terminal_columns: Optional[Tuple[str, ...]] = None):
        # Lazy import to avoid circular dep (templates.py imports from grammar.py
        # which doesn't depend on evaluator.py — but evaluator.py importing
        # templates.py would re-trigger the chain).
        from layer2.templates import Template, Leg
        if not isinstance(template, Template):
            raise TypeError(f"expected Template, got {type(template).__name__}")
        self.template = template
        self.notional = notional
        self.max_bars_in_trade = max_bars_in_trade
        self.warmup_bars = warmup_bars
        self.fee_per_leg = fee_per_leg
        self.default_iv = default_iv
        self.default_minutes_to_expiry = default_minutes_to_expiry
        # L2 grammar fix: minimum holding period prevents "enter and exit
        # immediately for tiny profit" which doesn't survive real execution.
        self.min_bars_in_trade = min_bars_in_trade
        # L2 grammar fix: slippage as fraction of option value per leg.
        self.slippage_pct = slippage_pct
        # F1 fix (2026-04-25): arm-aware filter for bar_data construction.
        # When the scalar-only arm runs, this is set to the encoder-derived
        # column list (PredRegime, RegimeProb*, probe scalars). The
        # backtester drops these columns from each row's bar_data BEFORE
        # ctx.update(), preventing EvaluationContext.update from setting
        # ctx.current_regime from PredRegime — which would otherwise leak
        # the L1 4-class regime label into scalar-only's IN_REGIME logic.
        # Even though F1 also strips IN_REGIME/REGIME_IS from
        # SCALAR_ONLY_FUNCTIONS, this filter is a defense-in-depth: it
        # ensures `ctx.current_regime` is None for scalar-only regardless
        # of grammar shape.
        self.forbidden_terminal_columns: Tuple[str, ...] = tuple(
            forbidden_terminal_columns or ()
        )

    @staticmethod
    def _delta_to_strike(spot: float, delta_target: float, iv: float,
                         minutes_to_expiry: float,
                         skew_slope_override: float = -0.15) -> float:
        """Compute strike that achieves the target BS delta UNDER SKEW on a 0DTE option.

        Uses Newton-Raphson iteration: start with ATM-IV strike, compute the
        actual delta under skew-adjusted IV at that strike, adjust, repeat.
        Converges in 3-5 iterations since the ATM-IV starting point is close.

        Pure-math normal distribution (no scipy) for 270x speedup per call.
        L2 grammar fix: round to $5 increments (SPX option strike grid).

        skew_slope_override: Skew slope for strike-finding.
            History: -0.35 (original) → -0.22 (calibrated with pull-to-ATM)
            → 0.0 (flat, overcorrected) → -0.15 (2026-05-19, post-pull-to-ATM
            removal). The 0.0 value priced both legs at same IV, producing
            11-14% credit/width ratios with 86%+ break-even win rates.
            -0.15 is conservative: real SPX put skew is steeper but
            we don't want to overstate credit. Credit correction factors
            may need recalibration after this change.
        """
        from layer2.evaluator_vectorized import _skew_iv

        if iv <= 0 or minutes_to_expiry <= 0:
            raw = spot * (1.0 - delta_target * 0.01)  # tiny offset fallback
        else:
            tau = minutes_to_expiry / (252 * 390)
            is_call = delta_target > 0

            # Initial guess: ATM-IV based inversion (ignoring skew)
            sigma_sqrt_tau = iv * math.sqrt(max(tau, 1e-12))
            if is_call:
                d_clamped = max(min(delta_target, 0.999), 0.001)
                d1 = _norm_ppf(d_clamped)
            else:
                d_clamped = max(min(1.0 + delta_target, 0.999), 0.001)
                d1 = _norm_ppf(d_clamped)
            raw = spot * math.exp(-d1 * sigma_sqrt_tau + 0.5 * sigma_sqrt_tau ** 2)

            # Newton-Raphson: refine strike so that BS delta under skew-adjusted IV
            # matches the target delta. 5 iterations is more than sufficient.
            _SQRT2 = math.sqrt(2.0)
            _INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
            for _ in range(5):
                skew_vol = _skew_iv(iv, spot, raw, minutes_to_expiry,
                                    skew_slope=skew_slope_override)
                s_sqrt_t = skew_vol * math.sqrt(max(tau, 1e-12))
                if s_sqrt_t < 1e-9:
                    break
                d1_skew = (math.log(spot / max(raw, 1.0)) + 0.5 * s_sqrt_t ** 2) / s_sqrt_t
                # Pure-math norm.cdf: 0.5 * erfc(-x / sqrt(2))
                cdf_d1 = 0.5 * math.erfc(-d1_skew / _SQRT2)
                if is_call:
                    actual_delta = cdf_d1
                else:
                    actual_delta = cdf_d1 - 1.0
                err = actual_delta - delta_target
                if abs(err) < 0.001:
                    break
                # Pure-math norm.pdf
                pdf_d1 = _INV_SQRT_2PI * math.exp(-0.5 * d1_skew * d1_skew)
                d_delta_dK = -pdf_d1 / (max(raw, 1.0) * s_sqrt_t)
                if abs(d_delta_dK) < 1e-12:
                    break
                raw = raw - err / d_delta_dK
                raw = max(raw, spot * 0.7)
                raw = min(raw, spot * 1.3)

        # Pull-to-ATM correction REMOVED (2026-05-19): the original per-template
        # credit correction factors (BPC=0.76, BCC=0.71, IC=0.79, IB=0.86)
        # were calibrated on 2026-05-15 with pull-to-ATM already active.
        # Applying both double-counts the same friction, pushing break-even
        # win rate from ~65% to ~84%. Factors recalibrated post-removal to
        # BPC=0.80, BCC=0.73, IC=0.78, IB=0.81 (630 matched trades).

        # Round to nearest $5 increment (SPX option strike grid)
        return round(raw / 5.0) * 5.0

    @staticmethod
    def _enforce_strike_separation(strikes: List[float],
                                   legs: tuple,
                                   min_sep: float = 5.0) -> List[float]:
        """Enforce minimum $5 separation between all leg strikes.

        For 0DTE options with <60 min to expiry, delta-based strike selection
        can put adjacent legs on the same $5 strike. This function sorts
        strikes, then spreads them apart so that consecutive sorted strikes
        differ by at least min_sep.

        Algorithm: sort legs by their INTENDED strike order (using
        delta_to_strike's monotonic mapping: higher delta_target → higher
        strike for calls; more negative delta_target → lower strike for
        puts). Then separate in that order, preserving the template's
        structural semantics so credit/debit spread identity is never
        inverted.
        """
        if len(strikes) <= 1:
            return strikes

        n = len(strikes)
        # Sort by (strike, -delta_target) to break ties correctly.
        # The intended strike ordering (lowest to highest) is:
        # further-OTM put < closer-to-ATM put < closer-to-ATM call < further-OTM call
        # Using NEGATIVE delta as tiebreaker achieves this:
        # puts: -(-0.05)=+0.05 < -(-0.15)=+0.15 → -0.05 (lower strike) first ✓
        # calls: -(+0.15)=-0.15 < -(+0.05)=-0.05 → +0.15 (lower strike) first ✓
        # Fall back to index-based tiebreaking if legs tuple is too short.
        if len(legs) >= n:
            indexed = sorted(
                range(n),
                key=lambda j: (strikes[j], -legs[j].delta_target),
            )
        else:
            indexed = sorted(range(n), key=lambda j: (strikes[j], j))
        sorted_strikes = [strikes[j] for j in indexed]

        # Walk upward, ensuring min_sep between consecutive strikes
        for i in range(1, n):
            if sorted_strikes[i] - sorted_strikes[i - 1] < min_sep - 0.01:
                sorted_strikes[i] = sorted_strikes[i - 1] + min_sep
            # Re-round to $5 grid
            sorted_strikes[i] = round(sorted_strikes[i] / 5.0) * 5.0
            # After rounding, may need to push further
            if sorted_strikes[i] - sorted_strikes[i - 1] < min_sep - 0.01:
                sorted_strikes[i] = sorted_strikes[i - 1] + min_sep
                sorted_strikes[i] = math.ceil(sorted_strikes[i] / 5.0) * 5.0

        # Restore original order
        result = [0.0] * n
        for i, orig_idx in enumerate(indexed):
            result[orig_idx] = sorted_strikes[i]
        return result

    def _leg_value(self, leg, spot: float, strike: float,
                   minutes_to_expiry: float, iv: float) -> float:
        """Per-leg option value (single contract) using the simplified-BS model
        from the original OptionsBacktester."""
        is_call = (leg.option_type == "call")
        return OptionsBacktester._option_value(
            spot, strike, minutes_to_expiry, iv, is_call
        )

    def _net_position_value(self, spot: float, strikes: List[float],
                            minutes_to_expiry: float, iv: float) -> float:
        """Net per-contract value of all legs combined.
        Σ over legs: qty_sign × ratio × leg_value.
        """
        total = 0.0
        for leg, strike in zip(self.template.legs, strikes):
            val = self._leg_value(leg, spot, strike, minutes_to_expiry, iv)
            total += leg.qty_sign * leg.ratio * val
        return total

    def _gross_position_value(self, spot: float, strikes: List[float],
                              minutes_to_expiry: float, iv: float) -> float:
        """Gross per-contract value: Σ over legs of |qty_sign × ratio × leg_value|.

        v9 Bug 2 fix: used to size positions so leverage is bounded by the
        ABSOLUTE premium exposure rather than the (often near-zero) NET
        premium. For defined-risk structures (e.g. ATM 0DTE call spread),
        the net entry value is typically near zero — long ATM call ~$5,
        short OTM call ~$5 ⇒ net ~$0.05. Sizing on `notional / abs(net) =
        notional / 0.10` produced 10× notional contracts, leading to the
        ~200% leverage observed in P0-C investigation. Sizing on gross
        instead keeps `n_contracts × gross ≤ notional`, i.e. the position's
        absolute exposure stays bounded by the capital budget.
        """
        total = 0.0
        for leg, strike in zip(self.template.legs, strikes):
            val = self._leg_value(leg, spot, strike, minutes_to_expiry, iv)
            total += abs(leg.qty_sign * leg.ratio * val)
        return total

    def _entry_costs(self, spot: float, strikes: List[float],
                     minutes_to_expiry: float = 60.0, iv: float = 0.20) -> float:
        """Entry transaction cost: per-leg fee + per-leg half-spread + slippage.

        L2 grammar fix: added slippage_pct (0.3% of option value per leg)
        to make proxy costs more realistic. Fee increased from $1.30 to $2.50.
        """
        cost = 0.0
        for leg, strike in zip(self.template.legs, strikes):
            cost += self.fee_per_leg * leg.ratio
            moneyness_pct = abs(spot - strike) / spot if spot > 0 else 0.0
            cost += OptionsBacktester._spread_cost(moneyness_pct) * leg.ratio
            # L2 grammar fix: slippage proportional to option value
            is_call = (leg.option_type == "call")
            leg_val = OptionsBacktester._option_value(
                spot, strike, minutes_to_expiry, iv, is_call
            )
            cost += self.slippage_pct * leg_val * leg.ratio
        return cost

    def run(self, entry_tree: Node = None, exit_tree: Node = None,
            size_tree: Node = None, data: pd.DataFrame = None,
            terminal_columns: Optional[List[str]] = None) -> BacktestResult:
        """Run the template through the data, evolving entry/exit/size via the
        provided trees (or template seeds if None).
        """
        if data is None:
            raise ValueError("data DataFrame required")
        # Reset per-run state to avoid stale values from prior run() calls
        for _attr in ('_session_open', '_session_date', '_session_high_ml',
                       '_session_low_ml', '_prev_day_vix_ml', '_curr_day_vix_ml',
                       '_vix_date'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        # Default to template seed trees if not provided
        if entry_tree is None:
            entry_tree = self.template.entry_seed
        if exit_tree is None:
            exit_tree = self.template.exit_seed
        if size_tree is None:
            size_tree = self.template.size_seed

        evaluator = TreeEvaluator()
        # v9: per-row fold_id list for EmbProj recenter dispatch.
        _fold_ids = _resolve_fold_ids_for_data(data)
        ctx = EvaluationContext(max_lag=30, emb_dim=384)

        if terminal_columns is None:
            # Use the L2 IO canonical column list to avoid passing date/index cols
            try:
                from layer2.io import L1_TERMINAL_COLUMNS
                cols = [c for c in L1_TERMINAL_COLUMNS if c in data.columns]
            except ImportError:
                cols = list(data.columns)
        else:
            cols = list(terminal_columns)
        # F1 fix (2026-04-25): drop forbidden columns (encoder-derived
        # signals that the scalar-only arm must NOT consume). For real-l1
        # / shuffled-l1 this is empty; for scalar-only it is
        # PredRegime/PredRV15/PredRV30/PredSpread/RegimeProb0..3 (all 4
        # typed-vector cols and probes are encoder-derived). With these
        # excluded from bar_data, ctx.current_regime stays None and
        # scalar-only's IN_REGIME / REGIME_IS operators (also stripped at
        # grammar level by F1) cannot consume L1's 4-class regime label.
        if self.forbidden_terminal_columns:
            forbidden = set(self.forbidden_terminal_columns)
            cols = [c for c in cols if c not in forbidden]

        price_col = next(
            (c for c in ("SPXClose", "spx_close", "close") if c in data.columns),
            "SPXClose"
        )
        iv_col = next(
            (c for c in ("ATM_IV", "atm_iv", "ATMIV") if c in data.columns), None
        )
        # M1 fix: use the MinutesToClose column directly if present instead of
        # decrementing minutes_left by 1.0 per bar. The old decrement assumed
        # 1-minute bars; L1 data can be 1-min or 5-min, and at end-of-day the
        # column is the authoritative time-to-close signal.
        has_minutes_col = "MinutesToClose" in data.columns
        has_date = "date" in data.columns

        # Compute session length and bar cadence for BarOfDay synthesis
        if has_minutes_col:
            _mtc_vals = data["MinutesToClose"].values
            self._session_len = float(np.nanmax(_mtc_vals)) if len(_mtc_vals) > 0 else 390.0
            if self._session_len < 100:
                self._session_len = 390.0
            # Derive cadence from consecutive intra-day MTC diffs
            self._bar_cadence = 5.0  # default for L1 5-bar stride
            if has_date and len(data) > 2:
                _dates = data["date"].values
                for _j in range(1, min(len(data), 100)):
                    if str(_dates[_j]) == str(_dates[_j - 1]):
                        _diff = abs(float(_mtc_vals[_j - 1]) - float(_mtc_vals[_j]))
                        if 0.1 < _diff < 100:
                            self._bar_cadence = _diff
                            break
        else:
            self._session_len = 390.0
            self._bar_cadence = 5.0

        n_bars = len(data)
        bar_returns = np.zeros(n_bars)
        trades: List[Trade] = []

        # Position state
        in_position = False
        entry_bar = 0
        entry_spot = 0.0
        strikes: List[float] = []
        entry_net_value = 0.0     # net per-contract value at entry
        entry_n_contracts = 0.0   # how many "units" of the structure
        entry_cost = 0.0
        minutes_left = 0.0
        # No explicit direction_sign: `_net_position_value = Σ qty_sign ×
        # ratio × leg_value` already embeds long/short semantics via
        # qty_sign, so the holder's mark-to-market P&L over time is simply
        # `curr_net_value - entry_net_value` regardless of whether the
        # template is credit or debit. The previous `direction_sign =
        # -1 if is_credit else +1` double-counted the sign for credit
        # structures (B1 fix). `Template.is_credit` is retained as
        # structural metadata only — it is NOT consulted here.

        # BarOfDay: compute from bar_position column if present, else from
        # MinutesToClose. Gives GP direct access to time-of-day position.
        _has_bar_of_day = "BarOfDay" in data.columns
        _has_bar_position = "bar_position" in data.columns

        _prev_date_ml = None
        for i in range(n_bars):
            row = data.iloc[i]
            bar_data = {c: row[c] for c in cols if c in row.index}
            # Session boundary reset (0DTE: each day is independent)
            if has_date:
                _cur_date_ml = str(row["date"])
                if _prev_date_ml is not None and _cur_date_ml != _prev_date_ml:
                    ctx.reset_session()
                _prev_date_ml = _cur_date_ml
            # Synthesize BarOfDay if not already in the data
            if not _has_bar_of_day:
                if _has_bar_position:
                    bar_data["BarOfDay"] = float(row["bar_position"])
                elif has_minutes_col:
                    # BarOfDay from MTC: use session length from max MTC in data.
                    # SPX 0DTE: 9:30-16:15 ET = 405 min, so max MTC ≈ 404.
                    _mtc = float(row["MinutesToClose"]) if "MinutesToClose" in row.index else 0.0
                    _sess = getattr(self, '_session_len', 390.0)
                    bar_data["BarOfDay"] = (_sess - _mtc) / self._bar_cadence
                else:
                    bar_data["BarOfDay"] = float(i)
            # Synthesize missing terminals that the grammar defines
            spot = float(row[price_col]) if price_col in row.index else 0.0
            iv = float(row[iv_col]) if iv_col and iv_col in row.index else self.default_iv
            if "SessionReturn" not in bar_data and spot > 0:
                if has_date:
                    _cur = str(row["date"])
                    if not hasattr(self, '_session_open') or _cur != getattr(self, '_session_date', ''):
                        self._session_open = spot
                        self._session_date = _cur
                        self._session_high_ml = spot
                        self._session_low_ml = spot
                    self._session_high_ml = max(self._session_high_ml, spot)
                    self._session_low_ml = min(self._session_low_ml, spot)
                    bar_data["SessionReturn"] = (spot - self._session_open) / self._session_open
                    _range = self._session_high_ml - self._session_low_ml
                    if "SessionPosition" not in bar_data:
                        bar_data["SessionPosition"] = (spot - self._session_low_ml) / _range if _range > 0.01 else 0.5
            if "VIXChange" not in bar_data:
                _vix = float(row.get("VIXSpot", 0.0)) if "VIXSpot" in row.index else 0.0
                if not hasattr(self, '_prev_day_vix_ml'):
                    self._prev_day_vix_ml = _vix
                if has_date:
                    _cur = str(row["date"])
                    if not hasattr(self, '_vix_date') or _cur != self._vix_date:
                        self._prev_day_vix_ml = getattr(self, '_curr_day_vix_ml', _vix)
                        self._curr_day_vix_ml = _vix
                        self._vix_date = _cur
                bar_data["VIXChange"] = _vix - self._prev_day_vix_ml
            if "RegimeAboveLow" not in bar_data and "PredRegime" in bar_data:
                _r = int(bar_data["PredRegime"])
                bar_data["RegimeAboveLow"] = 1.0 if _r >= 1 else 0.0
                bar_data["RegimeIsHigh"] = 1.0 if _r >= 2 else 0.0
                bar_data["RegimeIsPremium"] = 1.0 if _r == 3 else 0.0
            if _fold_ids is not None:
                ctx.set_current_fold_id(_fold_ids[i])
            ctx.update(bar_data)

            if i < self.warmup_bars:
                continue

            # Day-boundary force-close
            if in_position and has_date and i > 0:
                if str(data.iloc[i]["date"]) != str(data.iloc[i - 1]["date"]):
                    prev_spot = float(data.iloc[i - 1][price_col])
                    close_val = self._net_position_value(
                        prev_spot, strikes, 0.0, iv
                    )
                    pnl_per_unit = (close_val - entry_net_value)
                    pnl = pnl_per_unit * entry_n_contracts
                    exit_cost = self._entry_costs(prev_spot, strikes)
                    bar_returns[i - 1] += (pnl - exit_cost) / self.notional
                    trades.append(Trade(
                        entry_bar, i - 1, Side.NEUTRAL, entry_spot, prev_spot,
                        pnl - exit_cost - entry_cost, i - 1 - entry_bar,
                    ))
                    in_position = False

            if in_position:
                bars_held = i - entry_bar
                # M1 fix: prefer the MinutesToClose column (authoritative,
                # bar-width-agnostic) over a fixed 1.0 decrement per bar.
                # Falls back to decrement only when the column is missing.
                if has_minutes_col:
                    minutes_left = max(0.0, float(row["MinutesToClose"]))
                    prev_minutes_left = max(0.0, float(data.iloc[i - 1]["MinutesToClose"]))
                else:
                    minutes_left = max(0.0, minutes_left - 1.0)
                    prev_minutes_left = minutes_left + 1.0
                curr_val = self._net_position_value(spot, strikes, minutes_left, iv)
                prev_spot = float(data.iloc[i - 1][price_col])
                prev_val = self._net_position_value(
                    prev_spot, strikes, prev_minutes_left, iv
                )
                bar_pnl_per_unit = (curr_val - prev_val)
                bar_pnl = bar_pnl_per_unit * entry_n_contracts

                # L2 grammar fix: min_bars_in_trade prevents premature exits
                # that don't survive real execution (enter-and-exit-immediately).
                should_exit = (
                    (bars_held >= self.min_bars_in_trade
                     and _safe_bool(evaluator.evaluate(exit_tree, ctx)))
                    or bars_held >= self.max_bars_in_trade
                    or minutes_left <= 0
                )
                if should_exit:
                    total_pnl_per_unit = (curr_val - entry_net_value)
                    total_pnl = total_pnl_per_unit * entry_n_contracts
                    exit_cost = self._entry_costs(spot, strikes)
                    bar_returns[i] = (bar_pnl - exit_cost) / self.notional
                    trades.append(Trade(
                        entry_bar, i, Side.NEUTRAL, entry_spot, spot,
                        total_pnl - exit_cost - entry_cost, bars_held,
                    ))
                    in_position = False
                else:
                    bar_returns[i] = bar_pnl / self.notional
            else:
                # Entry decision
                if _safe_bool(evaluator.evaluate(entry_tree, ctx)):
                    size_val = _safe_real(evaluator.evaluate(size_tree, ctx))
                    # Clamp size to [0, 1]
                    size = max(0.0, min(1.0, size_val))
                    if size <= 1e-6:
                        continue  # zero-size entry skipped
                    in_position = True
                    entry_bar = i
                    entry_spot = spot
                    # M1 fix: use MinutesToClose at entry bar if available,
                    # else fall back to the configured default (pre-fix only
                    # behaviour).
                    if has_minutes_col:
                        minutes_left = max(0.0, float(row["MinutesToClose"]))
                    else:
                        minutes_left = self.default_minutes_to_expiry
                    # Resolve strikes from delta-targets + enforce minimum separation
                    strikes = [
                        self._delta_to_strike(spot, leg.delta_target, iv, minutes_left)
                        for leg in self.template.legs
                    ]
                    strikes = self._enforce_strike_separation(
                        strikes, self.template.legs, self.MIN_STRIKE_SEPARATION
                    )
                    entry_net_value = self._net_position_value(
                        spot, strikes, minutes_left, iv
                    )
                    # v9 Bug 2 fix: size contracts by GROSS per-unit value
                    # (Σ |qty × ratio × leg_value|), not net. Pre-fix sized on
                    # `max(abs(net), 0.10)` which floored to 0.10 for
                    # defined-risk structures with near-zero net entry value
                    # (long ATM call $5 + short OTM call $5 ⇒ net ~$0.05).
                    # Floor engagement gave n_contracts = notional / 0.10 =
                    # 10× notional, and a 0.5% intra-bar SPX move yielded
                    # mean trade PnL ≈ $4061 on $1000 notional. Sizing on
                    # gross keeps `n_contracts × gross ≤ notional × size`,
                    # i.e. effective leverage ≤ 1.0 of capital. The
                    # MAX_CONTRACTS hard cap is retained as a belt-and-
                    # suspenders bound for cases where gross is itself
                    # tiny (deep-OTM 0DTE wings ~$0.05 per leg).
                    entry_gross_value = self._gross_position_value(
                        spot, strikes, minutes_left, iv
                    )
                    # L2 grammar fix: raise gross-value floor from $0.50 to
                    # $2.00 and add 2x leverage cap. The $0.50 floor allowed
                    # excessive leverage on cheap wings.
                    abs_val = max(entry_gross_value, 2.0)
                    MAX_CONTRACTS = self.notional / 2.0
                    entry_n_contracts = min(
                        (self.notional * size) / abs_val,
                        MAX_CONTRACTS,
                    )
                    # L2 grammar fix: leverage cap — total notional exposure
                    # must not exceed 2x capital.
                    max_notional = entry_n_contracts * abs_val
                    if max_notional > 2.0 * self.notional:
                        entry_n_contracts = int(2.0 * self.notional / abs_val)
                    entry_cost = self._entry_costs(spot, strikes, minutes_left, iv)
                    bar_returns[i] -= entry_cost / self.notional

        # Force-close at end of data
        if in_position:
            last_spot = float(data.iloc[-1][price_col])
            close_val = self._net_position_value(last_spot, strikes, 0.0, iv)
            total_pnl = (close_val - entry_net_value) * entry_n_contracts
            exit_cost = self._entry_costs(last_spot, strikes)
            bar_returns[-1] += (total_pnl - exit_cost) / self.notional
            trades.append(Trade(
                entry_bar, n_bars - 1, Side.NEUTRAL, entry_spot, last_spot,
                total_pnl - exit_cost - entry_cost, n_bars - 1 - entry_bar,
            ))

        eq = np.cumsum(bar_returns)
        bpd = SimpleBacktester._derive_bars_per_day(data)
        _n_days = data["date"].nunique() if "date" in data.columns else max(1, len(data) // bpd)
        return BacktestResult(
            returns=bar_returns, trades=trades, equity_curve=eq,
            max_drawdown=SimpleBacktester._max_drawdown(eq),
            sharpe=SimpleBacktester._sharpe(bar_returns, bars_per_day=bpd),
            sortino=SimpleBacktester._sortino(bar_returns, bars_per_day=bpd),
            total_trades=len(trades),
            win_rate=sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0.0,
            n_days=_n_days,
        )

    def compute_fitness(self, result: BacktestResult) -> Dict[str, float]:
        """Multi-objective fitness compatible with NSGA-III."""
        from layer2.fitness import trade_count_score
        return {
            "sharpe": result.sharpe,
            "neg_max_drawdown": -result.max_drawdown,
            "trade_count_score": trade_count_score(result.total_trades, result.n_days),
            "win_rate": result.win_rate,
        }


# ---------------------------------------------------------------------------
# GP Train/Val/Test Split
# ---------------------------------------------------------------------------

def split_gp_data(data: pd.DataFrame,
                  train_end: str = "2024-09-30",
                  val_end: str = "2025-01-31",
                  embargo_days: int = 5) -> dict:
    """Split L1Output DataFrame into train/val/test for GP evolution.

    Train: 2022-09-19 to train_end (in-distribution for L1 encoder)
    Val: train_end + embargo to val_end (L1 val period — tune GP hyperparams)
    Test: val_end + embargo to end (true OOT — report only this)

    Args:
        data: DataFrame with a 'date' column (string YYYY-MM-DD format).
        train_end: Last date (inclusive) for training split.
        val_end: Last date (inclusive) for validation split.
        embargo_days: Number of calendar days to skip between splits.

    Returns:
        Dict with keys 'train', 'val', 'test', each a DataFrame.
    """
    dates = pd.to_datetime(data["date"])

    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)
    embargo = pd.Timedelta(days=embargo_days)

    train_mask = dates <= train_end_dt
    val_mask = (dates > train_end_dt + embargo) & (dates <= val_end_dt)
    test_mask = dates > val_end_dt + embargo

    splits = {
        "train": data.loc[train_mask].reset_index(drop=True),
        "val": data.loc[val_mask].reset_index(drop=True),
        "test": data.loc[test_mask].reset_index(drop=True),
    }

    print(f"GP data split — Train: {len(splits['train'])} rows, "
          f"Val: {len(splits['val'])} rows, Test: {len(splits['test'])} rows")
    print(f"  Train: ... to {train_end}")
    print(f"  Val:   {train_end} + {embargo_days}d embargo to {val_end}")
    print(f"  Test:  {val_end} + {embargo_days}d embargo to end")

    return splits
