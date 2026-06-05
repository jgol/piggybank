# ===================================================================
# 1. IMPORTS AND DEVICE
# ===================================================================

import json, io, base64, gc, os
from datetime import date, timedelta

missing = []
for pkg in ("numpy", "torch", "sklearn", "matplotlib"):
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"Missing packages: {missing}")
    raise ImportError(f"Missing: {missing}")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

_qb = None
LOCAL_MODE = bool(os.environ.get("LOCAL_MODE"))

def get_qb():
    global _qb
    if _qb is None:
        if LOCAL_MODE:
            # Route ObjectStore calls to the local-disk shim when training
            # outside QC. The shim serves layer1_feat_v3/* keys from the
            # reassembled corpus bundle and backs all other keys on disk
            # under raw_data/local_store/.
            from layer1.training.local_backend import get_local_qb
            _qb = get_local_qb()
        else:
            from QuantConnect.Research import QuantBook
            _qb = QuantBook()
    return _qb

try:
    import clr
    IN_QC = True
except ImportError:
    IN_QC = False

# MLflow in-loop hooks (local runs only — no-op in QC where mlflow is absent).
# Imported with a fallback stub so the QC-deployed notebook keeps working
# even if mlflow isn't installed in the Research environment.
try:
    from layer1.training import mlflow_hooks as _mlf_hooks
except Exception:
    class _NoopMlfHooks:
        def __getattr__(self, name):
            return lambda *a, **kw: None
    _mlf_hooks = _NoopMlfHooks()

print(f"PyTorch {torch.__version__}  |  IN_QC: {IN_QC}  |  LOCAL_MODE: {LOCAL_MODE}")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


# ===================================================================
# 2. CONFIGURATION
# ===================================================================

TOTAL_VARIATES = 141
OBJECTSTORE_PREFIX = "layer1_feat_v2"  # : post-TZ-fix corpus

# (Exp 10e): Leakage-free feature set — 41 bar-level features.
# Used as raw-41 baseline in probe evaluation. SSL uses the full 129/147-variate set.
#
# 2026-04-28 PATH A extension (literature-grounded decision #2): added v41/v105/
# v111/v112/v113 to fix HC-360 baseline feature-set asymmetry. Several of these
# variates are probe LABEL SOURCES — excluding them from the handcrafted baseline
# while exposing them through the SSL encoder produces a non-comparable evaluation
# (Caruana 2015; Recht et al. 2019; Demšar 2006; Sculley et al. 2018; Bouthillier
# et al. 2021 — feature-set parity is mandatory for valid method comparison).
#
# 41-variate set yields 10 stats × 41 = HC-410 baseline. For the spread_5 probe,
# additionally exclude v46/v47 (ATM spread + spread_chg) to avoid label leakage —
# spread_5 target is the 5-bar forward percentile of v46. That gives HC-390 for
# spread_5 only. See `compute_handcrafted_features` for the per-probe variant.
MODEL_FEATURES = [
    # IV surface — wing skew + ATM (5; +v41 ATM IV)
    9, 17, 41, 65, 73,                       # 10Dp IV, 25Dp IV, ATM IV, 25Dc IV, 10Dc IV
    # ATM microstructure (4)
    42, 43, 46, 47,                          # theta, gamma, spread, spread_chg
    # Strike aggregates (6)
    99, 100, 101, 102, 103, 104,             # max_imb, max_liq, n_quoted, set_width, pc_imb, liq_wt
    # SPX derived — leakage-free (3; +v105 session_log_return)
    105, 114, 118,                           # session_log_return, mins_to_close, vol_surge
    # SPX volatility — Parkinson rolling RV (3; +v111/v112/v113)
    111, 112, 113,                           # rv_15, rv_30, rv_60 (Parkinson, all backward-looking)
    # VIX term structure (12)
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    # Order flow (8)
    131, 132, 133, 134, 135, 136, 137, 139,  # pc_liq, net_prem, spread_asym, liq_conc, q_imb, spread_skew, net_gamma, imb_chg
    # /C1: v140 (options_staleness) REMOVED — permanently zero in v2 data,
    # trivially solvable SSL target. Was a phantom signal from CT/ET bug ().
]
# Per-probe label-leakage exclusions for the handcrafted / raw_36 baseline.
# Lookup by probe name; default = no exclusion. Encoder is unaffected — these
# only restrict the flat-baseline stat columns to preserve target/feature
# separation. Two distinct cases addressed:
#
# (A) Direct label leak (target derived deterministically from variate):
# spread_5 target = 5-bar forward percentile of v46 (ATM spread).
# Excluding v46 + v47 from the flat baseline forces a non-tautological
# comparator.
#
# (B) Trailing-RV autocorrelation leak (added 2026-04-29 evening, post-
# linear-arm agent analysis): v111/v112/v113 are Parkinson realized
# volatility at 15/30/60-bar TRAILING windows. log_rv_15 / log_rv_30
# targets are FORWARD realized vol at 15/30 bars. Volatility persistence
# makes the trailing variate a near-tautological predictor of the
# forward target — a flat ridge over v111-v113 gets R² ≈ 0.59 on
# log_rv_15 mostly via this single feature. Per L1 agent analysis of
# the linear-arm Azure run (full_linear_20260429_141509.log: emb_fine
# 0.5087 vs raw_36 0.5870), this rendered the encoder-vs-baseline
# comparison uninterpretable on log_rv probes. Excluding v111/v112/v113
# restores the flat baseline as a non-trivial adversary; the encoder
# still has access to all variates via the full SSL_FEATURES_V3 input.
#
# Note: regime probe is intentionally NOT in the exclusion list. v41 and v111
# ARE the regime label sources, but the encoder also has them as inputs —
# excluding them only on the baseline side would create feature-set asymmetry
# in the encoder's favor. The regime probe remains a feature-availability
# test (already disclosed at audit task #203, not a representation-quality
# headline).
HC_PROBE_LEAKAGE_EXCLUSIONS = {
    "spread_5":  frozenset({46, 47}),         # (A) direct: target = forward percentile of v46
    "log_rv_15": frozenset({111, 112, 113}),  # (B) trailing-RV autocorrelation leak
    "log_rv_30": frozenset({111, 112, 113}),  # (B) trailing-RV autocorrelation leak
}

# ---- SSL Configuration ----
# Full feature set for SSL masked-variate reconstruction.
# Expanded from classifier's 36 features. Grid reliability (88-98) excluded
# from encoder input — used internally by compute_ssl_mask_weights() for
# quality gating. v140 excluded (permanently zero, C1).
SSL_FEATURES = sorted(
    list(range(0, 88)) +       # Options grid (mid, iv, theta, gamma, qliq, qimb, spread, spread_chg) x 11
    list(range(99, 105)) +     # Strike aggregates
    list(range(105, 119)) +    # SPX derived (v114 = mins_to_close kept as context, excluded from targets)
    list(range(119, 131)) +    # VIX term structure
    list(range(131, 140))      # Order flow
)
N_SSL_FEATURES = len(SSL_FEATURES)  # 129

# Variates NEVER selected as SSL mask targets. Stay visible as encoder inputs
# so the model can attend to them, but masking them would be trivially solvable.
SSL_MASK_INELIGIBLE = frozenset({
    # Deterministic / arithmetic duplicates (original 6)
    114,  # mins_to_close — deterministic linear ramp, reconstructible from position
    128,  # fut_spread = f1 - f2
    129,  # f1_vix_basis = (f1 - vix) / vix
    130,  # ts_slope = regression of VIX term structure
    138,  # tick_mom = sign(ret_1m), categorical (~3 values post-z-score)
    139,  # imb_change = delta of order flow accumulator
    # C2 remediation (2026-05-02): v105 (session_log_return) RESTORED as eligible.
    # Original exclusion rationale was "cumsum(ret_1m) across T=60 is trivially
    # solvable." However, the CUMULATIVE effect of excluding ALL 14 spx_derived
    # variates was catastrophic -- zero SPX reconstruction gradient signal. v105
    # is the most informative SPX variate (meaningful variance, directly consumed
    # by GP), and "trivially solvable" overstated the difficulty -- cumsum over
    # noisy z-scored ret_1m is not a trivial identity. Other spx_derived variates
    # (v106-v113, v115-v118) remain ineligible for their original valid reasons
    # (near-zero variance or deterministic).
    # 105, # session_log_return — NOW ELIGIBLE (C2 remediation)
    # Grid spread_chg = spread[t] - spread[t-1] — trivially reconstructible from spread
    7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87,
    # Training-objective-v2 (2026-04-30): near-zero-variance spx_derived variates.
    # After z-scoring with MIN_STD_FLOOR=0.01, these cluster tightly around 0.
    # Predict-zero MSE ≈ 0.006 on them → any reconstruction noise produces
    # negative R² (-1.22 measured in SSL-017) and distorts the loss gradient.
    # Excluding from targets redirects decoder capacity to the ~99 variates
    # that carry real information. They remain as encoder context inputs.
    106,  # high_low_range — near-zero after z-score
    107,  # log_return_1m — near-zero, trivially derivable from v105 (session_log_return)
    108,  # log_return_5m
    109,  # log_return_15m
    110,  # log_return_30m
    111,  # parkinson_rv_15m (also rv_block_masked with v106)
    112,  # parkinson_rv_30m
    113,  # parkinson_rv_60m
    115,  # range_compression
    116,  # vwap_distance
    117,  # cumulative_return
    118,  # volume_surge (borderline — spikes exist but rare)
    # : Daily-constant VIX CBOE variates — context-only.
    # These have ZERO intraday variation (0% of 10,584 windows show any change
    # across T=60 bars). MVR on flat targets produces R²=-0.465 because:
    # (a) no temporal signal to exploit — reconstruction is pure cross-variate regression
    # (b) shared decoder Linear(128,60) optimized for options_grid temporal dynamics
    # produces "temporal ripple" on flat targets, creating ANTI-correlated predictions
    # (c) low predict-zero MSE (0.09 due to val regime near training mean) amplifies
    # any error into deeply negative R²
    # Keeping as context: the encoder benefits from SEEING VIX (r=0.77 with grid IV)
    # to reconstruct options_grid. GP has VIXSpot/VIXTermSlope as raw bypass terminals.
    119,  # VIX spot — CBOE daily, constant within session
    120,  # VIX9D — CBOE daily, constant within session
    121,  # VIX3M — CBOE daily, constant within session
    122,  # VIX6M — CBOE daily, constant within session
    125,  # VIX9D/VIX ratio — derived from daily CBOE, constant
    126,  # VIX/VIX3M ratio — derived from daily CBOE, constant
    127,  # VIX/VIX6M ratio — derived from daily CBOE, constant
})

# -------------------------------------------------------------------
# L1-SSL-009 v3 feature set (147 variates = 129 v2 + 18 flow aggregates)
# -------------------------------------------------------------------
# Adds multi-scale rolling-mean aggregates of the 9 order-flow variates
# (v131-v139). See layer1/data/BACKFILL_PLAN.md for derivation and log1p
# handling. Stored in `layer1_feat_v3/` corpus as float32 (matches
# patch_float16_variates.py convention, preserves patched-day precision).
#
# Aggregate layout:
# v141-v149 = flow_roll3_v131..139 (3-bar trailing rolling mean)
# v150-v158 = flow_roll15_v131..139 (15-bar trailing rolling mean)
SSL_FLOW_AGG_INDICES = list(range(141, 159))   # 18 aggregates
SSL_FEATURES_V3 = sorted(SSL_FEATURES + SSL_FLOW_AGG_INDICES)  # 147
N_SSL_FEATURES_V3 = len(SSL_FEATURES_V3)
# Dedup guard: `sorted(list_a + list_b)` does NOT deduplicate in Python.
# If SSL_FEATURES and SSL_FLOW_AGG_INDICES ever overlap, this assertion
# catches the duplicate variate entries at module load rather than at
# training-runtime with confusing shape mismatches.
assert len(set(SSL_FEATURES_V3)) == len(SSL_FEATURES_V3), \
    "SSL_FEATURES and SSL_FLOW_AGG_INDICES contain duplicate variate indices"

# Flow aggregates are NEVER selected as primary MVR mask targets. They
# enter the encoder as context-only inputs AND are block-masked as
# derivatives when their base raw flow variate (v131-v139) is masked.
# Block-mask expansion: _expand_flow_aggregate_mask() below. Without it,
# the encoder trivially inverts the rolling mean to reconstruct masked
# raw flow cells (leak identified in L1-SSL-009 spec §4.3 and AI Engineer
# Round 1 review).
SSL_MASK_INELIGIBLE_V3 = SSL_MASK_INELIGIBLE | frozenset(SSL_FLOW_AGG_INDICES)

# Raw → (roll3, roll15) aggregate variate-index map
FLOW_RAW_TO_AGG_RAW = {
    raw_v: (raw_v - 131 + 141, raw_v - 131 + 150)
    for raw_v in range(131, 140)
}
ROLLING_WINDOW_ROLL3 = 3
ROLLING_WINDOW_ROLL15 = 15

# Parkinson rolling-RV → high-low-range dependency (v111-v113 Option B block-mask).
# v106 = (H - L) / C ─ single-bar high/low range derivative
# v111 = sqrt(mean over 15 bars of log(H/L)^2 / (4 ln 2))
# v112 = same over 30 bars
# v113 = same over 60 bars
# log(H/L)^2 ≈ ((H-L)/C)^2 = v106^2 for small ranges, so v111-v113 are
# trailing-window aggregations of v106^2. Without block-mask propagation the
# encoder can re-derive a masked v111[t] from unmasked v106[t-14..t]
# (rolling-window inversion leak — analogous to the flow-aggregate leak).
RV_RAW_TO_AGG_RAW = {
    111: (106, 15),  # v111 ← rolling 15 of v106
    112: (106, 30),  # v112 ← rolling 30 of v106
    113: (106, 60),  # v113 ← rolling 60 of v106
}

OBJECTSTORE_PREFIX_V3 = "layer1_feat_v3"

# -------------------------------------------------------------------
# Active feature-set selector (v2 baseline vs v3 SSL-004-FE).
# -------------------------------------------------------------------
# Flip USE_V3_FEATURES to switch between:
# False (default) — SSL-004 baseline: 129 variates, layer1_feat_v2 corpus
# True — SSL-004-FE: 147 variates, layer1_feat_v3 corpus
#
# All training-path references below resolve through ACTIVE_* so only
# this block changes between experiments. Checkpoint keys include a
# version suffix so v2 and v3 checkpoints cannot clobber each other.
USE_V3_FEATURES = True  # SSL-004-FE run — flip to False for v2 baseline reproducibility

if USE_V3_FEATURES:
    ACTIVE_SSL_FEATURES = SSL_FEATURES_V3
    ACTIVE_N_SSL_FEATURES = N_SSL_FEATURES_V3
    ACTIVE_MASK_INELIGIBLE = SSL_MASK_INELIGIBLE_V3
    ACTIVE_OBJECTSTORE_PREFIX = OBJECTSTORE_PREFIX_V3
    ACTIVE_CKPT_SUFFIX = "_v3"
else:
    ACTIVE_SSL_FEATURES = SSL_FEATURES
    ACTIVE_N_SSL_FEATURES = N_SSL_FEATURES
    ACTIVE_MASK_INELIGIBLE = SSL_MASK_INELIGIBLE
    ACTIVE_OBJECTSTORE_PREFIX = OBJECTSTORE_PREFIX
    ACTIVE_CKPT_SUFFIX = ""

# Checkpoint key resolution — prevents v2/v3 cross-contamination.
# v2 training writes "ssl_model"; v3 writes "ssl_model_v3".
ACTIVE_CKPT_KEY = f"ssl_model{ACTIVE_CKPT_SUFFIX}"
ACTIVE_HISTORY_KEY = f"ssl_history{ACTIVE_CKPT_SUFFIX}"

# -------------------------------------------------------------------
# Known concerns for v3 training — NOT addressed in pipeline.py; these
# are training-loop config and diagnostics for the next PR (task 37).
# -------------------------------------------------------------------
#
# 1) Mask-rate inflation (Data Engineer Round 6 review).
# With default variate_ratio=0.3 + cell_ratio=0.15 + block expansion,
# effective mask rate on roll15 aggregates reaches ~90%:
# P(roll15 cell masked) ≈ 0.30 · 1.0 + 0.70 · (1 − 0.85^15) ≈ 0.90
# → aggregates serve as visible context only ~10% of training samples.
# Recommendation: run a short mask-rate sweep (variate_ratio ∈
# {0.15, 0.20, 0.25, 0.30}) on SSL-004-FE before committing to full
# training. Lower ratios preserve more aggregate context for
# multi-scale pattern learning.
#
# 2) Masking-artifact prior (Model QA Round 6 review, HIGH severity).
# When an aggregate cell is block-masked its value is zeroed (z-score
# mean). This is indistinguishable from a GENUINE zero aggregate on
# quiet-flow days. The encoder can learn `E[v131 | roll3_input=0] ≈ 0`
# from unmasked training data, then use this prior to "reconstruct"
# masked v131 cells where roll3 was also zeroed by block-masking. Loss
# decreases without genuine learning; representation memorizes a
# masking-artifact prior.
# ADDRESSED (telemetry, task 37): compute_masking_artifact_diagnostic
# runs at end of run_ssl_pipeline (Step 7). Quantile-split MSE on
# masked raw-flow cells + Pearson corr(|true|, MSE). Currently log-
# only (assert_pass=False); flip to True after first SSL-004-FE run
# to make this a hard training gate.
# Stronger defenses (deferred): mask-indicator channel so encoder
# distinguishes "block-masked" from "genuine zero", or fill masked
# positions with a distinguished value outside the natural roll3
# distribution.
#
# 3) Train/probe distribution mismatch (Model QA Round 6, Medium).
# During training, encoder sees aggregate=0 in ~30% of samples
# (block-masked). At probe/inference time, aggregates are always
# visible with real values. Attention weights learned under training
# distribution may under-weight aggregates at probe time.
# ADDRESSED (telemetry, task 37): compute_embedding_cosine_shift
# runs at end of run_ssl_pipeline (Step 8). Encodes val twice —
# once with aggregates visible, once with aggregates zeroed — and
# measures cosine similarity of the mean-pooled embedding. Heavy
# dependence → low cosine → probe-time shift. Thresholds: mean >=
# 0.90, p5 >= 0.80. Currently log-only (assert_pass=False); flip
# to True after first SSL-004-FE run to make this a hard gate.

# -b: Heavy-tailed variates that benefit from log1p compression before z-score.
# These have long right tails where ±5σ clipping discards meaningful information.
# Applied as sign(x)*log1p(|x|) BEFORE z-scoring in make_ssl_split_and_loaders().
# Indices are RAW variate numbers (from the 141-layout), resolved to SSL_FEATURES
# positions at runtime.
SSL_LOG1P_VARIATES = frozenset({
    111, 112, 113,  # Parkinson RV (15m, 30m, 60m) — heavy right tail during volatility spikes
    131,            # put/call liquidity ratio — unbounded ratio, right-skewed
})

# ---- SSL-010-LOCAL canonical reproduction config (FROZEN, DO NOT MODIFY) ----
# Bit-equality reproduction target for the H2 L1 input checkpoint
# (commit 812ed7d, val_recon=0.2606, probes 3/4 reportable on emb_fine).
# Anchored here so any drift in EXPERIMENT_BASELINE below is detectable.
# Gate: any future SSL run that wants to replay SSL-010-LOCAL must pass these
# exact values (modulo the new SSL-012 architectural-fix flags whose defaults
# preserve SSL-010 behavior — see EXPERIMENT_BASELINE comment).
EXPERIMENT_BASELINE_SSL010 = {
    "seq_len": 60,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "d_ff": 512,
    "dropout": 0.2,
    "learning_rate": 1e-4,
    "batch_size": 64,
    "num_epochs": 100,
    "warmup_epochs": 10,
    "variate_ratio": 0.3,
    "cell_ratio": 0.15,
    "contrastive_weight": 0.0,
    "contrastive_temp": 0.1,
    "forecast_mask_bars": 0,
    "forecast_mask_prob": 0.0,
    "temporal_weight": 0.1,           # SSL-010 ran with temporal-stats head
    "tokenizer": "linear",
    "patch_size": 5,
    # Architectural-fix flags MUST be at their SSL-010 defaults:
    "use_temporal_attn": False,
    "use_film_recon": False,
    "rv_block_mask": False,
    "use_mask_indicator": False,
    "mask_fill_value": 0.0,
    "tail_weight_alpha": 0.0,
    "huber_delta": 0.0,        # 0.0 = MSE (SSL-010 legacy); >0 = Huber loss
    "use_group_recon": False,  # False = shared recon head (SSL-010 legacy)
    "spectral_rank_weight": 0.0,  # 0.0 = disabled (SSL-010 legacy)
}

# ---- Frozen baseline config (DO NOT MODIFY) ----
# Reference config for the tokenizer ablation series.
# d_model=128 (target architecture); all tokenizer experiments use this base.
# The pre-training config diff flags any unintended deviations.
EXPERIMENT_BASELINE = dict(
    seq_len=60,
    d_model=128,
    n_heads=4,
    n_layers=3,
    d_ff=512,             # 4× d_model
    dropout=0.2,
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=100,
    warmup_epochs=10,
    variate_ratio=0.3,         # v3: see module-header concern #1 — mask-rate inflation on roll15 aggregates at 0.3
    cell_ratio=0.15,
    contrastive_weight=0.0,
    contrastive_temp=0.1,
    forecast_mask_bars=0,
    forecast_mask_prob=0.0,
    temporal_weight=0.0,
    tokenizer="linear",
    patch_size=12,             # V2: larger patches for rank parity with linear (was 5)
    # SSL-012 architectural-fix flags. Defaults preserve SSL-010/SSL-011
    # behavior so legacy runs reproduce exactly when overrides are absent.
    use_temporal_attn=False,   # SSL-012: prepend temporal-axis attention block (CrossFormer hybrid)
    use_film_recon=False,      # SSL-012: per-variate FiLM conditioning on recon head (identity-init)
    rv_block_mask=False,       # SSL-012 Option B: symmetric v106 ↔ v111-v113 block-mask
    # SSL-012 mean-collapse mitigations (M1+M2+M3) targeting MQA.6 shrinkage prior
    # on heavy-tailed flow variates (v133/v135/v136/v137). See module-header
    # concerns #2 and #3 for the design lineage (deferred at SSL-004-FE).
    use_mask_indicator=False,  # M1: feed encoder a binary "is_masked" channel via mask_proj head
    mask_fill_value=0.0,       # M2: value substituted at masked cells (-8.0 in SSL-012 — outside z-score natural [-5,5])
    tail_weight_alpha=0.0,     # M3: per-cell loss weight *= (1 + alpha * |z_target|) on heavy-tail variates
    huber_delta=0.0,           # 0.0 = MSE (legacy); >0 = Huber loss transition point
    use_group_recon=False,     # False = shared recon head (legacy)
    spectral_rank_weight=0.0,  # R1: 0.0 = disabled; >0 = log-det covariance penalty weight
)

# Drift check (#211, A-H2 fix 2026-04-29): bidirectional integrity.
# Direction A — every SSL-010 baseline key (except temporal_weight, which
# SSL-010 explicitly overrode to 0.1) must continue to match EXPERIMENT_BASELINE.
# Direction B — any NEW key added to EXPERIMENT_BASELINE must be acknowledged
# in EXPERIMENT_BASELINE_SSL010 (either at SSL-010's value, or by adding it
# to _NEW_KEYS_ALLOWED with explicit reason). Without direction B, an
# always-on `use_X=True` added later would silently break SSL-010 reproduction.
_SSL010_DRIFT_GUARD_KEYS = set(EXPERIMENT_BASELINE_SSL010) - {"temporal_weight", "patch_size"}
# Forward-direction allowlist: keys that may legitimately appear in
# EXPERIMENT_BASELINE without an SSL-010 counterpart (e.g., a flag whose
# default-False is a no-op for SSL-010 reproduction). Any key NOT in this
# allowlist AND not in EXPERIMENT_BASELINE_SSL010 will raise.
_NEW_KEYS_ALLOWED: set = set()  # currently empty — every key tracked
for _k in _SSL010_DRIFT_GUARD_KEYS:
    if EXPERIMENT_BASELINE.get(_k) != EXPERIMENT_BASELINE_SSL010[_k]:
        raise AssertionError(
            f"EXPERIMENT_BASELINE drift (Direction A) on key {_k!r}: "
            f"got {EXPERIMENT_BASELINE.get(_k)!r}, "
            f"SSL-010 baseline expects {EXPERIMENT_BASELINE_SSL010[_k]!r}. "
            f"To reproduce SSL-010, build hyperparams directly from "
            f"EXPERIMENT_BASELINE_SSL010 instead of EXPERIMENT_BASELINE."
        )
for _k in EXPERIMENT_BASELINE:
    if _k not in EXPERIMENT_BASELINE_SSL010 and _k not in _NEW_KEYS_ALLOWED:
        raise AssertionError(
            f"EXPERIMENT_BASELINE drift (Direction B) — new key {_k!r}={EXPERIMENT_BASELINE[_k]!r} "
            f"is not declared in EXPERIMENT_BASELINE_SSL010 and not in _NEW_KEYS_ALLOWED. "
            f"Either add it to EXPERIMENT_BASELINE_SSL010 at SSL-010's value (preserves "
            f"reproduction), or add to _NEW_KEYS_ALLOWED with a comment explaining why "
            f"it's a no-op for SSL-010."
        )
del _k

# Heavy-tail flow variates eligible for tail-importance weighting in
# MaskedVariateLoss when tail_weight_alpha > 0. Selected per Pass 2 Model QA
# diagnostic (active-tail MSE 17-75× calm-middle on these variates in
# SSL-010-LOCAL). v131/v132/v134/v138/v139 had marginal or no shrinkage so
# are excluded here.
HEAVY_TAIL_VARIATES = (133, 135, 136, 137)

# ---- Active SSL training hyperparameters ----
# Start from EXPERIMENT_BASELINE, override ONLY the variables you intend to change.
# The pre-training config diff (in train_ssl) will flag any differences.
#
# SSL-012 (Path β, 2026-04-28):
# - Temporal attention DEFERRED to SSL-013 3-arm comparison (no clear job
# for linear-arm-only — see Pass 2 hostile review).
# - FiLM, RV block-mask kept (real bug fixes, near-free compute).
# - M1+M2+M3 added to attack the MQA.6 mean-collapse on heavy-tail flow.
SSL_HYPERPARAMS = dict(
    EXPERIMENT_BASELINE,
    # === EXPERIMENT OVERRIDES ===
    # C3 remediation (2026-05-02): reduce variate_ratio 0.30 -> 0.20.
    # At 0.30, 77.8% of gradient concentration falls on options_grid (77 of
    # 99 eligible variates). Reducing to 0.20 decreases total eligible masked
    # variates, proportionally reducing options_grid dominance. Combined with
    # the C1 threshold block-mask fix, this improves minority group visibility.
    # EXPERIMENT_BASELINE retains 0.30 for SSL-010 reproduction.
    variate_ratio=0.20,
    # temporal_weight dropped to 0.0 (2026-04-30): the TemporalStatsHead's
    # std target correlates with log_rv probe targets (task #204), confounding
    # probe results. Contribution was only ~4% of val_loss. All 3 tokenizer
    # arms run without it for clean probe evaluation.
    # NOTE: this is INDEPENDENT of use_temporal_attn (TemporalAttentionBlock
    # in the encoder architecture), which stays off for linear, on for patch/CNN.
    temporal_weight=0.0,
    # === SSL-012 bundle (Path β v2) ===
    use_temporal_attn=False,   # SSL-013 will flip this for patch/CNN — linear-arm doesn't need it
    use_film_recon=True,       # FiLM conditioning on shared recon head (identity-init, drift = diagnostic)
    rv_block_mask=True,        # Option B: symmetric v106 ↔ v111-v113 block-mask
    use_mask_indicator=True,   # M1: encoder receives binary "is_masked" channel
    mask_fill_value=-8.0,      # M2: distinguished value, outside z-score [-5,5] natural range
    # M3 DROPPED 2026-04-28 (literature-grounded decision #1).
    # Cell-level (1+α|z|) reweighting + global gradient clipping is refuted by:
    # - Lin et al. 2017 (focal loss) — modulator must vanish on confident predictions
    # - Cao et al. 2019 (LDAM) — prefers margin-based reweighting tied to class prior
    # - Wettig et al. 2023 — masking-rate sensitivity dominates over loss reweighting
    # - He et al. 2022 (MAE) — uniform per-cell loss is the canonical MVR objective
    # - Karras et al. 2022 — variance-of-target weighting is loss-shape redesign,
    # not a multiplier
    # Heavy-tail mean-collapse fix should redesign loss SHAPE (Huber / log-cosh /
    # Gaussian-mixture NLL), not multiply MSE by |z|. Tracked under future work.
    tail_weight_alpha=0.0,     # M3: REMOVED — kept for backward-compat key ordering
    # === Training objective v5 (SSL-017, 2026-04-30) ===
    # Return to Huber δ=1.0 + shared decoder (SSL-014 v2 config): agent analysis
    # showed (a) Huber achieves 13% better true MSE than direct MSE training,
    # (b) shared decoder provides implicit anti-collapse via bottleneck pressure,
    # (c) dropout=0.1 avoids the gradient-noise amplification that caused SSL-016
    # val spikes. Added: spectral covariance penalty (R1) for explicit anti-collapse.
    huber_delta=1.0,
    use_group_recon=False,     # TESTED SSL-024: FAILED (flow worsened). Do NOT enable. Shared decoder preferred.
    equal_group_loss=False,    # TESTED SSL-025: FAILED (flow worsened). Do NOT enable.
    quantile_flow=False,       # TESTED SSL-026: WORKED on linear only; SSL-028: CATASTROPHIC on patch. Do NOT enable without arm-specific testing.
    use_logcosh=False,         # TESTED SSL-027: MIXED (strike +12pp but VIX -20pp). Not worth the trade-off.
    num_epochs=100,
    dropout=0.1,
    spectral_rank_weight=0.06, # R1: base weight (used as seed for adaptive, or fixed if target_spectral_fraction=0)
    target_spectral_fraction=0.20,  # : adaptive loss-fraction targeting. Spectral contributes this fraction of total loss.
                                    # Self-calibrates across tokenizer architectures. Set 0.0 to disable (use fixed weight).
                                    # SSL-030 showed 0.06 fixed was catastrophic for patch (VIX=-1.729). Adaptive fixes this.
)

# SSL-012 ckpt routing: when ANY architectural / loss / mask change is enabled,
# write checkpoints under a distinct suffix so SSL-010-LOCAL canonical artifacts
# (ssl_model_v3 + ssl_history_v3) are NOT overwritten. This preserves the
# H1/RQ1/RQ2 baseline for any future paired comparison or rollback. Resolves
# to ssl_model_v3_ssl012 / ssl_history_v3_ssl012 / ssl_resume_v3_ssl012.
_ssl012_flags = (
    SSL_HYPERPARAMS.get("use_temporal_attn", False),
    SSL_HYPERPARAMS.get("rv_block_mask", False),
    SSL_HYPERPARAMS.get("use_film_recon", False),
    SSL_HYPERPARAMS.get("use_mask_indicator", False),
    SSL_HYPERPARAMS.get("mask_fill_value", 0.0) != 0.0,
    SSL_HYPERPARAMS.get("tail_weight_alpha", 0.0) > 0.0,
)
if any(_ssl012_flags):
    ACTIVE_CKPT_SUFFIX = ACTIVE_CKPT_SUFFIX + "_ssl012"
    ACTIVE_CKPT_KEY = f"ssl_model{ACTIVE_CKPT_SUFFIX}"
    ACTIVE_HISTORY_KEY = f"ssl_history{ACTIVE_CKPT_SUFFIX}"
    print(f"  SSL-012 routing: ACTIVE_CKPT_KEY={ACTIVE_CKPT_KEY}")

# Tokenizer-arm routing (2026-04-29): distinguish per-tokenizer primary
# checkpoints so a 3-arm ablation (linear / patch / CNN) doesn't have arms
# clobbering each other's primary keys (`ssl_model{...}`, `ssl_history{...}`,
# `ssl_resume{...}`, `probe_results{...}` — every key derived from
# ACTIVE_CKPT_SUFFIX). Linear stays at the legacy key (no extra suffix
# appended) so the in-progress Azure linear training continues to write to
# `ssl_model_v3_ssl012` as already in flight; patch and CNN get a
# distinguishing suffix when they're configured.
#
# Net keys after this routing:
# linear → ssl_model_v3_ssl012 (legacy, continues current training)
# patch → ssl_model_v3_ssl012_patch
# cnn → ssl_model_v3_ssl012_cnn
#
# Existing post-training archive at line ~4775 (`ssl_model_{tok}{SUFFIX}`)
# remains as a tokenizer-tagged secondary copy — unchanged.
_active_tokenizer = SSL_HYPERPARAMS.get("tokenizer", "linear")
if _active_tokenizer != "linear":
    ACTIVE_CKPT_SUFFIX = ACTIVE_CKPT_SUFFIX + f"_{_active_tokenizer}"
    ACTIVE_CKPT_KEY = f"ssl_model{ACTIVE_CKPT_SUFFIX}"
    ACTIVE_HISTORY_KEY = f"ssl_history{ACTIVE_CKPT_SUFFIX}"
    print(f"  Tokenizer routing: ACTIVE_CKPT_KEY={ACTIVE_CKPT_KEY}")

# N4 3-arm ablation guard (2026-04-29): the 3-arm tokenizer comparison is
# valid ONLY if the non-tokenizer overrides in SSL_HYPERPARAMS are identical
# across all arms. Log them loudly at module load so any drift is visible.
# use_temporal_attn is tokenizer-coupled (off for linear/patch, on for CNN)
# per revised design, so it's excluded from the guard alongside tokenizer
# itself. patch_size is only consumed by the patch arm (ignored by linear/CNN).
_TOKENIZER_COUPLED_KEYS = {"tokenizer", "use_temporal_attn", "patch_size"}
_NON_TOKENIZER_OVERRIDES = {
    k: SSL_HYPERPARAMS[k] for k in SSL_HYPERPARAMS
    if k not in _TOKENIZER_COUPLED_KEYS and SSL_HYPERPARAMS[k] != EXPERIMENT_BASELINE.get(k)
}
if _NON_TOKENIZER_OVERRIDES:
    print(f"  3-arm ablation guard: {len(_NON_TOKENIZER_OVERRIDES)} non-tokenizer "
          f"overrides from EXPERIMENT_BASELINE (must be IDENTICAL across arms):")
    for _k, _v in sorted(_NON_TOKENIZER_OVERRIDES.items()):
        print(f"    {_k}: {EXPERIMENT_BASELINE.get(_k)!r} → {_v!r}")

# ---- Probe Configuration (§11.3) ----
# Linear probes on frozen encoder embeddings to test representation quality.
# Walk-forward expanding-window CV with embargo; sklearn closed-form solvers
# eliminate SGD hyperparameter confounds (Alain & Bengio 2017).
PROBE_HYPERPARAMS = dict(
    n_folds=8,            # Expanding-window walk-forward folds (8 for tighter CI on paired deltas)
    n_seeds=3,            # Random seeds per fold (shuffle baseline + init)
    embargo_days=5,       # de Prado (2018): gap between train/test prevents window leakage
    # Forward horizons (bars = minutes)
    probe_a_horizons=[5, 15, 30],  # Forward return direction: sign(Δ session_log_return)
    probe_b_horizons=[15, 30],     # Forward realized vol: log(Parkinson RV at t+k)
    probe_c_horizon=30,            # Future regime: expert label at t+30 (not concurrent)
    probe_d_horizon=5,             # ATM spread delta: v46[t+5] - v46[t]
    dead_zone_std_multiplier=0.1,  # Skip targets where |value| < 0.1 * train_fold_stdev
    last_live_bar=389,             # 16:00 ET — conservative cutoff for forward targets
)

# Raw variate indices for probe target extraction (141-variate layout)
RAW_SESSION_LOG_RET_INDEX = 105   # v105: ln(close_t / close_open), for forward return Δ
RAW_ATM_SPREAD_INDEX = 46         # v46: ATM spread (grid point 5, feature 6)
# RAW_PARKINSON_RV_15M_INDEX (111) and RAW_PARKINSON_RV_30M_INDEX (112) already defined above

# new probe target variates (grid layout: point*8 + feature_idx)
RAW_ATM_GAMMA_INDEX = 43          # v43: ATM gamma (grid point 5, feature 3)
RAW_ATM_THETA_INDEX = 42          # v42: ATM theta (grid point 5, feature 2)
RAW_IV_25DP_INDEX = 17            # v17: 25-delta put IV (grid point 2, feature 1)
RAW_IV_25DC_INDEX = 65            # v65: 25-delta call IV (grid point 8, feature 1)
RAW_IV_ATM_INDEX = 41             # v41: ATM IV (grid point 5, feature 1)
RAW_NET_PREMIUM_FLOW_INDEX = 132  # v132: net premium flow (order flow channel)

# Grid-only token range for selective mean-pooling (: options R²=0.474, dominant group)
# SSL_FEATURES positions 0:N_GRID_TOKENS correspond to raw variates 0-87
N_GRID_TOKENS = 88

# Log RV floor: log(max(rv, floor)) prevents log(0) in Probe B targets
LOG_RV_FLOOR = 1e-10

# Feature group boundaries in SSL_FEATURES (for per-group probe analysis)
# These match the baseline groups exactly
PROBE_FEATURE_GROUPS = {
    "options_grid": (0, 88),     # SSL_FEATURES positions 0-87
    "strike_agg": (88, 94),      # positions 88-93 (raw 99-104)
    "spx_derived": (94, 108),    # positions 94-107 (raw 105-118)
    "vix_term": (108, 120),      # positions 108-119 (raw 119-130)
    "order_flow": (120, 129),    # positions 120-128 (raw 131-139)
}

# : Expert-defined regime labeling (replaces GMM)
# VIX spot remains in MODEL_FEATURES for comparison (legitimate — vol axis uses ATM IV, not VIX)

# : ALL label-input features extracted from raw 141-variate data for labeling only.
# None of these appear in MODEL_FEATURES — eliminates label-to-input leakage.
RAW_ATM_IV_INDEX = 41            # : ATM IV → vol axis EMA
RAW_PARKINSON_RV_15M_INDEX = 111 # : PRV15m → new movement axis EWMA
RAW_PARKINSON_RV_30M_INDEX = 112 # : PRV30m → vol axis RV30 override

ATM_IV_EMA_SPAN = 60  # 60-bar EMA (~1 hour of minute bars), smooths noise

# : Movement axis uses EWMA(parkinson_rv_15m) instead of EWMA(abs(log_ret_15m)).
# Parkinson RV measures movement magnitude from high/low ranges — not reconstructible
# from any remaining MODEL_FEATURES (all price-range features removed).
# Warm-up bars: RV30 needs 30 bars, PRV15m needs 15 bars, movement EWMA needs ~15
# bars. First WARMUP_BARS of each day have structurally unreliable labels.
WARMUP_BARS = 45  # Conservative: max(RV30=30, EWMA=15) + margin

# Percentile thresholds (computed from training bars)
VOL_LOW_PCT = 33       # P33: below → LOW volatility (on ATM IV EMA)
VOL_HIGH_PCT = 75      # P75: above → HIGH volatility (on ATM IV EMA)
RV30_HIGH_PCT = 75     # P75: override to HIGH if exceeded
# MOVEMENT_PCT = 60 # [Pre-Exp11] Replaced by EWMA + hysteresis below

# (Exp 11): EWMA smoothing + Schmitt trigger hysteresis for movement axis.
# Fixes median dwell=3 bars / 43 transitions per day from stateless P60 threshold.
# (Exp 10e-b): Reduced from 45→15 to fix double-smoothing. PRV15m is already
# a 15-bar rolling Parkinson estimator; EWMA(45) on top created ~60-bar effective
# smoothing that compressed movement IQR 8× and collapsed DIRECTIONAL/VOLATILE.
MOVEMENT_EWMA_SPAN = 15   # : Reduced from 45 — effective window ~25 bars
MOVEMENT_UPPER_PCT = 65    # EWMA percentile: cross above → enter TRENDING
MOVEMENT_LOWER_PCT = 55    # EWMA percentile: drop below → exit TRENDING

TRAIN_START = "2022-09-19"
TRAIN_END = "2024-09-30"
VAL_START = "2024-10-08"
VAL_END = "2025-01-31"
TEST_START = "2025-02-08"
# v8 (RF-1): corpus NPZ bundle covers through 2026-04-09; the prior
# TEST_END='2025-06-30' constant was stale and silently dropped 192
# trading days from every Parquet regeneration. Aligned to actual
# corpus coverage. Test window row count grows from 1,224 to ~3,600.
TEST_END = "2026-04-09"

WINDOW_STRIDE = 30  # : reduced from 5 — closer to ~2,800 ESS

# : Regime taxonomy — 4-class (CALM, DIRECTIONAL, VOLATILE, CRISIS).
REGIME_TAXONOMY = {
    0: "CALM",
    1: "DIRECTIONAL",
    2: "VOLATILE",
    3: "CRISIS",
}

# Preprocessing constants
GAMMA_INDICES = [i * 8 + 3 for i in range(11)]
OPTIONS_GRID_END = 88
RELIABILITY_END = 99

_results = {}


# ===================================================================
# 3. PREPROCESSING
# ===================================================================

def should_skip_day(X):
    # Count both all-zero and all-NaN columns as dead. v3 backfill writes
    # NaN into aggregates v141-v158 on dead days (raw flow was fully
    # forward-filled → rolling means are undefined). Without the NaN
    # clause, a dead day with 18 NaN aggregates + 130 zero raw columns
    # reports dead=130 (just under threshold) and gets admitted, then
    # poisons global_mean/global_std via np.mean propagation.
    all_zero = np.all(X == 0, axis=0)
    all_nan = np.all(np.isnan(X), axis=0)
    dead = int(np.sum(all_zero | all_nan))
    return dead > 130


def preprocess_day(X):
    """Clean a single day's raw tensor. Returns (data, liveness_mask).

    NOTE: VIX forward-fill is NOT applied here. Call apply_vix_forwardfill()
    separately after extracting labeling features (VIX spot, RV30), so labels
    reflect raw values before forward-fill.
    """
    n_bars = X.shape[0]
    X = X.astype(np.float32)

    # Pad 140->141 if old data
    if X.shape[1] < TOTAL_VARIATES:
        pad_width = TOTAL_VARIATES - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)

    # Gamma clamp [0, 100]
    for gi in GAMMA_INDICES:
        X[:, gi] = np.clip(X[:, gi], 0.0, 100.0)

    # : v132 (net_premium_liquidity) is ALREADY stored signed_log1p'd
    # by the collector (research_collector.py:497 — `out[1] = _signed_log1p(...)`).
    # The previous `X[:, 132] = sign * log1p(|x|)` block here was a
    # pre-existing double compression that produced v132_effective =
    # signed_log1p(signed_log1p(raw)). Mismatch against the v3 aggregates
    # v141/v150 (single log1p, applied at backfill) — removed.
    # v137 (net_gamma_weighted_liq) is also collector-stored log1p'd and
    # is correctly NOT re-compressed here. SSL-004 baseline must be
    # re-run on v2 to recalibrate z-score stats before SSL-004-FE results
    # can be compared against the old (doubly-compressed) checkpoint.

    # : Signed log1p for skewed features (strike aggs + order flow).
    # Raw means: 101=163, 102=98, 104=-12, 139=124. These are stored RAW
    # by the collector (no prior log1p), so this compression is correct.
    # Confirmed v101/v102/v104 ∈ strike_aggregates (raw), v139 ∈ order
    # flow out[8]=imb_change (raw).
    for vi in [101, 102, 104, 139]:
        col = X[:, vi]
        X[:, vi] = np.sign(col) * np.log1p(np.abs(col))

    # Detect adaptive cutoff: last bar with any options grid variate nonzero
    grid_data = X[:, :OPTIONS_GRID_END]
    any_nonzero = np.any(grid_data != 0, axis=1)
    nonzero_bars = np.where(any_nonzero)[0]
    cutoff_int = int(nonzero_bars[-1]) if len(nonzero_bars) > 0 else 0

    # +: Forward-fill options variates (0-98), strike aggregates (99-104),
    # and order flow (131-139) from last live bar after cutoff.
    if cutoff_int < n_bars - 1:
        X[cutoff_int + 1:, :RELIABILITY_END] = X[cutoff_int, :RELIABILITY_END]
        X[cutoff_int + 1:, 99:105] = X[cutoff_int, 99:105]    # strike aggregates
        X[cutoff_int + 1:, 131:140] = X[cutoff_int, 131:140]   # order flow

    # /C1: Staleness ramp REMOVED. v140 is permanently zero in v2 data
    # ( CT/ET phantom retired). The old ramp injected a synthetic signal
    # that contradicted the collector's semantics. v140 is also dropped from
    # MODEL_FEATURES, so it no longer reaches the model.

    # Liveness mask: 1.0 for live bars, 0.0 for dead bars (all forward-filled ranges).
    # Width tracks X.shape[1] so v3 tensors (159 cols) don't under-allocate —
    # compute_ssl_mask_weights indexes liveness[:, vi] for every feature in
    # ACTIVE_SSL_FEATURES including aggregates v141-v158.
    V_actual = X.shape[1]
    liveness = np.ones((n_bars, V_actual), dtype=np.float32)
    if cutoff_int < n_bars - 1:
        liveness[cutoff_int + 1:, :RELIABILITY_END] = 0.0
        liveness[cutoff_int + 1:, 99:105] = 0.0    # strike aggregates
        liveness[cutoff_int + 1:, 131:140] = 0.0   # order flow
        # v3: flow aggregates v141-v158 — rolling means whose trailing
        # window contained non-live raw values are distributionally
        # inconsistent with training (raw gets forward-filled, aggregate
        # reflects pre-ffill rolling mean). Zero the C10 weight on these
        # positions in the dead tail.
        if V_actual > 140:
            liveness[cutoff_int + 1:, 141:V_actual] = 0.0

    return X, liveness


def apply_vix_forwardfill(X):
    """Apply VIX f1/f2 forward-fill and recompute derived variates in-place.

    Separated from preprocess_day() so labeling features (VIX spot, RV30)
    reflect raw values before forward-fill.
    """
    n_bars = X.shape[0]

    # VIX f1/f2 forward-fill if zero (variates 123, 124)
    for vi in [123, 124]:
        col = X[:, vi]
        last_good = 0.0
        for b in range(n_bars):
            if col[b] != 0.0:
                last_good = col[b]
            elif last_good != 0.0:
                col[b] = last_good
        X[:, vi] = col

    # Recompute derived VIX variates (128=futures_spread, 129=f1_vix_basis, 130=term_structure_slope)
    vix = X[:, 119]
    f1 = X[:, 123]
    f2 = X[:, 124]
    X[:, 128] = f1 - f2
    safe_vix = np.where(vix > 0, vix, 1.0)
    X[:, 129] = np.where(vix > 0, (f1 - vix) / safe_vix, 0.0)

    # Term structure slope: linear regression of [vix9d, vix, vix3m, vix6m] on [9,30,90,180]
    tenors = np.array([9.0, 30.0, 90.0, 180.0])
    x_c = tenors - tenors.mean()
    denom = (x_c ** 2).sum()
    if denom > 0:
        for b in range(n_bars):
            levels = np.array([X[b, 120], X[b, 119], X[b, 121], X[b, 122]])
            if levels.sum() > 0:
                y_c = levels - levels.mean()
                X[b, 130] = np.dot(x_c, y_c) / denom


# ===================================================================
# 3B. SSL MASK WEIGHTS (C10)
# ===================================================================

def compute_ssl_mask_weights(X_day, liveness, feature_indices):
    """Compute per-cell SSL reconstruction weights for one day.

    C10 unified allowlist: controls which (bar, variate) cells contribute
    to SSL reconstruction loss. Subsumes C3, C4, C7, C8, and all static
    exclusions. Called AFTER preprocess_day() and apply_vix_forwardfill().

    Weight semantics:
      1.0  = valid reconstruction target
      0.25 = downweighted (partial data, e.g. VIX dead day)
      0.0  = excluded (dead, structural zero, trivially solvable)

    Args:
        X_day: (n_bars, 141) float32, preprocessed + VIX-forward-filled
        liveness: (n_bars, 141) float32, from preprocess_day()
        feature_indices: list[int], which variates are SSL encoder features

    Returns:
        weights: (n_bars, len(feature_indices)) float32
    """
    n_bars = X_day.shape[0]
    n_feat = len(feature_indices)
    weights = np.ones((n_bars, n_feat), dtype=np.float32)

    # Pre-extract bar-level signals used by multiple conditions
    mins_to_close = X_day[:, 114]                     # (n_bars,)
    reliability = X_day[:, 88:99]                     # (n_bars, 11) grid reliability
    settle_mask = mins_to_close < 15                  # C3: settle window (16:01-16:15)
    late_mask = mins_to_close < 30                    # C7: late-session theta/gamma

    # Pre-detect VIX f1 dead day for C8 (check first 60 bars of raw f1)
    f1_front = X_day[:min(60, n_bars), 123]
    vix_day_dead = (f1_front.sum() == 0.0)

    for fi, vi in enumerate(feature_indices):
        col = X_day[:, vi]

        # --- Static exclusions (ACTIVE_MASK_INELIGIBLE) ---
        # v3 flow aggregates (v141-v158) land here: weight=0 at the loss
        # layer. Block-mask propagation in sample_ssl_mask still applies,
        # but the reconstruction loss gradient on aggregates is zeroed
        # so the encoder is not trained to reconstruct derivative cells.
        if vi in ACTIVE_MASK_INELIGIBLE:
            weights[:, fi] = 0.0
            continue

        # --- Forward-filled bars (from liveness mask) ---
        live = liveness[:, vi]
        weights[:, fi] *= live

        # --- Day-level: dead variate (all zero for the full day) ---
        if np.all(col == 0.0):
            weights[:, fi] = 0.0
            continue

        # --- Day-level: flatline (all nonzero values identical) ---
        nonzero = col[col != 0.0]
        if len(nonzero) > 0 and np.all(nonzero == nonzero[0]):
            weights[:, fi] = 0.0
            continue

        # --- C4: first-bar structural zeros (warmup) ---
        if vi in (107, 108, 111, 112, 113):
            weights[0, fi] = 0.0

        # --- C3: settle-window SPX derived exclusion ---
        if 105 <= vi <= 118:
            weights[settle_mask, fi] = 0.0

        # --- C7: late-session theta/gamma clip exclusion ---
        if vi < 88 and (vi % 8) in (2, 3):  # theta=offset 2, gamma=offset 3
            weights[late_mask, fi] = 0.0

        # --- C8: VIX futures dead-day downweight ---
        if vi in (123, 124) and vix_day_dead:
            weights[:, fi] = np.minimum(weights[:, fi], 0.25)

        # --- Grid reliability downweight ---
        # When reliability < 0.5, the grid point's option values are
        # interpolated or stale. Halve their reconstruction weight to
        # prevent copy-shortcut learning from stale forward-fills.
        if vi < 88:
            grid_idx = vi // 8  # which of the 11 grid points (0-10)
            rely = reliability[:, grid_idx]
            low_rely = rely < 0.5
            if np.any(low_rely):
                weights[low_rely, fi] *= 0.5

    return weights


# ===================================================================
# 3C. SSL MASK SAMPLER
# ===================================================================

def _build_eligible_mask(feature_indices, mask_ineligible=None):
    """Build boolean array of which SSL features are eligible for masking.

    Returns (n_features,) numpy bool. True = can be masked.

    Args:
        feature_indices: list of raw variate indices (e.g., SSL_FEATURES
                         or SSL_FEATURES_V3)
        mask_ineligible: optional set of ineligible raw indices.
                         Default: SSL_MASK_INELIGIBLE (v2 layout).
                         Pass SSL_MASK_INELIGIBLE_V3 for v3 (147-variate)
                         layout — aggregates v141-v158 are ineligible as
                         primary targets (they're block-masked as
                         derivatives instead, via _expand_flow_aggregate_mask).
    """
    if mask_ineligible is None:
        mask_ineligible = SSL_MASK_INELIGIBLE
    return np.array([vi not in mask_ineligible for vi in feature_indices])


def _build_flow_agg_position_map(feature_indices):
    """Build raw→aggregate position map for variate-block masking.

    For each of the 9 raw flow variates (v131..v139), locate its position
    in feature_indices AND the positions of its two aggregate derivatives
    (roll3 at v{i+10}, roll15 at v{i+19}). Returns None if any of the 9
    raw vars or 18 aggregates is missing from feature_indices — this is
    the SSL-004 case (129-variate layout, no aggregates). In that case
    _expand_flow_aggregate_mask is a no-op.

    If the feature set contains SOME aggregates but not all 18, a
    `RuntimeWarning` is emitted: partial aggregate sets would silently
    skip the fix, opening a leak. Full-set-or-none is the contract.

    Returns:
        None, or a dict containing:
          "raw_positions_py":    list[int] length 9 — positions of v131..v139
          "roll3_positions_py":  list[int] length 9 — positions of v141..v149
          "roll15_positions_py": list[int] length 9 — positions of v150..v158

    Both Python-int lists AND tensor forms would be ideal, but for the
    hot-path masking code we store only Python ints. This avoids 27×
    `.item()` GPU syncs per batch (27 = 9 flow × 3 positions), which on
    a ~100-epoch × ~15k-batch run amounts to ~40M unnecessary syncs.
    """
    import warnings

    pos_map = {v: i for i, v in enumerate(feature_indices)}
    raw_flow = list(range(131, 140))
    roll3 = list(range(141, 150))
    roll15 = list(range(150, 159))
    all_agg_vars = roll3 + roll15
    present_aggs = [v for v in all_agg_vars if v in pos_map]
    if len(present_aggs) == 0:
        # SSL-004 layout: no aggregates at all → clean None
        return None
    if any(v not in pos_map for v in raw_flow + all_agg_vars):
        # Partial aggregate set → silent-skip would leak; warn loudly
        if 0 < len(present_aggs) < len(all_agg_vars):
            warnings.warn(
                f"_build_flow_agg_position_map: feature_indices contains "
                f"{len(present_aggs)}/{len(all_agg_vars)} flow aggregates — "
                f"partial set. Returning None (no block-masking). If you "
                f"intended v3 layout, ensure all 18 aggregates (v141-v158) "
                f"and all 9 raw flow variates (v131-v139) are present.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None
    return {
        "raw_positions_py":   [pos_map[v] for v in raw_flow],
        "roll3_positions_py": [pos_map[v] for v in roll3],
        "roll15_positions_py": [pos_map[v] for v in roll15],
    }


def _trailing_window_or(x, window):
    """Trailing-window boolean OR along the T axis.

    Computes result[b, t] = OR(x[b, max(0, t-window+1) : t+1]) — i.e., if
    x[b, s] is True for any s in the window [t-w+1, t], then result[b, t]
    is True.

    Used to propagate raw-flow cell masks forward to aggregate cells whose
    trailing rolling window contained the masked raw cell. For a roll-k
    aggregate, if raw[t] is masked, aggregate cells at t, t+1, ..., t+k-1
    (clipped to T-1) are contaminated. This function computes that
    propagation in vectorized form.

    Args:
        x: (B, T) bool tensor
        window: int ≥ 1. Rolling window size.

    Returns:
        (B, T) bool tensor — a FRESH tensor, never aliased to input. This
        matters for the caller (`_expand_flow_aggregate_mask`) which may
        subsequently OR-assign into `result`; aliasing to the source would
        corrupt the source. Returning `x.clone()` in the identity branch
        keeps the contract symmetric for all window values ≥ 1.
    """
    if window <= 1:
        return x.clone()
    result = x.clone()
    T = x.shape[-1]
    for shift in range(1, window):
        if T - shift > 0:
            result[:, shift:] |= x[:, :T - shift]
    return result


def _trailing_window_count(x, window):
    """Trailing-window SUM (count of True values) along the T axis.

    Computes result[b, t] = sum(x[b, max(0, t-window+1) : t+1]).

    Used by threshold-based block-mask expansion: instead of masking an
    aggregate cell whenever ANY source cell in its rolling window is masked
    (which destroys roll15 visibility at ~93%), we count how many source
    cells are masked and only block-mask if the count exceeds a threshold.

    Args:
        x: (B, T) bool tensor — raw cell mask
        window: int >= 1. Rolling window size.

    Returns:
        (B, T) int tensor — count of masked cells in each trailing window.
        FRESH tensor, never aliased to input.
    """
    if window <= 1:
        return x.int().clone()
    # Use cumulative sum for efficient sliding-window count
    x_int = x.int()  # (B, T)
    cumsum = torch.zeros_like(x_int)
    cumsum[:, 0] = x_int[:, 0]
    for t in range(1, x_int.shape[-1]):
        cumsum[:, t] = cumsum[:, t - 1] + x_int[:, t]
    # count[b, t] = cumsum[t] - cumsum[t - window] (with boundary handling)
    result = cumsum.clone()
    T = x_int.shape[-1]
    for t in range(window, T):
        result[:, t] = cumsum[:, t] - cumsum[:, t - window]
    # For t < window, result[t] = cumsum[t] which is correct (partial window from start)
    return result


def _expand_flow_aggregate_mask(combined, variate_mask, flow_positions,
                                block_threshold=0.5):
    """Block-mask aggregate cells whose rolling window has >threshold masked raw cells.

    Mutates `combined` and `variate_mask` IN PLACE. No-op when
    flow_positions is None (SSL-004 mode, no aggregates in feature set).

    For each of the 9 raw flow variates v131..v139:
      1. Whole-variate propagation: if variate_mask[b, :, raw_pos] is all
         True across T (raw variate whole-masked for sample b), also set
         variate_mask[b, :, roll3_pos] and [..., roll15_pos] all True.
         We use `.all(dim=1)` (not `variate_mask[:, 0, :]`) so the check
         is robust to future callers that might write non-uniform
         variate-level masks.
      2. Cell-wise THRESHOLD propagation (C1 remediation 2026-05-02):
         For each aggregate cell at bar t, count how many source cells in
         the trailing window [t-w+1, t] are masked. Only block-mask the
         aggregate cell if the fraction exceeds `block_threshold` (default
         0.5). This replaces the old ANY-masked logic which destroyed
         roll15 visibility (~93% masked at cell_ratio=0.15).

         Methodological justification: flow aggregates are mask-INELIGIBLE
         (never reconstruction targets). The block-mask controls CONTEXT
         visibility only. Rolling means are lossy summaries -- seeing
         roll15 when some source cells are masked does NOT reveal individual
         cell values. The threshold approach preserves anti-leak protection
         while giving ~50% context visibility.

    Args:
        combined: (B, T, V) bool -- combined mask from sample_ssl_mask
        variate_mask: (B, T, V) bool -- whole-variate-only subset of combined
        flow_positions: dict from _build_flow_agg_position_map() or None
        block_threshold: float in (0, 1]. Fraction of source cells in the
            rolling window that must be masked before the aggregate cell is
            block-masked. Default 0.5 (>50% of source cells masked).
    """
    if flow_positions is None:
        return
    # Python-int position lists (see _build_flow_agg_position_map docstring
    # for the GPU-sync rationale).
    raw_pos_py = flow_positions["raw_positions_py"]
    r3_pos_py = flow_positions["roll3_positions_py"]
    r15_pos_py = flow_positions["roll15_positions_py"]
    # Structural validation: a user-built or corrupted dict with wrong-length
    # position lists would silently block-mask the wrong columns and leak.
    # Flow layout is invariant: 9 raw x 2 scales. Assert cheaply at entry.
    assert len(raw_pos_py) == 9, \
        f"_expand_flow_aggregate_mask: expected 9 raw positions, got {len(raw_pos_py)}"
    assert len(r3_pos_py) == 9, \
        f"_expand_flow_aggregate_mask: expected 9 roll3 positions, got {len(r3_pos_py)}"
    assert len(r15_pos_py) == 9, \
        f"_expand_flow_aggregate_mask: expected 9 roll15 positions, got {len(r15_pos_py)}"

    for raw_pos, r3_pos, r15_pos in zip(raw_pos_py, r3_pos_py, r15_pos_py):
        # Whole-variate propagation -- robust to non-uniform variate_mask
        raw_whole = variate_mask[:, :, raw_pos].all(dim=1)  # (B,)
        if raw_whole.any():
            broadcast = raw_whole.unsqueeze(1)  # (B, 1)
            variate_mask[:, :, r3_pos] |= broadcast
            variate_mask[:, :, r15_pos] |= broadcast
            combined[:, :, r3_pos] |= broadcast
            combined[:, :, r15_pos] |= broadcast

        # Cell-wise THRESHOLD propagation (C1 remediation):
        # Count how many source cells in each trailing window are masked.
        # Only block-mask the aggregate cell if the count exceeds the
        # threshold fraction of the window size.
        raw_cells = combined[:, :, raw_pos]  # (B, T) -- includes whole-var + cell + forecast

        # Roll3: window=3, threshold count = ceil(3 * 0.5) = 2
        r3_count = _trailing_window_count(raw_cells, ROLLING_WINDOW_ROLL3)
        # Effective window size at each position (handles boundary: first
        # window-1 bars have partial windows)
        T = raw_cells.shape[-1]
        r3_effective_window = torch.clamp(
            torch.arange(1, T + 1, device=raw_cells.device),
            max=ROLLING_WINDOW_ROLL3
        )  # (T,) — values 1,2,3,3,3,...,3
        r3_threshold_count = torch.ceil(r3_effective_window.float() * block_threshold).int()
        combined[:, :, r3_pos] |= (r3_count >= r3_threshold_count.unsqueeze(0))

        # Roll15: window=15, threshold count = ceil(15 * 0.5) = 8
        r15_count = _trailing_window_count(raw_cells, ROLLING_WINDOW_ROLL15)
        r15_effective_window = torch.clamp(
            torch.arange(1, T + 1, device=raw_cells.device),
            max=ROLLING_WINDOW_ROLL15
        )  # (T,) — values 1,2,...,15,15,...,15
        r15_threshold_count = torch.ceil(r15_effective_window.float() * block_threshold).int()
        combined[:, :, r15_pos] |= (r15_count >= r15_threshold_count.unsqueeze(0))

    # Invariant post-condition: variate_mask <= combined at exit.
    # Within this function the invariant is structurally preserved -- the
    # whole-var branch writes to both in lockstep; the cell-wise branch
    # widens only `combined`. So this assert catches callers that pass in
    # an already-violating `variate_mask` (not bugs in this function's
    # body). Kept as a cheap defense against future refactors of
    # sample_ssl_mask that might inadvertently break the subset contract.
    if __debug__:
        assert (variate_mask <= combined).all(), \
            "invariant violated: variate_mask is not a subset of combined"


def _leading_window_or(x, window):
    """Leading-window boolean OR along the T axis.

    Symmetric counterpart to _trailing_window_or: result[b, t] = OR(x[b, t..t+window-1]).
    Used for backward propagation of v111-v113 RV masks onto v106 cells whose
    rolling window included the masked RV cell. Without backward propagation,
    masking v111[t] alone is trivially recoverable from unmasked v106[t-14..t]
    (the encoder reconstructs Parkinson(t) directly from the visible HL range
    sequence). Symmetric two-way masking closes the leak.

    Returns a fresh tensor — never aliased to input. See _trailing_window_or
    docstring for the aliasing contract.
    """
    if window <= 1:
        return x.clone()
    result = x.clone()
    T = x.shape[-1]
    for shift in range(1, window):
        if T - shift > 0:
            result[:, :T - shift] |= x[:, shift:]
    return result


def _build_rv_position_map(feature_indices):
    """Build v106 ↔ v111-v113 position map for variate-block masking.

    Returns None if neither v106 nor v111/v112/v113 is present (legacy SSL-004
    layouts that don't include the Parkinson RV variates). Returns the
    position map when ALL of {v106, v111, v112, v113} are present.

    A partial set (any of v106/v111/v112/v113 missing while at least one is
    present) raises ValueError — the leak-guard analogue of
    _build_flow_agg_position_map's RuntimeWarning. We escalate to ValueError
    here because the partial-RV case is a misconfigured ablation layout, not
    a backwards-compatible smaller feature set.

    Returns:
        None, or a dict with:
          "hl_range_pos_py": int — position of v106
          "rv15_pos_py":     int — position of v111
          "rv30_pos_py":     int — position of v112
          "rv60_pos_py":     int — position of v113

    Stored as Python ints to avoid GPU-sync hot-path costs on every batch
    (matches _build_flow_agg_position_map convention).
    """
    pos_map = {v: i for i, v in enumerate(feature_indices)}
    rv_vars = (106, 111, 112, 113)
    present = [v for v in rv_vars if v in pos_map]
    if len(present) == 0:
        return None
    if len(present) < len(rv_vars):
        missing = [v for v in rv_vars if v not in pos_map]
        raise ValueError(
            f"_build_rv_position_map: feature_indices contains "
            f"{len(present)}/{len(rv_vars)} of v106/v111/v112/v113 — "
            f"partial set (missing: {missing}). RV block-mask propagation "
            f"requires all four. Either include the full set or exclude all."
        )
    return {
        "hl_range_pos_py": pos_map[106],
        "rv15_pos_py":     pos_map[111],
        "rv30_pos_py":     pos_map[112],
        "rv60_pos_py":     pos_map[113],
    }


def _expand_rv_mask(combined, variate_mask, rv_positions):
    """Whole-variate block-mask coupling between v106 and v111-v113.

    Mutates `combined` and `variate_mask` IN PLACE. No-op when rv_positions
    is None.

    REDESIGNED 2026-04-29 from symmetric cell-wise window propagation.
    Reality Checker post-run audit demonstrated the previous (cell-wise
    bidirectional) implementation produced effective mask rates ~99.5% on
    v106 and v111-v113 under standard cell_ratio=0.15, variate_ratio=0.30
    config. Closed-form: P(any v111[t..t+14] cell-masked | not whole-masked)
    = 1 − 0.85^15 ≈ 0.91, applied to backward propagation onto v106 over a
    15-bar window — composed across w∈{15,30,60} ⇒ ~99% effective masking
    on v106. This destroyed the variate group log_rv probes target and was
    the dominant mechanical cause of the SSL-012 v2 regression.

    Whole-variate-only propagation: closes the leak at the structurally
    load-bearing case (Stage-1 whole-variate masks) without destroying
    cell-level information across rolling windows.

      Forward: if v106 whole-masked → v111/v112/v113 whole-masked
      Backward: if any of v111/v112/v113 whole-masked → v106 whole-masked

    The leak this prevents: when the encoder is asked to reconstruct
    whole-masked v111, having v106 visible across the 60-bar window
    trivially solves Parkinson(t) ≈ ((H-L)/C)² ≈ v106². The whole-variate
    coupling forces both source and aggregate to be unavailable together.

    Cell-level leak: when a single v106[t] cell is masked but v111[t]
    visible, the encoder COULD compute v111 from neighboring v106 cells.
    This residual leak is acceptable because (a) v111 is a 15-bar mean,
    so the encoder needs 14 visible v106 cells around t — substantial
    information transfer, not algebraic inversion, and (b) Stage-2 cell
    masking on this scale (15% of cells) doesn't enable systematic
    recovery; the encoder still has to do real representation learning.
    """
    if rv_positions is None:
        return
    hl_pos = rv_positions["hl_range_pos_py"]
    rv15_pos = rv_positions["rv15_pos_py"]
    rv30_pos = rv_positions["rv30_pos_py"]
    rv60_pos = rv_positions["rv60_pos_py"]
    rv_positions_list = (rv15_pos, rv30_pos, rv60_pos)

    # Whole-variate propagation — symmetric.
    hl_whole = variate_mask[:, :, hl_pos].all(dim=1)  # (B,)
    rv_whole_any = torch.zeros_like(hl_whole)
    for rv_pos in rv_positions_list:
        rv_whole_any |= variate_mask[:, :, rv_pos].all(dim=1)

    if hl_whole.any():
        broadcast = hl_whole.unsqueeze(1)
        for rv_pos in rv_positions_list:
            variate_mask[:, :, rv_pos] |= broadcast
            combined[:, :, rv_pos] |= broadcast

    if rv_whole_any.any():
        broadcast = rv_whole_any.unsqueeze(1)
        variate_mask[:, :, hl_pos] |= broadcast
        combined[:, :, hl_pos] |= broadcast

    if __debug__:
        assert (variate_mask <= combined).all(), \
            "invariant violated in _expand_rv_mask: variate_mask is not a subset of combined"


def sample_ssl_mask(batch_size, seq_len, n_features, eligible_mask,
                    variate_ratio=0.3, cell_ratio=0.15, generator=None,
                    forecast_bars=0, forecast_prob=0.0,
                    flow_agg_positions=None, feature_indices=None,
                    rv_positions=None):
    """Generate hybrid SSL mask for one batch (D71 spec section 11.1).

    Four-stage masking:
      1. Whole-variate: ~30% of eligible variates masked across all T (D78-c)
      2. Per-cell: ~15% of remaining eligible (t, v) cells masked independently
      3. Temporal forecast (Phase 2): mask last N bars for ~50% of samples,
         forcing encoder to learn predictive temporal structure
      4. Flow-aggregate block expansion (L1-SSL-009 v3): when the feature
         set includes flow aggregates (v141-v158), derivative cells whose
         rolling-mean window contained a masked raw flow cell are also
         masked. Prevents the encoder from trivially inverting the rolling
         mean to reconstruct masked raw flow cells. No-op when
         flow_agg_positions is None (SSL-004 layout with no aggregates).

    Masked positions are zeroed in the input (post-z-score, 0 ~ mean).
    Note: no mask indicator channel — the variate_embed provides implicit identity
    so the encoder can distinguish "masked variate v" from "genuinely zero variate v".

    Args:
        batch_size: int
        seq_len: int (T=60)
        n_features: int (129 for SSL_FEATURES, 147 for SSL_FEATURES_V3)
        eligible_mask: (n_features,) bool tensor — True for mask-eligible variates
        variate_ratio: fraction of eligible variates to whole-mask (default 0.3)
        cell_ratio: fraction of remaining eligible cells to mask (default 0.15)
        generator: optional torch.Generator for deterministic masking (val eval)
        forecast_bars: int — number of trailing bars to mask (Phase 2, default 0 = off)
        forecast_prob: float — probability of applying forecast mask per sample
        flow_agg_positions: optional dict from _build_flow_agg_position_map().
              When provided, applies variate-block masking of flow aggregates
              (v141-v158 derived from v131-v139) to prevent rolling-mean
              inversion leak. Build once per training run via
              _build_flow_agg_position_map(feature_indices) and pass through.
              When None (default, SSL-004 backward compat), no block
              expansion — standard 3-stage mask.
        feature_indices: optional list[int] of raw variate indices for the
              feature layout being trained (e.g., SSL_FEATURES or
              SSL_FEATURES_V3). When provided, enables a content-based
              leak guard: any aggregate (v141-v158) in the layout without
              flow_agg_positions raises ValueError. Strongly preferred over
              the size-based fallback — catches ablation layouts of any
              total size that include aggregates.

    Returns:
        mask: (batch_size, seq_len, n_features) bool tensor
              True = masked (hidden from encoder, target for reconstruction)
        variate_mask: (batch_size, seq_len, n_features) bool tensor
              True = whole-variate masked positions (subset of mask).
              Use (mask & ~variate_mask) for per-cell-only positions.
    """
    dev = eligible_mask.device

    # Leak guard: any flow aggregate (v141-v158) in the feature set
    # REQUIRES flow_agg_positions. Silent omission would train with the
    # rolling-mean inversion leak described in L1-SSL-009 §4.3 — raw v131
    # masked + v141 visible = trivial aggregate-inversion.
    #
    # Prefer content-based check (feature_indices) over size-based (147 ==
    # v3) — a future 150-variate ablation (e.g., SSL_FEATURES + partial
    # aggregate subset) would bypass a size check silently. Content check
    # catches ANY feature layout carrying aggregates. When feature_indices
    # is not supplied, fall back to size-based as a last line of defense
    # for legacy callers.
    if flow_agg_positions is None:
        if feature_indices is not None:
            agg_present = any(v in SSL_FLOW_AGG_INDICES for v in feature_indices)
            if agg_present:
                raise ValueError(
                    f"sample_ssl_mask: feature_indices contains flow "
                    f"aggregates (v141-v158) but flow_agg_positions=None. "
                    f"This would train with the rolling-mean inversion leak. "
                    f"Pass flow_agg_positions="
                    f"_build_flow_agg_position_map(feature_indices)."
                )
        elif n_features == N_SSL_FEATURES_V3:
            raise ValueError(
                f"sample_ssl_mask called with n_features="
                f"{N_SSL_FEATURES_V3} (v3 layout) but flow_agg_positions="
                f"None and feature_indices not provided. This would train "
                f"with the rolling-mean inversion leak. Pass "
                f"flow_agg_positions=_build_flow_agg_position_map("
                f"feature_indices) AND feature_indices=<your layout> for "
                f"content-based validation."
            )

    # Stage 1: whole-variate mask — same across all T for each sample
    variate_probs = torch.rand(batch_size, n_features, device=dev, generator=generator)
    variate_selected = (variate_probs < variate_ratio) & eligible_mask.unsqueeze(0)

    # Stage 1b: flow-group anti-correlation cap (SSL-017+, structured masking).
    # Ensure at least 5 of 9 order_flow variates (v131-v139) remain visible,
    # guaranteeing sufficient within-group context for cross-variate prediction.
    # Without this, ~10% of batches mask 5+ flow variates simultaneously,
    # making flow reconstruction unlearnable (encoder defaults to predict-zero).
    max_flow_masked = 4
    if feature_indices is not None:
        flow_pos = [i for i, v in enumerate(feature_indices) if 131 <= v <= 139]
        if len(flow_pos) > 0:
            flow_selected = variate_selected[:, flow_pos]
            n_flow = flow_selected.sum(dim=1)
            for b in (n_flow > max_flow_masked).nonzero(as_tuple=True)[0]:
                masked_idx = flow_selected[b].nonzero(as_tuple=True)[0]
                n_deselect = int(n_flow[b].item()) - max_flow_masked
                perm = torch.randperm(len(masked_idx), device='cpu')[:n_deselect]
                for fi in perm:
                    variate_selected[b, flow_pos[masked_idx[fi]]] = False

    variate_mask = variate_selected.unsqueeze(1).expand(-1, seq_len, -1).clone()  # (B, T, V)

    # Stage 2: per-cell mask on remaining eligible variates
    remaining = eligible_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
    remaining = remaining & ~variate_mask  # exclude already-masked variates
    cell_probs = torch.rand(batch_size, seq_len, n_features, device=dev, generator=generator)
    cell_mask = (cell_probs < cell_ratio) & remaining

    combined = variate_mask | cell_mask

    # Stage 3 (Phase 2): temporal forecast mask — mask last N bars on eligible variates
    if forecast_bars > 0 and forecast_prob > 0:
        sample_probs = torch.rand(batch_size, device=dev, generator=generator)
        apply_forecast = sample_probs < forecast_prob  # (B,) bool
        if apply_forecast.any():
            forecast_region = torch.zeros(batch_size, seq_len, n_features,
                                          dtype=torch.bool, device=dev)
            # Last forecast_bars timesteps, all eligible variates
            forecast_region[:, -forecast_bars:, :] = eligible_mask.unsqueeze(0).unsqueeze(0)
            forecast_region = forecast_region & apply_forecast.unsqueeze(1).unsqueeze(2)
            combined = combined | forecast_region

    # Stage 4 (L1-SSL-009 v3): block-expand mask over flow aggregate derivatives
    if flow_agg_positions is not None:
        _expand_flow_aggregate_mask(combined, variate_mask, flow_agg_positions)

    # Stage 5 (SSL-012 Option B): symmetric block-expand across v106 ↔ v111-v113.
    # No-op when rv_positions is None (legacy SSL-004 layout). When rv_positions
    # is provided, propagates masks bidirectionally over the 15/30/60-bar
    # Parkinson windows.
    if rv_positions is not None:
        _expand_rv_mask(combined, variate_mask, rv_positions)

    return combined, variate_mask


# ===================================================================
# 4. DATA LOADING
# ===================================================================

def _load_tensor(qb, date_str):
    # ACTIVE_OBJECTSTORE_PREFIX routes to v2 (layer1_feat_v2, 141 cols) or
    # v3 (layer1_feat_v3, 159 cols with flow aggregates). preprocess_day's
    # pad-up-to-141 branch is a no-op on v3 data (already ≥141 cols).
    key = f"{ACTIVE_OBJECTSTORE_PREFIX}/{date_str}.npz"
    if not qb.ObjectStore.ContainsKey(key):
        return None
    raw = json.loads(qb.ObjectStore.Read(key))
    data = base64.b64decode(raw["data"])
    buf = io.BytesIO(data)
    npz = np.load(buf)
    X = npz["X"].astype(np.float32)
    if X.shape[1] < TOTAL_VARIATES:
        pad = np.zeros((X.shape[0], TOTAL_VARIATES - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)
    return X


def iter_trading_dates(start, end):
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)



# ===================================================================
# 4B. SSL DATA LOADING
# ===================================================================

def load_ssl_days(qb, start_date, end_date):
    """Load days for SSL pretraining with C10 mask weights.
    Returns sliding windows of ACTIVE_SSL_FEATURES (129 for v2, 147 for v3)
    plus per-cell C10 mask weights for reconstruction loss weighting.

    Returns dict with:
        X_windows: (N, T, ACTIVE_N_SSL_FEATURES) float32
        W_windows: (N, T, ACTIVE_N_SSL_FEATURES) float32 — C10 mask weights
        window_dates: (N,) date strings
    """
    T = SSL_HYPERPARAMS["seq_len"]
    stride = WINDOW_STRIDE
    n_ssl = ACTIVE_N_SSL_FEATURES

    all_dates = list(iter_trading_dates(start_date, end_date))
    est_bars = len(all_dates) * 400
    est_windows = est_bars // stride + 256

    X_windows = np.empty((est_windows, T, n_ssl), dtype=np.float32)
    W_windows = np.empty((est_windows, T, n_ssl), dtype=np.float32)
    window_dates_list = []

    win_idx = 0
    n_days = 0
    n_skipped = 0

    for d in all_dates:
        ds = d.isoformat()
        raw = _load_tensor(qb, ds)
        if raw is None:
            continue

        if should_skip_day(raw):
            n_skipped += 1
            continue

        data, liveness = preprocess_day(raw)
        del raw
        apply_vix_forwardfill(data)

        # C10 mask weights for this day
        day_weights = compute_ssl_mask_weights(data, liveness, ACTIVE_SSL_FEATURES)

        n_bars = data.shape[0]
        n_days += 1

        # Extract SSL features (v2: 129 variates, v3: 147 incl. flow aggregates)
        day_ssl = data[:, ACTIVE_SSL_FEATURES]  # (n_bars, n_ssl)

        # Build sliding windows (data + weights)
        for start_pos in range(0, n_bars - T + 1, stride):
            end_pos = start_pos + T
            if win_idx >= len(X_windows):
                ext_x = np.empty((est_windows, T, n_ssl), dtype=np.float32)
                X_windows = np.concatenate([X_windows, ext_x], axis=0)
                ext_w = np.empty((est_windows, T, n_ssl), dtype=np.float32)
                W_windows = np.concatenate([W_windows, ext_w], axis=0)

            X_windows[win_idx] = day_ssl[start_pos:end_pos]
            W_windows[win_idx] = day_weights[start_pos:end_pos]
            window_dates_list.append(ds)
            win_idx += 1

        del data, liveness, day_ssl, day_weights
        gc.collect()

        if n_days % 50 == 0:
            print(f"  {n_days} days loaded, {win_idx} windows...")

    X_windows = X_windows[:win_idx]
    W_windows = W_windows[:win_idx]
    window_dates = np.array(window_dates_list)

    gb = win_idx * T * n_ssl * 4 * 2 / 1e9  # x2 for X_windows + W_windows
    print(f"Done. {n_days} days ({n_skipped} skipped), "
          f"{win_idx} windows, {gb:.2f} GB (X+W)")

    return dict(
        X_windows=X_windows,
        W_windows=W_windows,
        window_dates=window_dates,
    )


# ===================================================================
# 4C. PROBE DATA LOADING
# ===================================================================

def load_probe_days(qb, start_date, end_date):
    """Load days for probe evaluation with forward target extraction.

    Builds SSL windows (129 features) AND extracts per-window forward targets
    for all four probes at each configured horizon. Also computes per-bar
    regime labels (D60) with fixed thresholds from the training period for
    Probe C (future regime at t+30).

    Bar-position validity: a forward target at horizon k is valid iff the
    target bar exists in the same day AND its position <= last_live_bar (389).

    Returns dict with:
        X_windows: (N, T, 129) float32 — SSL feature windows
        W_windows: (N, T, 129) float32 — C10 mask weights
        window_dates: (N,) str
        window_last_bar_pos: (N,) int32 — last bar position within day

        Forward targets (NaN / -1 where invalid):
        fwd_ret: dict[int, (N,) float32] — forward returns per horizon
        fwd_rv: dict[int, (N,) float32] — forward Parkinson RV (raw, pre-log)
        fwd_regime: (N,) int64 — future regime label (-1 if invalid/warmup)
        fwd_spread: (N,) float32 — forward ATM spread delta
        fwd_valid: dict[int, (N,) bool] — per-horizon validity masks

        Regime metadata:
        regime_thresholds: dict — thresholds used for labeling
    """
    T = SSL_HYPERPARAMS["seq_len"]
    stride = WINDOW_STRIDE
    n_ssl = ACTIVE_N_SSL_FEATURES
    last_live = PROBE_HYPERPARAMS["last_live_bar"]
    ph = PROBE_HYPERPARAMS

    # All unique horizons across probes
    all_horizons = sorted(set(
        ph["probe_a_horizons"] + ph["probe_b_horizons"] +
        [ph["probe_c_horizon"], ph["probe_d_horizon"]]
    ))  # [5, 15, 30]

    all_dates = list(iter_trading_dates(start_date, end_date))
    est_bars = len(all_dates) * 405
    est_windows = est_bars // stride + 256

    # --- Pre-allocate SSL windows ---
    X_windows = np.empty((est_windows, T, n_ssl), dtype=np.float32)
    W_windows = np.empty((est_windows, T, n_ssl), dtype=np.float32)
    window_dates_list = []
    window_last_bar_pos_list = []

    # --- Pre-allocate forward targets ---
    fwd_ret = {k: np.full(est_windows, np.nan, dtype=np.float32)
               for k in ph["probe_a_horizons"]}
    fwd_rv = {k: np.full(est_windows, np.nan, dtype=np.float32)
              for k in ph["probe_b_horizons"]}
    fwd_spread = np.full(est_windows, np.nan, dtype=np.float32)
    # new probe forward arrays
    fwd_gamma_accel = np.full(est_windows, np.nan, dtype=np.float32)
    fwd_smile_convex = np.full(est_windows, np.nan, dtype=np.float32)
    fwd_flow_tox = np.full(est_windows, np.nan, dtype=np.float32)
    # PredJump uses fwd_ret[30] (already computed), no new array needed
    fwd_valid = {k: np.zeros(est_windows, dtype=bool) for k in all_horizons}

    # --- Pre-allocate bar-level labeling features (for Probe C regime labels) ---
    bar_atm_iv_ema_arr = np.empty(est_bars, dtype=np.float32)
    bar_movement_ewma_arr = np.empty(est_bars, dtype=np.float32)
    bar_rv30_arr = np.empty(est_bars, dtype=np.float32)
    bar_positions_arr = np.empty(est_bars, dtype=np.int32)
    bar_dates_list = []

    # Per-window → global bar index for Probe C post-hoc assignment
    window_last_bar_global = np.empty(est_windows, dtype=np.int64)

    win_idx = 0
    bar_idx = 0
    n_days = 0
    n_skipped = 0

    def _extend_bar_arrays(needed):
        """Extend bar-level arrays if they're too short."""
        nonlocal bar_atm_iv_ema_arr, bar_movement_ewma_arr, bar_rv30_arr, bar_positions_arr
        if bar_idx + needed <= len(bar_atm_iv_ema_arr):
            return
        ext = max(est_bars, needed)
        bar_atm_iv_ema_arr = np.concatenate([bar_atm_iv_ema_arr, np.empty(ext, dtype=np.float32)])
        bar_movement_ewma_arr = np.concatenate([bar_movement_ewma_arr, np.empty(ext, dtype=np.float32)])
        bar_rv30_arr = np.concatenate([bar_rv30_arr, np.empty(ext, dtype=np.float32)])
        bar_positions_arr = np.concatenate([bar_positions_arr, np.empty(ext, dtype=np.int32)])

    def _extend_win_arrays():
        """Extend window-level arrays if they're too short."""
        nonlocal X_windows, W_windows, window_last_bar_global, fwd_spread
        nonlocal fwd_ret, fwd_rv, fwd_valid
        nonlocal fwd_gamma_accel, fwd_smile_convex, fwd_flow_tox
        ext = est_windows
        X_windows = np.concatenate([X_windows, np.empty((ext, T, n_ssl), dtype=np.float32)])
        W_windows = np.concatenate([W_windows, np.empty((ext, T, n_ssl), dtype=np.float32)])
        window_last_bar_global = np.concatenate([window_last_bar_global, np.full(ext, -1, dtype=np.int64)])
        fwd_spread = np.concatenate([fwd_spread, np.full(ext, np.nan, dtype=np.float32)])
        fwd_gamma_accel = np.concatenate([fwd_gamma_accel, np.full(ext, np.nan, dtype=np.float32)])
        fwd_smile_convex = np.concatenate([fwd_smile_convex, np.full(ext, np.nan, dtype=np.float32)])
        fwd_flow_tox = np.concatenate([fwd_flow_tox, np.full(ext, np.nan, dtype=np.float32)])
        for k in fwd_ret:
            fwd_ret[k] = np.concatenate([fwd_ret[k], np.full(ext, np.nan, dtype=np.float32)])
        for k in fwd_rv:
            fwd_rv[k] = np.concatenate([fwd_rv[k], np.full(ext, np.nan, dtype=np.float32)])
        for k in fwd_valid:
            fwd_valid[k] = np.concatenate([fwd_valid[k], np.zeros(ext, dtype=bool)])

    print(f"Loading probe data: {start_date} to {end_date}")
    for d in all_dates:
        ds = d.isoformat()
        raw = _load_tensor(qb, ds)
        if raw is None:
            continue
        if should_skip_day(raw):
            n_skipped += 1
            continue

        data, liveness = preprocess_day(raw)
        del raw
        n_bars = data.shape[0]
        n_days += 1

        # ---- Extract labeling features BEFORE VIX forward-fill ----
        day_iv_raw = data[:, RAW_ATM_IV_INDEX].copy()
        day_iv_raw[day_iv_raw == 0] = np.nan
        day_iv_ema = _ema_1d(day_iv_raw, ATM_IV_EMA_SPAN)
        day_prv15m = data[:, RAW_PARKINSON_RV_15M_INDEX].copy()
        day_movement_ewma = _ema_1d(day_prv15m, MOVEMENT_EWMA_SPAN)
        day_rv30 = data[:, RAW_PARKINSON_RV_30M_INDEX].copy()

        _extend_bar_arrays(n_bars)
        bar_atm_iv_ema_arr[bar_idx:bar_idx + n_bars] = day_iv_ema
        bar_movement_ewma_arr[bar_idx:bar_idx + n_bars] = day_movement_ewma
        bar_rv30_arr[bar_idx:bar_idx + n_bars] = day_rv30
        bar_positions_arr[bar_idx:bar_idx + n_bars] = np.arange(n_bars, dtype=np.int32)
        bar_dates_list.extend([ds] * n_bars)

        # ---- Apply VIX forward-fill for SSL features ----
        apply_vix_forwardfill(data)

        # C10 mask weights
        day_weights = compute_ssl_mask_weights(data, liveness, ACTIVE_SSL_FEATURES)

        # Extract target variates (VIX ffill doesn't affect these SPX/options variates)
        day_v105 = data[:, RAW_SESSION_LOG_RET_INDEX]
        day_v111 = data[:, RAW_PARKINSON_RV_15M_INDEX]
        day_v112 = data[:, RAW_PARKINSON_RV_30M_INDEX]
        day_v46 = data[:, RAW_ATM_SPREAD_INDEX]
        # new probe variates
        day_gamma = data[:, RAW_ATM_GAMMA_INDEX]
        day_theta = data[:, RAW_ATM_THETA_INDEX]
        day_iv25dp = data[:, RAW_IV_25DP_INDEX]
        day_iv25dc = data[:, RAW_IV_25DC_INDEX]
        day_iv_atm = data[:, RAW_IV_ATM_INDEX]
        day_flow = data[:, RAW_NET_PREMIUM_FLOW_INDEX]

        # SSL features for windows (v2: 129, v3: 147 incl. flow aggregates)
        day_ssl = data[:, ACTIVE_SSL_FEATURES]

        # ---- Build sliding windows + extract forward targets ----
        for start_pos in range(0, n_bars - T + 1, stride):
            end_pos = start_pos + T
            last_bar_pos = end_pos - 1

            if win_idx >= len(X_windows):
                _extend_win_arrays()

            X_windows[win_idx] = day_ssl[start_pos:end_pos]
            W_windows[win_idx] = day_weights[start_pos:end_pos]
            window_dates_list.append(ds)
            window_last_bar_pos_list.append(last_bar_pos)
            window_last_bar_global[win_idx] = bar_idx + last_bar_pos

            # Per-horizon forward target extraction
            for k in all_horizons:
                fwd_bar = last_bar_pos + k
                if fwd_bar < n_bars and fwd_bar <= last_live:
                    fwd_valid[k][win_idx] = True

                    # Probe A: forward return = Δ session_log_return
                    if k in fwd_ret:
                        fwd_ret[k][win_idx] = day_v105[fwd_bar] - day_v105[last_bar_pos]

                    # Probe B: forward Parkinson RV (raw, pre-log)
                    # v111 is trailing 15-bar RV; at bar t+15, it covers [t+1..t+15] = forward RV.
                    # v112 is trailing 30-bar RV; at bar t+30, it covers [t+1..t+30] = forward RV.
                    # INVARIANT: horizon k must equal the RV window length for correct semantics.
                    if k in fwd_rv:
                        fwd_rv[k][win_idx] = day_v111[fwd_bar] if k <= 15 else day_v112[fwd_bar]

                    # Probe D: ATM spread delta
                    if k == ph["probe_d_horizon"]:
                        fwd_spread[win_idx] = day_v46[fwd_bar] - day_v46[last_bar_pos]

                    # new probes (all at k=15 horizon)
                    if k == 15:
                        # PredGammaAccel: gamma/theta ratio change
                        g_now = day_gamma[last_bar_pos]
                        t_now = day_theta[last_bar_pos]
                        g_fwd = day_gamma[fwd_bar]
                        t_fwd = day_theta[fwd_bar]
                        if abs(t_now) > 0.001 and abs(t_fwd) > 0.001:
                            g2t_now = g_now / abs(t_now)
                            g2t_fwd = g_fwd / abs(t_fwd)
                            fwd_gamma_accel[win_idx] = g2t_fwd - g2t_now

                        # PredSmileConvexity: IV butterfly spread change
                        bf_now = (day_iv25dp[last_bar_pos] + day_iv25dc[last_bar_pos]) / 2 - day_iv_atm[last_bar_pos]
                        bf_fwd = (day_iv25dp[fwd_bar] + day_iv25dc[fwd_bar]) / 2 - day_iv_atm[fwd_bar]
                        if day_iv_atm[last_bar_pos] > 0.001 and day_iv_atm[fwd_bar] > 0.001:
                            fwd_smile_convex[win_idx] = bf_fwd - bf_now

                        # PredFlowToxicity: forward flow z-score
                        # Use absolute value of flow as toxicity proxy
                        fwd_flow_tox[win_idx] = abs(day_flow[fwd_bar])

            win_idx += 1

        bar_idx += n_bars
        del data, liveness, day_ssl, day_weights
        gc.collect()

        if n_days % 50 == 0:
            print(f"  {n_days} days loaded, {win_idx} windows...")

    # ---- Trim arrays ----
    X_windows = X_windows[:win_idx]
    W_windows = W_windows[:win_idx]
    window_dates = np.array(window_dates_list)
    window_last_bar_pos = np.array(window_last_bar_pos_list, dtype=np.int32)
    window_last_bar_global = window_last_bar_global[:win_idx]
    fwd_spread = fwd_spread[:win_idx]
    fwd_gamma_accel = fwd_gamma_accel[:win_idx]
    fwd_smile_convex = fwd_smile_convex[:win_idx]
    fwd_flow_tox = fwd_flow_tox[:win_idx]
    for k in all_horizons:
        fwd_valid[k] = fwd_valid[k][:win_idx]
    for k in fwd_ret:
        fwd_ret[k] = fwd_ret[k][:win_idx]
    for k in fwd_rv:
        fwd_rv[k] = fwd_rv[k][:win_idx]

    bar_atm_iv_ema_arr = bar_atm_iv_ema_arr[:bar_idx]
    bar_movement_ewma_arr = bar_movement_ewma_arr[:bar_idx]
    bar_rv30_arr = bar_rv30_arr[:bar_idx]
    bar_positions_arr = bar_positions_arr[:bar_idx]
    bar_dates = np.array(bar_dates_list)

    # ---- Compute regime labels for Probe C (future regime at t+30) ----
    print(f"\nComputing regime labels for Probe C ({bar_idx} bars)...")
    # _compute_thresholds uses bar_dates <= TRAIN_END for training bars
    thresholds = _compute_thresholds(
        bar_atm_iv_ema_arr,  # bar_feats param (only used for length)
        bar_atm_iv_ema_arr, bar_movement_ewma_arr,
        bar_dates, bar_positions_arr, bar_rv30_arr)

    vol_class, move_class, warmup_mask = _classify_bars(
        bar_atm_iv_ema_arr,  # bar_feats param (only used for length)
        bar_atm_iv_ema_arr, bar_movement_ewma_arr,
        bar_dates, bar_positions_arr, thresholds, bar_rv30_arr)

    cfg = GRID_CONFIGS[4]  # : 4-class regime taxonomy
    bar_regime_labels = _apply_grid(vol_class, move_class, warmup_mask, cfg)

    # Assign Probe C: regime label at t + probe_c_horizon
    c_horizon = ph["probe_c_horizon"]
    fwd_regime = np.full(win_idx, -1, dtype=np.int64)
    n_c_valid = 0
    for wi in range(win_idx):
        lb_global = int(window_last_bar_global[wi])
        fwd_global = lb_global + c_horizon
        if (fwd_global < bar_idx and
                bar_dates[fwd_global] == bar_dates[lb_global] and
                bar_positions_arr[fwd_global] <= last_live and
                bar_regime_labels[fwd_global] >= 0):
            fwd_regime[wi] = bar_regime_labels[fwd_global]
            n_c_valid += 1

    # Concurrent regime ( Objective 1): regime at the LAST BAR of the
    # current window (horizon=0). This is a pure current-state encoding test
    # --- if the encoder cannot classify the regime from its embedding at the
    # window end, it has not learned the market state. No forward-prediction
    # involved; validates representation quality independent of transfer.
    concurrent_regime = np.full(win_idx, -1, dtype=np.int64)
    n_concurrent_valid = 0
    for wi in range(win_idx):
        lb_global = int(window_last_bar_global[wi])
        if (lb_global < bar_idx and
                bar_regime_labels[lb_global] >= 0):
            concurrent_regime[wi] = bar_regime_labels[lb_global]
            n_concurrent_valid += 1

    # ---- Summary statistics ----
    n_valid = {k: int(fwd_valid[k].sum()) for k in all_horizons}
    gb = win_idx * T * n_ssl * 4 * 2 / 1e9
    print(f"\nDone. {n_days} days ({n_skipped} skipped), {win_idx} windows, {gb:.2f} GB (X+W)")
    print(f"Forward target validity:")
    for k in all_horizons:
        print(f"  horizon={k}: {n_valid[k]}/{win_idx} ({100*n_valid[k]/max(win_idx,1):.1f}%)")
    print(f"  Probe C (regime at t+{c_horizon}): {n_c_valid}/{win_idx} ({100*n_c_valid/max(win_idx,1):.1f}%)")
    print(f"  Concurrent regime (t=0): {n_concurrent_valid}/{win_idx} ({100*n_concurrent_valid/max(win_idx,1):.1f}%)")

    return dict(
        X_windows=X_windows,
        W_windows=W_windows,
        window_dates=window_dates,
        window_last_bar_pos=window_last_bar_pos,
        fwd_ret=fwd_ret,
        fwd_rv=fwd_rv,
        fwd_regime=fwd_regime,
        concurrent_regime=concurrent_regime,
        fwd_spread=fwd_spread,
        fwd_gamma_accel=fwd_gamma_accel,
        fwd_smile_convex=fwd_smile_convex,
        fwd_flow_tox=fwd_flow_tox,
        fwd_valid=fwd_valid,
        regime_thresholds=thresholds,
        # #210 fix (2026-04-29): bar-level data for per-fold regime threshold
        # recomputation. Previously _compute_thresholds used a global TRAIN_END
        # cutoff; per-fold refit requires access to the raw bar signals so
        # thresholds can be derived from each fold's training bars only.
        _bar_atm_iv_ema=bar_atm_iv_ema_arr,
        _bar_movement_ewma=bar_movement_ewma_arr,
        _bar_rv30=bar_rv30_arr,
        _bar_dates=bar_dates,
        _bar_positions=bar_positions_arr,
        _window_last_bar_global=window_last_bar_global,
        _bar_regime_labels_global=bar_regime_labels,  # from global thresholds (legacy fallback)
    )


def compute_probe_targets(probe_data):
    """Transform raw forward values into probe-ready target arrays.

    Fold-independent transforms applied here:
    - Probe A: raw forward returns stored as-is (sign + dead-zone applied per-fold)
    - Probe B: log(max(rv, LOG_RV_FLOOR)) transform
    - Probe C: future regime labels passed through (int64, -1 = invalid)
    - Probe D: raw spread deltas stored as-is (dead-zone applied per-fold)

    Dead-zone is NOT applied here — it requires per-fold train stdev.
    Use apply_dead_zone() during per-fold evaluation.

    Returns dict with:
        targets: dict[str, ndarray] — target arrays keyed by probe name
        validity: dict[str, ndarray(bool)] — per-target validity masks
    """
    ph = PROBE_HYPERPARAMS
    n_windows = len(probe_data["X_windows"])
    targets = {}
    validity = {}

    # Probe A: raw forward returns (sign extraction deferred to per-fold dead-zone)
    for k in ph["probe_a_horizons"]:
        key = f"ret_{k}"
        targets[key] = probe_data["fwd_ret"][k].copy()
        validity[key] = probe_data["fwd_valid"][k].copy()

    # Probe B: log-transform forward RV with floor guard
    for k in ph["probe_b_horizons"]:
        key = f"log_rv_{k}"
        raw_rv = probe_data["fwd_rv"][k]
        targets[key] = np.log(np.maximum(raw_rv, LOG_RV_FLOOR)).astype(np.float32)
        # Invalid if horizon out of range OR raw RV is exactly 0 (dead/warmup bar)
        validity[key] = probe_data["fwd_valid"][k] & (raw_rv > 0)

    # Probe C: future regime at t+probe_c_horizon
    targets["regime"] = probe_data["fwd_regime"].copy()
    validity["regime"] = probe_data["fwd_regime"] >= 0

    # Probe C2: Concurrent regime at t=0 ( Objective 1 — pure current-state test)
    # Uses regime label at the LAST BAR of the window, not a forward prediction.
    # If the encoder cannot classify the current regime from its embedding, it
    # has not learned the market state — regardless of forward-prediction ability.
    # Backward-compatible: if probe_data was generated before this addition,
    # skip gracefully (older windowing code lacks 'concurrent_regime' key).
    if "concurrent_regime" in probe_data:
        targets["regime_concurrent"] = probe_data["concurrent_regime"].copy()
        validity["regime_concurrent"] = probe_data["concurrent_regime"] >= 0

    # Probe D: raw spread delta
    d_k = ph["probe_d_horizon"]
    key = f"spread_{d_k}"
    targets[key] = probe_data["fwd_spread"].copy()
    validity[key] = probe_data["fwd_valid"][d_k].copy()

    # new probes (backward-compatible: skip if not in probe_data)
    if "fwd_gamma_accel" in probe_data:
        targets["gamma_accel"] = probe_data["fwd_gamma_accel"].copy()
        validity["gamma_accel"] = ~np.isnan(probe_data["fwd_gamma_accel"]) & probe_data["fwd_valid"].get(15, np.ones(n_windows, dtype=bool))

    if "fwd_smile_convex" in probe_data:
        targets["smile_convex"] = probe_data["fwd_smile_convex"].copy()
        validity["smile_convex"] = ~np.isnan(probe_data["fwd_smile_convex"]) & probe_data["fwd_valid"].get(15, np.ones(n_windows, dtype=bool))

    # PredJump: binary classification from forward 30-bar returns
    if 30 in probe_data.get("fwd_ret", {}):
        jump_threshold = 0.01  # 1% SPX move
        fwd_ret_30 = probe_data["fwd_ret"][30]
        targets["jump_30"] = (np.abs(fwd_ret_30) > jump_threshold).astype(np.float32)
        validity["jump_30"] = probe_data["fwd_valid"].get(30, np.ones(n_windows, dtype=bool)) & ~np.isnan(fwd_ret_30)

    if "fwd_flow_tox" in probe_data:
        # Normalize flow toxicity to z-score using training statistics
        raw_flow = probe_data["fwd_flow_tox"].copy()
        targets["flow_tox"] = raw_flow
        validity["flow_tox"] = ~np.isnan(raw_flow) & probe_data["fwd_valid"].get(15, np.ones(n_windows, dtype=bool))

    # Summary
    print(f"\nProbe targets ({n_windows} windows):")
    for k, mask in sorted(validity.items()):
        n = int(mask.sum())
        print(f"  {k}: {n}/{n_windows} valid ({100*n/max(n_windows,1):.1f}%)")

    return dict(targets=targets, validity=validity)


def apply_dead_zone(target_values, valid_mask, train_mask, multiplier=None):
    """Compute dead-zone mask for continuous probe targets.

    BLOCKER fix (agent review): uses FIXED stdev from the train fold's valid
    windows, not within-day stdev (which has look-ahead bias since it
    includes future bars from the same day).

    Windows where |target| < multiplier * train_stdev are excluded from
    both training and evaluation to avoid noise-dominated labels.

    Args:
        target_values: (N,) float32 — raw target values (e.g., forward returns)
        valid_mask: (N,) bool — pre-existing validity (bar-position, etc.)
        train_mask: (N,) bool — which windows are in this fold's training set
        multiplier: float — dead-zone width in stdev units (default from config)

    Returns:
        (N,) bool — updated validity mask (valid_mask AND above dead-zone)
    """
    if multiplier is None:
        multiplier = PROBE_HYPERPARAMS["dead_zone_std_multiplier"]

    # Stdev from valid training windows only
    train_valid = valid_mask & train_mask
    train_vals = target_values[train_valid]
    train_vals = train_vals[~np.isnan(train_vals)]

    if len(train_vals) < 10:
        print(f"    [dead-zone] WARNING: only {len(train_vals)} valid train samples, skipping")
        return valid_mask.copy()

    train_std = float(np.std(train_vals))
    threshold = multiplier * train_std
    above_dz = np.abs(target_values) >= threshold
    combined = valid_mask & above_dz
    n_removed = int(valid_mask.sum() - combined.sum())
    print(f"    [dead-zone] std={train_std:.6f}, threshold={threshold:.6f}, "
          f"removed {n_removed} ({100*n_removed/max(int(valid_mask.sum()),1):.1f}%)")

    return combined


def make_probe_folds(probe_data, n_folds=None, embargo_days=None):
    """Generate expanding-window walk-forward fold splits for probe evaluation.

    Protocol (de Prado 2018, §11.4):
    - Dates split into (n_folds + 1) blocks of roughly equal size
    - Fold i: train on blocks 0..i, test on block i+1
    - embargo_days TRADING DAYS gap between train end and test start
    - Training window expands monotonically; test blocks are non-overlapping

    Returns list of n_folds (train_idx, test_idx) tuples, where each is
    an ndarray of integer indices into probe_data arrays.
    """
    if n_folds is None:
        n_folds = PROBE_HYPERPARAMS["n_folds"]
    if embargo_days is None:
        embargo_days = PROBE_HYPERPARAMS["embargo_days"]

    window_dates = probe_data["window_dates"]
    unique_dates = np.array(sorted(set(window_dates)))
    n_dates = len(unique_dates)

    # Split dates into (n_folds + 1) blocks
    block_size = n_dates // (n_folds + 1)
    if block_size < 10:
        raise ValueError(
            f"Only {n_dates} unique dates for {n_folds} folds "
            f"(block_size={block_size} < 10). Reduce n_folds or add data."
        )

    folds = []
    for i in range(n_folds):
        # Train: all dates in blocks 0 through i
        train_end_idx = (i + 1) * block_size - 1
        train_end_date = unique_dates[train_end_idx]

        # Test: block (i + 1). Last fold gets all remaining dates.
        test_start_idx = (i + 1) * block_size
        if i == n_folds - 1:
            test_end_idx = n_dates - 1
        else:
            test_end_idx = (i + 2) * block_size - 1

        if test_start_idx >= n_dates:
            break

        # Apply embargo: skip embargo_days trading days after train end
        embargoed_test_start_idx = min(test_start_idx + embargo_days, n_dates)
        if embargoed_test_start_idx > test_end_idx:
            print(f"  Fold {i}: embargo ate entire test block, skipping")
            continue

        test_start_date = unique_dates[embargoed_test_start_idx]
        test_end_date = unique_dates[test_end_idx]

        # Map to window indices
        train_idx = np.where(window_dates <= train_end_date)[0]
        test_idx = np.where(
            (window_dates >= test_start_date) &
            (window_dates <= test_end_date)
        )[0]

        if len(train_idx) < 100 or len(test_idx) < 10:
            print(f"  Fold {i}: insufficient data (train={len(train_idx)}, "
                  f"test={len(test_idx)}), skipping")
            continue

        folds.append((train_idx, test_idx))

    # Summary
    print(f"\nWalk-forward folds: {len(folds)}/{n_folds} valid "
          f"(embargo={embargo_days} trading days)")
    for i, (tr, te) in enumerate(folds):
        tr_dates = sorted(set(window_dates[tr]))
        te_dates = sorted(set(window_dates[te]))
        print(f"  Fold {i}: train {len(tr)} win ({len(tr_dates)} days: "
              f"{tr_dates[0]}..{tr_dates[-1]}), "
              f"test {len(te)} win ({len(te_dates)} days: "
              f"{te_dates[0]}..{te_dates[-1]})")

    return folds


def encode_windows(qb, X_windows, seeds=None):
    """Encode SSL windows through frozen encoder, returning mean-pooled embeddings.

    Loads the SSL checkpoint from ObjectStore, applies its z-score stats
    (encoder scope — fixed, not per-fold), runs the frozen encoder in eval
    mode with torch.no_grad(), and mean-pools variate tokens.

    Args:
        qb: QuantBook instance (for ObjectStore access)
        X_windows: (N, T, V=129) float32 — raw (pre-z-score) SSL feature windows
        seeds: list[int] or None — if provided, compute temporally-shuffled
               embeddings for each seed (mandatory baseline 2: temporal shuffle)

    Returns dict with:
        emb_full: (N, d_model) float32 — mean-pool over all 129 tokens
        emb_grid: (N, d_model) float32 — mean-pool over first 88 tokens (options grid)
        emb_shuffled: dict[int, (N, d_model)] — per-seed shuffled embeddings (if seeds given)
        z_stats: dict — z-score stats from checkpoint
        model: iTransformerEncoder — frozen encoder (for inspection if needed)
    """
    # Load checkpoint
    print(f"Loading SSL checkpoint from ObjectStore ({ACTIVE_CKPT_KEY})...")
    raw = qb.ObjectStore.Read(ACTIVE_CKPT_KEY)
    buf = io.BytesIO(base64.b64decode(raw))
    checkpoint = torch.load(buf, map_location=device, weights_only=False)

    if "z_mean" not in checkpoint or "z_std" not in checkpoint:
        raise RuntimeError(
            "Checkpoint missing z_mean/z_std — training was likely interrupted "
            "before save_ssl_artifacts() ran. Re-run run_ssl_pipeline().")
    z_mean = checkpoint["z_mean"]
    z_std = checkpoint["z_std"]
    log1p_cols = checkpoint.get("log1p_cols", [])
    hp = checkpoint.get("ssl_hyperparams", SSL_HYPERPARAMS)
    n_variates = checkpoint.get("n_ssl_features", N_SSL_FEATURES)
    d_model = hp["d_model"]

    # Reconstruct and load encoder
    model = iTransformerEncoder(hp, n_variates=n_variates)
    model.load_state_dict(checkpoint["encoder_state_dict"])
    model.eval()
    model.to(device)
    print(f"  Encoder loaded: {sum(p.numel() for p in model.parameters())} params, "
          f"d_model={d_model}, n_variates={n_variates}")

    # Apply same preprocessing as make_ssl_split_and_loaders (encoder scope):
    # 1. -b: log1p compression for heavy-tailed variates
    X = X_windows.copy()
    if log1p_cols:
        X[:, :, log1p_cols] = np.sign(X[:, :, log1p_cols]) * np.log1p(np.abs(X[:, :, log1p_cols]))
    # 2. z-score with checkpoint stats + clip
    X = ((X - z_mean) / z_std).astype(np.float32)
    X = np.clip(X, -5.0, 5.0)

    N = len(X)
    batch_size = 256

    # Group boundaries in SSL_FEATURES positions (for group-concat pooling)
    group_bounds = [
        (PROBE_FEATURE_GROUPS["options_grid"][0], PROBE_FEATURE_GROUPS["options_grid"][1]),
        (PROBE_FEATURE_GROUPS["strike_agg"][0],   PROBE_FEATURE_GROUPS["strike_agg"][1]),
        (PROBE_FEATURE_GROUPS["spx_derived"][0],  PROBE_FEATURE_GROUPS["spx_derived"][1]),
        (PROBE_FEATURE_GROUPS["vix_term"][0],     PROBE_FEATURE_GROUPS["vix_term"][1]),
        (PROBE_FEATURE_GROUPS["order_flow"][0],   PROBE_FEATURE_GROUPS["order_flow"][1]),
    ]
    n_groups = len(group_bounds)
    group_dim = n_groups * d_model  # 5 × d_model

    # Fine-grained sub-groups within the options grid (for spread diagnostic)
    # Grid has 11 points × 8 features. Feature offsets: mid=0, iv=1, theta=2,
    # gamma=3, qliq=4, qimb=5, spread=6, spread_chg=7
    grid_iv_idx = [i * 8 + 1 for i in range(11)]       # 11 IV tokens
    grid_spread_idx = [i * 8 + 6 for i in range(11)] + [i * 8 + 7 for i in range(11)]  # 22 spread+spread_chg tokens
    grid_greeks_idx = [i * 8 + 2 for i in range(11)] + [i * 8 + 3 for i in range(11)]  # 22 theta+gamma tokens
    grid_other_idx = [i for i in range(88) if i not in grid_iv_idx + grid_spread_idx + grid_greeks_idx]  # remaining 33
    fine_group_indices = [
        grid_iv_idx,        # 11 tokens: IV across moneyness
        grid_spread_idx,    # 22 tokens: spread + spread_chg
        grid_greeks_idx,    # 22 tokens: theta + gamma
        grid_other_idx,     # 33 tokens: mid, qliq, qimb
    ]
    # Plus the 4 non-grid groups from the original grouping
    for gs, ge in group_bounds[1:]:  # skip options_grid, add strike_agg, spx, vix, order_flow
        fine_group_indices.append(list(range(gs, ge)))
    n_fine_groups = len(fine_group_indices)  # 8 total

    n_layers = hp.get("n_layers", 3)
    ml_dim = d_model * n_layers  # Phase 2: multi-layer concat dimension
    ml_group_dim = n_groups * ml_dim  # group-concat with multi-layer

    def _encode_batch(x_np, full_only=False):
        """Encode a batch of z-scored windows → multiple pooled embedding types.

        Phase 2: multi-layer extraction (concatenate all layer outputs) for
        all pooled embeddings. emb_flat (V*D, was 18,816-d on v3) was
        removed — consistent OOM source on QC Research during RidgeCV;
        emb_fine covers the spatial-detail role and is well-conditioned.

        Args:
            x_np: (N, T, V) z-scored input
            full_only: if True, only compute emb_full (for shuffle baseline)
        """
        n = len(x_np)
        emb_f = np.empty((n, ml_dim), dtype=np.float32)
        if full_only:
            with torch.no_grad():
                for s in range(0, n, batch_size):
                    e = min(s + batch_size, n)
                    x_batch = torch.from_numpy(x_np[s:e]).to(device)
                    z = model.encode_variates(x_batch, multi_layer=True)  # (B, V, D*L)
                    emb_f[s:e] = z.mean(dim=1).cpu().numpy()
            return emb_f
        emb_g = np.empty((n, ml_dim), dtype=np.float32)
        emb_grp = np.empty((n, ml_group_dim), dtype=np.float32)
        emb_mx = np.empty((n, ml_dim), dtype=np.float32)
        emb_fine = np.empty((n, n_fine_groups * ml_dim), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                x_batch = torch.from_numpy(x_np[s:e]).to(device)
                # Multi-layer for pooled embeddings
                z_ml = model.encode_variates(x_batch, multi_layer=True)  # (B, V, D*L)
                emb_f[s:e] = z_ml.mean(dim=1).cpu().numpy()
                emb_g[s:e] = z_ml[:, :N_GRID_TOKENS, :].mean(dim=1).cpu().numpy()
                emb_mx[s:e] = z_ml.max(dim=1)[0].cpu().numpy()
                # Group-concat: mean-pool within each group, concatenate
                parts = []
                for gs, ge in group_bounds:
                    parts.append(z_ml[:, gs:ge, :].mean(dim=1))
                emb_grp[s:e] = torch.cat(parts, dim=1).cpu().numpy()
                # Fine-grained group-concat: 8 sub-groups (4 grid + 4 non-grid)
                fine_parts = []
                for idx_list in fine_group_indices:
                    fine_parts.append(z_ml[:, idx_list, :].mean(dim=1))
                emb_fine[s:e] = torch.cat(fine_parts, dim=1).cpu().numpy()
        return emb_f, emb_g, emb_grp, emb_mx, emb_fine

    # Encode all windows
    print(f"  Encoding {N} windows...")
    emb_full, emb_grid, emb_group, emb_max, emb_fine = _encode_batch(X)
    print(f"  emb_full:  {emb_full.shape[1]}-d (multi-layer: {n_layers}L × {d_model}), "
          f"norm range [{np.linalg.norm(emb_full, axis=1).min():.3f}, "
          f"{np.linalg.norm(emb_full, axis=1).max():.3f}]")
    print(f"  emb_group: {emb_group.shape[1]}-d (5 groups × {n_layers}L × {d_model})")
    print(f"  emb_fine:  {emb_fine.shape[1]}-d (8 fine groups × {n_layers}L × {d_model})")
    print(f"  emb_max:   {emb_max.shape[1]}-d (multi-layer)")

    result = dict(
        emb_full=emb_full,
        emb_grid=emb_grid,
        emb_group=emb_group,
        emb_fine=emb_fine,
        emb_max=emb_max,
        z_stats=dict(mean=z_mean, std=z_std, log1p_cols=log1p_cols),
        model=model,
        hp=hp,
        n_variates=n_variates,
    )

    # Shuffled baseline: permute temporal dimension within each window per seed
    # Only need mean-pooled full embedding for shuffle baseline comparison
    if seeds:
        result["emb_shuffled"] = {}
        for seed in seeds:
            print(f"  Encoding shuffled (seed={seed})...")
            rng = np.random.default_rng(seed)
            X_shuf = X.copy()
            for i in range(N):
                perm = rng.permutation(X_shuf.shape[1])
                X_shuf[i] = X_shuf[i, perm, :]
            emb_s = _encode_batch(X_shuf, full_only=True)
            result["emb_shuffled"][seed] = emb_s
            del X_shuf

    del X
    gc.collect()
    return result


# ===================================================================
# 4D-bis. HELD-OUT MASKED-CELL RECONSTRUCTION PROBE (#217, 2026-04-29)
# ===================================================================

def evaluate_held_out_mvr_probe(qb, X_windows, fold_indices, n_seeds=3,
                                 batch_size=64, W_windows=None):
    """Held-out masked-variate reconstruction probe — directly tests the SSL
    pretraining task.

    For each walk-forward fold's TEST set, applies the same masking strategy
    used in training (variate + cell + RV/flow block expansion), forwards
    masked windows through the frozen encoder + recon head, and reports
    per-variate MSE on masked cells. Multiple mask seeds per fold to reduce
    sampling variance.

    Why this probe matters: every other probe tests downstream usefulness of
    pooled embeddings. None directly tests whether the encoder solves the
    pretraining task it was trained on. Without this probe, we cannot
    distinguish "good representations whose pooling collapses signal" from
    "bad representations that simply didn't learn".

    Args:
        qb: QuantBook instance (for ObjectStore checkpoint load).
        X_windows: (N, T, V) float32 — raw (pre-z-score) windows aligned with
            the indices in `fold_indices`.
        fold_indices: list of (train_idx, test_idx) — walk-forward CV folds.
            Only the test_idx side is used; train_idx is ignored.
        n_seeds: int — number of mask-resamples per fold (default 3).
        batch_size: int — forward batch size (matches encode_windows default).
        W_windows: optional (N, T, V) float32 — C10 quality weights matching
            X_windows. When provided, the per-cell MSE is weighted by W,
            matching the trained loss exactly. When None (default), unweighted
            uniform MSE — easier to compute but does NOT match the trained
            objective. Pass W_windows from probe_data["W_windows"] (output of
            load_probe_days) to make the probe headline directly comparable
            to the training-time `val_loss_unweighted` metric.

    Returns dict with:
        per_fold_per_variate_mse: (n_folds, V) float32 — fold-mean across
            seeds of per-variate masked-MSE, NaN where no cells were masked
            for that variate in that fold.
        per_fold_overall_mse: (n_folds,) float32 — per-fold mean across
            variates and seeds (the headline number).
        mean_mse, median_mse: floats — across-fold aggregates.
        n_seeds: int (echo back for provenance).
        weighted: bool — whether C10 weights were applied.
    """
    print(f"Loading SSL checkpoint from ObjectStore ({ACTIVE_CKPT_KEY})...")
    raw = qb.ObjectStore.Read(ACTIVE_CKPT_KEY)
    buf = io.BytesIO(base64.b64decode(raw))
    ckpt = torch.load(buf, map_location=device, weights_only=False)
    if "z_mean" not in ckpt or "z_std" not in ckpt:
        raise RuntimeError("Checkpoint missing z_mean/z_std — re-run training.")
    z_mean = ckpt["z_mean"]
    z_std = ckpt["z_std"]
    log1p_cols = ckpt.get("log1p_cols", [])
    hp = ckpt.get("ssl_hyperparams", SSL_HYPERPARAMS)
    T = int(hp.get("seq_len", 60))

    # Pass-A B2 fix (2026-04-29): build eligibility + block-mask infrastructure
    # FROM THE CHECKPOINT, not module globals. Otherwise a probe run against an
    # SSL-010 (v2, 129 variates) checkpoint while module is configured for v3
    # (147 variates) would silently mismatch shapes/masks.
    ckpt_features = list(ckpt.get("ssl_features", ACTIVE_SSL_FEATURES))
    ckpt_use_v3 = bool(ckpt.get("use_v3_features", USE_V3_FEATURES))
    n_variates = ckpt.get("n_ssl_features", len(ckpt_features))
    if n_variates != len(ckpt_features):
        raise RuntimeError(
            f"Checkpoint inconsistency: n_ssl_features={n_variates} but "
            f"len(ssl_features)={len(ckpt_features)}.")
    # Reconstruct ineligible set from the *same* construction logic the
    # training run used — derived purely from feature membership, not from
    # the live module-level frozenset.
    if ckpt_use_v3:
        ckpt_flow_agg = [v for v in ckpt_features if 141 <= v < 159]
        ckpt_ineligible = SSL_MASK_INELIGIBLE | frozenset(ckpt_flow_agg)
    else:
        ckpt_ineligible = SSL_MASK_INELIGIBLE

    # Reconstruct encoder + recon head, load weights, freeze.
    model = iTransformerEncoder(hp, n_variates=n_variates).to(device)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.eval()
    recon_head = _build_recon_head_from_hp(
        hp, int(hp["d_model"]), T, n_variates, ckpt_features).to(device)
    recon_head.load_state_dict(ckpt["recon_head_state_dict"])
    recon_head.eval()

    # Mask-eligibility + block-expansion derived from CHECKPOINT feature_indices.
    eligible_np = _build_eligible_mask(ckpt_features, ckpt_ineligible)
    eligible = torch.from_numpy(eligible_np).to(device)
    flow_agg_positions = _build_flow_agg_position_map(ckpt_features) \
        if ckpt_use_v3 else None
    rv_positions = _build_rv_position_map(ckpt_features) \
        if hp.get("rv_block_mask", False) else None
    mask_fill = float(hp.get("mask_fill_value", 0.0))
    use_mask_indicator = bool(hp.get("use_mask_indicator", False))

    # Pass-C NEW-B1 fix (2026-04-29): C10 quality-weight tensor for matching
    # the trained loss. When None, fall back to unweighted MSE (same as before
    # this fix) — but the headline is no longer directly comparable to
    # val_loss_unweighted. Pass W_windows to make the probe a faithful
    # replica of the trained objective.
    weighted = W_windows is not None
    if weighted:
        if W_windows.shape != X_windows.shape:
            raise ValueError(
                f"W_windows shape {W_windows.shape} != X_windows {X_windows.shape}")

    # Apply preprocessing matching encode_windows / make_ssl_split_and_loaders.
    X = X_windows.copy()
    if log1p_cols:
        X[:, :, log1p_cols] = np.sign(X[:, :, log1p_cols]) * np.log1p(np.abs(X[:, :, log1p_cols]))
    X = (X - z_mean) / z_std
    X = np.clip(X, -5.0, 5.0).astype(np.float32)

    F = len(fold_indices)
    V = n_variates
    fold_var_mse = np.full((F, V), np.nan, dtype=np.float32)
    fold_overall = np.full(F, np.nan, dtype=np.float32)
    fold_overall_cell_ratio = np.full(F, np.nan, dtype=np.float32)

    for fi, (_train_idx, test_idx) in enumerate(fold_indices):
        test_idx = np.asarray(test_idx)
        X_te = X[test_idx]
        W_te = W_windows[test_idx] if weighted else None
        n = len(X_te)
        # Pass-A H5 fix (2026-04-29): allow folds with n < batch_size; previously
        # silently skipped. Now we just process the whole fold in one micro-batch.
        if n < 1:
            continue
        seed_var_mse = np.full((n_seeds, V), np.nan, dtype=np.float32)
        for si in range(n_seeds):
            gen = torch.Generator(device=device).manual_seed(1000 * fi + si)
            sq_sum = torch.zeros(V, device=device)
            wn_sum = torch.zeros(V, device=device)  # weighted cell count when weighted, else raw
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                Xb = torch.from_numpy(X_te[s:e]).to(device)
                Xb = torch.nan_to_num(Xb, nan=0.0)
                B = Xb.shape[0]
                mask, _vmask = sample_ssl_mask(
                    B, T, V, eligible,
                    variate_ratio=float(hp.get("variate_ratio", 0.3)),
                    cell_ratio=float(hp.get("cell_ratio", 0.15)),
                    generator=gen,
                    flow_agg_positions=flow_agg_positions,
                    feature_indices=ckpt_features,
                    rv_positions=rv_positions,
                )
                Xm = Xb.clone()
                Xm[mask] = mask_fill
                with torch.no_grad():
                    tokens = model.encode_variates(
                        Xm, mask=mask if use_mask_indicator else None,
                    )
                    x_hat = recon_head(tokens)  # (B, V, T) — variate-major
                # Align targets to (B, V, T) variate-major to match x_hat shape.
                Xb_vt = Xb.permute(0, 2, 1)
                mask_vt = mask.permute(0, 2, 1).float()
                sq = (x_hat - Xb_vt) ** 2
                if weighted:
                    Wb = torch.from_numpy(W_te[s:e]).to(device).permute(0, 2, 1)
                    eff = mask_vt * Wb
                    sq_sum += (sq * eff).sum(dim=(0, 2))
                    wn_sum += eff.sum(dim=(0, 2))
                else:
                    sq_sum += (sq * mask_vt).sum(dim=(0, 2))
                    wn_sum += mask_vt.sum(dim=(0, 2))
            per_var = (sq_sum / wn_sum.clamp_min(1e-6)).cpu().numpy()
            per_var = np.where(wn_sum.cpu().numpy() > 0, per_var, np.nan)
            seed_var_mse[si] = per_var
            # Pass-A H3 / Pass-C B3 fix (2026-04-29): the trained val_loss
            # uses GLOBAL CELL RATIO `sum(sq*eff) / sum(eff)` over (B,T,V)
            # cells (pipeline.py:MaskedVariateLoss.forward). Per-variate-mean
            # is a DIFFERENT scalar — variates with more masked cells
            # dominate in the trained loss but receive equal weight in
            # variate-mean. Track BOTH so the headline can be either:
            # `mean_mse_cell_ratio` is comparable to val_loss_unweighted
            # `mean_mse_variate_mean` is the legacy headline (uniform
            # per-variate weight — useful for per-variate diagnostics).
            if si == 0:
                # First seed seeds these accumulators; later seeds add.
                seed_cell_sq_sum = sq_sum.clone()
                seed_cell_wn_sum = wn_sum.clone()
            else:
                seed_cell_sq_sum += sq_sum
                seed_cell_wn_sum += wn_sum
        # Mean over seeds, ignoring NaN cells
        fold_var_mse[fi] = np.nanmean(seed_var_mse, axis=0)
        fold_overall[fi] = float(np.nanmean(fold_var_mse[fi]))
        # Cell-ratio headline (matches trained val_loss_unweighted aggregation).
        cell_ratio_mse = float(
            (seed_cell_sq_sum.sum() / seed_cell_wn_sum.sum().clamp_min(1e-6))
            .cpu().item()
        )
        # Stash on the fold_overall slot via a parallel array.
        if fi == 0:
            fold_overall_cell_ratio = np.full(F, np.nan, dtype=np.float32)
        fold_overall_cell_ratio[fi] = cell_ratio_mse
        print(f"  fold {fi}: held-out MVR MSE = {fold_overall[fi]:.4f} "
              f"(variate-mean) | {cell_ratio_mse:.4f} (cell-ratio, matches "
              f"val_loss_unweighted) | n_test={n}, seeds={n_seeds}")

    finite = np.isfinite(fold_overall)
    finite_cr = np.isfinite(fold_overall_cell_ratio)
    return dict(
        per_fold_per_variate_mse=fold_var_mse.tolist(),
        per_fold_overall_mse=fold_overall.tolist(),
        per_fold_overall_mse_cell_ratio=fold_overall_cell_ratio.tolist(),
        mean_mse=float(fold_overall[finite].mean()) if finite.any() else float("nan"),
        median_mse=float(np.median(fold_overall[finite])) if finite.any() else float("nan"),
        # Cell-ratio aggregation matches the trained `val_loss_unweighted`.
        # Use this when comparing across runs to avoid the per-variate-mean
        # vs cell-ratio scalar mismatch (Pass-A H3 / Pass-C B3 fix).
        mean_mse_cell_ratio=float(fold_overall_cell_ratio[finite_cr].mean())
            if finite_cr.any() else float("nan"),
        median_mse_cell_ratio=float(np.median(fold_overall_cell_ratio[finite_cr]))
            if finite_cr.any() else float("nan"),
        n_seeds=n_seeds,
        n_folds=int(finite.sum()),
        weighted=weighted,
    )


# ===================================================================
# 4E. SKLEARN LINEAR PROBES
# ===================================================================

def fit_regression_probe(X_train, y_train, X_test, y_test):
    """Fit RidgeCV regression probe (Probes B, D).

    Uses sklearn's closed-form leave-one-out CV for alpha selection —
    no SGD, no learning rate, deterministic (Alain & Bengio 2017).
    L2 regularization applied to all inputs including raw baselines
    (fixes dimensionality fairness BLOCKER).

    Inputs are StandardScaler-normalized (per-fold train stats).
    Returns test-set R² and selected alpha.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = RidgeCV(alphas=np.logspace(-3, 5, 20))  # #216 unified grid (was logspace(-3,3,10), saturating ceiling)
    model.fit(X_tr, y_train)
    r2 = float(model.score(X_te, y_test))

    return dict(r2=r2, alpha=float(model.alpha_))


def fit_classification_probe(X_train, y_train, X_test, y_test):
    """Fit LogisticRegressionCV classification probe (Probes A, C).

    Uses sklearn's built-in CV for C (inverse L2 strength) selection.
    L2 regularization on all inputs including raw baselines.
    Reports balanced accuracy + AUC-ROC.

    Inputs are StandardScaler-normalized (per-fold train stats).
    """
    import warnings
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    n_classes = len(np.unique(y_train))
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Inner CV folds for C selection — cap at 3 or fewer if data is small
    inner_cv = min(3, max(2, len(y_train) // max(n_classes * 5, 1)))
    n_features = X_tr.shape[1]
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 10),
        cv=inner_cv,
        scoring="balanced_accuracy",
        max_iter=2000 if n_features > 1000 else 1000,
        solver="lbfgs",
        class_weight="balanced",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)      # sklearn convergence
        warnings.filterwarnings("ignore", category=FutureWarning)    # sklearn deprecations
        model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    # AUC-ROC (may fail if test split lacks some classes)
    try:
        y_proba = model.predict_proba(X_te)
        if n_classes == 2:
            auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            auc = float(roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro"))
    except (ValueError, IndexError):
        auc = float("nan")

    return dict(
        balanced_accuracy=bal_acc,
        auc_roc=auc,
        C=float(model.C_[0]),
    )


def fit_mlp_classification_probe(X_train, y_train, X_test, y_test):
    """Fit MLP classification probe (nonlinear diagnostic).

    Tests whether representation contains nonlinearly separable signal that
    linear probes miss. 1 hidden layer (128), early stopping.

    #221 fix 2026-04-29: replaced "best of 2 seeds, fixed alpha=1e-3" (which
    biases the metric upward by ~0.005-0.02 vs unbiased) with seed-averaged
    metrics over a small alpha grid. The reported metric is the alpha-grid
    mean over seeds — not the max. This matches how linear probes report
    (RidgeCV inner-CV picks alpha; we don't take "best of seeds" anywhere).
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    alpha_grid = (1e-4, 1e-3, 1e-2)  # short L2 reg sweep
    seeds = (0, 1)
    accs = []
    for alpha in alpha_grid:
        for seed in seeds:
            m = MLPClassifier(
                hidden_layer_sizes=(128,), max_iter=300,
                early_stopping=True, validation_fraction=0.15,
                random_state=seed, alpha=alpha,
            )
            m.fit(X_tr, y_train)
            acc = float(balanced_accuracy_score(y_test, m.predict(X_te)))
            accs.append(acc)
    mean_acc = float(np.mean(accs))
    # Note: MLPClassifier doesn't support class_weight='balanced' natively.
    # This asymmetry with linear probes is acceptable for a nonlinear diagnostic.
    return dict(balanced_accuracy=mean_acc, auc_roc=float("nan"))


def fit_mlp_regression_probe(X_train, y_train, X_test, y_test):
    """Fit MLP regression probe (nonlinear diagnostic).

    Tests whether representation contains nonlinearly accessible signal.
    1 hidden layer (128), early stopping.

    #221 fix 2026-04-29: replaced "best of 2 seeds, fixed alpha=1e-3" with
    seed-averaged metrics over a small alpha grid (see classification probe).
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    alpha_grid = (1e-4, 1e-3, 1e-2)
    seeds = (0, 1)
    r2s = []
    for alpha in alpha_grid:
        for seed in seeds:
            m = MLPRegressor(
                hidden_layer_sizes=(128,), max_iter=300,
                early_stopping=True, validation_fraction=0.15,
                random_state=seed, alpha=alpha,
            )
            m.fit(X_tr, y_train)
            r2s.append(float(m.score(X_te, y_test)))
    return dict(r2=float(np.mean(r2s)))


# Positions of MODEL_FEATURES within SSL_FEATURES (for raw-36 baseline)
_MF_SSL_POS = [SSL_FEATURES.index(v) for v in MODEL_FEATURES]

# Log1p column indices within SSL_FEATURES for raw baseline preprocessing
_LOG1P_SSL_POS = sorted(
    SSL_FEATURES.index(v) for v in SSL_LOG1P_VARIATES if v in SSL_FEATURES
)

# Return-direction probes are expected null (all representations ≈ chance).
# They are tracked for monitoring but excluded from the reportability tally.
_EXPECTED_NULL_PROBES = frozenset({"ret_5", "ret_15", "ret_30"})


def compute_handcrafted_features(X_windows_36):
    """Compute hand-crafted temporal summary statistics for encoder ablation.

    Extracts 10 statistics per variate from (N, T=60, 36) windows, producing
    a (N, 360) feature matrix. Tests whether the SSL encoder provides signal
    beyond what targeted summary statistics capture.

    Statistics per variate:
      Level:      last, mean, mean_5
      Trend:      slope (full window), slope_10 (last 10 bars)
      Volatility: std, range (max-min)
      Shape:      skew
      Dynamics:   acf_1 (autocorrelation at lag 1), last_minus_mean

    Args:
        X_windows_36: (N, T, 36) float32 — MODEL_FEATURES windows
                      (log1p already applied to heavy-tailed variates)

    Returns:
        (N, 360) float32 — 10 summary statistics x 36 variates
    """
    N, T, V = X_windows_36.shape
    n_stats = 10
    out = np.empty((N, V * n_stats), dtype=np.float32)

    # Linreg design matrices (precomputed)
    t_full = np.arange(T, dtype=np.float64)
    t_mean = t_full.mean()
    t_var = ((t_full - t_mean) ** 2).sum()

    t_10 = np.arange(10, dtype=np.float64)
    t10_mean = t_10.mean()
    t10_var = ((t_10 - t10_mean) ** 2).sum()

    for vi in range(V):
        col = X_windows_36[:, :, vi]  # (N, T)
        base = vi * n_stats

        # Level
        out[:, base + 0] = col[:, -1]                          # last
        col_mean = col.mean(axis=1)
        out[:, base + 1] = col_mean                            # mean
        out[:, base + 2] = col[:, -5:].mean(axis=1)           # mean_5

        # Trend
        x_centered = col - col_mean[:, None]
        out[:, base + 3] = (x_centered * (t_full - t_mean)).sum(axis=1) / t_var  # slope
        col_10 = col[:, -10:]
        x10_centered = col_10 - col_10.mean(axis=1, keepdims=True)
        out[:, base + 4] = (x10_centered * (t_10 - t10_mean)).sum(axis=1) / t10_var  # slope_10

        # Volatility
        out[:, base + 5] = col.std(axis=1)                    # std
        out[:, base + 6] = col.max(axis=1) - col.min(axis=1)  # range

        # Shape (moment-based skewness, no scipy needed)
        std_safe = np.maximum(out[:, base + 5], 1e-10)
        out[:, base + 7] = ((x_centered ** 3).mean(axis=1)) / (std_safe ** 3)  # skew

        # Dynamics: autocorrelation at lag 1
        x1 = col[:, :-1]
        x2 = col[:, 1:]
        x1c = x1 - x1.mean(axis=1, keepdims=True)
        x2c = x2 - x2.mean(axis=1, keepdims=True)
        num = (x1c * x2c).sum(axis=1)
        den = np.sqrt((x1c ** 2).sum(axis=1) * (x2c ** 2).sum(axis=1))
        den = np.maximum(den, 1e-10)
        out[:, base + 8] = num / den                           # acf_1

        # Deviation from mean
        out[:, base + 9] = col[:, -1] - col_mean              # last_minus_mean

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def run_handcrafted_probes():
    """Run all probes on hand-crafted summary statistics (no encoder).

    This is Run 0 of the tokenizer comparison: tests whether the SSL encoder
    provides signal beyond domain-expert feature engineering. Uses 10 summary
    statistics per variate x 36 MODEL_FEATURES = 360 features.

    Usage (in QC Research notebook):
        run_handcrafted_probes()
    """
    global _results
    qb = get_qb()
    ph = PROBE_HYPERPARAMS

    print("=" * 60)
    print("HAND-CRAFTED BASELINE (NO ENCODER)")
    print("=" * 60)

    # Step 1: Load probe data
    print("\n=== Step 1: Load probe data ===")
    if "probe_data" in _results:
        print("Reusing cached probe data")
        probe_data = _results["probe_data"]
    else:
        probe_data = load_probe_days(qb, TRAIN_START, TEST_END)
        _results["probe_data"] = probe_data

    # Step 2: Targets
    print("\n=== Step 2: Compute probe targets ===")
    probe_targets = compute_probe_targets(probe_data)
    _results["probe_targets"] = probe_targets

    # Step 3: Compute hand-crafted features
    print("\n=== Step 3: Compute hand-crafted features ===")
    X_raw = probe_data["X_windows"].copy()
    if _LOG1P_SSL_POS:
        X_raw[:, :, _LOG1P_SSL_POS] = (
            np.sign(X_raw[:, :, _LOG1P_SSL_POS])
            * np.log1p(np.abs(X_raw[:, :, _LOG1P_SSL_POS]))
        )
    # N1 fix: fill NaN warmup bars from flow aggregates v141-v158.
    np.nan_to_num(X_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X_mf = X_raw[:, :, _MF_SSL_POS]  # (N, T=60, |MODEL_FEATURES|)
    X_hc = compute_handcrafted_features(X_mf)
    n_mf = len(_MF_SSL_POS)
    n_hc = X_hc.shape[1]
    print(f"  Hand-crafted features: {n_hc}-d "
          f"(10 stats x {n_mf} MODEL_FEATURES)")

    # Also prepare raw baseline (T*V flat) for comparison
    N, T, V = X_raw.shape
    X_raw_mf = X_mf.reshape(N, T * n_mf)

    # Per-probe HC variants — drop label-source variates (decision #2 Path A,
    # 2026-04-28). Mapping declared in HC_PROBE_LEAKAGE_EXCLUSIONS at module top.
    # Each entry maps probe_name → (X_hc_variant, X_raw_variant, hc_label, raw_label).
    probe_hc_variants = {}
    for probe_name_var, excluded in HC_PROBE_LEAKAGE_EXCLUSIONS.items():
        keep_local_pos = [i for i, v in enumerate(MODEL_FEATURES) if v not in excluded]
        keep_ssl_pos = [SSL_FEATURES.index(MODEL_FEATURES[i]) for i in keep_local_pos]
        X_mf_v = X_raw[:, :, keep_ssl_pos]
        X_hc_v = compute_handcrafted_features(X_mf_v)
        X_raw_v = X_mf_v.reshape(N, T * len(keep_local_pos))
        probe_hc_variants[probe_name_var] = (
            X_hc_v, X_raw_v,
            f"handcrafted_{X_hc_v.shape[1]}",
            f"raw_{len(keep_local_pos)}",
        )
        print(f"  Per-probe HC variant for {probe_name_var}: "
              f"{X_hc_v.shape[1]}-d HC + {X_raw_v.shape[1]}-d raw "
              f"(excluded variates: {sorted(excluded)})")

    # Step 4: Folds
    print("\n=== Step 4: Walk-forward folds ===")
    folds = make_probe_folds(probe_data)
    _results["probe_folds"] = folds

    # Step 5: Run ALL probes on hand-crafted + raw_36
    print("\n=== Step 5: Run probes on hand-crafted features ===")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV

    probe_names = []
    for k in ph["probe_a_horizons"]:
        probe_names.append(f"ret_{k}")
    for k in ph["probe_b_horizons"]:
        probe_names.append(f"log_rv_{k}")
    probe_names.append("regime")
    probe_names.append("regime_concurrent")
    probe_names.append(f"spread_{ph['probe_d_horizon']}")

    all_results = {}

    for probe_name in probe_names:
        # Skip probes whose targets are not available (backward-compat)
        if probe_name not in probe_targets["targets"]:
            continue
        target = probe_targets["targets"][probe_name]
        valid = probe_targets["validity"][probe_name]
        is_cls = probe_name.startswith("ret_") or probe_name.startswith("regime")
        needs_sign = probe_name.startswith("ret_")
        needs_dz = probe_name.startswith("ret_") or probe_name.startswith("spread_")

        print(f"\n  Probe: {probe_name} ({'classification' if is_cls else 'regression'})")

        # Default representation pair = the full MODEL_FEATURES set.
        hc_label_default = f"handcrafted_{n_hc}"
        raw_label_default = f"raw_{n_mf}"
        if probe_name in probe_hc_variants:
            X_hc_p, X_raw_p, hc_label, raw_label = probe_hc_variants[probe_name]
            print(f"    [Per-probe HC variant active: {hc_label} / {raw_label}]")
        else:
            X_hc_p, X_raw_p = X_hc, X_raw_mf
            hc_label, raw_label = hc_label_default, raw_label_default

        for rep_name, X_rep in [(hc_label, X_hc_p), (raw_label, X_raw_p)]:
            # Bug fix (proactive hunt 2026-04-29): track fold INDEX alongside
            # metric so the bracket gate's paired-Δ can intersect by fold_idx.
            # Previously fold_metrics was a flat list with skipped folds
            # missing — positional-to-fold-index mapping broke when any fold
            # was skipped (same class of bug as Pass-A B1).
            fold_metrics = []  # list of (fold_idx, metric)
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                tr_valid = valid[train_idx]
                te_valid = valid[test_idx]

                X_tr = X_rep[train_idx][tr_valid]
                X_te = X_rep[test_idx][te_valid]
                y_tr = target[train_idx][tr_valid]
                y_te = target[test_idx][te_valid]

                # N/p eligibility: probe must have enough training samples
                # relative to its dimensionality for Ridge/Logistic to be
                # a reliable measurement instrument (not underfitting noise).
                min_n_per_p = 1.5
                n_p_ratio = len(X_tr) / max(X_tr.shape[1], 1)
                if n_p_ratio < min_n_per_p or len(X_te) < 20:
                    print(f"    [{rep_name}: N/p={n_p_ratio:.2f} < {min_n_per_p}, "
                          f"fold {fi} skipped (N={len(X_tr)}, p={X_tr.shape[1]})]")
                    continue

                # Dead zone
                if needs_dz:
                    # Pass-C NEW-B3 fix (2026-04-29): use the configurable
                    # PROBE_HYPERPARAMS multiplier so HC and main-probe dead
                    # zones agree (was hardcoded 0.1 — silent fairness gap
                    # with apply_dead_zone in the main probe path which
                    # honors the configured multiplier).
                    tr_std = float(np.std(y_tr[~np.isnan(y_tr)]))
                    thresh = ph["dead_zone_std_multiplier"] * tr_std
                    tr_mask = np.abs(y_tr) >= thresh
                    te_mask = np.abs(y_te) >= thresh
                    X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
                    X_te, y_te = X_te[te_mask], y_te[te_mask]

                # Sign for return-direction probes
                if needs_sign:
                    y_tr = (y_tr > 0).astype(int)
                    y_te = (y_te > 0).astype(int)

                n_p_ratio_post = len(X_tr) / max(X_tr.shape[1], 1)
                if n_p_ratio_post < min_n_per_p or len(X_te) < 20:
                    continue

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                if is_cls:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf = LogisticRegressionCV(
                            Cs=10, cv=3, max_iter=1000, solver="lbfgs",
                            scoring="balanced_accuracy",
                            class_weight="balanced",
                        )
                        classes = np.unique(y_tr)
                        if len(classes) < 2:
                            continue
                        clf.fit(X_tr_s, y_tr)
                        from sklearn.metrics import balanced_accuracy_score
                        y_pred = clf.predict(X_te_s)
                        metric = float(balanced_accuracy_score(y_te, y_pred))
                else:
                    model_ridge = RidgeCV(alphas=np.logspace(-3, 5, 20))  # #216 unified grid (was logspace(-3,3,10), saturating ceiling)
                    model_ridge.fit(X_tr_s, y_tr)
                    metric = float(model_ridge.score(X_te_s, y_te))

                fold_metrics.append((fold_idx, metric))

            scores = [m for _, m in fold_metrics]
            idxs = [fi for fi, _ in fold_metrics]
            mean_m = np.mean(scores) if scores else float("nan")
            std_m = np.std(scores) if scores else float("nan")
            key = f"{probe_name}__{rep_name}"
            all_results[key] = {"mean": mean_m, "std": std_m,
                                "folds": scores, "fold_idx": idxs}
            metric_name = "bal_acc" if is_cls else "R²"
            print(f"    {rep_name:20s}: {metric_name} = {mean_m:.4f} +/- {std_m:.4f}")

    # Step 6: Summary comparison (per-probe HC dimensionality, decision #2 Path A)
    print("\n" + "=" * 60)
    print("HAND-CRAFTED vs RAW BASELINE COMPARISON")
    print("=" * 60)
    print(f"  {'Probe':12s} {'HC':>16s} {'raw':>16s} {'HC wins?':>10s}")
    print("  " + "-" * 56)
    for probe_name in probe_names:
        if probe_name in probe_hc_variants:
            _, _, hc_lbl, raw_lbl = probe_hc_variants[probe_name]
        else:
            hc_lbl = f"handcrafted_{n_hc}"
            raw_lbl = f"raw_{n_mf}"
        hc = all_results.get(f"{probe_name}__{hc_lbl}", {}).get("mean", float("nan"))
        raw = all_results.get(f"{probe_name}__{raw_lbl}", {}).get("mean", float("nan"))
        wins = "YES" if hc > raw + 0.005 else ("~SAME" if abs(hc - raw) <= 0.005 else "NO")
        hc_disp = f"{hc_lbl}={hc:.4f}"
        raw_disp = f"{raw_lbl}={raw:.4f}"
        print(f"  {probe_name:12s} {hc_disp:>16s} {raw_disp:>16s} {wins:>10s}")

    print(f"\n  Note: Compare these against encoder emb_group results from L1-SSL-004:")
    print(f"    log_rv_15: encoder=0.523, log_rv_30: encoder=0.493")
    print(f"    regime: encoder=0.589, spread_5: encoder=0.087")
    print(f"    (emb_fine spread: 0.252)")

    # Save
    _results["handcrafted_probes"] = all_results
    try:
        qb.ObjectStore.Save("handcrafted_probe_results", json.dumps(
            {k: {"mean": v["mean"], "std": v["std"]} for k, v in all_results.items()}
        ))
        print("\n  Saved to ObjectStore: handcrafted_probe_results")
    except Exception as e:
        print(f"\n  WARNING: ObjectStore save failed: {e}")

    return _results


def run_single_probe(probe_name, probe_data, probe_targets, folds,
                     encodings, seeds):
    """Run a single probe across all folds and input types.

    For each fold: fits linear probes on all embedding variants (full, grid,
    group-concat, max-pool, flat) and raw baselines (36, 129). Also fits MLP
    probes on emb_group as a nonlinear diagnostic. Applies dead-zone for
    continuous→classification probes and regression probes near zero.

    Args:
        probe_name: str — key into probe_targets, e.g., "ret_15", "log_rv_30"
        probe_data: dict from load_probe_days (X_windows needed for raw baselines)
        probe_targets: dict from compute_probe_targets (targets + validity)
        folds: list of (train_idx, test_idx) from make_probe_folds
        encodings: dict from encode_windows (emb_full, emb_grid, emb_group, etc.)
        seeds: list[int]

    Returns dict with:
        probe_name: str
        probe_type: "classification" or "regression"
        fold_results: dict[input_type → list[dict or None per fold]]
        summary: dict[input_type → dict with mean/std of primary metric]
    """
    targets = probe_targets["targets"][probe_name]
    valid = probe_targets["validity"][probe_name]

    # Determine probe type and transformations
    is_classification = probe_name.startswith("ret_") or probe_name.startswith("regime")
    needs_dead_zone = probe_name.startswith("ret_") or probe_name.startswith("spread_")
    needs_sign = probe_name.startswith("ret_")
    probe_type = "classification" if is_classification else "regression"
    primary_metric = "balanced_accuracy" if is_classification else "r2"
    is_expected_null = probe_name in _EXPECTED_NULL_PROBES

    print(f"\n{'='*60}")
    print(f"Probe: {probe_name} ({probe_type})"
          f"{'  [EXPECTED NULL — monitoring only]' if is_expected_null else ''}")
    print(f"  Valid targets: {int(valid.sum())}/{len(valid)}")
    print(f"  Dead-zone: {needs_dead_zone}, Sign: {needs_sign}")
    print(f"{'='*60}")

    # Input sources — embedding variants
    emb_full = encodings["emb_full"]
    emb_grid = encodings["emb_grid"]
    emb_group = encodings["emb_group"]
    emb_fine = encodings.get("emb_fine")  # Fine-grained 8-group pooling (spread diagnostic)
    emb_max = encodings["emb_max"]
    # emb_flat removed from encodings (OOM on QC RidgeCV); no access needed.
    emb_shuffled = encodings.get("emb_shuffled", {})

    # Raw baseline inputs: apply log1p to heavy-tailed variates (fair comparison)
    # then flatten (N, T, V) → (N, T*V)
    X_raw = probe_data["X_windows"].copy()
    if _LOG1P_SSL_POS:
        X_raw[:, :, _LOG1P_SSL_POS] = (
            np.sign(X_raw[:, :, _LOG1P_SSL_POS])
            * np.log1p(np.abs(X_raw[:, :, _LOG1P_SSL_POS]))
        )
    # N1 fix (2026-04-29): flow aggregates v141-v158 have NaN warmup bars (per
    # (internal doc) "NaN on warm-up bars where the window reaches into the prior
    # day"). Flattening NaN cells into RidgeCV's design matrix → NaN coefficients
    # → catastrophic R² (−16 to −30 on raw_129 in the linear Azure run). Fill
    # with 0.0 AFTER log1p so the ridge sees "no info" not "NaN".
    np.nan_to_num(X_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    N, T, V = X_raw.shape
    X_raw_129 = X_raw.reshape(N, T * V)
    # raw_36 baseline: leakage-free MODEL_FEATURES variates flattened.
    # Per-probe variant: when probe_name is in HC_PROBE_LEAKAGE_EXCLUSIONS
    # (e.g., spread_5), drop the label-source variates (e.g., v46/v47) to
    # prevent direct label leak (decision #2 Path A, 2026-04-28).
    _hc_excl = HC_PROBE_LEAKAGE_EXCLUSIONS.get(probe_name, frozenset())
    if _hc_excl:
        _keep_pos = [SSL_FEATURES.index(v) for v in MODEL_FEATURES if v not in _hc_excl]
        X_raw_36 = X_raw[:, :, _keep_pos].reshape(N, T * len(_keep_pos))
        print(f"  [raw_36 per-probe variant for {probe_name}]: {len(_keep_pos)} variates "
              f"(excluded {sorted(_hc_excl)})")
    else:
        X_raw_36 = X_raw[:, :, _MF_SSL_POS].reshape(N, T * len(_MF_SSL_POS))

    # Linear probe input types (ordered: embedding variants, then raw baselines)
    # emb_flat (V*D-dimensional) excluded from classification probes:
    # LogisticRegressionCV LBFGS on high-d input takes hours per fold in QC Research.
    # Regression probes use RidgeCV (closed-form), which handles high-d efficiently.
    linear_inputs_base = [
        ("emb_full",  emb_full),
        ("emb_grid",  emb_grid),
        ("emb_group", emb_group),
        ("emb_max",   emb_max),
        ("raw_36",    X_raw_36),
        ("raw_129",   X_raw_129),
    ]
    # Add fine-grained pooling if available (spread diagnostic)
    if emb_fine is not None:
        linear_inputs_base.insert(4, ("emb_fine", emb_fine))
    # emb_flat (V*D = 18,816-d on v3) previously added to regression probes
    # was the recurring OOM culprit on QC Research kernels: its X'X Gram
    # matrix alone is ~2.8GB at float32 (18816² × 4 bytes), and RidgeCV
    # internally holds multiple copies during alpha search. Dropped from
    # the probe input list — emb_fine (3072-d, 8-group CVPE-style pooling)
    # already covers the "rich spatial representation" role and is well-
    # conditioned against the 1380–4140 per-fold training sizes.
    linear_inputs = linear_inputs_base
    linear_type_names = [name for name, _ in linear_inputs]
    shuffle_types = [f"shuffle_{s}" for s in seeds]
    all_types = linear_type_names + shuffle_types
    # MLP probe types (run on group-concat embedding only to limit runtime)
    mlp_type = "mlp_group"
    all_types_with_mlp = all_types + [mlp_type]
    fold_results = {t: [] for t in all_types_with_mlp}

    for fi, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  Fold {fi}: train={len(train_idx)}, test={len(test_idx)}")

        # Per-fold dead-zone
        v_fold = valid.copy()
        if needs_dead_zone:
            train_mask_full = np.zeros(len(targets), dtype=bool)
            train_mask_full[train_idx] = True
            v_fold = apply_dead_zone(targets, v_fold, train_mask_full)

        # Build train/test targets with validity
        tr_valid = v_fold[train_idx]
        te_valid = v_fold[test_idx]

        if needs_sign:
            y_all = (targets > 0).astype(np.int64)
        elif probe_name.startswith("regime"):
            # AUDIT DISCLOSURE 2026-04-29: regime label is derived from v41 (ATM IV)
            # + v111 (Parkinson RV 15m), both of which are encoder INPUT variates.
            # Treat regime as a "encoder didn't destroy v41/v111" gate (task #203).
            #
            # #210 fix (2026-04-29): recompute regime thresholds PER FOLD using
            # only the fold's training bars. Previously used a global TRAIN_END
            # cutoff that leaked test-fold bars into threshold computation for
            # folds 0-2, contaminating the one statistically-tight probe verdict.
            #
            # : regime_concurrent uses horizon=0 (label at window end);
            # regime uses probe_c_horizon (label at t+30). Same per-fold
            # threshold logic applies to both.
            _regime_horizon = (0 if probe_name == "regime_concurrent"
                               else PROBE_HYPERPARAMS["probe_c_horizon"])
            _bar_data = probe_data.get("_bar_atm_iv_ema")
            if _bar_data is not None:
                _wlbg = probe_data["_window_last_bar_global"]
                _bar_iv = probe_data["_bar_atm_iv_ema"]
                _bar_mew = probe_data["_bar_movement_ewma"]
                _bar_rv = probe_data["_bar_rv30"]
                _bar_d = probe_data["_bar_dates"]
                _bar_p = probe_data["_bar_positions"]
                # Build per-fold train_bar_mask: a bar is "training" if ANY
                # training window's last_bar_global falls on or after it.
                # Conservative: mark all bars reachable from train windows.
                _train_bars_set = set()
                for _wi in train_idx:
                    _lb = int(_wlbg[_wi])
                    # The window covers bars [_lb - T + 1, _lb]; mark all
                    _train_bars_set.update(range(max(0, _lb - 59), _lb + 1))
                _train_bar_mask = np.zeros(len(_bar_iv), dtype=bool)
                for _bi in _train_bars_set:
                    if _bi < len(_train_bar_mask):
                        _train_bar_mask[_bi] = True
                # Recompute thresholds on train bars only (suppress print)
                import io as _io, contextlib as _ctx
                _n_train_bars = int(_train_bar_mask.sum())
                if _n_train_bars < 200:
                    print(f"    [{probe_name} #210 WARNING] fold {fi}: only {_n_train_bars} "
                          f"train bars for threshold computation — risk of "
                          f"degenerate percentiles (vol_p33==vol_p75)")
                with _ctx.redirect_stdout(_io.StringIO()):
                    _fold_thresh = _compute_thresholds(
                        _bar_iv, _bar_iv, _bar_mew, _bar_d, _bar_p, _bar_rv,
                        train_bar_mask=_train_bar_mask)
                # Re-derive labels for ALL bars using per-fold thresholds
                with _ctx.redirect_stdout(_io.StringIO()):
                    _vol_c, _mov_c, _wm = _classify_bars(
                        _bar_iv, _bar_iv, _bar_mew, _bar_d, _bar_p,
                        _fold_thresh, _bar_rv)
                _cfg = GRID_CONFIGS[4]
                _fold_regime_labels = _apply_grid(_vol_c, _mov_c, _wm, _cfg)
                # Map bar labels → per-window targets via horizon
                _n_win = len(targets)
                _fold_targets = np.full(_n_win, -1, dtype=np.int64)
                for _wi in range(_n_win):
                    _lb = int(_wlbg[_wi])
                    _fwd = _lb + _regime_horizon
                    if (_fwd < len(_fold_regime_labels) and
                            _fold_regime_labels[_fwd] >= 0):
                        _fold_targets[_wi] = _fold_regime_labels[_fwd]
                y_all = _fold_targets
                # Update validity for this fold (invalid where label == -1)
                v_fold = v_fold & (_fold_targets >= 0)
                tr_valid = v_fold[train_idx]
                te_valid = v_fold[test_idx]
                if fi == 0:
                    print(f"    [#210 per-fold regime thresholds active, horizon={_regime_horizon}]")
            else:
                # Fallback if bar-level data not in probe_data (legacy path)
                y_all = targets.astype(np.int64)
        else:
            y_all = targets.astype(np.float32)

        y_train = y_all[train_idx][tr_valid]
        y_test = y_all[test_idx][te_valid]

        n_tr = len(y_train)
        n_te = len(y_test)
        if n_tr < 50 or n_te < 10:
            print(f"    Skipping: insufficient valid samples (train={n_tr}, test={n_te})")
            for t in all_types_with_mlp:
                fold_results[t].append(None)
            continue

        if is_classification:
            n_cls = len(np.unique(y_train))
            print(f"    Classes in train: {n_cls}, samples: {n_tr} train, {n_te} test")
            if n_cls < 2:
                # Degenerate percentile diagnostic: when #210 per-fold
                # thresholds collapse (vol_p33==vol_p75 on tiny fold-0),
                # all bars become one class → single-class training set.
                # This is a SILENT statistical power loss — the fold
                # contributes no signal to the Bouthillier paired-Δ.
                if probe_name.startswith("regime"):
                    print(f"    [DEGENERATE REGIME THRESHOLD — {probe_name} fold {fi} "
                          f"has only {n_cls} class(es). Per-fold #210 "
                          f"thresholds likely collapsed on a small training "
                          f"bar set. This fold will NOT contribute to the "
                          f"paired-Δ test — effective n_paired may fall "
                          f"below 3, rendering the probe inconclusive.]")
                else:
                    print(f"    Skipping: only {n_cls} class(es) in training set")
                for t in all_types_with_mlp:
                    fold_results[t].append(None)
                continue

        # --- Linear probes ---
        fit_fn = fit_classification_probe if is_classification else fit_regression_probe

        for input_name, X_source in linear_inputs:
            X_tr = X_source[train_idx][tr_valid]
            X_te = X_source[test_idx][te_valid]
            p_dim = X_tr.shape[1]
            n_p_ratio = len(X_tr) / max(p_dim, 1)
            if fi == 0 and p_dim > 1000:
                print(f"    [{input_name}: {p_dim}-d input]")
            if n_p_ratio < 1.5:
                print(f"    {input_name:12s}: SKIPPED (N/p={n_p_ratio:.2f} < 1.5, "
                      f"N={len(X_tr)}, p={p_dim})")
                fold_results[input_name].append(None)
                continue
            try:
                m = fit_fn(X_tr, y_train, X_te, y_test)
                print(f"    {input_name:12s}: {primary_metric}={m[primary_metric]:.4f}")
            except Exception as e:
                print(f"    {input_name:12s}: FAILED ({e})")
                m = None
                _probe_health["fold_fit_failures"] += 1
            fold_results[input_name].append(m)

        # --- Shuffle baselines (per seed) ---
        for seed in seeds:
            key = f"shuffle_{seed}"
            X_s = emb_shuffled.get(seed)
            if X_s is None:
                fold_results[key].append(None)
                continue
            X_tr = X_s[train_idx][tr_valid]
            X_te = X_s[test_idx][te_valid]
            s_n_p = len(X_tr) / max(X_tr.shape[1], 1)
            if s_n_p < 1.5:
                fold_results[key].append(None)
                continue
            try:
                m = fit_fn(X_tr, y_train, X_te, y_test)
                print(f"    {key:12s}: {primary_metric}={m[primary_metric]:.4f}")
            except Exception as e:
                print(f"    {key:12s}: FAILED ({e})")
                m = None
                _probe_health["fold_fit_failures"] += 1
            fold_results[key].append(m)

        # --- MLP probe on emb_group (nonlinear diagnostic) ---
        mlp_fn = fit_mlp_classification_probe if is_classification else fit_mlp_regression_probe
        X_tr = emb_group[train_idx][tr_valid]
        X_te = emb_group[test_idx][te_valid]
        try:
            m = mlp_fn(X_tr, y_train, X_te, y_test)
            print(f"    {'mlp_group':12s}: {primary_metric}={m[primary_metric]:.4f}")
        except Exception as e:
            print(f"    {'mlp_group':12s}: FAILED ({e})")
            m = None
            _probe_health["fold_fit_failures"] += 1
        fold_results[mlp_type].append(m)

    # Aggregate results: mean ± std of primary metric across valid folds
    summary = {}
    for t in all_types_with_mlp:
        vals = [r[primary_metric] for r in fold_results[t] if r is not None]
        if vals:
            summary[t] = dict(
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                n_folds=len(vals),
            )
        else:
            summary[t] = dict(mean=float("nan"), std=float("nan"), n_folds=0)

    # Combine shuffle seeds into single shuffle summary
    shuffle_vals = []
    for seed in seeds:
        for r in fold_results[f"shuffle_{seed}"]:
            if r is not None:
                shuffle_vals.append(r[primary_metric])
    if shuffle_vals:
        summary["shuffle_combined"] = dict(
            mean=float(np.mean(shuffle_vals)),
            std=float(np.std(shuffle_vals)),
            n_samples=len(shuffle_vals),
        )

    # Last-3-fold mean (reduces expanding-window early-fold bias)
    for t in all_types_with_mlp:
        vals = [r[primary_metric] for r in fold_results[t][-3:] if r is not None]
        if vals:
            summary[f"{t}_last3"] = dict(
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                n_folds=len(vals),
            )

    # Print summary table (emb_flat removed — see encode_windows docstring).
    # 2026-04-29 #220: Last3 column DEMOTED from the default print table to
    # prevent comparison-statistic-identity errors (cherry-picking Last3 vs a
    # baseline that uses 5-fold mean). Last3 still computed and persisted in
    # `summary` for later analysis but no longer rendered next to Mean.
    display_types = ["emb_full", "emb_group", "emb_fine", "emb_max",
                     "emb_grid", "raw_36", "raw_129",
                     "shuffle_combined", "mlp_group"]
    print(f"\n  Summary ({probe_name}, {primary_metric}):")
    print(f"  {'Input':16s} {'Mean':>8s} {'Std':>8s} {'Folds':>6s}")
    print(f"  {'-'*44}")
    for t in display_types:
        s = summary.get(t, {})
        print(f"  {t:16s} {s.get('mean', float('nan')):8.4f} "
              f"{s.get('std', float('nan')):8.4f} "
              f"{s.get('n_folds', s.get('n_samples', 0)):6d}")

    # Reportability check (literature-grounded decision #4, 2026-04-28):
    # Bouthillier et al. 2021 (Accounting for Variance in ML Benchmarks) +
    # Cohen 1988 (effect-size convention) + Lakens 2013 (paired d_z).
    #
    # Replaces the post-hoc VARIANCE_GATE_STD=0.20 (calibrated on SSL-010
    # only; arbitrary for any other run) with paired statistics on the actual
    # comparison: per-fold delta of emb_fine vs raw_36 (the adversarial
    # baseline the L2 GP downstream contract consumes — emb_fine per
    # batch_forecast.py:712).
    #
    # Two conditions, ALL on emb_fine vs raw_36:
    # (1) 80% CI of paired per-fold delta EXCLUDES zero (Bouthillier 2021
    # — confidence-interval based comparison is robust to small-N
    # variance asymmetries unlike paired-t hypothesis-testing alone)
    # (2) Cohen's d_z > 0.5 on per-fold paired delta (Cohen 1988 medium-
    # effect threshold; Lakens 2013 d_z = mean_diff / std_diff for
    # paired designs)
    # PLUS the legacy sanity floor:
    # (3) mean(emb_fine) > mean(shuffle) (sign-test floor against the
    # within-day shuffled-encodings null)
    fine_mean = summary["emb_fine"].get("mean", float("nan"))
    fine_std = summary["emb_fine"].get("std", float("nan"))
    raw36_mean = summary["raw_36"].get("mean", float("nan"))
    shuffle_mean = summary.get("shuffle_combined", {}).get("mean", float("nan"))

    # Per-fold paired delta: emb_fine[fold_i] - raw_36[fold_i] for matched folds.
    # Pass-A B1 fix (2026-04-29): the previous implementation filtered each
    # series independently with `if r is not None`, then aligned by position
    # — a silent fold-misalignment bug. If emb_fine survived folds {0,1,3,4}
    # and raw_36 survived {0,2,3,4}, position 1 paired emb_fine[fold=1] with
    # raw_36[fold=2], destroying the paired-statistic identity (the exact
    # failure pattern from feedback_evaluation_methodology_blind_spots.md).
    # Now we zip by fold-index FIRST, then drop pairs where either side failed.
    paired_pairs = [
        (f[primary_metric], r[primary_metric])
        for f, r in zip(fold_results["emb_fine"], fold_results["raw_36"])
        if f is not None and r is not None
    ]
    n_paired = len(paired_pairs)
    if n_paired >= 3:
        fine_arr = np.asarray([p[0] for p in paired_pairs], dtype=np.float64)
        raw_arr = np.asarray([p[1] for p in paired_pairs], dtype=np.float64)
        deltas = fine_arr - raw_arr
        delta_mean = float(deltas.mean())
        delta_std = float(deltas.std(ddof=1)) if n_paired > 1 else float("nan")
        # 80%-confidence two-sided CI half-width (== one-sided 90% lower bound
        # for the gate's `ci_low > 0` test, which is a one-sided claim of
        # "emb_fine is better than raw_36"). Pass-A B3 fix: corrected fallback
        # t_critical table (one-sided 90% / two-sided 80%, df = n_paired - 1).
        try:
            from scipy import stats as _sp_stats
            t_crit = float(_sp_stats.t.ppf(0.90, df=n_paired - 1))
        except Exception:
            # Pre-tabulated one-sided 90% / two-sided 80% Student-t critical
            # values for df = 1..10. Use df=10 for n_paired > 11 as a
            # conservative-from-above approximation (true t shrinks toward 1.282).
            _T_CRIT_10_90 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476,
                             6: 1.440, 7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372}
            t_crit = _T_CRIT_10_90.get(n_paired - 1, 1.372)
        ci_half = t_crit * delta_std / np.sqrt(n_paired) if n_paired > 1 else float("nan")
        ci_low = delta_mean - ci_half
        ci_high = delta_mean + ci_half
        # Cohen's d_z (Lakens 2013): mean of paired differences / std of differences.
        d_z = delta_mean / delta_std if (delta_std and delta_std > 1e-12) else float("nan")
        ci_excludes_zero = (ci_low > 0.0) if not np.isnan(ci_low) else False
        d_z_medium = (not np.isnan(d_z)) and (d_z > 0.5)
    else:
        delta_mean = float("nan"); ci_low = float("nan"); ci_high = float("nan")
        d_z = float("nan"); ci_excludes_zero = False; d_z_medium = False

    beats_shuffle = fine_mean > shuffle_mean if not np.isnan(fine_mean) else False
    reportable = ci_excludes_zero and d_z_medium and beats_shuffle

    if is_expected_null:
        print(f"\n  [EXPECTED NULL] Not counted in reportability tally.")
        print(f"  (emb_fine mean={fine_mean:.4f} std={fine_std:.4f}, "
              f"raw_36={raw36_mean:.4f}, shuffle={shuffle_mean:.4f})")
        reportable = None  # sentinel — excluded from count
    else:
        # Feature-availability tag (agent audit mandate, tasks #203 + N4):
        # regime is NOT a representation-quality test — its label sources
        # (v41 ATM IV + v111 Parkinson RV) are in both encoder AND baseline
        # inputs. The probe only tests "did pooling destroy this info."
        if probe_name.startswith("regime"):
            print(f"\n  [FEATURE-AVAILABILITY TEST — NOT representation quality]")
            print(f"  (v41 + v111 are label sources AND encoder inputs; "
                  f"{probe_name} tests pooling preservation, not learned abstraction)")
        print(f"\n  Reportable [Bouthillier 2021 + Cohen d_z]:")
        print(f"    paired Δ(emb_fine − raw_36) mean={delta_mean:+.4f}, "
              f"80% CI=[{ci_low:+.4f}, {ci_high:+.4f}], d_z={d_z:+.3f}, n={n_paired}")
        print(f"    CI excludes 0: {ci_excludes_zero},  "
              f"d_z > 0.5: {d_z_medium},  "
              f"beats shuffle: {beats_shuffle}")
        print(f"  Reportable: {'YES' if reportable else 'NO'}")

    return dict(
        probe_name=probe_name,
        probe_type=probe_type,
        primary_metric=primary_metric,
        fold_results=fold_results,
        summary=summary,
        reportable=reportable,
        is_expected_null=is_expected_null,
    )


# ===================================================================
# 5. EXPERT REGIME LABELING (, , , )
# ===================================================================

def _ema_1d(values, span):
    """Causal EMA over a 1D array (single day). No look-ahead."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            out[i] = out[i - 1]  # forward-fill NaN input
        elif np.isnan(out[i - 1]):
            out[i] = values[i]   # first valid value seeds the EMA
        else:
            out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _hysteresis_classify(bar_movement_ewma, bar_dates, enter_thresh, exit_thresh):
    """Classify movement axis using Schmitt trigger hysteresis on EWMA signal.

    Per-day state machine. Starts QUIET each day. Transitions:
      QUIET → TRENDING  when ewma >= enter_thresh  (upper band)
      TRENDING → QUIET  when ewma < exit_thresh    (lower band)
    Dead zone (exit_thresh <= ewma < enter_thresh): hold current state.

    Returns move_class: (N,) int8, 0=QUIET, 1=TRENDING.
    """
    n_bars = len(bar_movement_ewma)
    move_class = np.zeros(n_bars, dtype=np.int8)

    unique_days, day_starts = np.unique(bar_dates, return_index=True)
    day_ends = np.append(day_starts[1:], n_bars)

    for ds, de in zip(day_starts, day_ends):
        state = 0  # start QUIET each day
        for b in range(ds, de):
            val = bar_movement_ewma[b]
            if np.isnan(val):
                pass  # hold state through NaN
            elif state == 0 and val >= enter_thresh:
                state = 1
            elif state == 1 and val < exit_thresh:
                state = 0
            move_class[b] = state

    return move_class


# Grid configurations: maps (vol_class, movement_class) → regime index
# Each config defines: name, num_regimes, taxonomy dict, grid mapping
GRID_CONFIGS = {
    3: {
        "name": "k=3 (2 vol × 2 movement, no VOLATILE)",
        "taxonomy": {0: "CALM", 1: "DIRECTIONAL", 2: "CRISIS"},
        # LOW/MID+QUIET→CALM, LOW/MID+TRENDING→DIRECTIONAL,
        # HIGH+QUIET→DIRECTIONAL, HIGH+TRENDING→CRISIS
        "grid": {
            (0, 0): 0, (0, 1): 1,   # LOW + QUIET/TRENDING
            (1, 0): 0, (1, 1): 1,   # MID + QUIET/TRENDING
            (2, 0): 1, (2, 1): 2,   # HIGH + QUIET/TRENDING
        },
    },
    4: {
        "name": "k=4 (2 vol × 2 movement)",
        "taxonomy": {0: "CALM", 1: "DIRECTIONAL", 2: "VOLATILE", 3: "CRISIS"},
        # LOW/MID collapsed, HIGH separate
        "grid": {
            (0, 0): 0, (0, 1): 1,   # LOW + QUIET/TRENDING
            (1, 0): 0, (1, 1): 1,   # MID + QUIET/TRENDING
            (2, 0): 2, (2, 1): 3,   # HIGH + QUIET/TRENDING
        },
    },
    6: {
        "name": "k=6 (3 vol × 2 movement)",
        "taxonomy": {
            0: "LOW_CALM", 1: "LOW_DIRECTIONAL",
            2: "MID_CALM", 3: "MID_DIRECTIONAL",
            4: "HIGH_CALM", 5: "HIGH_CRISIS",
        },
        # No collapse — every cell is its own regime
        "grid": {
            (0, 0): 0, (0, 1): 1,   # LOW + QUIET/TRENDING
            (1, 0): 2, (1, 1): 3,   # MID + QUIET/TRENDING
            (2, 0): 4, (2, 1): 5,   # HIGH + QUIET/TRENDING
        },
    },
}


def _compute_thresholds(bar_feats, bar_atm_iv_ema, bar_movement_ewma,
                        bar_dates, bar_positions, bar_rv30,
                        train_bar_mask=None):
    """Compute labeling thresholds from training bars (excluding warm-up).

    D63: Volatility axis uses ATM IV EMA (bar-level) instead of daily VIX spot.
    D64: Movement axis uses EWMA + Schmitt trigger hysteresis (two thresholds).
    D68: RV30 extracted from raw data (no longer in MODEL_FEATURES).

    #210 fix (2026-04-29): accepts optional `train_bar_mask` for per-fold
    threshold computation. When None (default), uses the legacy global
    `bar_dates <= TRAIN_END` cutoff. When provided, uses the caller-supplied
    boolean mask to select training bars — enabling walk-forward folds to
    derive thresholds from ONLY their training split.
    """
    if train_bar_mask is not None:
        is_train = train_bar_mask
    else:
        is_train = np.array([d <= TRAIN_END for d in bar_dates])
    is_warmed = bar_positions >= WARMUP_BARS
    valid = is_train & is_warmed & ~np.isnan(bar_atm_iv_ema)

    atm_iv = bar_atm_iv_ema[valid]
    rv30 = bar_rv30[valid]

    # ATM IV EMA: NaN already excluded by valid mask; zeros converted to NaN in load_probe_days
    vol_p33 = np.percentile(atm_iv, VOL_LOW_PCT)
    vol_p75 = np.percentile(atm_iv, VOL_HIGH_PCT)
    rv30_p75 = np.percentile(rv30, RV30_HIGH_PCT)

    # : Movement thresholds on EWMA-smoothed signal (Schmitt trigger bands)
    mew_valid = bar_movement_ewma[valid]
    mew_valid = mew_valid[~np.isnan(mew_valid)]  # guard against NaN propagation from _ema_1d
    movement_upper = np.percentile(mew_valid, MOVEMENT_UPPER_PCT)
    movement_lower = np.percentile(mew_valid, MOVEMENT_LOWER_PCT)

    n_valid = int(valid.sum())
    n_iv_nan = int(np.isnan(bar_atm_iv_ema[is_train & is_warmed]).sum())
    print(f"  Thresholds from {n_valid} train bars (warmup-excluded)"
          f"{f', {n_iv_nan} NaN ATM IV excluded' if n_iv_nan else ''}:")
    print(f"    ATM IV EMA: LOW < {vol_p33:.4f} < MID < {vol_p75:.4f} < HIGH"
          f"  (span={ATM_IV_EMA_SPAN})")
    print(f"    RV30:       override to HIGH if > {rv30_p75:.6f}")
    print(f"    Movement EWMA (span={MOVEMENT_EWMA_SPAN}):")
    print(f"      QUIET < {movement_lower:.6f} (P{MOVEMENT_LOWER_PCT}) "
          f"< dead zone < {movement_upper:.6f} (P{MOVEMENT_UPPER_PCT}) < TRENDING")

    return dict(
        vol_p33=float(vol_p33), vol_p75=float(vol_p75),
        rv30_p75=float(rv30_p75),
        movement_upper=float(movement_upper),
        movement_lower=float(movement_lower),
    )


def _classify_bars(bar_feats, bar_atm_iv_ema, bar_movement_ewma,
                   bar_dates, bar_positions, thresholds, bar_rv30):
    """Classify each bar on the vol and movement axes.

    D63: Volatility axis uses ATM IV EMA (bar-level) instead of daily VIX spot.
    D64: Movement axis uses EWMA + Schmitt trigger hysteresis.
    D68: RV30 extracted from raw data (no longer in MODEL_FEATURES).
    Returns (vol_class, move_class, warmup_mask).
    """
    n_bars = len(bar_feats)
    atm_iv = bar_atm_iv_ema
    rv30 = bar_rv30

    # Volatility: 0=LOW, 1=MID, 2=HIGH (: ATM IV EMA replaces VIX spot)
    vol_class = np.ones(n_bars, dtype=np.int8)  # default MID
    vol_class[atm_iv < thresholds["vol_p33"]] = 0
    vol_class[atm_iv >= thresholds["vol_p75"]] = 2
    vol_class[rv30 > thresholds["rv30_p75"]] = 2  # RV override
    # NaN ATM IV → mark as warm-up (will get label -1)
    vol_class[np.isnan(atm_iv)] = 1  # won't matter — warmup_mask handles it

    # : Movement via EWMA + hysteresis (0=QUIET, 1=TRENDING)
    move_class = _hysteresis_classify(
        bar_movement_ewma, bar_dates,
        thresholds["movement_upper"], thresholds["movement_lower"])

    # Warm-up mask: first WARMUP_BARS of each day + NaN ATM IV
    warmup_mask = (bar_positions < WARMUP_BARS) | np.isnan(atm_iv)

    return vol_class, move_class, warmup_mask


def _apply_grid(vol_class, move_class, warmup_mask, grid_config):
    """Apply a grid configuration to produce regime labels. Warm-up bars get label -1."""
    n_bars = len(vol_class)
    grid = grid_config["grid"]
    bar_labels = np.full(n_bars, -1, dtype=np.int64)
    for b in range(n_bars):
        if warmup_mask[b]:
            continue
        bar_labels[b] = grid[(int(vol_class[b]), int(move_class[b]))]
    return bar_labels








# ===================================================================
# 6. SSL SPLIT AND LOADERS
# ===================================================================

def make_ssl_split_and_loaders(data):
    """Split SSL windows by date, apply global z-score, build DataLoaders.

    No labels needed — each DataLoader yields (X, W) where W is C10 weights.
    Z-score stats saved for inference.
    """
    X_all = data["X_windows"].copy()  # : defensive copy — don't mutate cached ssl_data on re-run
    W_all = data["W_windows"]
    dates = data["window_dates"]
    BS = SSL_HYPERPARAMS["batch_size"]

    train_mask = dates <= TRAIN_END
    val_mask = (dates >= VAL_START) & (dates <= VAL_END)

    # -b: log1p compression for heavy-tailed variates BEFORE z-scoring.
    # Resolves raw variate indices to positions in ACTIVE_SSL_FEATURES.
    log1p_cols = sorted(
        ACTIVE_SSL_FEATURES.index(v)
        for v in SSL_LOG1P_VARIATES if v in ACTIVE_SSL_FEATURES
    )
    if log1p_cols:
        print(f"  D78-b: Applying log1p to {len(log1p_cols)} heavy-tailed variates "
              f"(raw: {sorted(SSL_LOG1P_VARIATES & set(ACTIVE_SSL_FEATURES))})")
        X_all[:, :, log1p_cols] = np.sign(X_all[:, :, log1p_cols]) * np.log1p(np.abs(X_all[:, :, log1p_cols]))

    # : Global z-score from train windows. Use nanmean/nanstd — v3
    # aggregates carry NaN on dead days that squeak past should_skip_day
    # (NaN != 0 so np.all(X == 0) misses NaN-only columns). nanmean/nanstd
    # ignore NaN per-column so the aggregate stats come from live bars
    # only, and downstream torch.nan_to_num at batch load produces a
    # deterministic 0 at masked positions instead of NaN-poisoned stats.
    X_train_raw = X_all[train_mask]
    N_tr, T, V = X_train_raw.shape
    flat = X_train_raw.reshape(-1, V)
    global_mean = np.nanmean(flat, axis=0)
    global_std = np.nanstd(flat, axis=0)
    # Warn if any column was fully-NaN (all samples NaN on that variate)
    fully_nan = np.isnan(global_mean)
    if fully_nan.any():
        bad_positions = np.where(fully_nan)[0].tolist()
        bad_raws = [ACTIVE_SSL_FEATURES[i] for i in bad_positions]
        print(f"  WARNING: {fully_nan.sum()} variate(s) have all-NaN training "
              f"data; falling back to mean=0, std=1. raw: {bad_raws}")
        global_mean = np.where(fully_nan, 0.0, global_mean)
        global_std = np.where(fully_nan, 1.0, global_std)
    # Warnings on raw stats BEFORE flooring (so near-zero check isn't masked)
    raw_std = global_std.copy()
    for i, fidx in enumerate(ACTIVE_SSL_FEATURES):
        if abs(global_mean[i]) > 10:
            print(f"  WARNING: SSL feature v{fidx} extreme mean={global_mean[i]:.2f}")
        if raw_std[i] < 1e-4:
            print(f"  WARNING: SSL feature v{fidx} near-zero std={raw_std[i]:.6f}")

    # -a: Min std floor 0.01 (was 1e-8). Prevents 10^8x magnification for
    # tiny-std features. Features with std < 0.01 get divided by 0.01 instead.
    MIN_STD_FLOOR = 0.01
    n_floored = int((global_std < MIN_STD_FLOOR).sum())
    global_std = np.where(global_std < MIN_STD_FLOOR, MIN_STD_FLOOR, global_std)
    print(f"  SSL z-score: mean [{global_mean.min():.4f}, {global_mean.max():.4f}], "
          f"std [{global_std.min():.4f}, {global_std.max():.4f}]")
    if n_floored > 0:
        floored_vars = [f"v{ACTIVE_SSL_FEATURES[i]}" for i in range(V) if raw_std[i] < MIN_STD_FLOOR]
        print(f"  D78-a: {n_floored} variates floored at std={MIN_STD_FLOOR}: {floored_vars}")

    # OF-3: Quantile preprocessing for flow variates. Replaces z-score with
    # QuantileTransformer(output_distribution='normal') for order_flow positions.
    # This converts the bimodal (70% zero, 30% heavy tail) distribution into
    # proper Gaussian, eliminating the distributional mismatch with Huber loss.
    # Literature: Gorishniy et al. (2021) "Revisiting Deep Learning Models for
    # Tabular Data" — quantile normalization substantially improves DL on heavy tails.
    _quantile_flow = SSL_HYPERPARAMS.get("quantile_flow", False)
    _qt_models = {}
    if _quantile_flow:
        from sklearn.preprocessing import QuantileTransformer
        # Flow variate positions in ACTIVE_SSL_FEATURES
        _flow_raw = list(range(131, 140))  # v131-v139
        _flow_positions = sorted(
            ACTIVE_SSL_FEATURES.index(v) for v in _flow_raw if v in ACTIVE_SSL_FEATURES
        )
        if _flow_positions:
            print(f"  OF-3: Quantile preprocessing for {len(_flow_positions)} flow variates "
                  f"(positions {_flow_positions[:3]}...)")
            # Fit on training data (flattened across time)
            X_train_flow = X_all[train_mask][:, :, _flow_positions].reshape(-1, len(_flow_positions))
            # Remove NaN rows for fitting
            valid_rows = ~np.isnan(X_train_flow).any(axis=1)
            qt = QuantileTransformer(
                n_quantiles=min(1000, valid_rows.sum()),
                output_distribution='normal',
                random_state=42,
            )
            qt.fit(X_train_flow[valid_rows])
            _qt_models['flow'] = (qt, _flow_positions)
            # Transform ALL data (train + val) using the train-fitted transformer
            N_all = X_all.shape[0]
            T_all = X_all.shape[1]
            flow_flat = X_all[:, :, _flow_positions].reshape(-1, len(_flow_positions))
            flow_transformed = qt.transform(np.nan_to_num(flow_flat, nan=0.0))
            X_all[:, :, _flow_positions] = flow_transformed.reshape(N_all, T_all, len(_flow_positions))
            print(f"    Quantile-transformed flow: range [{X_all[:,:,_flow_positions].min():.2f}, "
                  f"{X_all[:,:,_flow_positions].max():.2f}]")

    # : z-score + clip
    X_normed = (X_all - global_mean) / global_std
    X_normed = np.clip(X_normed, -5.0, 5.0)

    # OF-3 post-step: flow variates already N(0,1) from QuantileTransformer.
    # The z-score above corrupted them — restore from the pre-z-score quantile
    # output (X_all already has quantile-transformed flow values, just clip).
    if _quantile_flow and _qt_models:
        _, positions = _qt_models['flow']
        # X_all[:, :, positions] is already quantile-transformed to N(0,1)
        X_normed[:, :, positions] = np.clip(X_all[:, :, positions], -5.0, 5.0)

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask)]:
        X = X_normed[mask].astype(np.float32)
        W = W_all[mask].astype(np.float32)
        n = len(X)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(W))
        loader = DataLoader(
            ds, batch_size=BS,
            shuffle=(name == "train"),
            drop_last=(name == "train"),
            num_workers=0,
        )
        splits[name] = loader
        print(f"  {name}: {n} windows")

    splits["z_stats"] = {
        "mean": global_mean,
        "std": global_std,
        "log1p_cols": log1p_cols,       # -b: columns that were log1p-transformed
        "min_std_floor": MIN_STD_FLOOR,  # -a: reproducibility
    }
    return splits


# ===================================================================
# 7. MODEL
# ===================================================================

class iTransformerEncoderBlock(nn.Module):
    """D56: Standard attention (no de-stationary tau), matching official iTransformer."""
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        z2 = self.norm1(z)
        a, w = self.attn(z2, z2, z2, need_weights=True)
        z = z + self.drop(a)
        z = z + self.ffn(self.norm2(z))
        return z, w


class TemporalAttentionBlock(nn.Module):
    """SSL-012: temporal-axis self-attention block (CrossFormer two-stage hybrid).

    Operates on (N=B*V, T, D) per-variate temporal sequences. Each variate's
    T-step trajectory is attended INDEPENDENTLY along the time axis. No
    cross-variate mixing happens here — that's reserved for the variate-axis
    stack downstream.

    Inserted BEFORE the variate-axis blocks when SSL_HYPERPARAMS["use_temporal_attn"]
    is True. Without this block the encoder has no temporal-axis attention path
    (the legacy iTransformer recipe collapses T into D inside the tokenizer
    in one step), so tokenizer-injected temporal differences cannot be exploited
    by attention. CrossFormer (Zhang & Yan, ICLR 2023) and TimesNet (Wu et al.,
    ICLR 2023) both motivate explicit time-axis attention before cross-dim
    mixing for multivariate forecasting.

    Standard pre-norm transformer block: LN → MHA → residual → LN → FFN → residual.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, z, return_attn_weights=False):
        """z: (N, T, D). Returns (N, T, D) with temporal mixing applied per-variate."""
        z2 = self.norm1(z)
        if return_attn_weights:
            a, w = self.attn(z2, z2, z2, need_weights=True)
        else:
            a, w = self.attn(z2, z2, z2, need_weights=False)
        z = z + self.drop(a)
        z = z + self.ffn(self.norm2(z))
        if return_attn_weights:
            return z, w
        return z


class TemporalAttentionPool(nn.Module):
    """Learned attention pooling over temporal positions.

    Replaces AdaptiveAvgPool1d(1) which uniformly averages all positions,
    destroying temporal location information. Instead, learns content-dependent
    weights: positions where the upstream feature extractor detected meaningful
    patterns (e.g., spread events) get higher weight.

    Adds only d_model + 1 parameters (~65 at d_model=64).
    """
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, z):
        """
        Args:
            z: (N, T, D) — temporal feature sequence
        Returns:
            pooled: (N, D) — attention-weighted sum
        """
        scores = self.score(z).squeeze(-1)           # (N, T)
        weights = torch.softmax(scores, dim=-1)      # (N, T)
        return (z * weights.unsqueeze(-1)).sum(dim=1) # (N, D)


class PatchTokenizer(nn.Module):
    """L2.8 Step 1a: Patch tokenizer preserving local temporal structure.

    Divides each variate's T-bar temporal profile into non-overlapping patches
    of patch_size bars, projects each patch to d_model dimensions, then
    attention-pools across patches. Preserves 5-bar local patterns while
    using efficient GEMM operations (not Conv1d).

    At T=60, patch_size=5: 12 patches per variate, each capturing one
    5-bar local window. The attention pool learns which temporal windows
    matter — windows with spread events or IV shifts get higher weight.

    Literature: Nie et al. 2023 (PatchTST — patching retains local semantic
    information); Huo et al. 2025 (CT-PatchTST — dual attention on patches).
    """
    def __init__(self, seq_len, d_model, patch_size=5, dropout=0.2):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"seq_len={seq_len} must be divisible by patch_size={patch_size}"
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.pool = TemporalAttentionPool(d_model)

    def forward_embed(self, x_vt):
        """Patch embedding WITHOUT pooling — returns per-patch D-dim embeddings.

        Used when temporal attention operates AFTER patching (correct architecture
        for patch tokenizer). The caller applies temporal attention on the 12
        patch embeddings, THEN pools.

        Args:
            x_vt: (B, V, T) — per-variate temporal profiles
        Returns:
            z: (B*V, n_patches, D) — per-patch embeddings before pooling
            BV_shape: tuple (B, V) for reshaping after pooling
        """
        B, V, T = x_vt.shape
        # Reshape into patches: (B, V, n_patches, patch_size)
        x = x_vt.reshape(B, V, self.n_patches, self.patch_size)
        # Project each patch: (B, V, n_patches, D)
        z = self.drop(torch.nn.functional.gelu(self.proj(x)))
        # Reshape for downstream temporal attention: (B*V, n_patches, D)
        z = z.reshape(B * V, self.n_patches, -1)
        return z, (B, V)

    def forward(self, x_vt):
        """
        Args:
            x_vt: (B, V, T) — per-variate temporal profiles
        Returns:
            z: (B, V, D) — per-variate token embeddings
        """
        z, (B, V) = self.forward_embed(x_vt)
        # Attention pool across patches: (B*V, D)
        z = self.pool(z)
        return z.reshape(B, V, -1)                   # (B, V, D)


class PatchTokenizerV2(nn.Module):
    """PatchTST-style: larger patches + concat + project (no temporal attention, no mean-pool).

    Fixes the rank-5 bottleneck of PatchTokenizerV1 (nn.Linear(5, 128) = rank 5).
    With patch_size=12: rank per patch = 12. 5 patches concatenated: effective rank = 60.
    This matches the linear tokenizer's rank = min(60, 128) = 60.

    Literature: Nie et al. (2023) "A Time Series is Worth 64 Words" (PatchTST, ICLR).
    """
    def __init__(self, seq_len, d_model, patch_size=12, dropout=0.2):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"seq_len={seq_len} must be divisible by patch_size={patch_size}"
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size  # 60/12 = 5

        # Per-patch projection: rank = min(patch_size, d_model) = 12
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.patch_norm = nn.LayerNorm(d_model)

        # Positional encoding for patches (scaled stronger than the old 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.1)

        # Concatenate all patches + project: (n_patches * d_model) -> d_model
        # This replaces mean-pool with a rank-preserving learned projection
        self.concat_proj = nn.Linear(self.n_patches * d_model, d_model)
        self.concat_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_vt):
        """
        Args:
            x_vt: (B, V, T) — per-variate temporal profiles
        Returns:
            z: (B, V, D) — per-variate token embeddings
        """
        B, V, T = x_vt.shape
        # Reshape into patches: (B*V, n_patches, patch_size)
        x = x_vt.reshape(B * V, self.n_patches, self.patch_size)

        # Project each patch: (B*V, n_patches, d_model)
        z = self.patch_proj(x)
        z = self.patch_norm(z)
        z = torch.nn.functional.gelu(z)

        # Add positional encoding
        z = z + self.pos_embed

        # NO temporal attention — it was inert across 3 runs (98% entropy, never learned)

        # Concatenate all patches: (B*V, n_patches * d_model) -> (B*V, d_model)
        z = z.reshape(B * V, self.n_patches * z.shape[-1])
        z = self.concat_proj(z)
        z = self.concat_norm(z)
        z = self.drop(z)

        return z.reshape(B, V, -1)  # (B, V, D)


class CNNTokenizer(nn.Module):
    """L2.8 Step 1b: CNN tokenizer preserving local temporal structure.

    Two 1D convolutions with kernel_size=5 (~5-bar receptive field per layer,
    ~9-bar effective receptive field stacked) followed by attention pooling.
    Each variate is processed independently (channel-independent), preserving
    the iTransformer's design where cross-variate interaction happens in the
    attention layers.

    Slower than PatchTokenizer (~1300x vs ~8x over linear on CPU) due to
    the B*V batch expansion with Conv1d. Use checkpoint/resume for multi-
    session training on QC Research.

    Literature: Nagrath & Panigrahy 2026 (CNN on temporal patches preserves
    short-range dynamics).
    """
    def __init__(self, seq_len, d_model, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        mid = d_model // 2  # bottleneck: 1→32→64 at d_model=64 (~+4% total params)
        self.conv = nn.Sequential(
            nn.Conv1d(1, mid, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = TemporalAttentionPool(d_model)

    def forward_embed(self, x_vt):
        """CNN embedding WITHOUT pooling — returns per-position D-dim features.

        Used when temporal attention operates AFTER convolution (correct
        architecture for CNN tokenizer). The caller applies temporal attention
        on the T=60 position embeddings, THEN pools.

        Args:
            x_vt: (B, V, T) — per-variate temporal profiles
        Returns:
            z: (B*V, T, D) — per-position CNN features before pooling
            BV_shape: tuple (B, V) for reshaping after pooling
        """
        B, V, T = x_vt.shape
        x = x_vt.reshape(B * V, 1, T)               # (B*V, 1, T)
        z = self.conv(x)                              # (B*V, d_model, T)
        z = z.transpose(1, 2)                         # (B*V, T, d_model)
        return z, (B, V)

    def forward(self, x_vt):
        """
        Args:
            x_vt: (B, V, T) — per-variate temporal profiles
        Returns:
            z: (B, V, D) — per-variate token embeddings
        """
        z, (B, V) = self.forward_embed(x_vt)
        z = self.pool(z)                              # (B*V, d_model)
        return z.reshape(B, V, -1)                    # (B, V, D)


class iTransformerEncoder(nn.Module):
    """D55-D59: Simplified iTransformer aligned with reference implementation.
    L2.8: Temporal tokenizer ablation (linear / patch / CNN).

    D55: Global z-score replaces NST (normalization moved to data pipeline)
    D56: Standard MultiheadAttention replaces DeStationaryAttention (no tau_mlp)
    D57: Final LayerNorm before pooling (matches official repo)
    D58: Post-embedding dropout after tokenizer + variate embed
    L2.8: Tokenizer comparison — linear (baseline) vs patch vs CNN, all with
          attention pooling. Tests whether preserving local temporal structure
          recovers the spread probe gap (R²=0.17 encoded vs 0.44 raw).
    """
    def __init__(self, hp=None, n_variates=None):
        super().__init__()
        p = hp or SSL_HYPERPARAMS
        V = n_variates or N_SSL_FEATURES
        T = p["seq_len"]
        D = p["d_model"]
        # Legacy classifier heads — kept for checkpoint compatibility
        R = p.get("num_regimes", 4)
        E = p.get("embedding_dim", 32)

        # SSL-012: optional temporal-axis attention block.
        #
        # ARCHITECTURE (2026-04-30 refactor):
        # - For tokenizer="linear": temporal attention operates BEFORE the tokenizer
        # on raw (B, V, T=60) scalar input. Uses cell_proj (1→D) + pos_embed(60,D)
        # + TemporalAttentionBlock + readout(D→1) + residual to produce temporally-
        # mixed (B, V, T) scalars, preserving the linear tokenizer's T→D contract.
        # - For tokenizer="patch" or "cnn": temporal attention operates AFTER the
        # tokenizer's embedding step, on already D-dimensional representations:
        # * Patch: (B*V, 12, D) — selects which 5-bar patches are informative
        # * CNN: (B*V, 60, D) — selects which convolutional positions matter
        # No cell_proj or readout needed — patches/CNN outputs are already D-dim.
        # Replaces the tokenizer's built-in TemporalAttentionPool: temporal
        # attention IS the selective aggregation, followed by mean-pool.
        #
        # SSL-012 M1: mask-indicator channel. When enabled, the encoder receives
        # a per-cell binary "is_masked" signal projected to D and added to the
        # tokenizer output. Breaks the indistinguishability between genuine
        # near-zero cells and masked-and-filled cells (concern #2 in module
        # header, deferred at SSL-004-FE design time). Operates independent of
        # the M2 distinguished mask fill value — M1 alone tells the encoder
        # WHICH cells are masked even if M2 isn't on; the two together are the
        # canonical pairing.
        self.use_mask_indicator = p.get("use_mask_indicator", False)
        if self.use_mask_indicator:
            self.mask_proj = nn.Linear(T, D)
            # N2 KNOWN LIMITATION (2026-04-29, documented per agent audit):
            # At training, mask_proj receives varied binary masks (~30% density).
            # At probe/inference (mask=None), zeros are materialized so
            # mask_proj.bias fires but weight contribution is zero — a constant
            # shift the encoder NEVER sees during training (every training batch
            # has non-trivial mask density). This is a train→probe distribution
            # shift on the M1 channel. UNIFORM across all 3 tokenizer arms so
            # cross-arm comparison is not biased. Absolute "encoder beats baseline"
            # claims are slightly weakened. Two possible future fixes:
            # (a) zero-init weight AND bias so mask_proj is a no-op when input
            # is zero — but training will still push bias ≠ 0.
            # (b) at probe time, sample K masks at training-time density and
            # average embeddings — eliminates the shift but K× more compute.
            # Neither is implemented this cycle. Tracked in session-state memory.
            print(f"  Mask-indicator (M1): ENABLED ({T*D + D:,} params)")

        self.use_temporal_attn = p.get("use_temporal_attn", False)
        tok_type = p.get("tokenizer", "linear")
        # design (revised): PatchTokenizerV2 does NOT require temporal attention
        # (it uses concat+project instead of attention pooling). CNN still requires it.
        if tok_type == "cnn" and not p.get("use_temporal_attn", False):
            raise ValueError(
                f"tokenizer='cnn' requires use_temporal_attn=True per D93 "
                f"design (temporal attention operates post-tokenizer on CNN "
                f"embeddings). Set use_temporal_attn=True in SSL_HYPERPARAMS."
            )
        if tok_type == "patch" and p.get("use_temporal_attn", False):
            raise ValueError(
                f"tokenizer='patch' (V2) is incompatible with use_temporal_attn=True. "
                f"V2 uses concat+project aggregation internally; post-tokenizer "
                f"temporal attention would require forward_embed() which V2 does not "
                f"implement. Set use_temporal_attn=False for patch arm."
            )
        # Track which temporal-attention mode we're in for forward routing
        self._temporal_attn_mode = None
        if self.use_temporal_attn:
            if tok_type == "linear":
                # PRE-TOKENIZER mode: temporal attention on raw scalar bars.
                # cell_proj lifts each scalar to D-dim, pos_embed adds position,
                # temporal_block attends, readout projects back to scalar + residual.
                self._temporal_attn_mode = "pre_tokenizer"
                self.cell_proj = nn.Linear(1, D)
                # Use temporal_pos_embed (not temporal_pos) so the existing
                # 'embed' substring in the no_decay rule (train_ssl) catches it
                # by name and exempts the positional embedding from weight decay,
                # consistent with variate_embed treatment.
                self.temporal_pos_embed = nn.Parameter(torch.randn(T, D) * 0.02)
                self.temporal_block = TemporalAttentionBlock(
                    D, p["n_heads"], p["d_ff"], p["dropout"]
                )
                # Per-cell scalar readout: (B, V, T, D) → (B, V, T). Allows the
                # downstream tokenizer to operate on its native scalar input
                # while still having received temporal mixing.
                #
                # ZERO-INIT for identity equivalence at epoch 0: combined with the
                # residual `x_vt + delta` in _temporal_mix, this makes SSL-012
                # forward output match SSL-010 exactly at init. Any later
                # divergence is then attributable to learned temporal signal.
                self.temporal_readout = nn.Linear(D, 1)
                with torch.no_grad():
                    self.temporal_readout.weight.zero_()
                    self.temporal_readout.bias.zero_()
                print(f"  Temporal attention: ENABLED pre-tokenizer (linear mode, "
                      f"+{T*D + D*D*4 + D*p['d_ff']*2 + D*2:,} params approx, "
                      f"identity-init via zero-init readout + residual)")
            else:
                # POST-TOKENIZER mode: temporal attention on D-dim embeddings
                # from the patch/CNN tokenizer's embedding step.
                # For patch: T_temporal = n_patches = T // patch_size (12 at T=60, ps=5)
                # For CNN: T_temporal = T (same-padding conv preserves length)
                self._temporal_attn_mode = "post_tokenizer"
                if tok_type == "patch":
                    patch_size = p.get("patch_size", 12)
                    self._temporal_seq_len = T // patch_size  # 5 at T=60, ps=12
                else:  # cnn
                    self._temporal_seq_len = T  # 60
                T_temp = self._temporal_seq_len
                self.temporal_pos_embed = nn.Parameter(
                    torch.randn(T_temp, D) * 0.02
                )
                self.temporal_block = TemporalAttentionBlock(
                    D, p["n_heads"], p["d_ff"], p["dropout"]
                )
                # No cell_proj needed: patch/CNN outputs are already D-dimensional.
                # No temporal_readout needed: we don't project back to scalar.
                # After temporal attention, we mean-pool the attended positions
                # to get (B*V, D). The TemporalAttentionBlock's attention weights
                # already perform soft selection; mean-pool is the aggregation
                # (simpler than learned attention pool — the temporal block IS
                # the learned weighting mechanism).
                print(f"  Temporal attention: ENABLED post-tokenizer ({tok_type} mode, "
                      f"T_temporal={T_temp}, +{T_temp*D + D*D*4 + D*p['d_ff']*2 + D*2:,} "
                      f"params approx)")

        # : tokenizer selection — "linear" (default), "patch", or "cnn"
        tok_type = p.get("tokenizer", "linear")
        if tok_type == "patch":
            patch_size = p.get("patch_size", 12)  # V2: larger patches for rank parity with linear (was 5)
            self.tok = PatchTokenizerV2(T, D, patch_size, p["dropout"])
        elif tok_type == "cnn":
            self.tok = CNNTokenizer(T, D, p["dropout"])
        else:
            self.tok = nn.Linear(T, D)
        print(f"  Tokenizer: {tok_type}")

        # : learnable variate identity embedding
        self.variate_embed = nn.Parameter(torch.randn(V, D) * 0.02)

        # : post-embedding dropout
        self.embed_drop = nn.Dropout(p["dropout"])

        self.blocks = nn.ModuleList([
            iTransformerEncoderBlock(D, p["n_heads"], p["d_ff"], p["dropout"])
            for _ in range(p["n_layers"])
        ])

        # : final LayerNorm before pooling (matches official iTransformer)
        self.final_norm = nn.LayerNorm(D)

        self.pool_q = nn.Parameter(torch.randn(1, 1, D))
        self.pool_attn = nn.MultiheadAttention(D, num_heads=1, batch_first=True)
        self.regime_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(), nn.Dropout(p["dropout"]),
            nn.Linear(D // 2, R),
        )
        self.emb_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, E),
        )

    def _temporal_mix(self, x_vt):
        """PRE-TOKENIZER temporal attention (linear tokenizer only).

        Apply temporal-axis attention per-variate on raw scalar bars,
        returning (B, V, T) scalars for the linear tokenizer's T->D contract.

        Identity-init contract: temporal_readout has zero weights and zero bias
        (set in __init__), so the temporal-block branch produces zeros at init,
        and the residual `x_vt + 0` exactly preserves raw input. This makes
        SSL-012 forward output identical to SSL-010 at epoch 0; any deviation
        through training is a learned signal, not a random scrambling.

        Only called when _temporal_attn_mode == "pre_tokenizer" (linear arm).

        Args:
            x_vt: (B, V, T) — per-variate temporal scalar input
        Returns:
            (B, V, T) — input + temporally-mixed correction (residual).
        """
        B, V_dim, T = x_vt.shape
        # (B, V, T, 1) → cell_proj → (B, V, T, D) + pos → reshape (B*V, T, D)
        z = self.cell_proj(x_vt.unsqueeze(-1))                          # (B, V, T, D)
        z = z + self.temporal_pos_embed.unsqueeze(0).unsqueeze(0)       # broadcast over (B, V)
        D_dim = z.shape[-1]
        z = z.reshape(B * V_dim, T, D_dim)
        z = self.temporal_block(z)                              # (B*V, T, D)
        z = z.reshape(B, V_dim, T, D_dim)
        # Per-cell scalar readout (D → 1) + residual. With temporal_readout
        # zero-initialized, this returns x_vt at epoch 0.
        delta = self.temporal_readout(z).squeeze(-1)           # (B, V, T)
        return x_vt + delta

    def _temporal_mix_post_tok(self, z_seq):
        """POST-TOKENIZER temporal attention (patch/CNN tokenizers).

        Apply temporal-axis self-attention on already D-dimensional embeddings
        from the patch/CNN tokenizer's embedding step. The attention selects
        WHICH temporal positions (patches or convolutional positions) are
        informative, then mean-pools to produce (N, D).

        For patch tokenizer: z_seq is (B*V, 12, D) — 12 patch embeddings.
        For CNN tokenizer:   z_seq is (B*V, 60, D) — 60 convolutional positions.

        The temporal_pos_embed is sized to match (12 or 60 positions).

        Args:
            z_seq: (N, T_temporal, D) — post-tokenizer temporal embeddings
        Returns:
            z_pooled: (N, D) — temporally-attended and pooled embeddings
        """
        # Add learned positional embedding: (T_temporal, D) broadcast over N
        z_seq = z_seq + self.temporal_pos_embed.unsqueeze(0)
        # Temporal self-attention: each variate's temporal positions attend
        # to each other — patches/positions that are informative get amplified.
        z_seq = self.temporal_block(z_seq)                  # (N, T_temporal, D)
        # Mean pool over temporal positions. The attention block has already
        # performed soft selection (informative positions have larger norms
        # post-attention), so mean-pool aggregates the attended representation.
        return z_seq.mean(dim=1)                            # (N, D)

    def _tokenize_with_temporal(self, x_vt):
        """Tokenize with post-tokenizer temporal attention (patch/CNN path).

        Calls the tokenizer's forward_embed to get per-position D-dim
        embeddings, applies temporal attention + pos embed, then mean-pools.
        Returns (B, V, D) matching the standard tokenizer output contract.

        Args:
            x_vt: (B, V, T) — per-variate temporal profiles
        Returns:
            z: (B, V, D) — per-variate token embeddings
        """
        # Get embeddings without pooling
        z_seq, (B, V) = self.tok.forward_embed(x_vt)    # (B*V, T_temporal, D)
        # Apply temporal attention and pool
        z_pooled = self._temporal_mix_post_tok(z_seq)    # (B*V, D)
        return z_pooled.reshape(B, V, -1)                # (B, V, D)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, V) globally z-scored input (D55). When SSL training,
               masked positions have already been replaced with mask_fill_value
               (M2) by the caller — this method does not perform masking.
            mask: optional (B, T, V) bool — True where cells are masked. When
               use_mask_indicator (M1) is True AND mask is None, an all-zeros
               mask is materialized so mask_proj.bias still fires (matches the
               training-time forward distribution where mask_proj contributes
               every batch). Bug fix 2026-04-29: previously this path skipped
               mask_proj entirely on mask=None, creating a train/probe input
               distribution mismatch and an apparent regression in cross-fold
               probe metrics. The L2 inference adapter (batch_forecast.py)
               always materialized a zeros mask; this path is now consistent.
        Returns: (logits, probs, embedding, attn_weights_all)
        """
        B = x.shape[0]
        x_vt = x.transpose(1, 2)                             # (B, V, T)
        if self.use_temporal_attn and self._temporal_attn_mode == "pre_tokenizer":
            x_vt = self._temporal_mix(x_vt)                  # (B, V, T) post-temporal
            z = self.tok(x_vt)                               # (B, V, D)
        elif self.use_temporal_attn and self._temporal_attn_mode == "post_tokenizer":
            z = self._tokenize_with_temporal(x_vt)           # (B, V, D)
        else:
            z = self.tok(x_vt)                               # (B, V, D)
        if self.use_mask_indicator:
            if mask is None:
                # Probe / baseline / classifier-eval path: no cells masked at
                # inference. Materialize zeros so mask_proj.bias contributes,
                # matching training-time forward distribution.
                mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2],
                                    dtype=torch.float32, device=x.device)
            else:
                # Shape contract: mask must be (B, T, V) bool/float so the
                # transpose maps to (B, V, T) → mask_proj(60→D). Loud failure if
                # any caller passes (B, V, T) directly or a packed-flag shape.
                assert mask.shape == x.shape, \
                    f"mask shape {tuple(mask.shape)} must equal input x shape {tuple(x.shape)}"
            z = z + self.mask_proj(mask.transpose(1, 2).float())
        z = z + self.variate_embed.unsqueeze(0)
        z = self.embed_drop(z)  #

        attn_weights_all = []
        for block in self.blocks:
            z, w = block(z)
            attn_weights_all.append(w)

        z = self.final_norm(z)  #

        pq = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool_attn(pq, z, z)
        pooled = pooled.squeeze(1)

        logits = self.regime_head(pooled)
        probs = torch.softmax(logits, dim=-1)
        emb = self.emb_head(pooled)
        return logits, probs, emb, attn_weights_all

    def encode_variates(self, x, multi_layer=False, return_attn=False, mask=None):
        """SSL mode: return per-variate token embeddings.

        Args:
            x: (B, T, V) — masked, z-scored input. Caller must have already
               replaced masked cells with mask_fill_value (M2) before calling.
            multi_layer: if True, return concatenated embeddings from all layers
                         (B, V, D*n_layers) for richer probe representations.
                         Default False preserves backward compatibility for SSL training.
            return_attn: if True, also return a list of attention weight
                         tensors (one per layer, shape (B, V, V) averaged
                         over heads by nn.MultiheadAttention default). Used
                         by the attention-entropy diagnostic.
            mask: optional (B, T, V) bool mask (True where cells were masked).
                  When use_mask_indicator (M1) is enabled AND mask is None,
                  an all-zeros mask is materialized so mask_proj.bias still
                  fires (matches training-time forward distribution). Bug fix
                  2026-04-29: previously skipped mask_proj entirely on
                  mask=None, creating a train/probe input distribution
                  mismatch.
        Returns:
            z: (B, V, D) or (B, V, D*n_layers) — per-variate embeddings
            (optional) attn_weights: list[Tensor(B, V, V)] — one per layer
        """
        x_vt = x.transpose(1, 2)                  # (B, V, T)
        if self.use_temporal_attn and self._temporal_attn_mode == "pre_tokenizer":
            x_vt = self._temporal_mix(x_vt)       # (B, V, T) post-temporal
            z = self.tok(x_vt)                    # (B, V, D)
        elif self.use_temporal_attn and self._temporal_attn_mode == "post_tokenizer":
            z = self._tokenize_with_temporal(x_vt)  # (B, V, D)
        else:
            z = self.tok(x_vt)                    # (B, V, D)
        if self.use_mask_indicator:
            if mask is None:
                # Probe / baseline / classifier-eval path: no cells masked at
                # inference. Materialize zeros so mask_proj.bias contributes,
                # matching training-time forward distribution. Bug fix
                # 2026-04-29: previously skipped mask_proj entirely on
                # mask=None, creating a probe-time distribution mismatch.
                mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2],
                                    dtype=torch.float32, device=x.device)
            else:
                # Shape contract: mask must be (B, T, V) bool/float.
                assert mask.shape == x.shape, \
                    f"mask shape {tuple(mask.shape)} must equal input x shape {tuple(x.shape)}"
            z = z + self.mask_proj(mask.transpose(1, 2).float())
        z = z + self.variate_embed.unsqueeze(0)
        z = self.embed_drop(z)
        attn_weights_all = [] if return_attn else None
        if multi_layer:
            layer_outputs = []
            for block in self.blocks:
                z, w = block(z)
                if return_attn:
                    attn_weights_all.append(w)
                layer_outputs.append(self.final_norm(z))
            out = torch.cat(layer_outputs, dim=-1)  # (B, V, D*n_layers)
            return (out, attn_weights_all) if return_attn else out
        for block in self.blocks:
            z, w = block(z)
            if return_attn:
                attn_weights_all.append(w)
        z = self.final_norm(z)
        return (z, attn_weights_all) if return_attn else z


def benchmark_tokenizers(n_iters=5):
    """Benchmark all three tokenizers on this machine. Run before committing
    to a multi-session CNN training run to get actual QC Research timings.

    Prints forward+backward time per batch for linear, patch, and CNN
    tokenizers at the current SSL_HYPERPARAMS configuration.
    """
    import time as _time

    p = SSL_HYPERPARAMS
    D, T = p["d_model"], p["seq_len"]
    B, V = p["batch_size"], N_SSL_FEATURES
    x = torch.randn(B, T, V)  # same shape as training batches

    results = {}
    for tok_name in ["linear", "patch", "cnn"]:
        hp = dict(p, tokenizer=tok_name)
        model = iTransformerEncoder(hp, n_variates=V)
        recon = VariateReconstructionHead(D, T)
        model.train()
        recon.train()

        # Warmup
        tokens = model.encode_variates(x)
        out = recon(tokens)
        out.sum().backward()

        # Timed runs
        times = []
        for _ in range(n_iters):
            model.zero_grad()
            recon.zero_grad()
            t0 = _time.perf_counter()
            tokens = model.encode_variates(x)
            out = recon(tokens)
            out.sum().backward()
            times.append(_time.perf_counter() - t0)

        avg_ms = sum(times) / len(times) * 1000
        n_params = sum(p_.numel() for p_ in model.parameters())
        results[tok_name] = avg_ms
        print(f"  {tok_name:>8s}: {avg_ms:>8.1f} ms/batch (fwd+bwd)  |  {n_params:,} params")

    # Estimate training time
    n_batches = 95  # ~6084 windows / batch 64
    n_epochs = p["num_epochs"]
    print(f"\n  Estimated training time ({n_epochs} epochs × {n_batches} batches):")
    for tok_name, ms in results.items():
        total_min = ms * n_batches * n_epochs / 1000 / 60
        sessions = max(1, int(total_min / 110) + 1)  # ~110 min usable per 2hr session
        print(f"    {tok_name:>8s}: ~{total_min:.0f} min ({sessions} QC session{'s' if sessions > 1 else ''})")


# ===================================================================
# 8. LOSS
# ===================================================================



class VariateReconstructionHead(nn.Module):
    """D71 SSL: reconstruct T time steps from each variate token.

    Baseline path (use_film=False or n_variates=None):
        Single shared `Linear(d_model, seq_len)` applied to (B, V, D) → (B, V, T).
        ~7.7K params at D=128, T=60 — the spec recipe (section 11.1).

    SSL-012 FiLM path (use_film=True with n_variates):
        After the shared projection, apply per-variate affine
            x_hat[b, v, t] = (proj(token)[b, v, t]) * scale[v, t] + shift[v, t]
        with scale initialized to 1.0 and shift to 0.0. **At init, mathematically
        equivalent to the baseline head** (multiplying by 1 and adding 0). Drift
        of (scale, shift) from init during training measures how much
        per-variate temporal conditioning the encoder needs that the shared
        head cannot absorb. If FiLM weights stay near init at end of training,
        the shared head was sufficient; if they drift substantially, variate
        conditioning was load-bearing — diagnostic on the recon-head bottleneck
        hypothesis. Adds 2*V*T params (17,640 at V=147, T=60).

    Input: (B, V, d_model) from iTransformerEncoder.encode_variates()
    Output: (B, V, T) predicted time series per variate
    """
    def __init__(self, d_model, seq_len, n_variates=None, use_film=False,
                 group_heads=None):
        """
        Args:
            group_heads: optional dict mapping group_name → list of variate
                POSITIONS (0-indexed into the V dimension) that belong to that
                group. When provided, replaces the shared `proj` with per-group
                `nn.Linear(D, T)` heads — each group's variates get decoded by
                their own head. This prevents the shared-decoder collapse
                (agent root-cause analysis 2026-04-30: shared Linear(D,T)
                forces a low-rank subspace, driving emb_var → 0.023).
                When None, uses the single shared head (backward compat).
        """
        super().__init__()
        self.use_group_heads = group_heads is not None and len(group_heads) > 0
        if self.use_group_heads:
            self.group_projs = nn.ModuleDict()
            self.group_indices = {}
            for gname, positions in group_heads.items():
                self.group_projs[gname] = nn.Linear(d_model, seq_len)
                self.group_indices[gname] = positions
            total_mapped = sum(len(v) for v in group_heads.values())
            print(f"  Recon head: {len(group_heads)} per-group heads "
                  f"({total_mapped} variates mapped, "
                  f"{len(group_heads) * (d_model * seq_len + seq_len):,} params)")
        else:
            self.proj = nn.Linear(d_model, seq_len)
        self.use_film = bool(use_film) and (n_variates is not None) and not self.use_group_heads
        if self.use_film:
            self.film_scale = nn.Parameter(torch.ones(n_variates, seq_len))
            self.film_shift = nn.Parameter(torch.zeros(n_variates, seq_len))

    def forward(self, variate_tokens):
        if self.use_group_heads:
            B, V, D = variate_tokens.shape
            T = next(iter(self.group_projs.values())).out_features
            out = torch.zeros(B, V, T, device=variate_tokens.device,
                              dtype=variate_tokens.dtype)
            for gname, proj in self.group_projs.items():
                idx = self.group_indices[gname]
                out[:, idx, :] = proj(variate_tokens[:, idx, :])
            return out
        out = self.proj(variate_tokens)  # (B, V, T)
        if self.use_film:
            out = out * self.film_scale.unsqueeze(0) + self.film_shift.unsqueeze(0)
        return out

    def film_drift(self):
        """Diagnostic: L2 norm of FiLM-weight deviation from identity init.

        Returns dict with scale_drift = ||scale - 1||, shift_drift = ||shift||.
        Drift > ~0.5 indicates per-variate conditioning was load-bearing.
        Returns None when FiLM is disabled.
        """
        if not self.use_film:
            return None
        with torch.no_grad():
            scale_drift = (self.film_scale - 1.0).norm().item()
            shift_drift = self.film_shift.norm().item()
        return {"film_scale_drift": scale_drift, "film_shift_drift": shift_drift}


def _build_recon_head_from_hp(hp, d_model, seq_len, n_variates, feature_indices):
    """Construct a VariateReconstructionHead matching the given hyperparams.

    Handles both shared-decoder and per-group-decoder configurations.
    Use this at every checkpoint-load site to avoid key mismatches.
    """
    group_heads = None
    use_group = hp.get("use_group_recon", False)
    if use_group:
        _pos_of = {v: i for i, v in enumerate(feature_indices)}
        group_heads = {
            "options_grid":  [_pos_of[v] for v in feature_indices if v < 88],
            "strike_agg":    [_pos_of[v] for v in feature_indices if 99 <= v < 105],
            "spx_derived":   [_pos_of[v] for v in feature_indices if 105 <= v < 119],
            "vix_term":      [_pos_of[v] for v in feature_indices if 119 <= v < 131],
            "order_flow":    [_pos_of[v] for v in feature_indices if 131 <= v < 140],
            "flow_roll3":    [_pos_of[v] for v in feature_indices if 141 <= v < 150],
            "flow_roll15":   [_pos_of[v] for v in feature_indices if 150 <= v < 159],
        }
    return VariateReconstructionHead(
        d_model, seq_len,
        n_variates=n_variates,
        use_film=hp.get("use_film_recon", False) and not use_group,
        group_heads=group_heads,
    )


class MaskedVariateLoss(nn.Module):
    """D71 SSL: MSE on masked positions, weighted by C10 mask weights.

    loss = sum( (x_hat - x_target)^2 * mask * weights ) / sum(mask * weights)

    Normalizes by effective masked count so loss magnitude is independent of
    mask ratio and batch size. If no positions are masked (edge case),
    returns 0.
    """
    def forward(self, x_hat_vt, x_target_tv, mask_tv, weights_tv,
                heavy_tail_positions=None, tail_weight_alpha=0.0,
                huber_delta=0.0, group_loss_weight=None,
                use_logcosh=False):
        """
        Args:
            x_hat_vt: (B, V, T) — reconstructed from encoder + recon head
            x_target_tv: (B, T, V) — original z-scored input
            mask_tv: (B, T, V) bool — True where masked
            weights_tv: (B, T, V) float — C10 quality weights [0, 1]
            heavy_tail_positions: optional iterable[int] — positions (in
                ACTIVE_SSL_FEATURES) of variates eligible for tail weighting.
                Used by SSL-012 M3 to apply per-cell weight boost on
                v133/v135/v136/v137 (Pass 2 active-tail-shrinkage variates).
            tail_weight_alpha: float — when > 0 AND heavy_tail_positions is
                provided, per-cell weight is multiplied by
                (1 + tail_weight_alpha * |z_target|) for cells in those
                variates. Forces the encoder to attempt magnitude prediction
                instead of mean-collapsing to ~0.

        Returns:
            loss: scalar
            info: dict with diagnostic keys
        """
        # Align shapes: transpose reconstruction to (B, T, V)
        x_hat = x_hat_vt.transpose(1, 2)

        # Huber loss (2026-04-30): replaces uniform MSE to address MQA.6
        # shrinkage on heavy-tail variates. MSE penalizes large errors
        # quadratically → encoder learns mean-collapse on flow variates.
        # Huber transitions to linear at |residual| > delta, capping gradient.
        # Literature: Huber 1964; Barron 2019 (adaptive robust loss).
        # delta=0.0 means use MSE (backward compat).
        # huber_delta is a named parameter on the function signature (line 4057).
        # Do NOT re-read from kwargs — that would shadow the named param with 0.0.
        residual = x_hat - x_target_tv
        if use_logcosh:
            # OF-4: Log-cosh loss. Gradient = tanh(r) which asymptotes to ±1
            # smoothly — never clips to zero like Huber, always provides
            # directional information proportional to |tanh(r)|.
            # Approximates MSE for |r|<<1, L1 for |r|>>1.
            sq_err = torch.log(torch.cosh(residual.clamp(-20, 20)))
        elif huber_delta > 0:
            abs_r = residual.abs()
            sq_err = torch.where(
                abs_r <= huber_delta,
                0.5 * residual ** 2,
                huber_delta * (abs_r - 0.5 * huber_delta))
        else:
            sq_err = residual ** 2
        mask_f = mask_tv.float()
        effective = mask_f * weights_tv

        # SSL-012 M3: tail-importance weighting on the heavy-tail flow variates.
        # Per-cell weight scales linearly with the absolute z-score target —
        # cells where the true value is large (|z| ≈ 3-5σ active-tail bursts)
        # get up to (1 + tail_weight_alpha * |z|)× their base C10 weight,
        # making them harder to "ignore" via mean-collapse. Quiet cells where
        # |z| ≈ 0 are unaffected (multiplier ≈ 1).
        if heavy_tail_positions is not None and tail_weight_alpha > 0.0:
            ht_idx = torch.as_tensor(list(heavy_tail_positions),
                                     dtype=torch.long, device=effective.device)
            target_mag = x_target_tv[..., ht_idx].abs()  # (B, T, n_ht)
            boost = 1.0 + tail_weight_alpha * target_mag
            effective[..., ht_idx] = effective[..., ht_idx] * boost

        # Per-group loss weighting (OF-2): normalize so each group contributes
        # equally regardless of variate count. Without this, options_grid (88 var)
        # produces ~89% of gradient while order_flow (7 var) gets ~7%.
        # group_loss_weight: dict of variate_position → weight (pre-computed).
        if group_loss_weight is not None:
            # Apply per-variate group weight to the effective mask
            effective = effective * group_loss_weight.unsqueeze(0).unsqueeze(0)

        n_eff = effective.sum()
        if n_eff > 0:
            loss = (sq_err * effective).sum() / n_eff
        else:
            loss = torch.tensor(0.0, device=x_hat.device, requires_grad=True)

        # Per-variate reconstruction error (for diagnostics).
        # Bug A1 fix (2026-04-30): when Huber is active, sq_err contains
        # Huber loss values (not squared errors). The n_beating_zero and
        # shortcut_suspect diagnostics in train_ssl assume MSE semantics
        # (threshold=1.0 for predict-zero MSE on z-scored data). Compute
        # true MSE alongside for diagnostic use.
        with torch.no_grad():
            per_var_eff = effective.sum(dim=(0, 1))         # (V,)
            per_var_err = (sq_err * effective).sum(dim=(0, 1))  # (V,)
            safe_denom = per_var_eff.clamp(min=1.0)
            per_var_loss = per_var_err / safe_denom  # Huber-scale when active
            # True MSE for diagnostics (always squared error, regardless of loss)
            true_mse = (residual ** 2 * effective).sum(dim=(0, 1)) / safe_denom

        info = {
            "ssl_mse": loss.item(),
            "n_masked": mask_f.sum().item(),
            "n_effective": n_eff.item(),
            "per_var_mse": true_mse,       # ALWAYS true MSE for diagnostics
            "per_var_loss": per_var_loss,   # loss-scale (Huber or MSE)
        }
        return loss, info


class ContrastiveLoss(nn.Module):
    """Phase 2: SimMTM-style InfoNCE on mean-pooled encoder representations.

    Bridges the reconstruction→downstream gap by encouraging the global
    embedding (mean-pool over variate tokens) to cluster similar market states.
    Positive pairs: augmented views from the same sample (different masks).
    Loss: NT-Xent (normalized temperature-scaled cross-entropy).

    Reference: Dong et al. (NeurIPS 2024) — SimMTM §3.3 series-wise contrastive.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1, z2: (B, D) — mean-pooled embeddings from two masked views
        Returns:
            loss: scalar InfoNCE loss
        """
        B = z1.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)

        # L2-normalize embeddings
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Similarity matrix: (2B, 2B)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = z @ z.T / self.temperature  # (2B, 2B)

        # Mask out self-similarity on diagonal
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        return nn.functional.cross_entropy(sim, labels)


class TemporalStatsHead(nn.Module):
    """Auxiliary head: predict temporal statistics from variate tokens.
    Forces encoder to preserve temporal dynamics in learned representations.

    Predicts 4 summary statistics per variate from the D-dimensional token:
      [slope, std, acf_1, last_minus_mean]
    Supervised with MSE against ground-truth computed from raw (unmasked) input.
    Only visible (non-whole-variate-masked) variates contribute to the loss.
    """
    def __init__(self, d_model, n_stats=4):
        super().__init__()
        self.proj = nn.Linear(d_model, n_stats)

    def forward(self, variate_tokens):
        """variate_tokens: (B, V, D) -> (B, V, 4)"""
        return self.proj(variate_tokens)


def compute_temporal_targets(x_raw):
    """Compute ground-truth temporal statistics from raw (unmasked) input.

    Args:
        x_raw: (B, T, V) tensor — the UNMASKED input batch (z-scored)

    Returns:
        targets: (B, V, 4) tensor with [slope, std, acf_1, last_minus_mean]
        All operations are pure PyTorch (runs on device, no numpy).
    """
    B, T, V = x_raw.shape
    # Transpose to (B, V, T) for per-variate temporal operations
    x = x_raw.transpose(1, 2)  # (B, V, T)

    # 1. slope: closed-form linear regression beta = sum((t-t_bar)(x-x_bar)) / sum((t-t_bar)^2)
    t = torch.arange(T, dtype=x.dtype, device=x.device)  # (T,)
    t_bar = t.mean()
    t_centered = t - t_bar  # (T,)
    x_bar = x.mean(dim=-1, keepdim=True)  # (B, V, 1)
    x_centered = x - x_bar  # (B, V, T)
    slope = (x_centered * t_centered).sum(dim=-1) / (t_centered ** 2).sum()  # (B, V)
    slope = slope * T  # Scale to "total predicted change over window" — O(0.1-1.0) vs O(0.003)

    # 2. std: standard deviation over time axis
    std = x.std(dim=-1)  # (B, V)

    # 3. acf_1: lag-1 autocorrelation = corr(x[:-1], x[1:])
    x0 = x[:, :, :-1]  # (B, V, T-1)
    x1 = x[:, :, 1:]   # (B, V, T-1)
    x0_m = x0 - x0.mean(dim=-1, keepdim=True)
    x1_m = x1 - x1.mean(dim=-1, keepdim=True)
    cov_01 = (x0_m * x1_m).sum(dim=-1)  # (B, V)
    std_0 = x0_m.pow(2).sum(dim=-1).sqrt()  # (B, V)
    std_1 = x1_m.pow(2).sum(dim=-1).sqrt()  # (B, V)
    denom = std_0 * std_1 + 1e-8
    acf_1 = cov_01 / denom  # (B, V)

    # 4. last_minus_mean: x[:, :, -1] - mean(x over T)
    last_minus_mean = x[:, :, -1] - x_bar.squeeze(-1)  # (B, V)

    # Stack: (B, V, 4)
    targets = torch.stack([slope, std, acf_1, last_minus_mean], dim=-1)
    return targets


# ===================================================================
# 9. SSL TRAINING LOOP ()
# ===================================================================

def _compute_effective_mask_rates(hp, feature_indices):
    """Closed-form effective mask rate per variate (Reality Checker safety
    diagnostic, 2026-04-29).

    Computed analytically rather than empirically — a single matrix of
    per-variate probabilities printed at training start. Ceiling assertion
    catches the v106 99.5%-mask-rate failure mode that the SSL-012 v2 Option
    B design produced.

    Returns:
        dict[raw_v: int → float] of effective mask rate ∈ [0, 1].
    """
    variate_ratio = float(hp.get("variate_ratio", 0.30))
    cell_ratio = float(hp.get("cell_ratio", 0.15))
    rv_block_mask = bool(hp.get("rv_block_mask", False))
    seq_len = int(hp.get("seq_len", 60))

    # Stage 1 whole-variate + Stage 2 per-cell: P(cell masked) = vr + (1-vr)*cr
    base_rate = variate_ratio + (1.0 - variate_ratio) * cell_ratio

    rates = {raw_v: base_rate for raw_v in feature_indices}

    # Flow aggregates v141-v158 are mask-INELIGIBLE as primary targets but get
    # block-mask propagation from raw v131-v139.
    # C1 remediation (2026-05-02): threshold-based expansion. An aggregate cell
    # is block-masked only when >50% of its source window cells are masked.
    # Closed-form approximation: P(agg cell masked) = P(Binomial(w, base_rate) > w/2)
    # = sum_{k=ceil(w*0.5)}^{w} C(w,k) * p^k * (1-p)^(w-k).
    if any(v in (range(141, 159)) for v in feature_indices):
        from math import comb, ceil
        def _threshold_mask_prob(window, base_p, threshold=0.5):
            """P(masked cells in window > threshold * window)."""
            min_masked = ceil(window * threshold)
            prob = 0.0
            for k in range(min_masked, window + 1):
                prob += comb(window, k) * (base_p ** k) * ((1 - base_p) ** (window - k))
            return prob

        for raw_v in feature_indices:
            if 141 <= raw_v < 150:  # roll3
                rates[raw_v] = _threshold_mask_prob(3, base_rate)
            elif 150 <= raw_v < 159:  # roll15
                rates[raw_v] = _threshold_mask_prob(15, base_rate)

    # SSL-012 redesign 2026-04-29: Option B is now whole-variate-only.
    # Cell-level v106/v111-v113 coupling REMOVED — base_rate applies. Whole-
    # variate coupling adds: P(any RV whole-masked) propagates to v106 whole.
    # Group membership is derived from RV_RAW_TO_AGG_RAW so this stays correct
    # if the RV group changes (e.g., extending to v110 or dropping v113).
    rv_block_group = (106,) + tuple(RV_RAW_TO_AGG_RAW.keys())
    # Bug A2 fix (2026-04-30): only count ELIGIBLE variates in the joint
    # exponent. v106 is in SSL_MASK_INELIGIBLE → never selected by Stage 1
    # → should not inflate the joint whole-mask rate. Use the mask-ineligible
    # set that's active for this feature set.
    ineligible = ACTIVE_MASK_INELIGIBLE if USE_V3_FEATURES else SSL_MASK_INELIGIBLE
    rv_eligible = [v for v in rv_block_group if v not in ineligible]
    k = len(rv_eligible)  # only eligible members contribute to joint rate
    if rv_block_mask and any(v in rv_block_group for v in feature_indices):
        # whole-var-rate is variate_ratio per variate; coupling makes the
        # joint = 1 - (1-vr)^k over the block group.
        joint_whole_rate = 1.0 - (1.0 - variate_ratio) ** k
        for raw_v in rv_block_group:
            if raw_v in rates:
                # Refine: whole-mask-from-coupling adds extra to the whole-var
                # component; per-cell unchanged. Effective rate becomes
                # joint_whole_rate + (1 - joint_whole_rate) * cell_ratio
                rates[raw_v] = joint_whole_rate + (1.0 - joint_whole_rate) * cell_ratio

    return rates


def _print_and_assert_mask_rate_ceiling(hp, feature_indices,
                                         eligible_ceiling=0.65,
                                         catastrophe_ceiling=0.95):
    """Print per-variate effective mask rate table; assert per-class ceilings.

    Two ceilings differentiate mask-eligible (loss target) variates from
    context-only (mask-ineligible) variates:

    - **eligible_ceiling (default 0.65)**: applies to variates IN the loss
      target set. If exceeded, encoder cannot learn that variate; raises.
    - **catastrophe_ceiling (default 0.95)**: applies to ALL variates. Even
      mask-ineligible aggregates shouldn't be 100% masked, because the encoder
      uses them as context. Catches the SSL-012 v2 v106/v111-v113 99.5%
      destruction failure mode.

    Mask-ineligible variates (per ACTIVE_MASK_INELIGIBLE) include flow
    aggregates v141-v158 — these are intentionally not loss targets; high
    mask rate on them via _expand_flow_aggregate_mask is by design (see
    module-header concern #1). They get the looser catastrophe_ceiling only.
    """
    rates = _compute_effective_mask_rates(hp, feature_indices)
    if not rates:
        return
    # Group variates for readable table
    groups = {
        "options_grid (0-87)": [v for v in feature_indices if v < 88],
        "strike_agg (99-104)": [v for v in feature_indices if 99 <= v < 105],
        "spx_derived (105-118)": [v for v in feature_indices if 105 <= v < 119],
        "vix_term (119-130)": [v for v in feature_indices if 119 <= v < 131],
        "flow_raw (131-139)": [v for v in feature_indices if 131 <= v < 140],
        "flow_roll3 (141-149)": [v for v in feature_indices if 141 <= v < 150],
        "flow_roll15 (150-158)": [v for v in feature_indices if 150 <= v < 159],
    }
    print("\n  Effective mask rates (closed-form):")
    print(f"  {'group':<25s} {'min':>7s} {'mean':>7s} {'max':>7s}  {'note':<20s}")

    eligible_violation_var = None
    eligible_violation_rate = 0.0
    catastrophe_var = None
    catastrophe_rate = 0.0

    for group_name, vars_in_group in groups.items():
        if not vars_in_group:
            continue
        group_rates = [rates[v] for v in vars_in_group if v in rates]
        if not group_rates:
            continue
        gmin, gmean, gmax = min(group_rates), sum(group_rates)/len(group_rates), max(group_rates)
        # Determine if any variate in group is eligible (in loss target)
        ineligible_set = ACTIVE_MASK_INELIGIBLE if USE_V3_FEATURES else SSL_MASK_INELIGIBLE
        any_eligible = any(v not in ineligible_set for v in vars_in_group)
        note = ""
        if gmax > catastrophe_ceiling:
            note = "🔴 CATASTROPHE"
            for v in vars_in_group:
                if rates.get(v, 0.0) > catastrophe_rate:
                    catastrophe_var = v
                    catastrophe_rate = rates[v]
        elif any_eligible and gmax > eligible_ceiling:
            note = "⚠ELIGIBLE-VAR HIGH"
            for v in vars_in_group:
                if v not in ineligible_set and rates.get(v, 0.0) > eligible_violation_rate:
                    eligible_violation_var = v
                    eligible_violation_rate = rates[v]
        elif gmax > eligible_ceiling:
            note = "  context-only OK"
        print(f"  {group_name:<25s} {gmin:>7.3f} {gmean:>7.3f} {gmax:>7.3f}  {note:<20s}")

    # Catastrophe ceiling: only raise on MASK-ELIGIBLE variates. Mask-ineligible
    # variates (flow aggregates v141-v158, etc.) being highly masked is by-design
    # context-only behavior -- flagging in the table as "context-only OK" but not
    # raising. SSL-010-LOCAL canonical ran with flow_roll15 at 100% rate and
    # produced the dissertation's reference numbers, so high rate alone is not a
    # catastrophe -- it must combine with mask-eligibility.
    ineligible_set = ACTIVE_MASK_INELIGIBLE if USE_V3_FEATURES else SSL_MASK_INELIGIBLE
    if eligible_violation_rate > eligible_ceiling:
        raise ValueError(
            f"Mask-eligible variate v{eligible_violation_var} has effective rate "
            f"{eligible_violation_rate:.3f} > ceiling {eligible_ceiling}. This variate is "
            f"a LOSS TARGET -- encoder needs to see it on enough cells to learn it. "
            f"Reduce variate_ratio / cell_ratio, or redesign block-mask propagation. "
            f"(See D92 for the precedent SSL-012 v2 v106 failure.)"
        )
    # Catastrophe check only applies if the catastrophic variate is ALSO eligible.
    if (catastrophe_rate > catastrophe_ceiling
            and catastrophe_var is not None
            and catastrophe_var not in ineligible_set):
        raise ValueError(
            f"CATASTROPHIC effective mask rate on mask-eligible v{catastrophe_var}: "
            f"{catastrophe_rate:.3f} > {catastrophe_ceiling}. Encoder cannot learn this "
            f"variate. (See PRAXIS_DECISIONS D92.)"
        )

    # ---------------------------------------------------------------
    # Training-start diagnostic assertions (C1/C2/C3 remediation 2026-05-02).
    # These would have caught ALL the reconstruction opportunity defects
    # before months of wasted experiments.
    # ---------------------------------------------------------------

    # D1: Context visibility floor -- no group should have < 20% context
    # visibility. "Visibility" = 1 - mask_rate. Even ineligible groups
    # are used as encoder context and need sufficient visibility.
    min_visibility_threshold = 0.20
    for group_name, vars_in_group in groups.items():
        if not vars_in_group:
            continue
        group_rates_list = [rates[v] for v in vars_in_group if v in rates]
        if not group_rates_list:
            continue
        max_rate_in_group = max(group_rates_list)
        min_visibility = 1.0 - max_rate_in_group
        if min_visibility < min_visibility_threshold:
            raise ValueError(
                f"D1 HARD STOP: group '{group_name}' has min context visibility "
                f"{min_visibility:.1%} (max mask rate {max_rate_in_group:.3f}). "
                f"Any group below {min_visibility_threshold:.0%} visibility means "
                f"the encoder cannot use that group as context. Reduce "
                f"variate_ratio / cell_ratio or redesign block-mask propagation."
            )

    # D2: Gradient concentration -- no single group should have > 70% of
    # total eligible variates (eligible = in the loss target).
    eligible_by_group = {}
    total_eligible = 0
    for group_name, vars_in_group in groups.items():
        n_elig = sum(1 for v in vars_in_group if v not in ineligible_set)
        eligible_by_group[group_name] = n_elig
        total_eligible += n_elig
    if total_eligible > 0:
        max_concentration = 0.70
        for group_name, n_elig in eligible_by_group.items():
            frac = n_elig / total_eligible
            if frac > max_concentration:
                print(f"  WARNING: group {group_name} has {frac:.1%} of eligible "
                      f"variates ({n_elig}/{total_eligible}). Gradient concentration "
                      f"risk -- consider reducing variate_ratio or rebalancing.")

    # D3: Zero-gradient variate audit -- count variates with zero
    # reconstruction gradient (in ineligible set).
    zero_gradient_count = sum(1 for v in feature_indices if v in ineligible_set)
    total_variates = len(feature_indices)
    zero_frac = zero_gradient_count / total_variates if total_variates > 0 else 0
    print(f"\n  Zero-gradient variates: {zero_gradient_count}/{total_variates} "
          f"({zero_frac:.1%}) are mask-ineligible (no reconstruction gradient)")
    if zero_frac > 0.30:
        print(f"  WARNING: > 30% of variates have zero reconstruction gradient. "
              f"This is expected for context-only variates but check that no "
              f"informative variates are accidentally excluded.")


def _validate_config_vs_baseline(hp, baseline=None):
    """Pre-training safeguard: print config diff against RUN2_BASELINE.

    Flags every parameter that differs from the validated Run 2 configuration.
    If more than 1 parameter changed, prints a prominent warning — the
    experiment may be confounded (feedback_one_variable.md).
    """
    baseline = baseline or EXPERIMENT_BASELINE
    diffs = []
    for key in sorted(set(list(hp.keys()) + list(baseline.keys()))):
        v_new = hp.get(key)
        v_base = baseline.get(key)
        if v_new != v_base:
            diffs.append((key, v_base, v_new))

    print("\n" + "=" * 60)
    print("CONFIG DIFF vs EXPERIMENT_BASELINE")
    print("=" * 60)
    if not diffs:
        print("  No differences — replicating Run 2 exactly.")
    else:
        for key, v_base, v_new in diffs:
            print(f"  CHANGED: {key}: {v_base} → {v_new}")
        print(f"\n  Total changes: {len(diffs)}")
        if len(diffs) > 1:
            print("  ⚠ WARNING: More than 1 parameter changed!")
            print("  ⚠ This may confound the experiment (one variable at a time).")
            print("  ⚠ If intentional (e.g., Phase 2), proceed. Otherwise, fix config.")
    print("=" * 60 + "\n")
    return diffs


def train_ssl(loaders, start_epoch=1, history=None, best_val_loss=float('inf'),
              best_state=None, model=None, recon_head=None,
              opt_state=None, sched_state=None, temporal_head_state=None,
              spectral_ema_state=None,
              disable_checkpoints=False):
    """D71 SSL pretraining: masked variate reconstruction.

    Trains iTransformerEncoder + VariateReconstructionHead using hybrid masking
    (30% whole-variate + 15% per-cell, D78-c) on 129 SSL variates. Loss = weighted MSE
    on masked positions, with C10 quality weights.

    Best checkpoint selected by validation reconstruction loss.
    """
    p = SSL_HYPERPARAMS
    _validate_config_vs_baseline(p)

    # config sanity check — verify critical fixes are active
    _d98_vr = p.get("variate_ratio", 0.30)
    _d98_srw = p.get("spectral_rank_weight", 0.0)
    _d99_tsf = p.get("target_spectral_fraction", 0.0)
    assert _d98_vr <= 0.25, f"D98 violation: variate_ratio={_d98_vr} > 0.25 (old defective value was 0.30)"
    assert _d98_srw > 0, f"D98 violation: spectral_rank_weight={_d98_srw} (must be > 0 for anti-collapse)"
    _spectral_mode = f"ADAPTIVE target={_d99_tsf}" if _d99_tsf > 0 else f"FIXED weight={_d98_srw}"
    print(f"  D98 config verified: variate_ratio={_d98_vr}, per-group spectral=ON ({_spectral_mode})")

    # SSL-012 v2 audit follow-up (2026-04-29): print effective mask rates and
    # assert ceiling. Catches design-level information-destruction errors
    # (e.g., the original Option B that produced 99.5% effective mask rate
    # on v106 → encoder never saw the variate → log_rv probes regressed) at
    # training start, BEFORE compute is spent.
    # eligible_ceiling=0.85: under whole-variate-only Option B, 4-way coupling
    # of {v106, v111, v112, v113} makes their joint whole-mask rate
    # 1 - 0.7^4 = 0.76, plus cell_ratio adds the unmasked-window cells →
    # ~0.80 effective. This is by-design under the redesigned Option B.
    # Anything ABOVE 0.85 indicates either (a) variate_ratio > 0.30 (which
    # would amplify coupling) or (b) cell-wise propagation reintroduced
    # (regression to catastrophe).
    _print_and_assert_mask_rate_ceiling(p, ACTIVE_SSL_FEATURES,
                                          eligible_ceiling=0.85,
                                          catastrophe_ceiling=0.95)
    T = p["seq_len"]
    V = ACTIVE_N_SSL_FEATURES
    D = p["d_model"]
    # SSL-012 M2: distinguished mask fill value (0.0 = legacy z-score mean).
    mask_fill_value = float(p.get("mask_fill_value", 0.0))
    # SSL-012 M3: tail-importance weighting on heavy-tail flow variates.
    tail_weight_alpha = float(p.get("tail_weight_alpha", 0.0))
    # SSL-012 M1: encoder receives mask flag if enabled.
    use_mask_indicator = bool(p.get("use_mask_indicator", False))
    # R1: spectral covariance penalty weight (anti-collapse).
    spectral_rank_weight = float(p.get("spectral_rank_weight", 0.0))
    # : adaptive spectral scaling (loss-fraction targeting).
    # Instead of fixed weight, auto-scale so spectral contributes target_fraction
    # of total loss. Self-adaptive across tokenizer architectures (patch/linear/CNN)
    # without arm-specific hyperparameters. Solves SSL-030 catastrophic VIX collapse
    # where fixed 0.06 was too aggressive for patch's double-LayerNorm gradient path.
    target_spectral_fraction = float(p.get("target_spectral_fraction", 0.0))
    # : restore EMA state from checkpoint on resume, or initialize fresh
    _spectral_ema_weight = spectral_ema_state  # None on fresh start, float on resume

    if model is None:
        model = iTransformerEncoder(p, n_variates=V)
    if recon_head is None:
        recon_head = _build_recon_head_from_hp(p, D, T, V, ACTIVE_SSL_FEATURES)
    model = model.to(device)
    recon_head = recon_head.to(device)

    # SSL-012 M3: resolve heavy-tail variate POSITIONS (vs raw variate IDs)
    # against ACTIVE_SSL_FEATURES so MaskedVariateLoss can apply the per-cell
    # weight boost without cross-version surgery.
    heavy_tail_positions = None
    if tail_weight_alpha > 0.0:
        _pos_map_ht = {v: i for i, v in enumerate(ACTIVE_SSL_FEATURES)}
        heavy_tail_positions = tuple(_pos_map_ht[v] for v in HEAVY_TAIL_VARIATES if v in _pos_map_ht)
        if heavy_tail_positions:
            print(f"  Tail-weight (M3): ENABLED — alpha={tail_weight_alpha}, "
                  f"variates={list(HEAVY_TAIL_VARIATES)}, positions={list(heavy_tail_positions)}")

    if spectral_rank_weight > 0:
        if target_spectral_fraction > 0:
            print(f"  Spectral covariance penalty (D99 adaptive): ENABLED — "
                  f"base_weight={spectral_rank_weight}, target_fraction={target_spectral_fraction}")
        else:
            print(f"  Spectral covariance penalty (R1): ENABLED — weight={spectral_rank_weight}")

    recon_loss_fn = MaskedVariateLoss()
    _use_logcosh = bool(p.get("use_logcosh", False))  # resolve once, pass to forward()

    # OF-2: Per-group loss weighting — equalize gradient contribution across groups
    _group_loss_weight_tensor = None
    if p.get("equal_group_loss", False):
        # Build per-variate weight: each eligible group gets equal total weight
        ssl_to_raw = {i: v for i, v in enumerate(ACTIVE_SSL_FEATURES)}
        _glw_groups = {
            "options_grid": range(0, 88),
            "strike_agg": range(99, 105),
            "vix_term": range(119, 131),
            "order_flow": range(131, 140),
        }
        # Assign each variate to a group
        var_to_group = {}
        for i in range(ACTIVE_N_SSL_FEATURES):
            raw_v = ssl_to_raw[i]
            for gname, rng in _glw_groups.items():
                if raw_v in rng:
                    var_to_group[i] = gname
                    break
        # Count variates per group (mask-eligible only)
        eligible_np_glw = _build_eligible_mask(ACTIVE_SSL_FEATURES, ACTIVE_MASK_INELIGIBLE)
        group_counts = {}
        for i, g in var_to_group.items():
            if eligible_np_glw[i]:
                group_counts[g] = group_counts.get(g, 0) + 1
        n_groups = len(group_counts)
        # Weight = (1/n_groups) / (count_in_group / total_eligible)
        # This makes each group contribute 1/n_groups of the loss
        total_eligible = sum(group_counts.values())
        weights = torch.ones(ACTIVE_N_SSL_FEATURES, device=device)
        for i, g in var_to_group.items():
            if eligible_np_glw[i] and g in group_counts:
                weights[i] = (total_eligible / n_groups) / group_counts[g]
        _group_loss_weight_tensor = weights
        print(f"  Per-group loss weighting (OF-2): ENABLED — {n_groups} groups, "
              f"counts={group_counts}")
        for g, c in group_counts.items():
            print(f"    {g:15s}: {c} variates, weight={weights[list(var_to_group.keys())[list(var_to_group.values()).index(g)]].item():.2f}x")

    cl_weight = p.get("contrastive_weight", 0.0)
    cl_temp = p.get("contrastive_temp", 0.1)
    cl_loss_fn = ContrastiveLoss(temperature=cl_temp) if cl_weight > 0 else None
    fc_bars = p.get("forecast_mask_bars", 0)
    fc_prob = p.get("forecast_mask_prob", 0.0)
    # SSL-012 v2 audit followup (2026-04-29): temporal_weight=0.1 was carried
    # over from SSL-011 without per-bundle calibration. The TemporalStatsHead
    # supervises encoder representations toward [slope, std, acf_1, last_minus_mean]
    # per variate. The `std` component is correlated (NOT identical to —
    # in-window vs forward) with the log_rv probe target. Probe scores on
    # log_rv are partially conflated with this auxiliary supervision. Cross-
    # experiment comparison is internally consistent (SSL-010, SSL-011, SSL-012
    # all carry temporal_weight=0.1) but absolute log_rv R² should be reported
    # with the disclosure that aux-supervision contributes some fraction.
    # Single-factor ablation (task #204) at temporal_weight=0.0 will quantify.
    temporal_weight = p.get("temporal_weight", 0.0)
    temporal_head = None
    if temporal_weight > 0:
        temporal_head = TemporalStatsHead(D).to(device)
        if temporal_head_state is not None:
            temporal_head.load_state_dict(temporal_head_state)
            print("  Temporal head: restored from checkpoint")

    # Eligible mask for SSL sampling (once, on device).
    # ACTIVE_MASK_INELIGIBLE excludes flow aggregates v141-v158 in v3 — they
    # are context-only inputs, block-masked as derivatives via
    # flow_agg_positions below (never as primary MVR targets).
    eligible_np = _build_eligible_mask(ACTIVE_SSL_FEATURES, ACTIVE_MASK_INELIGIBLE)
    eligible = torch.from_numpy(eligible_np).to(device)
    n_eligible = int(eligible.sum())
    flow_agg_positions = _build_flow_agg_position_map(ACTIVE_SSL_FEATURES)
    # SSL-012 Option B: symmetric block-mask propagation between v106 and
    # v111-v113. Gated by hyperparameter `rv_block_mask` so SSL-010/SSL-011
    # legacy runs reproduce exactly when the flag is off.
    rv_positions = (_build_rv_position_map(ACTIVE_SSL_FEATURES)
                    if p.get("rv_block_mask", False) else None)
    if rv_positions is not None:
        # H4 remediation (2026-05-02): all 4 variates (v106, v111-v113) are in
        # SSL_MASK_INELIGIBLE, so the coupling never actually fires during
        # Stage 1 whole-variate selection. The cell-wise propagation path in
        # _expand_rv_mask may still affect combined mask on non-whole-masked
        # cells, but the primary intended coupling is vacuous.
        print("  RV block-mask (v106 <-> v111-v113): ENABLED (note: all v106/v111-v113 "
              "are mask-ineligible, whole-variate coupling is currently vacuous)")
    ver_label = "v3 (SSL-004-FE, flow aggregates block-masked)" if USE_V3_FEATURES else "v2 (SSL-004 baseline)"
    print(f"SSL features: {V} total, {n_eligible} eligible for masking, "
          f"{V - n_eligible} ineligible (context-only) — {ver_label}")
    print(f"D78 config: variate_ratio={p['variate_ratio']}, cell_ratio={p['cell_ratio']}, "
          f"log1p={len(SSL_LOG1P_VARIATES)} variates")
    if cl_weight > 0:
        print(f"Phase 2: contrastive_weight={cl_weight}, contrastive_temp={cl_temp}")
    if fc_bars > 0:
        print(f"Phase 2: forecast_mask_bars={fc_bars}, forecast_mask_prob={fc_prob}")
    if temporal_weight > 0:
        print(f"Phase 2: temporal_weight={temporal_weight}")

    # Optimizer: only params on the encode_variates() path + recon_head.
    # Excludes pool_q, pool_attn, regime_head, emb_head (classifier-only).
    ssl_prefixes = (
        'tok.', 'variate_embed', 'embed_drop.', 'blocks.', 'final_norm.',
        # SSL-012: temporal-axis attention preprocessor (only present when use_temporal_attn=True)
        'cell_proj.', 'temporal_pos_embed', 'temporal_block.', 'temporal_readout.',
        # SSL-012 M1: mask-indicator projection (only present when use_mask_indicator=True)
        'mask_proj.',
    )
    ssl_named = [
        (n, p_) for n, p_ in model.named_parameters()
        if p_.requires_grad and any(n.startswith(pf) for pf in ssl_prefixes)
    ]
    ssl_named += [(f"recon.{n}", p_) for n, p_ in recon_head.named_parameters()
                  if p_.requires_grad]
    if temporal_head is not None:
        ssl_named += [(f"temporal_head.{n}", p_) for n, p_ in temporal_head.named_parameters()
                      if p_.requires_grad]

    decay_params, no_decay_params = [], []
    for name, param in ssl_named:
        # SSL-012: also exclude FiLM scale/shift (perturbation around identity,
        # decay pulls scale away from 1.0 → contaminates the FiLM-drift
        # diagnostic). temporal_pos_embed already matches via 'embed' substring.
        if ('norm' in name or 'bias' in name or 'embed' in name
                or 'film_scale' in name or 'film_shift' in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    opt = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 1e-2},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=p["learning_rate"])

    all_params = decay_params + no_decay_params
    n_trainable = sum(pp.numel() for pp in all_params)
    print(f"Trainable (SSL path): {n_trainable:,} params")

    # LR schedule: linear warmup → cosine decay
    warmup_epochs = p.get("warmup_epochs", 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=p["num_epochs"] - warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    if opt_state is not None:
        opt.load_state_dict(opt_state)
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)
    else:
        for _ in range(start_epoch - 1):
            scheduler.step()

    if history is None:
        history = {"train_loss": [], "val_loss": [], "n_effective": [],
                   "train_cl_loss": [], "val_cl_loss": [],
                   "train_temporal_loss": [], "val_temporal_loss": []}

    best_temporal_loss = float('inf')
    best_temporal_state = None
    emb_var = None           # set inside epoch diagnostic block if reached
    n_beating_zero = None    # set inside epoch diagnostic block if reached

    for epoch in range(start_epoch, p["num_epochs"] + 1):
        # --- Train ---
        model.train()
        recon_head.train()
        if temporal_head is not None:
            temporal_head.train()
        epoch_loss = 0.0
        epoch_cl_loss = 0.0
        epoch_temporal_loss = 0.0
        epoch_n_eff = 0.0
        epoch_samples = 0
        epoch_per_var = torch.zeros(V, device=device)
        epoch_var_count = 0

        for Xb, Wb in loaders["train"]:
            Xb = torch.nan_to_num(Xb.to(device), nan=0.0)
            Wb = Wb.to(device)
            B = Xb.shape[0]

            # Generate mask view 1 (with temporal forecast masking).
            # v3: flow_agg_positions triggers block-expansion on v141-v158
            # derivatives when raw v131-v139 is masked (L1-SSL-009 §4.3).
            mask, variate_mask = sample_ssl_mask(B, T, V, eligible,
                                      p["variate_ratio"], p["cell_ratio"],
                                      forecast_bars=fc_bars, forecast_prob=fc_prob,
                                      flow_agg_positions=flow_agg_positions,
                                      feature_indices=ACTIVE_SSL_FEATURES,
                                      rv_positions=rv_positions)

            # Replace masked cells with mask_fill_value (M2: -8.0 in SSL-012,
            # 0.0 = legacy z-score mean). Combined with M1 mask-indicator
            # signal passed through encode_variates(mask=...), the encoder can
            # distinguish "this cell is masked, predict me" from "this cell is
            # genuinely near zero".
            X_masked = Xb.clone()
            X_masked[mask] = mask_fill_value

            # Forward: encode → reconstruct → reconstruction loss
            tokens = model.encode_variates(
                X_masked, mask=mask if use_mask_indicator else None,
            )                                            # (B, V, D)
            x_hat = recon_head(tokens)                   # (B, V, T)
            recon_loss, info = recon_loss_fn(
                x_hat, Xb, mask, Wb,
                heavy_tail_positions=heavy_tail_positions,
                tail_weight_alpha=tail_weight_alpha,
                huber_delta=float(p.get("huber_delta", 0.0)),
                group_loss_weight=_group_loss_weight_tensor,
                use_logcosh=_use_logcosh,
            )

            loss = recon_loss

            # R1: Spectral covariance penalty (anti-collapse)
            # Penalizes -log(det(Cov(tokens))) which is minimized when all
            # eigenvalues are equal (full-rank). Gradient ∝ 1/eigenvalue so
            # strongest on collapsing dimensions. Ermolov et al. 2021.
            #
            # H2 fix: per-group spectral penalty. The global penalty was
            # dominated by options_grid (88/147 tokens). Small groups (flow=9,
            # strike=6) could collapse without triggering the penalty. Now
            # each group gets its own covariance penalty, ensuring anti-collapse
            # protection for ALL groups equally.
            #
            # fix: adaptive loss-fraction targeting. Fixed weight=0.06 was
            # too aggressive for patch tokenizer (double-LayerNorm gradient path
            # attenuates reconstruction gradients → spectral dominates →
            # VIX collapse in SSL-030). Adaptive scaling auto-calibrates the
            # effective weight so spectral contributes exactly target_fraction
            # of total loss, regardless of tokenizer architecture.
            if spectral_rank_weight > 0:
                D = tokens.shape[-1]
                eye_D = torch.eye(D, device=tokens.device)
                rank_loss = torch.tensor(0.0, device=tokens.device)
                n_groups = 0
                for _grp_name, _grp_range in [
                    ("options_grid", range(0, 88)),
                    ("strike_agg", range(99, 105)),
                    ("spx_derived", range(105, 119)),
                    ("vix_term", range(119, 131)),
                    ("flow_raw", range(131, 140)),
                    ("flow_agg", range(141, 159)),
                ]:
                    # Map raw variate ranges to SSL positions
                    _grp_positions = [
                        i for i, v in enumerate(ACTIVE_SSL_FEATURES)
                        if v in _grp_range
                    ]
                    if len(_grp_positions) < 3:  # need ≥3 tokens for meaningful cov
                        continue
                    z_grp = tokens[:, _grp_positions, :]  # (B, |grp|, D)
                    z_flat = z_grp.reshape(-1, D)
                    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
                    cov = (z_centered.T @ z_centered) / max(z_flat.shape[0] - 1, 1)
                    cov_reg = cov + 1e-4 * eye_D
                    L_chol = torch.linalg.cholesky(cov_reg)
                    log_det = 2.0 * L_chol.diagonal().log().sum()
                    rank_loss = rank_loss + (-log_det / D)
                    n_groups += 1
                if n_groups > 0:
                    rank_loss = rank_loss / n_groups  # average across groups

                    if target_spectral_fraction > 0:
                        # : Adaptive loss-fraction targeting.
                        # Compute effective weight so spectral ≈ target_fraction of total loss.
                        # stop-gradient on the scaling factor to prevent circular gradients.
                        with torch.no_grad():
                            _recon_detached = recon_loss.detach()
                            _rank_detached = rank_loss.detach()
                            # Target: w_eff * rank_loss = target_fraction * (recon + w_eff * rank_loss)
                            # Solving: w_eff = target_fraction * recon / ((1 - target_fraction) * rank_loss)
                            _desired_weight = (
                                target_spectral_fraction * _recon_detached
                                / ((1.0 - target_spectral_fraction) * _rank_detached + 1e-8)
                            )
                            # Safety clamp: prevent extreme weights
                            _desired_weight = _desired_weight.clamp(0.005, 0.15)

                            # EMA smoothing (momentum=0.95) to prevent oscillation
                            if _spectral_ema_weight is None:
                                _spectral_ema_weight = float(_desired_weight)
                            else:
                                _spectral_ema_weight = (
                                    0.95 * _spectral_ema_weight
                                    + 0.05 * float(_desired_weight)
                                )
                        _effective_weight = _spectral_ema_weight
                        loss = loss + _effective_weight * rank_loss
                    else:
                        # Legacy fixed-weight mode
                        loss = loss + spectral_rank_weight * rank_loss
            cl_val = 0.0

            # Phase 2: contrastive auxiliary loss on second masked view
            if cl_loss_fn is not None and B >= 4:
                mask2, _ = sample_ssl_mask(B, T, V, eligible,
                                           p["variate_ratio"], p["cell_ratio"],
                                           forecast_bars=fc_bars, forecast_prob=fc_prob,
                                           flow_agg_positions=flow_agg_positions,
                                           feature_indices=ACTIVE_SSL_FEATURES,
                                           rv_positions=rv_positions)
                X_masked2 = Xb.clone()
                X_masked2[mask2] = mask_fill_value
                tokens2 = model.encode_variates(
                    X_masked2, mask=mask2 if use_mask_indicator else None,
                )                                            # (B, V, D)
                # Mean-pool over variate dimension for global embedding
                z1 = tokens.mean(dim=1)   # (B, D)
                z2 = tokens2.mean(dim=1)  # (B, D)
                cl_loss = cl_loss_fn(z1, z2)
                loss = loss + cl_weight * cl_loss
                cl_val = cl_loss.item()

            # Temporal statistics auxiliary loss on visible variates
            temporal_val = 0.0
            if temporal_head is not None:
                # Compute targets from UNMASKED input (real temporal profile)
                with torch.no_grad():
                    t_targets = compute_temporal_targets(Xb)  # (B, V, 4)
                t_preds = temporal_head(tokens)  # (B, V, 4)
                # Visible variates: NOT whole-variate masked (constant across T, take slice at t=0)
                visible = ~variate_mask[:, 0, :]  # (B, V) — True = visible
                if visible.any():
                    # MSE only on visible variates
                    diff_sq = (t_preds - t_targets).pow(2)  # (B, V, 4)
                    # Mean over stats dim, then masked mean over (B, V)
                    per_variate_mse = diff_sq.mean(dim=-1)  # (B, V)
                    temporal_loss = (per_variate_mse * visible.float()).sum() / visible.float().sum()
                    loss = loss + temporal_weight * temporal_loss
                    temporal_val = temporal_loss.item()

            if not torch.isfinite(loss):
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            epoch_loss += recon_loss.item() * B
            epoch_cl_loss += cl_val * B
            epoch_temporal_loss += temporal_val * B
            epoch_n_eff += info["n_effective"]
            epoch_per_var += info["per_var_mse"] * B  # sample-weighted (Bug B2 fix)
            epoch_var_count += B
            epoch_samples += B

        scheduler.step()
        train_loss = epoch_loss / max(epoch_samples, 1)
        train_cl_loss = epoch_cl_loss / max(epoch_samples, 1)
        train_temporal_loss = epoch_temporal_loss / max(epoch_samples, 1)
        epoch_per_var /= max(epoch_var_count, 1)

        # --- Validate (deterministic mask for stable checkpoint comparison) ---
        model.eval()
        recon_head.eval()
        if temporal_head is not None:
            temporal_head.eval()
        val_loss_sum = 0.0
        val_loss_unweighted_sum = 0.0  # SSL-012: cross-experiment comparable metric
        val_cl_sum = 0.0
        val_temporal_sum = 0.0
        val_n_eff = 0.0
        val_samples = 0
        val_per_var = torch.zeros(V, device=device)
        val_var_count = 0
        val_gen = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            for Xb, Wb in loaders["val"]:
                Xb = torch.nan_to_num(Xb.to(device), nan=0.0)
                Wb = Wb.to(device)
                B = Xb.shape[0]

                # Val uses deterministic mask (no forecast masking — stable comparison)
                mask, variate_mask = sample_ssl_mask(B, T, V, eligible,
                                          p["variate_ratio"], p["cell_ratio"],
                                          generator=val_gen,
                                          flow_agg_positions=flow_agg_positions,
                                          feature_indices=ACTIVE_SSL_FEATURES,
                                          rv_positions=rv_positions)
                X_masked = Xb.clone()
                X_masked[mask] = mask_fill_value

                tokens = model.encode_variates(
                    X_masked, mask=mask if use_mask_indicator else None,
                )
                x_hat = recon_head(tokens)
                loss, info = recon_loss_fn(
                    x_hat, Xb, mask, Wb,
                    heavy_tail_positions=heavy_tail_positions,
                    tail_weight_alpha=tail_weight_alpha,
                    huber_delta=float(p.get("huber_delta", 0.0)),
                    use_logcosh=_use_logcosh,
                )
                # SSL-012: ALSO compute the unweighted (legacy MSE-only) loss
                # for cross-experiment comparability with SSL-010-LOCAL.
                # Without this, the M3 boost makes val_loss numerically
                # incomparable to the canonical baseline. Diagnostic only —
                # does NOT influence best-checkpoint selection (we still
                # select on the weighted `loss` since that's what the model
                # is being optimized against).
                # Always compute pure-MSE diagnostic for cross-experiment
                # comparability (agent-flagged: val_uw was reporting Huber
                # when huber_delta>0, making it identical to val_loss).
                _needs_mse_diag = (
                    tail_weight_alpha > 0.0
                    or heavy_tail_positions is not None
                    or float(p.get("huber_delta", 0.0)) > 0.0
                )
                if _needs_mse_diag:
                    loss_unweighted, _info_uw = recon_loss_fn(
                        x_hat, Xb, mask, Wb,
                        heavy_tail_positions=None, tail_weight_alpha=0.0,
                        huber_delta=0.0, use_logcosh=False,  # force pure MSE
                    )
                    info["ssl_mse_unweighted"] = loss_unweighted.item()
                else:
                    info["ssl_mse_unweighted"] = info["ssl_mse"]

                # Val contrastive loss (deterministic second view)
                cl_val = 0.0
                if cl_loss_fn is not None and B >= 4:
                    mask2, _ = sample_ssl_mask(B, T, V, eligible,
                                               p["variate_ratio"], p["cell_ratio"],
                                               generator=val_gen,
                                               flow_agg_positions=flow_agg_positions,
                                               feature_indices=ACTIVE_SSL_FEATURES,
                                               rv_positions=rv_positions)
                    X_masked2 = Xb.clone()
                    X_masked2[mask2] = mask_fill_value
                    tokens2 = model.encode_variates(
                        X_masked2, mask=mask2 if use_mask_indicator else None,
                    )
                    z1 = tokens.mean(dim=1)
                    z2 = tokens2.mean(dim=1)
                    cl_val = cl_loss_fn(z1, z2).item()

                # Val temporal statistics loss
                temporal_val = 0.0
                if temporal_head is not None:
                    t_targets = compute_temporal_targets(Xb)
                    t_preds = temporal_head(tokens)
                    visible = ~variate_mask[:, 0, :]
                    if visible.any():
                        diff_sq = (t_preds - t_targets).pow(2)
                        per_variate_mse = diff_sq.mean(dim=-1)
                        temporal_val = (per_variate_mse * visible.float()).sum().item() / visible.float().sum().item()

                val_loss_sum += loss.item() * B
                val_loss_unweighted_sum += info["ssl_mse_unweighted"] * B
                val_cl_sum += cl_val * B
                val_temporal_sum += temporal_val * B
                val_n_eff += info["n_effective"]
                val_per_var += info["per_var_mse"] * B  # sample-weighted (Bug B2 fix)
                val_var_count += B
                val_samples += B

        val_loss = val_loss_sum / max(val_samples, 1)
        val_loss_unweighted = val_loss_unweighted_sum / max(val_samples, 1)
        val_cl_loss = val_cl_sum / max(val_samples, 1)
        val_temporal_loss = val_temporal_sum / max(val_samples, 1)
        val_per_var /= max(val_var_count, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["n_effective"].append(float(epoch_n_eff))
        history["train_cl_loss"].append(train_cl_loss)
        history["val_cl_loss"].append(val_cl_loss)
        history["train_temporal_loss"].append(train_temporal_loss)
        history["val_temporal_loss"].append(val_temporal_loss)

        # MLflow in-loop logging (no-op when no active run / no mlflow installed).
        # Safe to call unconditionally — stub handles QC, real hooks handle local.
        _mlf_hooks.log_epoch(
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            best_val_loss=float(best_val_loss),
            lr=float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else None,
            n_effective=float(epoch_n_eff),
            extras={
                "val_loss_unweighted": float(val_loss_unweighted),  # SSL-012: SSL-010-comparable metric
                "train_cl_loss": train_cl_loss if cl_weight > 0 else None,
                "val_cl_loss":   val_cl_loss   if cl_weight > 0 else None,
                "train_temporal_loss": train_temporal_loss if temporal_weight > 0 else None,
                "val_temporal_loss":   val_temporal_loss   if temporal_weight > 0 else None,
            },
        )

        # Best checkpoint by val loss
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "recon_head": {k: v.cpu().clone() for k, v in recon_head.state_dict().items()},
            }

        # Secondary checkpoint: best temporal loss (separate from reconstruction best)
        temporal_improved = (temporal_weight > 0 and val_temporal_loss < best_temporal_loss)
        if temporal_improved:
            best_temporal_loss = val_temporal_loss
            best_temporal_state = {
                "encoder": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "recon_head": {k: v.cpu().clone() for k, v in recon_head.state_dict().items()},
            }

        if epoch % 5 == 0 or epoch == start_epoch:
            # Top-5 worst variates by val MSE
            top5_idx = val_per_var.argsort(descending=True)[:5]
            top5_str = ", ".join(
                f"v{ACTIVE_SSL_FEATURES[i]}={val_per_var[i]:.4f}" for i in top5_idx)
            # Predict-zero baseline: z-scored data has var≈1, so MSE<1 = better than mean
            n_beating_zero = int((val_per_var < 1.0).sum())
            cl_str = f"  cl={train_cl_loss:.4f}/{val_cl_loss:.4f}" if cl_weight > 0 else ""
            ts_str = f"  ts={train_temporal_loss:.4f}/{val_temporal_loss:.4f}" if temporal_weight > 0 else ""
            uw_str = (f"  val_uw={val_loss_unweighted:.4f}"
                      if (tail_weight_alpha > 0.0 or use_mask_indicator) else "")
            print(f"Ep {epoch:3d}/{p['num_epochs']}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}{uw_str}  "
                  f"best={best_val_loss:.4f}  "
                  f"eff={epoch_n_eff:.0f}  "
                  f"beat0={n_beating_zero}/{V}{cl_str}{ts_str}  "
                  f"worst=[{top5_str}]")

            # Embedding collapse detection: check variance of encoder outputs
            if epoch % 20 == 0:
                with torch.no_grad():
                    # Sample first val batch for diagnostic
                    Xb_diag, _ = next(iter(loaders["val"]))
                    Xb_diag = torch.nan_to_num(Xb_diag.to(device), nan=0.0)
                    z_diag = model.encode_variates(Xb_diag)  # (B, V, D)
                    emb_var = z_diag.var(dim=0).mean().item()
                    # Shortcut detection: relative threshold (2026-04-30).
                    # Old absolute threshold (MSE < 0.01) was miscalibrated —
                    # with Huber + 250 epochs, many variates legitimately reach
                    # MSE < 0.01 (RMS error 0.1σ on easy variates). Relative
                    # threshold: per-variate MSE < 5% of that variate's predict-
                    # zero MSE (≈ its z-scored variance ≈ 1.0 for most variates).
                    # This distinguishes "genuinely well-reconstructed" from
                    # "encoder outputs near-constant for this variate."
                    predict_zero_mse = torch.ones_like(val_per_var)  # z-scored → var≈1
                    n_shortcut = int((val_per_var < 0.05 * predict_zero_mse).sum())
                    n_wellrecon = int(((val_per_var >= 0.05 * predict_zero_mse) &
                                       (val_per_var < 1.0)).sum())  # between shortcut and predict-zero
                    # Temporal attention entropy (patch/CNN diagnostic)
                    temporal_ent_str = ""
                    if hasattr(model, 'use_temporal_attn') and model.use_temporal_attn:
                        with torch.no_grad():
                            x_vt_d = Xb_diag.transpose(1, 2)
                            if model._temporal_attn_mode == "pre_tokenizer":
                                # Linear path: temporal attention on raw T=60 scalar bars
                                B_d = Xb_diag.shape[0]
                                z_d = model.cell_proj(x_vt_d.unsqueeze(-1))
                                z_d = z_d + model.temporal_pos_embed.unsqueeze(0).unsqueeze(0)
                                z_d = z_d.reshape(-1, T, z_d.shape[-1])
                                _, attn_w = model.temporal_block(z_d, return_attn_weights=True)
                                T_ent = T
                            else:
                                # Post-tokenizer path: temporal attention on
                                # patch embeddings (12) or CNN positions (60)
                                z_seq, (B_d, V_d) = model.tok.forward_embed(x_vt_d)
                                T_ent = z_seq.shape[1]  # n_patches or T
                                z_seq = z_seq + model.temporal_pos_embed.unsqueeze(0)
                                _, attn_w = model.temporal_block(z_seq, return_attn_weights=True)
                            # attn_w: (N, T_ent, T_ent) — compute entropy of attention distribution
                            ent = -(attn_w * (attn_w + 1e-8).log()).sum(dim=-1).mean()
                            max_ent = float(np.log(T_ent))
                            temporal_ent_str = f"  temporal_attn_ent={ent.item():.3f}/{max_ent:.3f} (T={T_ent})"
                    _spectral_diag_str = ""
                    if target_spectral_fraction > 0 and _spectral_ema_weight is not None:
                        _spectral_diag_str = f"  spectral_w={_spectral_ema_weight:.4f}"
                    print(f"  [diag] emb_var={emb_var:.4f}  "
                          f"shortcut_suspect={n_shortcut}  "
                          f"well_reconstructed={n_wellrecon}"
                          f"{temporal_ent_str}{_spectral_diag_str}"
                          f"{'  WARNING: embedding collapse!' if emb_var < 0.01 else ''}")

                    # Persist diagnostic snapshot for post-training metrics pipeline
                    history.setdefault("diagnostic_snapshots", []).append({
                        "epoch": epoch,
                        "emb_var": emb_var,
                        "shortcut_suspect": n_shortcut,
                        "well_reconstructed": n_wellrecon,
                        "spectral_w": _spectral_ema_weight if target_spectral_fraction > 0 else None,
                    })

        # Checkpoint to ObjectStore (history every epoch, model only on improvement).
        # disable_checkpoints=True suppresses ALL ObjectStore writes — used by
        # variate_ratio_sweep so short-run sweep training doesn't clobber the
        # real ACTIVE_CKPT_KEY checkpoint from a preceding full run.
        if (IN_QC or LOCAL_MODE) and not disable_checkpoints:
            try:
                qb_ref = get_qb()
                qb_ref.ObjectStore.Save(ACTIVE_HISTORY_KEY, json.dumps(history))
                z_s = loaders.get("z_stats", {})

                # Best-model checkpoint (on improvement + every 5 epochs — probe-ready)
                # No optimizer/scheduler here — that's in ssl_resume.
                # Keeping ssl_model small (~2.5MB) for reliable saves.
                # Save frequency reduced to avoid ObjectStore API overhead.
                if improved and best_state is not None and (epoch % 5 == 0 or epoch <= 5):
                    buf = io.BytesIO()
                    torch.save({
                        "encoder_state_dict": best_state["encoder"],
                        "recon_head_state_dict": best_state["recon_head"],
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "ssl_hyperparams": p,
                        "ssl_features": ACTIVE_SSL_FEATURES,
                        "n_ssl_features": ACTIVE_N_SSL_FEATURES,
                        "use_v3_features": USE_V3_FEATURES,
                        "z_mean": z_s.get("mean"),
                        "z_std": z_s.get("std"),
                        "log1p_cols": z_s.get("log1p_cols", []),
                        "log1p_raw_variates": sorted(SSL_LOG1P_VARIATES),
                        "min_std_floor": z_s.get("min_std_floor", 0.01),
                    }, buf)
                    model_bytes = buf.getvalue()
                    qb_ref.ObjectStore.Save(
                        ACTIVE_CKPT_KEY,
                        base64.b64encode(model_bytes).decode())
                    print(f"  [best model saved as {ACTIVE_CKPT_KEY}, size={len(model_bytes)/1024:.0f}KB]")

                # Secondary checkpoint: best temporal loss
                if temporal_improved and best_temporal_state is not None:
                    buf_t = io.BytesIO()
                    torch.save({
                        "encoder_state_dict": best_temporal_state["encoder"],
                        "recon_head_state_dict": best_temporal_state["recon_head"],
                        "epoch": epoch,
                        "val_temporal_loss": best_temporal_loss,
                        "val_loss": val_loss,
                        "ssl_hyperparams": p,
                        "ssl_features": ACTIVE_SSL_FEATURES,
                        "n_ssl_features": ACTIVE_N_SSL_FEATURES,
                        "use_v3_features": USE_V3_FEATURES,
                        "z_mean": z_s.get("mean"),
                        "z_std": z_s.get("std"),
                        "log1p_cols": z_s.get("log1p_cols", []),
                        "log1p_raw_variates": sorted(SSL_LOG1P_VARIATES),
                        "min_std_floor": z_s.get("min_std_floor", 0.01),
                    }, buf_t)
                    temporal_bytes = buf_t.getvalue()
                    qb_ref.ObjectStore.Save(
                        f"ssl_model_temporal_best{ACTIVE_CKPT_SUFFIX}",
                        base64.b64encode(temporal_bytes).decode())
                    print(f"  [temporal-best model saved, size={len(temporal_bytes)/1024:.0f}KB, "
                          f"val_temporal={best_temporal_loss:.4f}]")

                # Resume checkpoint every 25 epochs — saves CURRENT state
                # (not best) so training can continue after QC timeout.
                # Best state is NOT duplicated here — it's in ssl_model.
                # This keeps the checkpoint under QC's 10MB ObjectStore limit.
                if epoch % 10 == 0:
                    buf_r = io.BytesIO()
                    resume_dict = {
                        "encoder_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "recon_head_state_dict": {k: v.cpu().clone() for k, v in recon_head.state_dict().items()},
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "best_in_ssl_model": improved or qb_ref.ObjectStore.ContainsKey(ACTIVE_CKPT_KEY),
                        "history": history,
                        "ssl_hyperparams": p,
                        "ssl_features": ACTIVE_SSL_FEATURES,
                        "n_ssl_features": ACTIVE_N_SSL_FEATURES,
                        "use_v3_features": USE_V3_FEATURES,
                        "z_mean": z_s.get("mean"),
                        "z_std": z_s.get("std"),
                        "log1p_cols": z_s.get("log1p_cols", []),
                        "log1p_raw_variates": sorted(SSL_LOG1P_VARIATES),
                        "min_std_floor": z_s.get("min_std_floor", 0.01),
                    }
                    if temporal_head is not None:
                        resume_dict["temporal_head_state_dict"] = {
                            k: v.cpu().clone() for k, v in temporal_head.state_dict().items()}
                    # : serialize adaptive spectral EMA weight for seamless resume
                    if _spectral_ema_weight is not None:
                        resume_dict["spectral_ema_weight"] = _spectral_ema_weight
                    torch.save(resume_dict, buf_r)
                    ckpt_bytes = buf_r.getvalue()
                    qb_ref.ObjectStore.Save(
                        f"ssl_resume{ACTIVE_CKPT_SUFFIX}",
                        base64.b64encode(ckpt_bytes).decode())
                    print(f"  [resume checkpoint saved at epoch {epoch}, "
                          f"size={len(ckpt_bytes)/1024:.0f}KB]")
            except Exception as e:
                print(f"  WARNING: checkpoint save failed: {e}")

    # Store final diagnostic values for post-training metrics pipeline
    history["final_emb_var"] = emb_var
    history["final_spectral_w"] = (
        _spectral_ema_weight
        if target_spectral_fraction > 0 and _spectral_ema_weight is not None
        else None
    )
    history["final_beat_zero"] = n_beating_zero
    history["best_val_loss"] = best_val_loss

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state["encoder"])
        recon_head.load_state_dict(best_state["recon_head"])

    # Archive best model under tokenizer-specific key (preserves across runs)
    # Suffix with version tag so v2/v3 archives don't clobber each other.
    tok_type = p.get("tokenizer", "linear")
    archive_key = f"ssl_model_{tok_type}{ACTIVE_CKPT_SUFFIX}"
    if (IN_QC or LOCAL_MODE) and not disable_checkpoints:
        try:
            qb_ref = get_qb()
            if qb_ref.ObjectStore.ContainsKey(ACTIVE_CKPT_KEY):
                raw_archive = qb_ref.ObjectStore.Read(ACTIVE_CKPT_KEY)
                qb_ref.ObjectStore.Save(archive_key, raw_archive)
                print(f"  [archived best model as '{archive_key}']")
        except Exception as e:
            print(f"  WARNING: archive save failed: {e}")

    return model, recon_head, history


# ===================================================================
# 9C. SSL RECONSTRUCTION BASELINES
# ===================================================================
# Peer-accepted baselines for contextualizing SSL reconstruction quality.
# References: Zerveas et al. (KDD 2021), Nie et al. (ICLR 2023),
# Dong et al. (NeurIPS 2024), Corsi (JFE 2009).
# Reports metrics separately for whole-variate vs per-cell masking.

def _persistence_predictions(X_target, mask, var_mask):
    """Persistence baseline: forward-fill last unmasked value per variate.

    Per-cell positions: copy nearest preceding unmasked timestep.
    Whole-variate positions: predict 0 (no within-variate signal).
    """
    B, T, V = X_target.shape
    pred = torch.zeros_like(X_target)
    cell_mask = mask & ~var_mask

    # Forward-fill then backward-fill for per-cell positions
    x = X_target.clone()
    x[cell_mask] = float('nan')
    # Also NaN whole-variate positions so they don't contaminate fill
    x[var_mask] = float('nan')

    for t in range(1, T):
        nan_m = torch.isnan(x[:, t, :])
        x[:, t, :] = torch.where(nan_m, x[:, t-1, :], x[:, t, :])
    for t in range(T-2, -1, -1):
        nan_m = torch.isnan(x[:, t, :])
        x[:, t, :] = torch.where(nan_m, x[:, t+1, :], x[:, t, :])
    x = torch.nan_to_num(x, nan=0.0)

    pred[cell_mask] = x[cell_mask]
    # whole-variate stays 0
    return pred


def _fit_ar1(train_loader, n_features):
    """Fit per-variate AR(1) coefficients: x_t = phi * x_{t-1}.

    On z-scored data, phi ≈ lag-1 autocorrelation.
    """
    sum_xy = np.zeros(n_features, dtype=np.float64)
    sum_xx = np.zeros(n_features, dtype=np.float64)

    for X_batch, _ in train_loader:
        X = X_batch.numpy()
        B, T, V = X.shape
        for t in range(1, T):
            x_prev = X[:, t-1, :]
            x_curr = X[:, t, :]
            sum_xy += (x_prev * x_curr).sum(axis=0)
            sum_xx += (x_prev ** 2).sum(axis=0)

    phi = sum_xy / np.maximum(sum_xx, 1e-8)
    return np.clip(phi, -1.0, 1.0).astype(np.float32)


def _ar1_predictions(X_target, mask, var_mask, phi):
    """AR(1) baseline: iterative phi propagation for per-cell; 0 for whole-variate.

    For per-cell masked positions, propagates phi^k decay correctly:
    if nearest unmasked value is k steps back, prediction = phi^k * x_{t-k}.
    Implemented as forward scan: at each masked t, pred = phi * prev (where
    prev is either the true value or the previous AR prediction).
    """
    B, T, V = X_target.shape
    pred = torch.zeros_like(X_target)
    cell_mask = mask & ~var_mask
    phi_t = torch.from_numpy(phi).to(X_target.device)  # (V,)

    # Forward scan: maintain running prediction per (sample, variate)
    prev = X_target[:, 0, :].clone()  # (B, V) — start with actual t=0
    for t in range(T):
        is_masked = cell_mask[:, t, :]  # (B, V)
        actual = X_target[:, t, :]
        # Where masked: predict phi * previous; where unmasked: use actual
        ar_pred = phi_t * prev
        pred[:, t, :] = torch.where(is_masked, ar_pred, torch.zeros_like(actual))
        prev = torch.where(is_masked, ar_pred, actual)

    # Backward pass for leading masked positions (before first unmasked)
    prev_back = X_target[:, T-1, :].clone()
    for t in range(T-1, -1, -1):
        is_masked = cell_mask[:, t, :]
        actual = X_target[:, t, :]
        ar_pred_back = phi_t * prev_back
        # Only fill if forward pass left it at 0 (no preceding unmasked value)
        no_forward = is_masked & (pred[:, t, :] == 0)
        pred[:, t, :] = torch.where(no_forward, ar_pred_back, pred[:, t, :])
        prev_back = torch.where(is_masked, ar_pred_back, actual)

    return pred


def _fit_cross_variate_linear(train_loader, n_features):
    """Fit per-variate Ridge regression: x_v = sum_j(beta_j * x_j) + bias.

    Uses all other variates at the same timestep as predictors.
    Light L2 regularization (alpha=1.0) for stability.
    """
    from sklearn.linear_model import Ridge

    all_data = []
    for X_batch, _ in train_loader:
        X = X_batch.numpy()
        all_data.append(X.reshape(-1, n_features))
    all_data = np.concatenate(all_data, axis=0)
    N, V = all_data.shape
    print(f"    Fitting {V} Ridge models on {N} samples...")

    beta = np.zeros((V, V), dtype=np.float32)
    bias = np.zeros(V, dtype=np.float32)

    for v in range(V):
        y = all_data[:, v]
        X_other = np.delete(all_data, v, axis=1)
        reg = Ridge(alpha=1.0)
        reg.fit(X_other, y)
        idx = list(range(V))
        idx.pop(v)
        beta[v, idx] = reg.coef_.astype(np.float32)
        bias[v] = reg.intercept_

    return beta, bias


def _fit_cross_variate_linear_masked(train_loader, n_features, n_augment=5,
                                      variate_ratio=0.20, seed=42):
    """Fit per-variate Ridge on MASKED training data.

    Unlike _fit_cross_variate_linear (which fits on complete data and breaks
    when evaluated on masked inputs), this fits on data with the same masking
    scheme the encoder sees during training. Each training sample is augmented
    n_augment times with random whole-variate masks, training Ridge to handle
    missing inputs.

    Fix for task #61: Ridge baseline was "broken by design" — fitted on
    complete data but evaluated on masked data. This produces a fair
    comparison where both Ridge and the encoder see the same degraded inputs.
    """
    from sklearn.linear_model import Ridge

    all_data = []
    for X_batch, _ in train_loader:
        X = X_batch.numpy()
        all_data.append(X.reshape(-1, n_features))
    all_data = np.concatenate(all_data, axis=0)
    N, V = all_data.shape

    rng = np.random.RandomState(seed)
    # Augment: apply random whole-variate masking n_augment times
    augmented_X = []
    augmented_Y = []
    n_mask_vars = max(1, int(V * variate_ratio))
    for aug in range(n_augment):
        mask_vars = rng.choice(V, size=n_mask_vars, replace=False)
        X_masked = all_data.copy()
        X_masked[:, mask_vars] = 0.0  # zero out masked variates
        augmented_X.append(X_masked)
        augmented_Y.append(all_data)  # targets are original (unmasked)

    X_aug = np.concatenate(augmented_X, axis=0)  # (N*n_augment, V)
    Y_aug = np.concatenate(augmented_Y, axis=0)  # (N*n_augment, V)

    print(f"    Fitting {V} masked-Ridge models on {len(X_aug)} samples "
          f"({n_augment} augmentations, {variate_ratio:.0%} mask rate)...")

    beta = np.zeros((V, V), dtype=np.float32)
    bias = np.zeros(V, dtype=np.float32)

    for v in range(V):
        y = Y_aug[:, v]
        X_other = np.delete(X_aug, v, axis=1)
        reg = Ridge(alpha=1.0)
        reg.fit(X_other, y)
        idx = list(range(V))
        idx.pop(v)
        beta[v, idx] = reg.coef_.astype(np.float32)
        bias[v] = reg.intercept_

    return beta, bias


def _cross_variate_predictions(X_target, mask, beta, bias):
    """Cross-variate linear predictions using only unmasked variates.

    For fair comparison with the transformer (which sees zeros at masked positions),
    masked variate contributions are zeroed. No renormalization — the transformer
    faces the same information loss.
    """
    B, T, V = X_target.shape
    beta_t = torch.from_numpy(beta).to(X_target.device)
    bias_t = torch.from_numpy(bias).to(X_target.device)
    unmasked = (~mask).float()

    pred = torch.zeros_like(X_target)
    for t in range(T):
        x_t = X_target[:, t, :] * unmasked[:, t, :]
        pred[:, t, :] = x_t @ beta_t.T + bias_t
    return pred


def _oracle_linear_predictions(X_target, beta, bias):
    """Oracle: predict each variate from ALL others (ignore masking).

    Theoretical ceiling — no model should consistently beat this on
    same-timestep reconstruction.
    """
    B, T, V = X_target.shape
    beta_t = torch.from_numpy(beta).to(X_target.device)
    bias_t = torch.from_numpy(bias).to(X_target.device)

    pred = torch.zeros_like(X_target)
    for t in range(T):
        pred[:, t, :] = X_target[:, t, :] @ beta_t.T + bias_t
    return pred


def _compute_split_mse(pred, target, mask, var_mask, weights):
    """Compute MSE for aggregate, whole-variate, and per-cell partitions."""
    sq_err = (pred - target) ** 2
    mask_f = mask.float()
    eff = mask_f * weights

    results = {}
    for name, part_mask in [("aggregate", mask),
                             ("whole_var", mask & var_mask),
                             ("per_cell", mask & ~var_mask)]:
        part_eff = part_mask.float() * weights
        n = part_eff.sum()
        if n > 0:
            mse = (sq_err * part_eff).sum() / n
            results[name] = mse.item()
        else:
            results[name] = 0.0  # zero contribution when zero effective weight
    return results


def run_ssl_baselines(loaders, model=None, recon_head=None, hp=None):
    """Evaluate reconstruction baselines on SSL validation set.

    Runs 6 baselines using identical deterministic masks as train_ssl() validation:
      1. Predict-zero (trivial floor)
      2. Persistence (forward-fill last unmasked value)
      3. Per-variate AR(1) (fitted on training data)
      4. Cross-variate linear (Ridge, fitted on training data)
      5. Hybrid (persistence for per-cell + CV linear for whole-variate)
      6. Oracle linear (ceiling — all variates visible)

    If model and recon_head are provided, also evaluates the transformer.

    Reports aggregate, whole-variate, and per-cell MSE for each.
    References: Zerveas et al. (KDD 2021), Nie et al. (ICLR 2023),
    Dong et al. (NeurIPS 2024).

    Args:
        hp: optional hyperparameter dict. Defaults to module SSL_HYPERPARAMS.
            Pass `ckpt["ssl_hyperparams"]` from a loaded checkpoint to ensure
            the diagnostic uses the SAME mask_fill_value / use_mask_indicator
            settings the encoder was trained with — A-H1 fix 2026-04-29
            (cross-path consistency). Without this, a probe-only re-run with
            mutated SSL_HYPERPARAMS would silently feed the encoder a
            different fill value than training did.
    """
    global _results
    p = hp if hp is not None else SSL_HYPERPARAMS
    T = p["seq_len"]
    V = ACTIVE_N_SSL_FEATURES

    print("=" * 60)
    print("SSL RECONSTRUCTION BASELINES")
    print("=" * 60)

    # Eligible mask (same as training)
    eligible_np = _build_eligible_mask(ACTIVE_SSL_FEATURES, ACTIVE_MASK_INELIGIBLE)
    eligible = torch.from_numpy(eligible_np).to(device)
    flow_agg_positions = _build_flow_agg_position_map(ACTIVE_SSL_FEATURES)
    # SSL-012: mirror training-time RV block-mask propagation in the diagnostic
    # mask so the transformer-vs-baselines comparison is on the same mask
    # distribution training used. Without this, the transformer is evaluated
    # on EASIER masks (no v106↔v111-v113 propagation) than it trained on,
    # biasing the comparison against the SSL model.
    rv_positions = (_build_rv_position_map(ACTIVE_SSL_FEATURES)
                    if p.get("rv_block_mask", False) else None)  # A-H1 followup

    # --- Fit baselines on training data ---
    print("\n  Fitting baselines on training data...")
    phi = _fit_ar1(loaders["train"], V)
    print(f"    AR(1) phi range: [{phi.min():.4f}, {phi.max():.4f}]")
    beta, bias = _fit_cross_variate_linear(loaders["train"], V)
    # Masked Ridge: same masking scheme as SSL training (task #61 fix)
    beta_m, bias_m = _fit_cross_variate_linear_masked(
        loaders["train"], V, n_augment=5,
        variate_ratio=p.get("variate_ratio", 0.20))

    # --- Evaluate on validation set with deterministic masks ---
    print("\n  Evaluating on validation set (deterministic mask, seed=42)...")
    val_gen = torch.Generator(device=device).manual_seed(42)

    # Accumulators for each baseline
    baselines = ["predict_zero", "persistence", "ar1",
                 "cv_linear", "masked_cv_linear", "hybrid", "oracle_linear"]
    if model is not None and recon_head is not None:
        baselines.append("transformer")
        model.eval()
        recon_head.eval()

    accum = {b: {"aggregate": 0.0, "whole_var": 0.0, "per_cell": 0.0,
                 "n_agg": 0.0, "n_wv": 0.0, "n_pc": 0.0}
             for b in baselines}

    with torch.no_grad():
        for Xb, Wb in loaders["val"]:
            Xb = torch.nan_to_num(Xb.to(device), nan=0.0)
            Wb = Wb.to(device)
            B = Xb.shape[0]

            mask, var_mask = sample_ssl_mask(B, T, V, eligible,
                                             p["variate_ratio"], p["cell_ratio"],
                                             generator=val_gen,
                                             flow_agg_positions=flow_agg_positions,
                                             feature_indices=ACTIVE_SSL_FEATURES,
                                             rv_positions=rv_positions)

            # Count effective masked positions per partition
            eff_agg = (mask.float() * Wb).sum().item()
            eff_wv = ((mask & var_mask).float() * Wb).sum().item()
            eff_pc = ((mask & ~var_mask).float() * Wb).sum().item()

            # 1. Predict-zero
            pred_zero = torch.zeros_like(Xb)
            r = _compute_split_mse(pred_zero, Xb, mask, var_mask, Wb)
            accum["predict_zero"]["aggregate"] += r["aggregate"] * eff_agg
            accum["predict_zero"]["whole_var"] += r["whole_var"] * eff_wv
            accum["predict_zero"]["per_cell"] += r["per_cell"] * eff_pc

            # 2. Persistence
            pred_pers = _persistence_predictions(Xb, mask, var_mask)
            r = _compute_split_mse(pred_pers, Xb, mask, var_mask, Wb)
            accum["persistence"]["aggregate"] += r["aggregate"] * eff_agg
            accum["persistence"]["whole_var"] += r["whole_var"] * eff_wv
            accum["persistence"]["per_cell"] += r["per_cell"] * eff_pc

            # 3. AR(1)
            pred_ar1 = _ar1_predictions(Xb, mask, var_mask, phi)
            r = _compute_split_mse(pred_ar1, Xb, mask, var_mask, Wb)
            accum["ar1"]["aggregate"] += r["aggregate"] * eff_agg
            accum["ar1"]["whole_var"] += r["whole_var"] * eff_wv
            accum["ar1"]["per_cell"] += r["per_cell"] * eff_pc

            # 4. Cross-variate linear
            pred_cv = _cross_variate_predictions(Xb, mask, beta, bias)
            r = _compute_split_mse(pred_cv, Xb, mask, var_mask, Wb)
            accum["cv_linear"]["aggregate"] += r["aggregate"] * eff_agg
            accum["cv_linear"]["whole_var"] += r["whole_var"] * eff_wv
            accum["cv_linear"]["per_cell"] += r["per_cell"] * eff_pc

            # 4b. Masked cross-variate linear (fair comparison — fitted on masked data)
            pred_mcv = _cross_variate_predictions(Xb, mask, beta_m, bias_m)
            r = _compute_split_mse(pred_mcv, Xb, mask, var_mask, Wb)
            accum["masked_cv_linear"]["aggregate"] += r["aggregate"] * eff_agg
            accum["masked_cv_linear"]["whole_var"] += r["whole_var"] * eff_wv
            accum["masked_cv_linear"]["per_cell"] += r["per_cell"] * eff_pc

            # 5. Hybrid (persistence for cell + CV linear for whole-variate)
            pred_hybrid = torch.zeros_like(Xb)
            cell_positions = mask & ~var_mask
            wv_positions = mask & var_mask
            pred_hybrid[cell_positions] = pred_pers[cell_positions]
            pred_hybrid[wv_positions] = pred_cv[wv_positions]
            r = _compute_split_mse(pred_hybrid, Xb, mask, var_mask, Wb)
            accum["hybrid"]["aggregate"] += r["aggregate"] * eff_agg
            accum["hybrid"]["whole_var"] += r["whole_var"] * eff_wv
            accum["hybrid"]["per_cell"] += r["per_cell"] * eff_pc

            # 6. Oracle linear (ceiling)
            pred_oracle = _oracle_linear_predictions(Xb, beta, bias)
            r = _compute_split_mse(pred_oracle, Xb, mask, var_mask, Wb)
            accum["oracle_linear"]["aggregate"] += r["aggregate"] * eff_agg
            accum["oracle_linear"]["whole_var"] += r["whole_var"] * eff_wv
            accum["oracle_linear"]["per_cell"] += r["per_cell"] * eff_pc

            # 7. Transformer (if provided) — A-H1 fix: read mask_fill_value /
            # use_mask_indicator from local `p` so a hp= override propagates.
            if "transformer" in baselines:
                X_masked = Xb.clone()
                X_masked[mask] = float(p.get("mask_fill_value", 0.0))
                _use_mi = bool(p.get("use_mask_indicator", False))
                tokens = model.encode_variates(X_masked, mask=mask if _use_mi else None)
                x_hat = recon_head(tokens)
                x_hat_tv = x_hat.transpose(1, 2)  # (B, V, T) -> (B, T, V)
                r = _compute_split_mse(x_hat_tv, Xb, mask, var_mask, Wb)
                accum["transformer"]["aggregate"] += r["aggregate"] * eff_agg
                accum["transformer"]["whole_var"] += r["whole_var"] * eff_wv
                accum["transformer"]["per_cell"] += r["per_cell"] * eff_pc

            # Track total effective counts (same for all baselines)
            for b in baselines:
                accum[b]["n_agg"] += eff_agg
                accum[b]["n_wv"] += eff_wv
                accum[b]["n_pc"] += eff_pc

    # --- Compute final MSE and print table ---
    print("\n  SSL Reconstruction Baselines (lower = better)")
    print(f"  {'Baseline':<20s} {'Aggregate':>10s} {'Whole-Var':>10s} {'Per-Cell':>10s}")
    print("  " + "-" * 52)

    results = {}
    for b in baselines:
        a = accum[b]
        agg = a["aggregate"] / max(a["n_agg"], 1.0)
        wv = a["whole_var"] / max(a["n_wv"], 1.0)
        pc = a["per_cell"] / max(a["n_pc"], 1.0)
        results[b] = {"aggregate": agg, "whole_var": wv, "per_cell": pc}
        marker = " <-- model" if b == "transformer" else ""
        print(f"  {b:<20s} {agg:>10.4f} {wv:>10.4f} {pc:>10.4f}{marker}")

    # Key diagnostic ratios
    if "transformer" in results:
        tr = results["transformer"]
        pz = results["predict_zero"]
        orc = results["oracle_linear"]
        hyb = results["hybrid"]
        print(f"\n  Diagnostic ratios:")
        print(f"    Transformer / predict-zero:  {tr['aggregate']/pz['aggregate']:.3f} "
              f"(1.0 = no skill)")
        print(f"    Transformer / hybrid:        {tr['aggregate']/hyb['aggregate']:.3f} "
              f"(<1.0 = model adds value over trivial baselines)")
        print(f"    Transformer / oracle:        {tr['aggregate']/orc['aggregate']:.3f} "
              f"(1.0 = at ceiling)")
        wv_ratio = tr['whole_var'] / max(hyb['whole_var'], 1e-8)
        print(f"    Whole-var: transformer/hybrid: {wv_ratio:.3f} "
              f"(<1.0 = genuine cross-variate learning)")

    _results["ssl_baselines"] = results
    _results["ssl_fitted_baselines"] = {"phi": phi, "beta": beta, "bias": bias}
    print()
    return results


def run_ssl_baseline_diagnostics(loaders, model=None, recon_head=None,
                                  fitted_baselines=None, hp=None):
    """Per-variate and per-group baseline decomposition for methodology narrative.

    Computes per-variate MSE for each baseline and transformer, groups by
    feature category, reports R² = 1 - MSE/MSE_predict_zero per group,
    and diagnoses the weighted predict-zero MSE anomaly.

    Must run AFTER run_ssl_baselines() or standalone with a loaded model.

    Args:
        hp: optional hyperparameter dict. Defaults to module SSL_HYPERPARAMS.
            Pass `ckpt["ssl_hyperparams"]` for cross-path consistency with
            training (A-H1 fix 2026-04-29).
    """
    global _results
    p = hp if hp is not None else SSL_HYPERPARAMS
    T = p["seq_len"]
    V = ACTIVE_N_SSL_FEATURES

    print("=" * 60)
    print("SSL BASELINE DIAGNOSTICS (per-variate / per-group)")
    print("=" * 60)

    # --- Feature group mapping (SSL index -> group name) ---
    # ACTIVE_SSL_FEATURES maps SSL index -> raw variate index
    ssl_to_raw = {i: v for i, v in enumerate(ACTIVE_SSL_FEATURES)}
    GROUP_RANGES = [
        ("options_grid", range(0, 88)),
        ("strike_agg",   range(99, 105)),
        ("spx_derived",  range(105, 119)),
        ("vix_term",     range(119, 131)),
        ("order_flow",   range(131, 140)),
        ("flow_roll3",   range(141, 150)),   # v3 only: rolling-mean-3 aggregates
        ("flow_roll15",  range(150, 159)),   # v3 only: rolling-mean-15 aggregates
    ]
    def _group_for(raw_v):
        for name, rng in GROUP_RANGES:
            if raw_v in rng:
                return name
        return "other"

    variate_group = [_group_for(ssl_to_raw[i]) for i in range(V)]
    group_names = ["options_grid", "strike_agg", "spx_derived", "vix_term", "order_flow"]
    if USE_V3_FEATURES:
        group_names += ["flow_roll3", "flow_roll15"]
    group_counts = {g: sum(1 for x in variate_group if x == g) for g in group_names}
    print(f"\n  Feature groups: {group_counts}")

    # --- Eligible mask and masking ---
    eligible_np = _build_eligible_mask(ACTIVE_SSL_FEATURES, ACTIVE_MASK_INELIGIBLE)
    eligible = torch.from_numpy(eligible_np).to(device)
    flow_agg_positions = _build_flow_agg_position_map(ACTIVE_SSL_FEATURES)
    rv_positions = (_build_rv_position_map(ACTIVE_SSL_FEATURES)
                    if p.get("rv_block_mask", False) else None)  # A-H1 followup
    val_gen = torch.Generator(device=device).manual_seed(42)

    # --- Reuse or fit baselines ---
    if fitted_baselines is None:
        fitted_baselines = _results.get("ssl_fitted_baselines")
    if fitted_baselines is not None:
        print("\n  Reusing pre-fitted baselines from run_ssl_baselines()")
        phi = fitted_baselines["phi"]
        beta = fitted_baselines["beta"]
        bias = fitted_baselines["bias"]
    else:
        print("\n  Fitting baselines on training data...")
        phi = _fit_ar1(loaders["train"], V)
        beta, bias = _fit_cross_variate_linear(loaders["train"], V)

    baselines = ["predict_zero", "persistence", "ar1", "cv_linear", "hybrid", "oracle_linear"]
    if model is not None and recon_head is not None:
        baselines.append("transformer")
        model.eval()
        recon_head.eval()

    # Per-variate accumulators: sum of squared errors and counts
    per_var_sse = {b: np.zeros(V, dtype=np.float64) for b in baselines}
    per_var_n   = {b: np.zeros(V, dtype=np.float64) for b in baselines}
    # Also accumulate for whole-var and per-cell separately
    per_var_sse_wv = {b: np.zeros(V, dtype=np.float64) for b in baselines}
    per_var_n_wv   = {b: np.zeros(V, dtype=np.float64) for b in baselines}
    per_var_sse_pc = {b: np.zeros(V, dtype=np.float64) for b in baselines}
    per_var_n_pc   = {b: np.zeros(V, dtype=np.float64) for b in baselines}

    # Also track weighted variance per variate (for R²)
    var_sum_x2 = np.zeros(V, dtype=np.float64)
    var_sum_w  = np.zeros(V, dtype=np.float64)

    print("  Evaluating per-variate MSE on validation set...")

    with torch.no_grad():
        for Xb, Wb in loaders["val"]:
            Xb = torch.nan_to_num(Xb.to(device), nan=0.0)
            Wb = Wb.to(device)
            B = Xb.shape[0]

            mask, var_mask = sample_ssl_mask(B, T, V, eligible,
                                             p["variate_ratio"], p["cell_ratio"],
                                             generator=val_gen,
                                             flow_agg_positions=flow_agg_positions,
                                             feature_indices=ACTIVE_SSL_FEATURES,
                                             rv_positions=rv_positions)
            cell_mask = mask & ~var_mask

            # Predictions for each baseline
            pred_zero = torch.zeros_like(Xb)
            pred_pers = _persistence_predictions(Xb, mask, var_mask)
            pred_ar1  = _ar1_predictions(Xb, mask, var_mask, phi)
            pred_cv   = _cross_variate_predictions(Xb, mask, beta, bias)
            pred_oracle = _oracle_linear_predictions(Xb, beta, bias)
            pred_hybrid = torch.zeros_like(Xb)
            pred_hybrid[cell_mask] = pred_pers[cell_mask]
            pred_hybrid[mask & var_mask] = pred_cv[mask & var_mask]

            preds = {
                "predict_zero": pred_zero, "persistence": pred_pers,
                "ar1": pred_ar1, "cv_linear": pred_cv,
                "hybrid": pred_hybrid, "oracle_linear": pred_oracle,
            }
            if "transformer" in baselines:
                X_masked = Xb.clone()
                X_masked[mask] = float(p.get("mask_fill_value", 0.0))  # A-H1 fix
                _use_mi = bool(p.get("use_mask_indicator", False))  # A-H1 followup fix
                tokens = model.encode_variates(X_masked, mask=mask if _use_mi else None)
                x_hat = recon_head(tokens).transpose(1, 2)
                preds["transformer"] = x_hat

            # Per-variate accumulation — convert shared tensors once
            mask_f = mask.float()
            wv_f   = (mask & var_mask).float()
            pc_f   = (mask & ~var_mask).float()
            Xb_np   = Xb.cpu().numpy()
            Wb_np   = Wb.cpu().numpy()
            mask_np = mask_f.cpu().numpy()
            wv_np   = wv_f.cpu().numpy()
            pc_np   = pc_f.cpu().numpy()

            # Effective weight arrays (shared across baselines)
            eff_agg = Wb_np * mask_np
            eff_wv  = Wb_np * wv_np
            eff_pc  = Wb_np * pc_np
            eff_agg_sum = eff_agg.sum(axis=(0, 1))
            eff_wv_sum  = eff_wv.sum(axis=(0, 1))
            eff_pc_sum  = eff_pc.sum(axis=(0, 1))

            for b in baselines:
                sq_err_raw = ((preds[b] - Xb) ** 2).cpu().numpy()

                per_var_sse[b]    += (sq_err_raw * eff_agg).sum(axis=(0, 1))
                per_var_sse_wv[b] += (sq_err_raw * eff_wv).sum(axis=(0, 1))
                per_var_sse_pc[b] += (sq_err_raw * eff_pc).sum(axis=(0, 1))

            # Denominators are identical for all baselines — accumulate once
            per_var_n[baselines[0]]    += eff_agg_sum
            per_var_n_wv[baselines[0]] += eff_wv_sum
            per_var_n_pc[baselines[0]] += eff_pc_sum

            # Weighted variance per variate (for predict-zero diagnosis)
            var_sum_x2 += (Xb_np ** 2 * eff_agg).sum(axis=(0, 1))
            var_sum_w  += eff_agg_sum

    # --- Compute per-variate MSE (shared denominator across baselines) ---
    n_agg = per_var_n[baselines[0]]    # (V,) — identical for all baselines
    n_wv  = per_var_n_wv[baselines[0]]
    n_pc  = per_var_n_pc[baselines[0]]

    per_var_mse = {}
    for b in baselines:
        mse = np.zeros(V, dtype=np.float64)
        for v in range(V):
            if n_agg[v] > 0:
                mse[v] = per_var_sse[b][v] / n_agg[v]
        per_var_mse[b] = mse

    # Weighted variance per variate
    per_var_variance = np.zeros(V, dtype=np.float64)
    for v in range(V):
        if var_sum_w[v] > 0:
            per_var_variance[v] = var_sum_x2[v] / var_sum_w[v]

    # --- Diagnose predict-zero MSE ---
    print("\n  Predict-zero MSE diagnosis (why != 1.0 on z-scored data):")
    pz_total = per_var_sse["predict_zero"].sum() / max(n_agg.sum(), 1e-8)
    print(f"    Properly-weighted predict-zero MSE: {pz_total:.4f}")
    print(f"    Mean per-variate predict-zero MSE (unweighted avg): {per_var_mse['predict_zero'].mean():.4f}")
    print(f"    Per-variate weighted variance (should be ~1.0 if z-scored):")
    n_low_var = np.sum(per_var_variance < 0.5)
    n_mid_var = np.sum((per_var_variance >= 0.5) & (per_var_variance < 0.9))
    n_unit_var = np.sum((per_var_variance >= 0.9) & (per_var_variance < 1.1))
    n_high_var = np.sum(per_var_variance >= 1.1)
    n_zero = np.sum(per_var_variance == 0.0)
    print(f"      Zero (mask-ineligible):  {n_zero}")
    print(f"      Low  (< 0.5):           {n_low_var}")
    print(f"      Mid  (0.5 - 0.9):       {n_mid_var}")
    print(f"      Unit (0.9 - 1.1):       {n_unit_var}")
    print(f"      High (> 1.1):           {n_high_var}")

    # Top 10 lowest non-zero variance variates
    nonzero_var = [(v, per_var_variance[v], ssl_to_raw[v], variate_group[v])
                   for v in range(V) if per_var_variance[v] > 0]
    nonzero_var.sort(key=lambda x: x[1])
    print(f"\n    10 lowest-variance variates (driving predict-zero MSE down):")
    print(f"    {'SSL':>4s} {'Raw':>4s} {'Group':<14s} {'Wt.Var':>8s}")
    for ssl_i, wvar, raw_i, grp in nonzero_var[:10]:
        print(f"    {ssl_i:>4d} v{raw_i:<3d} {grp:<14s} {wvar:>8.4f}")

    # --- Group-level summary ---
    print("\n  Per-group MSE and R² (R² = 1 - MSE_model / MSE_predict_zero):")
    print(f"\n  {'Group':<14s} {'#Feat':>5s}  {'PredZero':>8s} {'Persist':>8s} "
          f"{'AR(1)':>8s} {'CVLin':>8s} {'Hybrid':>8s} {'Oracle':>8s}", end="")
    if "transformer" in baselines:
        print(f" {'Xformer':>8s} {'R²':>6s}", end="")
    print()
    print("  " + "-" * (95 + (16 if "transformer" in baselines else 0)))

    group_results = {}
    for g in group_names:
        indices = [v for v in range(V) if variate_group[v] == g]
        if not indices:
            continue
        row = {}
        total_n = sum(n_agg[v] for v in indices)
        for b in baselines:
            total_sse = sum(per_var_sse[b][v] for v in indices)
            row[b] = total_sse / max(total_n, 1e-8)

        r2 = 1.0 - row.get("transformer", row["predict_zero"]) / max(row["predict_zero"], 1e-8) if "transformer" in baselines else 0.0
        group_results[g] = {**row, "r2": r2, "n_feat": len(indices)}

        print(f"  {g:<14s} {len(indices):>5d}  {row['predict_zero']:>8.4f} {row['persistence']:>8.4f} "
              f"{row['ar1']:>8.4f} {row['cv_linear']:>8.4f} {row['hybrid']:>8.4f} {row['oracle_linear']:>8.4f}", end="")
        if "transformer" in baselines:
            print(f" {row['transformer']:>8.4f} {r2:>6.3f}", end="")
        print()

    # --- Whole-variate group decomposition ---
    print(f"\n  Whole-variate MSE per group (cross-variate learning quality):")
    print(f"  {'Group':<14s} {'#Feat':>5s}  {'PredZero':>8s} {'CVLin':>8s} {'Oracle':>8s}", end="")
    if "transformer" in baselines:
        print(f" {'Xformer':>8s} {'R²_wv':>6s}", end="")
    print()
    print("  " + "-" * (52 + (16 if "transformer" in baselines else 0)))

    whole_var_results = {}
    for g in group_names:
        indices = [v for v in range(V) if variate_group[v] == g]
        if not indices:
            continue
        row_wv = {}
        total_n_wv_g = sum(n_wv[v] for v in indices)
        for b in ["predict_zero", "cv_linear", "oracle_linear"] + (["transformer"] if "transformer" in baselines else []):
            total_sse = sum(per_var_sse_wv[b][v] for v in indices)
            row_wv[b] = total_sse / max(total_n_wv_g, 1e-8)
        r2_wv = 1.0 - row_wv.get("transformer", 0) / max(row_wv["predict_zero"], 1e-8) if "transformer" in baselines else 0.0
        whole_var_results[g] = {**row_wv, "r2_wv": r2_wv, "n_feat": len(indices)}
        print(f"  {g:<14s} {len(indices):>5d}  {row_wv['predict_zero']:>8.4f} {row_wv['cv_linear']:>8.4f} "
              f"{row_wv['oracle_linear']:>8.4f}", end="")
        if "transformer" in baselines:
            print(f" {row_wv['transformer']:>8.4f} {r2_wv:>6.3f}", end="")
        print()

    # --- Top transformer wins (largest improvement over best simple baseline) ---
    if "transformer" in baselines:
        print(f"\n  Top 15 variates where transformer beats best simple baseline:")
        print(f"  {'SSL':>4s} {'Raw':>4s} {'Group':<14s} {'BestSimp':>8s} {'Xformer':>8s} {'Improve':>8s}")
        improvements = []
        for v in range(V):
            best_simple = min(per_var_mse["persistence"][v], per_var_mse["ar1"][v])
            if per_var_n[baselines[0]][v] > 0 and best_simple > 0.01:
                imp = (best_simple - per_var_mse["transformer"][v]) / best_simple
                improvements.append((v, ssl_to_raw[v], variate_group[v],
                                     best_simple, per_var_mse["transformer"][v], imp))
        improvements.sort(key=lambda x: -x[5])
        for ssl_i, raw_i, grp, bs, xf, imp in improvements[:15]:
            print(f"  {ssl_i:>4d} v{raw_i:<3d} {grp:<14s} {bs:>8.4f} {xf:>8.4f} {imp:>7.1%}")

        print(f"\n  Top 10 variates where transformer is WORST (biggest gap to oracle):")
        print(f"  {'SSL':>4s} {'Raw':>4s} {'Group':<14s} {'Xformer':>8s} {'Oracle':>8s} {'Gap':>8s}")
        gaps = []
        for v in range(V):
            if per_var_n[baselines[0]][v] > 0 and per_var_mse["transformer"][v] > 0.01:
                gap = per_var_mse["transformer"][v] - per_var_mse["oracle_linear"][v]
                gaps.append((v, ssl_to_raw[v], variate_group[v],
                             per_var_mse["transformer"][v], per_var_mse["oracle_linear"][v], gap))
        gaps.sort(key=lambda x: -x[5])
        for ssl_i, raw_i, grp, xf, orc, g in gaps[:10]:
            print(f"  {ssl_i:>4d} v{raw_i:<3d} {grp:<14s} {xf:>8.4f} {orc:>8.4f} {g:>8.4f}")

    # Compute eligible count per group for structured metrics
    eligible_by_group = {}
    for g in group_names:
        indices = [v for v in range(V) if variate_group[v] == g]
        eligible_by_group[g] = sum(1 for v in indices if eligible_np[v])

    _results["ssl_baseline_diagnostics"] = {
        "per_var_mse": {b: per_var_mse[b].tolist() for b in baselines},
        "per_var_variance": per_var_variance.tolist(),
        "group_results": group_results,
        "whole_var_results": whole_var_results,
        "eligible_by_group": eligible_by_group,
        "variate_group": variate_group,
        "ssl_to_raw": ssl_to_raw,
    }
    print()
    return _results["ssl_baseline_diagnostics"]



# ===================================================================
# 10A-2. STRUCTURED POST-TRAINING METRICS COLLECTION
# ===================================================================
# Assembles ALL metrics from _results into a single JSON-serializable
# dict and saves to disk. Runs automatically at the end of
# run_ssl_pipeline() to guarantee every training run produces a
# machine-readable metrics record.


def _safe_float(v):
    """Convert numpy/torch scalars to Python float for JSON serialization."""
    if v is None:
        return None
    try:
        import torch as _t
        if isinstance(v, _t.Tensor):
            return float(v.item()) if v.numel() == 1 else None
    except ImportError:
        pass
    if hasattr(v, 'item'):
        return float(v.item())
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect_training_metrics(history, hp, n_params, tokenizer="linear"):
    """Collect ALL post-training metrics into a standardized JSON structure.

    Assembles from _results (populated by run_ssl_baselines,
    run_ssl_baseline_diagnostics, and train_ssl diagnostic snapshots).
    Saves to raw_data/local_store/training_metrics_{tokenizer}.json.

    Returns the metrics dict.
    """
    global _results
    import json as _json
    import datetime as _dt
    import os as _os

    # --- Training summary ---
    val_losses = history.get("val_loss", [])
    if val_losses:
        best_val = min(val_losses)
        best_epoch = 1 + val_losses.index(best_val)
    else:
        best_val = history.get("best_val_loss")
        best_epoch = None

    metrics = {
        "schema_version": 1,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tokenizer": tokenizer,

        "training_summary": {
            "val_loss": _safe_float(best_val),
            "best_epoch": best_epoch,
            "total_epochs": hp.get("num_epochs", 100),
            "n_params": n_params,
            "emb_var": _safe_float(history.get("final_emb_var")),
            "beat_zero": history.get("final_beat_zero"),
            "spectral_w": _safe_float(history.get("final_spectral_w")),
        },

        "hyperparams": {
            "d_model": hp.get("d_model"),
            "n_layers": hp.get("n_layers"),
            "n_heads": hp.get("n_heads"),
            "seq_len": hp.get("seq_len"),
            "variate_ratio": hp.get("variate_ratio"),
            "cell_ratio": hp.get("cell_ratio"),
            "spectral_rank_weight": hp.get("spectral_rank_weight"),
            "target_spectral_fraction": hp.get("target_spectral_fraction"),
            "huber_delta": hp.get("huber_delta"),
            "lr": hp.get("lr"),
            "use_shared_decoder": hp.get("use_shared_decoder"),
            "tokenizer": hp.get("tokenizer", "linear"),
            "patch_size": hp.get("patch_size"),
            "use_temporal_attn": hp.get("use_temporal_attn"),
        },

        "diagnostic_snapshots": history.get("diagnostic_snapshots", []),
    }

    # --- Reconstruction baselines (aggregate table) ---
    baselines = _results.get("ssl_baselines")
    if baselines:
        bl_summary = {}
        for name, vals in baselines.items():
            if isinstance(vals, dict):
                bl_summary[name] = {
                    k: _safe_float(v) for k, v in vals.items()
                }
        metrics["baselines_aggregate"] = bl_summary

    # --- Per-group reconstruction breakdown ---
    diag = _results.get("ssl_baseline_diagnostics")
    if diag:
        group_results = diag.get("group_results", {})
        whole_var_results = diag.get("whole_var_results", {})
        eligible_by_group = diag.get("eligible_by_group", {})

        per_group = {}
        for g_name, g_data in group_results.items():
            n_feat = g_data.get("n_feat", 0)
            n_eligible = eligible_by_group.get(g_name, n_feat)

            mse_pz = _safe_float(g_data.get("predict_zero"))
            mse_enc = _safe_float(g_data.get("transformer"))
            mse_oracle = _safe_float(g_data.get("oracle_linear"))

            # pct_learned: (mse_pz - mse_enc) / (mse_pz - mse_oracle) * 100
            pct_learned = None
            if mse_pz is not None and mse_enc is not None and mse_oracle is not None:
                denom = mse_pz - mse_oracle
                if abs(denom) > 1e-12:
                    pct_learned = max(0.0, min(100.0,
                        (mse_pz - mse_enc) / denom * 100.0))

            # R-squared from whole-variate table
            wv = whole_var_results.get(g_name, {})
            r2_wv = _safe_float(wv.get("r2_wv"))

            per_group[g_name] = {
                "n_variates": n_feat,
                "n_eligible": n_eligible,
                "mse_predict_zero": mse_pz,
                "mse_persistence": _safe_float(g_data.get("persistence")),
                "mse_ar1": _safe_float(g_data.get("ar1")),
                "mse_cv_linear": _safe_float(g_data.get("cv_linear")),
                "mse_hybrid": _safe_float(g_data.get("hybrid")),
                "mse_oracle_linear": mse_oracle,
                "mse_transformer": mse_enc,
                "r_squared": _safe_float(g_data.get("r2")),
                "pct_learned": pct_learned,
                "r_squared_whole_var": r2_wv,
            }
        metrics["per_group_reconstruction"] = per_group

        # Aggregate weighted R-squared and pct_learned
        total_feat = sum(d["n_variates"] for d in per_group.values())
        if total_feat > 0:
            w_r2 = sum(
                d["r_squared"] * d["n_variates"]
                for d in per_group.values()
                if d["r_squared"] is not None
            ) / total_feat
            pct_values = [
                (d["pct_learned"], d["n_variates"])
                for d in per_group.values()
                if d["pct_learned"] is not None
            ]
            w_pct = (sum(p * n for p, n in pct_values) / sum(n for _, n in pct_values)
                     if pct_values else None)
            metrics["aggregate_quality"] = {
                "weighted_r_squared": _safe_float(w_r2),
                "weighted_pct_learned": _safe_float(w_pct),
                "total_variates": total_feat,
            }

    # --- Sanity gates G1-G6 (per (internal doc)) ---
    gates = {}
    ts = metrics["training_summary"]
    bl_agg = metrics.get("baselines_aggregate", {})

    # G1: beat_zero — count of variates where encoder MSE < predict-zero MSE >= 140/147
    # Catches catastrophic training failure.
    beat_zero = ts.get("beat_zero")
    if beat_zero is not None:
        n_total = hp.get("n_ssl_features", ACTIVE_N_SSL_FEATURES)
        gates["G1_beat_zero"] = {
            "pass": beat_zero >= 140,
            "beat_zero": beat_zero,
            "threshold": 140,
            "total": n_total,
        }

    # G2: masking_artifact — Spearman on flow variates (SOFT diagnostic, not hard gate)
    # Both arms fail due to pretext task design; tracked as trend, not disqualifier.
    mqa = _results.get("masking_artifact_diagnostic")
    if isinstance(mqa, dict) and not mqa.get("skipped"):
        g_mqa = mqa.get("global", {})
        spearman = _safe_float(g_mqa.get("spearman"))
        gates["G2_masking_artifact"] = {
            "pass": True,  # soft gate — always passes, value tracked for trends
            "spearman": spearman,
            "note": "soft_diagnostic",
        }

    # G3: stability_cv — CV(val_loss[-20:]) < 0.05
    # Catches training instability (oscillation/divergence in final epochs).
    val_losses = history.get("val_loss", []) if isinstance(history, dict) else []
    if len(val_losses) >= 20:
        import numpy as _np_g3
        tail = _np_g3.array(val_losses[-20:])
        cv = float(tail.std() / (tail.mean() + 1e-8))
        gates["G3_stability_cv"] = {
            "pass": cv < 0.05,
            "cv": round(cv, 6),
            "threshold": 0.05,
        }

    # G4: emb_var > 0.01 — representational collapse detection
    # Jing et al. (2022), Bardes et al. (2022 VICReg).
    emb_var = ts.get("emb_var")
    if emb_var is not None:
        gates["G4_emb_var"] = {
            "pass": emb_var > 0.01,
            "emb_var": emb_var,
            "threshold": 0.01,
        }

    # G5: per-group R² floor — all mask-eligible groups must have R² > 0
    # Catches group-level failure that G1 might miss (e.g., VIX R²=-0.452 while G1 passes).
    per_group = metrics.get("per_group_reconstruction", {})
    if per_group:
        group_r2s = {g: v.get("r_squared") for g, v in per_group.items()
                     if v.get("n_eligible", 0) > 0 and v.get("r_squared") is not None}
        failing_groups = [g for g, r2 in group_r2s.items() if r2 <= 0]
        gates["G5_per_group_floor"] = {
            "pass": len(failing_groups) == 0,
            "failing_groups": failing_groups,
            "n_groups_checked": len(group_r2s),
        }

    # G6: generalization gap — train_loss / val_loss > 0.5
    # Catches catastrophic overfitting (encoder memorized training masks).
    train_losses = history.get("train_loss", []) if isinstance(history, dict) else []
    if train_losses and val_losses:
        final_train = train_losses[-1] if train_losses else None
        final_val = val_losses[-1] if val_losses else None
        if final_train is not None and final_val is not None and final_val > 1e-8:
            ratio = final_train / final_val
            gates["G6_generalization_gap"] = {
                "pass": ratio > 0.5,
                "train_val_ratio": round(ratio, 4),
                "threshold": 0.5,
            }

    all_pass = all(g.get("pass", False) for g in gates.values()) if gates else False
    gates["all_pass"] = all_pass
    metrics["sanity_gates"] = gates

    # --- V3 diagnostics summary (scalar extracts) ---
    v3_diag = {}

    mqa = _results.get("masking_artifact_diagnostic")
    if isinstance(mqa, dict) and not mqa.get("skipped"):
        g = mqa.get("global", {})
        v3_diag["mqa6_passed"] = mqa.get("passed")
        v3_diag["mqa6_global_spearman"] = _safe_float(g.get("spearman"))
        v3_diag["mqa6_global_pearson"] = _safe_float(g.get("pearson"))

    cosine = _results.get("embedding_cosine_diagnostic")
    if isinstance(cosine, dict) and not cosine.get("skipped"):
        gmp = cosine.get("global_mean_pool", {})
        v3_diag["cosine_global_mean"] = _safe_float(gmp.get("mean"))
        v3_diag["cosine_global_p5"] = _safe_float(gmp.get("p5"))

    attn = _results.get("attention_entropy_diagnostic")
    if isinstance(attn, dict) and not attn.get("skipped"):
        v3_diag["attention_n_layers"] = attn.get("n_layers")

    if v3_diag:
        metrics["v3_diagnostics_summary"] = v3_diag

    # --- Save to disk ---
    metrics_dir = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "raw_data", "local_store"
    )
    _os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = _os.path.join(metrics_dir, f"training_metrics_{tokenizer}.json")

    with open(metrics_path, "w") as f:
        _json.dump(metrics, f, indent=2, default=str)
    print(f"  Structured metrics saved to: {metrics_path}")

    return metrics


# ===================================================================
# 10B. V3 MASKING-ARTIFACT DIAGNOSTIC (MQA.6 — asserted gate)
# ===================================================================
# L1-SSL-009 concern #2 (Model QA Round 6, HIGH severity): when a raw
# flow cell v131[t] is block-masked its value is zeroed, AND its
# aggregate derivatives v141[t..t+2] (roll3) and v150[t..t+14] (roll15)
# are ALSO block-masked to zero (per L1-SSL-009 §4.3 block-expansion).
# The encoder could learn a degenerate "predict 0 for masked flow cells"
# prior — loss decreases without genuine representation learning.
#
# This diagnostic probes that failure mode at end-of-training by:
# (a) Splitting masked raw-flow cells into quiet vs active cohorts by
# |true_value|, computing MSE per cohort. Large ratio (active >>
# quiet) is the shrink-to-zero signature.
# (b) Global Pearson corr(|true_value|, MSE_per_cell). An encoder that
# tracks magnitude should produce corr ≈ 0. Positive correlation
# indicates MSE growing with magnitude (= encoder predicts small
# values regardless of true magnitude = shrinkage prior).
#
# Asserted thresholds (v3 ship gate): corr < 0.4 AND active/quiet MSE
# ratio < 5.0. Loose thresholds for first run — tighten after SSL-004-FE
# baseline lands.


def _spearman(x, y):
    """Rank correlation (Pearson on ranks) — no scipy dependency.

    Rank-based correlation is robust to the monotone-but-nonlinear
    relationship between |true| and squared-MSE that a shrink-to-zero
    encoder produces (where MSE = true² up to a scale). Pearson on raw
    values is dominated by extreme cells; Spearman captures the rank
    relationship scale-free.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_masking_artifact_diagnostic(
    model, recon_head, loaders, device_,
    spearman_threshold=0.40,
    pearson_threshold=0.40,
    mse_ratio_threshold=3.0,
    quantile_low=0.25,
    quantile_high=0.75,
    assert_pass=True,
    max_batches=None,
    v2_reference_mode=False,
    include_raw_flow_on_v2=True,
    hp=None,
):
    """MQA.6 shrinkage-prior diagnostic on block-masked raw flow cells.

    Post-MQA review (Round 7+):
      - Quartile split (low/high percentiles, middle dead-zone dropped)
        amplifies the shrink-to-zero signal: quiet cohort is now
        {|z| ≤ 25pct}, active is {|z| ≥ 75pct}.
      - Spearman rank correlation replaces Pearson as the primary
        measure. MSE = (pred - true)² so Pearson(|true|, MSE) is a
        power-2 relationship dominated by tails; Spearman(|true|, MSE)
        directly captures the rank monotonicity of shrinkage.
      - Pearson still reported as secondary (for continuity with Round 6
        numbers and any external reference).

    v2 reference mode (USE_V3_FEATURES=False + v2_reference_mode=True):
      Masks raw flow cells directly via cell_ratio (no aggregates to
      block-expand) and reports the same statistics. Use to calibrate
      thresholds against a known-healthy SSL-004 checkpoint before
      tightening the v3 gates.

    Args:
        spearman_threshold: max allowed Spearman(|true|, MSE) per variate.
        pearson_threshold: max allowed Pearson(|true|, MSE) per variate.
        mse_ratio_threshold: max allowed MSE(active)/MSE(quiet). Tightened
            from 5.0 (median split) to 3.0 (quartile split — cleaner
            cohorts, so a smaller ratio carries the same meaning).
        quantile_low, quantile_high: define the cohorts. Default 0.25/0.75
            drops the middle 50% as a dead zone.
        v2_reference_mode: run on v2 checkpoint/corpus to collect
            baseline readings. Requires USE_V3_FEATURES=False.
        include_raw_flow_on_v2: also probe raw v131-v139 on v2 (which
            are maskable there). If False, v2 run is pure no-op and the
            function just logs the skip.

    Returns dict with per-variate Spearman/Pearson/MSE ratio and global
    aggregates. Raises AssertionError on threshold violation when
    assert_pass=True.
    """
    # v3 path: block-masked aggregates, standard diagnostic.
    # v2 path: either skip (default for current calls) or probe raw flow
    # directly in reference mode.
    if not USE_V3_FEATURES:
        if not v2_reference_mode:
            print("  [masking-artifact diagnostic: SKIPPED — v2 layout has no "
                  "block-masked aggregates, nothing to probe]")
            return {"skipped": True, "reason": "v2 layout (no aggregates)"}
        if not include_raw_flow_on_v2:
            return {"skipped": True, "reason": "v2 reference disabled"}

    # A-H1 fix 2026-04-29: read live config from `hp` if provided; default
    # to SSL_HYPERPARAMS for back-compat. Mid-run mutations of SSL_HYPERPARAMS
    # (or running this diagnostic from a probe-only re-load) no longer feed
    # the encoder a different fill value than training did.
    p = hp if hp is not None else SSL_HYPERPARAMS
    T = p["seq_len"]
    V = ACTIVE_N_SSL_FEATURES
    eligible_np = _build_eligible_mask(ACTIVE_SSL_FEATURES, ACTIVE_MASK_INELIGIBLE)
    eligible = torch.from_numpy(eligible_np).to(device_)
    flow_positions = _build_flow_agg_position_map(ACTIVE_SSL_FEATURES)
    rv_positions = (_build_rv_position_map(ACTIVE_SSL_FEATURES)
                    if p.get("rv_block_mask", False) else None)
    # v2 reference mode: build positions for raw v131-v139 directly
    # (ACTIVE_SSL_FEATURES on v2 has raw flow but no aggregates, so
    # _build_flow_agg_position_map returns None). Fall back to resolving
    # raw positions manually.
    if flow_positions is None:
        pos_map = {v: i for i, v in enumerate(ACTIVE_SSL_FEATURES)}
        raw_positions = [pos_map[v] for v in range(131, 140) if v in pos_map]
        if not raw_positions:
            return {"skipped": True, "reason": "no raw flow variates present"}
    else:
        raw_positions = flow_positions["raw_positions_py"]

    # Accumulate masked-cell (true, pred) pairs for each raw flow variate
    n_raw = len(raw_positions)
    per_variate_true = [[] for _ in range(n_raw)]
    per_variate_pred = [[] for _ in range(n_raw)]

    model.eval()
    recon_head.eval()
    val_gen = torch.Generator(device=device_).manual_seed(42)
    n_batches = 0
    with torch.no_grad():
        for Xb, Wb in loaders["val"]:
            if max_batches is not None and n_batches >= max_batches:
                break
            Xb = torch.nan_to_num(Xb.to(device_), nan=0.0)
            B = Xb.shape[0]

            mask, _ = sample_ssl_mask(B, T, V, eligible,
                                      p["variate_ratio"], p["cell_ratio"],
                                      generator=val_gen,
                                      flow_agg_positions=flow_positions,
                                      feature_indices=ACTIVE_SSL_FEATURES,
                                      rv_positions=rv_positions)

            X_masked = Xb.clone()
            X_masked[mask] = float(p.get("mask_fill_value", 0.0))  # A-H1 fix
            _use_mi = bool(p.get("use_mask_indicator", False))
            tokens = model.encode_variates(X_masked, mask=mask if _use_mi else None)
            x_hat = recon_head(tokens)              # (B, V, T)
            x_hat = x_hat.transpose(1, 2)            # (B, T, V)

            for i_flow, raw_pos in enumerate(raw_positions):
                cell_mask = mask[:, :, raw_pos]             # (B, T) bool
                if cell_mask.any():
                    per_variate_true[i_flow].append(Xb[:, :, raw_pos][cell_mask].cpu())
                    per_variate_pred[i_flow].append(x_hat[:, :, raw_pos][cell_mask].cpu())
            n_batches += 1

    # Compute diagnostic per raw flow variate
    results = {
        "per_variate": {},
        "global": {},
        "skipped": False,
        "v2_reference_mode": v2_reference_mode,
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
    }
    fails = []
    all_true_abs, all_mse = [], []
    for i_flow, raw_pos in enumerate(raw_positions):
        # On v3: raw variates v131-v139 at positions[0..8]
        # On v2-ref: whichever raw variates are present
        raw_v = 131 + i_flow if USE_V3_FEATURES else \
            [v for v, pos in zip(ACTIVE_SSL_FEATURES, range(len(ACTIVE_SSL_FEATURES)))
             if pos == raw_pos][0]
        if not per_variate_true[i_flow]:
            results["per_variate"][raw_v] = {"n_cells": 0, "status": "no_masked_cells"}
            continue
        true = torch.cat(per_variate_true[i_flow]).numpy()
        pred = torch.cat(per_variate_pred[i_flow]).numpy()
        residual = pred - true
        mse_per_cell = residual ** 2
        true_abs = np.abs(true)

        # Also compute Huber-loss per cell if huber_delta > 0. The encoder
        # was trained on Huber, so MSE-based shrinkage metrics evaluate a
        # different objective than the one the encoder optimized. Reporting
        # both gives an honest picture: MSE shows raw prediction quality,
        # Huber shows how the encoder sees its own performance.
        _hd = float(p.get("huber_delta", 0.0))
        if _hd > 0:
            abs_r = np.abs(residual)
            huber_per_cell = np.where(
                abs_r <= _hd,
                0.5 * residual ** 2,
                _hd * (abs_r - 0.5 * _hd))
        else:
            huber_per_cell = mse_per_cell  # MSE = Huber when delta=0

        # Quartile split (drop middle 50%) — cohorts are {bottom 25%} vs {top 25%}
        q_lo = np.quantile(true_abs, quantile_low)
        q_hi = np.quantile(true_abs, quantile_high)
        quiet = true_abs <= q_lo
        active = true_abs >= q_hi

        mse_quiet = float(mse_per_cell[quiet].mean()) if quiet.any() else 0.0
        mse_active = float(mse_per_cell[active].mean()) if active.any() else 0.0
        ratio = mse_active / max(mse_quiet, 1e-12)

        # Huber-scale ratio (matches training objective)
        hub_quiet = float(huber_per_cell[quiet].mean()) if quiet.any() else 0.0
        hub_active = float(huber_per_cell[active].mean()) if active.any() else 0.0
        hub_ratio = hub_active / max(hub_quiet, 1e-12)

        # Primary: Spearman rank correlation |true| vs error
        spearman = _spearman(true_abs, mse_per_cell)
        spearman_hub = _spearman(true_abs, huber_per_cell) if _hd > 0 else spearman
        # Secondary: Pearson on raw values (for continuity with Round 6)
        if len(true_abs) >= 2 and np.std(true_abs) > 0 and np.std(mse_per_cell) > 0:
            pearson = float(np.corrcoef(true_abs, mse_per_cell)[0, 1])
        else:
            pearson = 0.0

        results["per_variate"][raw_v] = {
            "n_cells": int(len(true)),
            "mse_quiet": mse_quiet,
            "mse_active": mse_active,
            "mse_ratio": ratio,
            "spearman": spearman,
            "pearson": pearson,
            "split_q_low": float(q_lo),
            "split_q_high": float(q_hi),
            # Huber-scale metrics (matches training objective when huber_delta > 0)
            "hub_quiet": hub_quiet,
            "hub_active": hub_active,
            "hub_ratio": hub_ratio,
            "spearman_hub": spearman_hub,
        }
        all_true_abs.append(true_abs)
        all_mse.append(mse_per_cell)

        spear_fail = spearman > spearman_threshold
        pear_fail = pearson > pearson_threshold
        ratio_fail = ratio > mse_ratio_threshold
        if spear_fail or pear_fail or ratio_fail:
            fails.append(
                f"v{raw_v}: sp={spearman:.3f} pe={pearson:.3f} ratio={ratio:.2f}"
                + (" [SP]" if spear_fail else "")
                + (" [PE]" if pear_fail else "")
                + (" [RATIO]" if ratio_fail else "")
            )

    # Global aggregated stats
    if all_true_abs:
        ga_true = np.concatenate(all_true_abs)
        ga_mse = np.concatenate(all_mse)
        results["global"] = {
            "n_cells": int(len(ga_true)),
            "spearman": _spearman(ga_true, ga_mse),
            "pearson": (
                float(np.corrcoef(ga_true, ga_mse)[0, 1])
                if np.std(ga_true) > 0 and np.std(ga_mse) > 0 else 0.0
            ),
            "mean_mse": float(ga_mse.mean()),
        }

    # Report
    print("\n" + "=" * 60)
    mode_label = "v2 REFERENCE" if v2_reference_mode else "v3 (SSL-004-FE)"
    print(f"MQA.6 MASKING-ARTIFACT DIAGNOSTIC  [{mode_label}]")
    print("=" * 60)
    print(f"  variate_ratio={p['variate_ratio']}  cell_ratio={p['cell_ratio']}  "
          f"cohorts=[<={quantile_low*100:.0f}%ile | >={quantile_high*100:.0f}%ile]  "
          f"batches={n_batches}")
    print(f"  thresholds (per-variate): spearman<{spearman_threshold}  "
          f"pearson<{pearson_threshold}  ratio<{mse_ratio_threshold}")
    print(f"  {'raw':<6}{'n_cells':>10}{'mse_quiet':>12}{'mse_active':>12}"
          f"{'ratio':>9}{'spear':>9}{'pears':>9}")
    for raw_v, d in results["per_variate"].items():
        if d.get("n_cells", 0) == 0:
            print(f"  v{raw_v:<5}{0:>10}  (no masked cells — insufficient val pass)")
            continue
        print(f"  v{raw_v:<5}{d['n_cells']:>10}{d['mse_quiet']:>12.4f}"
              f"{d['mse_active']:>12.4f}{d['mse_ratio']:>9.2f}"
              f"{d['spearman']:>9.3f}{d['pearson']:>9.3f}")
    g = results.get("global", {})
    if "spearman" in g:
        print(f"  {'GLOBAL':<6}{g['n_cells']:>10}{'':>12}{'':>12}{'':>9}"
              f"{g['spearman']:>9.3f}{g['pearson']:>9.3f}")

    if fails:
        msg = ("MQA.6 diagnostic FAILED — encoder exhibits shrinkage prior:\n"
               + "\n".join("    " + f for f in fails))
        print("\n  [FAIL] " + msg.replace("\n", "\n         "))
        results["passed"] = False
        if assert_pass:
            raise AssertionError(msg)
    else:
        print("\n  [PASS] no shrinkage-prior signature detected")
        results["passed"] = True

    return results


# ===================================================================
# 10C. V3 EMBEDDING-COSINE SHIFT DIAGNOSTIC (Concern #3)
# ===================================================================
# Train/probe distribution mismatch: during SSL training, block-masking
# zeroes flow aggregates v141-v158 in ~30% of samples. At probe/
# inference time, aggregates are always present with real values. If
# the encoder learns attention weights that heavily depend on aggregate
# values, probe-time embeddings will be systematically shifted relative
# to training distribution — probes degrade.
#
# Diagnostic: on a val pass, encode each batch twice — once with full
# input, once with aggregates zeroed. Compute cosine similarity on the
# mean-pooled embedding (what probes consume) per sample. A robust
# encoder produces cos ≈ 1; a heavily-aggregate-dependent encoder
# produces cos << 1. Thresholds: mean ≥ 0.90, p5 ≥ 0.80 (initial run
# uses log-only; tighten after baseline signal is understood).


def _v3_group_ranges_in_ssl_positions():
    """Resolve raw-variate group ranges to SSL-layout positions.

    Returns list of (group_name, [ssl_position_indices]) for ACTIVE_SSL_FEATURES.
    Groups with zero members in the active layout are omitted. Positions
    for flow_roll3/flow_roll15 only appear on v3.
    """
    groups_raw = [
        ("options_grid", range(0, 88)),
        ("strike_agg",   range(99, 105)),
        ("spx_derived",  range(105, 119)),
        ("vix_term",     range(119, 131)),
        ("order_flow",   range(131, 140)),
        ("flow_roll3",   range(141, 150)),
        ("flow_roll15",  range(150, 159)),
    ]
    pos_map = {v: i for i, v in enumerate(ACTIVE_SSL_FEATURES)}
    out = []
    for name, rng in groups_raw:
        positions = [pos_map[v] for v in rng if v in pos_map]
        if positions:
            out.append((name, positions))
    return out


def compute_embedding_cosine_shift(
    model, loaders, device_,
    min_mean_cosine=0.90,
    min_p5_cosine=0.80,
    min_flow_token_cosine=0.85,
    assert_pass=True,
    max_batches=None,
):
    """Concern #3 — train/probe embedding shift under aggregate absence.

    SCOPE CLARIFICATION (2026-04-29 audit): this diagnostic measures
    "encoder ignores agg-zero" — i.e., when flow aggregates are zeroed at
    probe time, do the raw-flow tokens stay close to their normal embeddings.
    A PASS proves the encoder is not READING aggregates as a primary signal
    for its raw-flow representations. A PASS does NOT prove "encoder doesn't
    cheat at training time via masked-raw → visible-aggregate inversion" —
    that property is enforced by `_expand_flow_aggregate_mask` which masks
    aggregates whenever their raw flow source is masked, not by this
    diagnostic. The two intervention distributions (training-time block-mask
    propagation vs probe-time aggregate-zeroing) are DIFFERENT, and PASS
    here is a necessary but not sufficient condition for shortcut absence.
    For the shortcut hypothesis itself, see the block-mask infrastructure.

    Only meaningful when flow aggregates are present in ACTIVE_SSL_FEATURES
    (USE_V3_FEATURES=True). Returns skipped=True on v2.

    Post-MQA review: the original single-number (mean-pool cosine)
    measurement diluted the signal 8× because only 18/147 tokens move
    under the intervention. This version adds:

    (a) **Per-variate cosine at raw-flow positions (v131-v139)**. These
        are the tokens whose attention is most aggregate-sensitive;
        shift here directly measures how much the encoder learned to
        rely on aggregate context when reconstructing raw flow.
    (b) **Per-variate cosine at aggregate positions (v141-v158)**. These
        ARE the zeroed tokens — sanity check that the intervention
        actually propagated. Expect low cosine here.
    (c) **Per-group mean-pool cosine**. Groups match probes' downstream
        pooling contract (options_grid, strike_agg, spx_derived,
        vix_term, order_flow, flow_roll3, flow_roll15). Shifts matter
        most on the groups probes actually consume.
    (d) **Global mean-pool cosine** (backward-compat with the original
        metric). Kept but no longer the sole gate.

    Only the primary gate (raw-flow per-variate cosine, min
    min_flow_token_cosine) is asserted. Other metrics are reported as
    context. Raw-flow variates are where aggregate-dependence has
    reconstruction-functional meaning; aggregate-token shift is
    expected and group-level shifts are directional but not
    calibrated against a v2 baseline yet.
    """
    if not USE_V3_FEATURES:
        print("  [embedding-cosine diagnostic: SKIPPED — v2 layout has no "
              "aggregates to zero]")
        return {"skipped": True, "reason": "v2 layout (no aggregates)"}

    flow_positions = _build_flow_agg_position_map(ACTIVE_SSL_FEATURES)
    assert flow_positions is not None, \
        "USE_V3_FEATURES=True but _build_flow_agg_position_map returned None"

    agg_positions = (
        flow_positions["roll3_positions_py"]
        + flow_positions["roll15_positions_py"]
    )
    raw_flow_positions = flow_positions["raw_positions_py"]  # v131-v139
    group_defs = _v3_group_ranges_in_ssl_positions()

    model.eval()
    # Per-variate cosine accumulators: (V,) lists of per-sample cosines
    V = ACTIVE_N_SSL_FEATURES
    per_variate_cos = [[] for _ in range(V)]
    # Per-group pooled-embedding cosine accumulators
    per_group_cos = {name: [] for name, _ in group_defs}
    # Global mean-pool cosine
    global_cos = []
    n_batches = 0
    with torch.no_grad():
        for Xb, _Wb in loaders["val"]:
            if max_batches is not None and n_batches >= max_batches:
                break
            Xb = torch.nan_to_num(Xb.to(device_), nan=0.0)

            # Path A: aggregates visible (probe-time distribution)
            tokens_full = model.encode_variates(Xb)            # (B, V, D)
            # Path B: aggregates zeroed (training-masked edge case)
            X_zeroed = Xb.clone()
            for pos in agg_positions:
                X_zeroed[:, :, pos] = 0.0
            tokens_zeroed = model.encode_variates(X_zeroed)

            # (a)/(b) per-variate cosine — each token as a (B, D) vector
            for v_pos in range(V):
                cos_v = torch.nn.functional.cosine_similarity(
                    tokens_full[:, v_pos, :],
                    tokens_zeroed[:, v_pos, :],
                    dim=-1,
                )
                per_variate_cos[v_pos].append(cos_v.cpu())

            # (c) per-group mean-pool cosine
            for name, positions in group_defs:
                pooled_full = tokens_full[:, positions, :].mean(dim=1)
                pooled_zeroed = tokens_zeroed[:, positions, :].mean(dim=1)
                cos_g = torch.nn.functional.cosine_similarity(
                    pooled_full, pooled_zeroed, dim=-1
                )
                per_group_cos[name].append(cos_g.cpu())

            # (d) global mean-pool cosine (backward compatibility)
            emb_full = tokens_full.mean(dim=1)
            emb_zeroed = tokens_zeroed.mean(dim=1)
            cos_g = torch.nn.functional.cosine_similarity(
                emb_full, emb_zeroed, dim=-1
            )
            global_cos.append(cos_g.cpu())

            n_batches += 1

    if not global_cos:
        return {"skipped": True, "reason": "no val batches"}

    def _stats(arr):
        if len(arr) == 0:
            return {"n": 0}
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p5": float(np.quantile(arr, 0.05)),
            "p25": float(np.quantile(arr, 0.25)),
            "median": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
            "p95": float(np.quantile(arr, 0.95)),
        }

    # Assemble results
    results = {"skipped": False}
    results["global_mean_pool"] = _stats(torch.cat(global_cos).numpy())
    results["per_variate"] = {}
    for v_pos in range(V):
        arr = torch.cat(per_variate_cos[v_pos]).numpy() if per_variate_cos[v_pos] else np.array([])
        raw_v = ACTIVE_SSL_FEATURES[v_pos]
        results["per_variate"][raw_v] = _stats(arr)
    results["per_group"] = {}
    for name, _positions in group_defs:
        arr = torch.cat(per_group_cos[name]).numpy()
        results["per_group"][name] = _stats(arr)

    # --- Reporting ---
    print("\n" + "=" * 60)
    print("CONCERN #3 EMBEDDING-COSINE SHIFT (per-variate + per-group)")
    print("=" * 60)
    print(f"  batches={n_batches}  "
          f"intervention: zero 18 aggregate positions (v141-v158)")

    # Global (back-compat)
    g = results["global_mean_pool"]
    print(f"\n  global mean-pool (147 tokens):  mean={g['mean']:.4f}  "
          f"p5={g['p5']:.4f}  p95={g['p95']:.4f}")

    # Per-group
    print(f"\n  per-group mean-pool cosine:")
    print(f"  {'group':<14s}{'mean':>8}{'p5':>8}{'p25':>8}{'median':>9}{'p75':>8}")
    for name, _ in group_defs:
        s = results["per_group"][name]
        if s["n"] > 0:
            print(f"  {name:<14s}{s['mean']:>8.4f}{s['p5']:>8.4f}"
                  f"{s['p25']:>8.4f}{s['median']:>9.4f}{s['p75']:>8.4f}")

    # Raw-flow tokens (PRIMARY GATE)
    print(f"\n  per-variate cosine at RAW FLOW positions (v131-v139) "
          f"— primary gate, min={min_flow_token_cosine}:")
    print(f"  {'raw':<6}{'mean':>8}{'p5':>8}{'p95':>8}{'status':>10}")
    flow_fails = []
    for i_flow, v_pos in enumerate(raw_flow_positions):
        raw_v = 131 + i_flow
        s = results["per_variate"][raw_v]
        if s["n"] == 0:
            continue
        status = "OK" if s["mean"] >= min_flow_token_cosine else "FAIL"
        print(f"  v{raw_v:<5}{s['mean']:>8.4f}{s['p5']:>8.4f}{s['p95']:>8.4f}{status:>10}")
        if s["mean"] < min_flow_token_cosine:
            flow_fails.append(f"v{raw_v}: mean={s['mean']:.4f}")

    # Aggregate tokens (SANITY CHECK — should shift substantially)
    print(f"\n  per-variate cosine at AGGREGATE positions (v141-v158) "
          f"— sanity check, expect low cosine (tokens were zeroed):")
    print(f"  {'raw':<6}{'mean':>8}{'p5':>8}{'p95':>8}")
    for raw_v in sorted(ACTIVE_SSL_FEATURES):
        if raw_v not in range(141, 159):
            continue
        s = results["per_variate"][raw_v]
        if s["n"] == 0:
            continue
        print(f"  v{raw_v:<5}{s['mean']:>8.4f}{s['p5']:>8.4f}{s['p95']:>8.4f}")

    # Pass/fail on primary gate (raw-flow tokens)
    fails = []
    if g["mean"] < min_mean_cosine:
        fails.append(f"global mean={g['mean']:.4f} < {min_mean_cosine} "
                     f"(backup gate)")
    if g["p5"] < min_p5_cosine:
        fails.append(f"global p5={g['p5']:.4f} < {min_p5_cosine} (backup gate)")
    if flow_fails:
        fails.append("raw-flow per-variate below threshold: "
                     + ", ".join(flow_fails))

    if fails:
        msg = ("Concern #3 diagnostic FAILED — " + "; ".join(fails)
               + ". Encoder is heavily dependent on aggregate values on "
               "raw-flow tokens; probe-time embeddings diverge from "
               "training distribution. Mitigations: add mask-indicator "
               "channel, reduce variate_ratio (concern #1), or fine-tune "
               "probes on a closer distribution.")
        print("\n  [FAIL] " + msg)
        results["passed"] = False
        if assert_pass:
            raise AssertionError(msg)
    else:
        print("\n  [PASS] raw-flow tokens robust to aggregate absence")
        results["passed"] = True

    return results


# ===================================================================
# 10D. V3 ATTENTION-ENTROPY DIAGNOSTIC (capacity dilution detector)
# ===================================================================
# MQA threat model A: v3 adds 18 aggregate tokens to a 147-token layout.
# Same d_model=128, same 3 layers — each token's share of the attention
# budget is now 129/147 = 88% of v2's. The encoder could spend capacity
# on aggregate tokens that carry only redundant (linear) information and
# degrade representations on the more-useful raw-flow tokens.
#
# Diagnostic: measure the average fraction of attention mass each variate
# GROUP receives across query positions. If groups receive ≈ proportional
# to their size, attention is uniform (baseline). If aggregate group
# receives << its proportional share, encoder is ignoring them (new
# tokens = dead weight). If aggregate group receives >> its share,
# encoder prefers them (they carry signal).
#
# No asserted thresholds — this is explanatory telemetry to interpret
# probe results. Expected behaviors:
# PASS-like: aggregates receive ≥ proportional share (12.2%) — signal.
# INFORM: aggregates receive 0.5–1.0× proportional — uniform, no
# preference; may or may not help probes.
# WARN: aggregates receive < 0.5× proportional — effectively
# unused; explains if v3 underperforms v2.


def compute_attention_entropy_diagnostic(
    model, loaders, device_,
    max_batches=20,
):
    """Attention-mass distribution across variate groups (last encoder layer).

    For each val batch, runs the encoder, captures per-layer (B, V, V)
    attention weights (softmax over keys per query, averaged over heads
    by nn.MultiheadAttention). Aggregates to a (V,) vector of "attention
    received" — each variate's mean column in the (V, V) attention matrix,
    averaged over queries and batches.

    Reports:
      - Per-group attention-received rate (sum within group / total).
      - Per-group over/under-sampling ratio vs uniform (group_size/V).
      - Per-layer breakdown (layer 1 / 2 / last).

    Returns dict; no threshold assertion.
    """
    if not USE_V3_FEATURES:
        print("  [attention-entropy diagnostic: SKIPPED — v2 layout]")
        return {"skipped": True, "reason": "v2 layout"}

    V = ACTIVE_N_SSL_FEATURES
    group_defs = _v3_group_ranges_in_ssl_positions()

    model.eval()
    # Per-layer (V,) attention-received accumulators
    layer_recv = None
    n_obs = 0
    n_layers = None
    with torch.no_grad():
        for batch_idx, (Xb, _Wb) in enumerate(loaders["val"]):
            if batch_idx >= max_batches:
                break
            Xb = torch.nan_to_num(Xb.to(device_), nan=0.0)
            _tokens, attn_list = model.encode_variates(Xb, return_attn=True)
            if n_layers is None:
                n_layers = len(attn_list)
                layer_recv = [torch.zeros(V, dtype=torch.float64)
                              for _ in range(n_layers)]
            for li, w in enumerate(attn_list):
                # w: (B, V, V). Rows are queries — each row sums to 1.
                # Column sum over queries = total attention received per key.
                # Normalize by V (number of queries) → mean-received per query.
                received = w.mean(dim=1)       # (B, V)
                received = received.mean(dim=0)  # (V,)
                layer_recv[li] += received.cpu().double()
            n_obs += 1

    if n_obs == 0:
        return {"skipped": True, "reason": "no val batches"}

    uniform_rate = 1.0 / V
    results = {
        "skipped": False,
        "n_batches": int(n_obs),
        "n_layers": int(n_layers),
        "uniform_rate": float(uniform_rate),
        "per_layer": {},
    }

    print("\n" + "=" * 60)
    print("ATTENTION-ENTROPY DIAGNOSTIC (capacity dilution detector)")
    print("=" * 60)
    print(f"  batches={n_obs}  layers={n_layers}  V={V}  "
          f"uniform rate = 1/V = {uniform_rate:.4f}")
    print(f"  group proportional share = group_size / V")
    print(f"  attention_ratio = group_received / proportional_share "
          f"(1.0 = uniform, >1 = preferred, <1 = neglected)")

    for li in range(n_layers):
        recv = (layer_recv[li] / n_obs).numpy()   # (V,) per-query mean
        per_group = {}
        for name, positions in group_defs:
            group_recv = float(recv[positions].sum())
            group_size = len(positions)
            share = group_size / V
            ratio = group_recv / max(share, 1e-12)
            per_group[name] = {
                "group_size": group_size,
                "share": share,
                "received": group_recv,
                "ratio_vs_uniform": ratio,
            }
        results["per_layer"][f"layer_{li}"] = per_group

    # Print last layer (most informative for downstream probes)
    last_li = n_layers - 1
    last = results["per_layer"][f"layer_{last_li}"]
    print(f"\n  Layer {last_li} (final) attention received by group:")
    print(f"  {'group':<14s}{'size':>6}{'share':>8}{'received':>10}"
          f"{'ratio':>8}{'status':>10}")
    for name, _ in group_defs:
        d = last[name]
        if d["ratio_vs_uniform"] >= 1.0:
            status = "preferred"
        elif d["ratio_vs_uniform"] >= 0.5:
            status = "uniform"
        else:
            status = "neglected"
        print(f"  {name:<14s}{d['group_size']:>6d}{d['share']:>8.4f}"
              f"{d['received']:>10.4f}{d['ratio_vs_uniform']:>8.2f}"
              f"{status:>10}")

    # Flow-aggregate callout
    agg_ratios = []
    for name in ("flow_roll3", "flow_roll15"):
        if name in last:
            agg_ratios.append(last[name]["ratio_vs_uniform"])
    if agg_ratios:
        avg_agg = sum(agg_ratios) / len(agg_ratios)
        if avg_agg < 0.5:
            verdict = ("flow aggregates appear NEGLECTED — encoder may not "
                       "be extracting signal from them. Check probe deltas "
                       "vs SSL-004 baseline to confirm whether aggregates "
                       "are dead weight.")
        elif avg_agg > 1.3:
            verdict = ("flow aggregates attract disproportionate attention — "
                       "encoder is actively relying on them.")
        else:
            verdict = "flow aggregates receive near-uniform attention."
        print(f"\n  flow aggregate avg ratio (roll3+roll15)/2 = "
              f"{avg_agg:.2f}  →  {verdict}")

    return results


# ===================================================================
# 11. SAVE ARTIFACTS
# ===================================================================


def save_ssl_artifacts(qb, model, recon_head, z_stats, history):
    """Save SSL encoder, reconstruction head, z-score stats, and history."""
    buf = io.BytesIO()
    torch.save({
        "encoder_state_dict": model.state_dict(),
        "recon_head_state_dict": recon_head.state_dict(),
        "ssl_hyperparams": SSL_HYPERPARAMS,
        "ssl_features": ACTIVE_SSL_FEATURES,
        "n_ssl_features": ACTIVE_N_SSL_FEATURES,
        "use_v3_features": USE_V3_FEATURES,
        "z_mean": z_stats["mean"],
        "z_std": z_stats["std"],
        "log1p_cols": z_stats.get("log1p_cols", []),              # -b: positional indices in ACTIVE_SSL_FEATURES
        "log1p_raw_variates": sorted(SSL_LOG1P_VARIATES),      # -b: raw 141-layout variate indices
        "min_std_floor": z_stats.get("min_std_floor", 0.01),   # -a
    }, buf)
    qb.ObjectStore.Save(ACTIVE_CKPT_KEY,
                        base64.b64encode(buf.getvalue()).decode())
    qb.ObjectStore.Save(ACTIVE_HISTORY_KEY, json.dumps(history))
    print(f"Saved to ObjectStore: {ACTIVE_CKPT_KEY}, {ACTIVE_HISTORY_KEY}")


# ===================================================================
# 12. MAIN
# ===================================================================


def run_ssl_pipeline():
    """D71 full SSL pretraining pipeline.

    Loads 129-variate SSL data with C10 mask weights, trains
    iTransformerEncoder via masked variate reconstruction.

    Usage (in QC Research notebook):
        run_ssl_pipeline()
    """
    global _results
    qb = get_qb()

    print("=" * 60)
    print("D71 SSL PRETRAINING")
    print("=" * 60)

    # Step 0: Validate config before anything else
    _validate_config_vs_baseline(SSL_HYPERPARAMS)

    # Step 0b: Clear stale probe checkpoints from previous experiments.
    # A new training run invalidates ALL prior training and probe artifacts.
    # ObjectStore persists across sessions — stale keys from a previous
    # experiment will contaminate resume, probes, and evaluation.
    # This was the root cause of L1-SSL-007 probe contamination and
    # L1-SSL-008 wrong-checkpoint resume.
    # Clean keys for the ACTIVE version only. Do NOT touch the other
    # version's keys — an in-flight v2 baseline run must not be clobbered
    # by starting a v3 experiment (or vice versa).
    stale_keys = (
        ACTIVE_CKPT_KEY,
        f"ssl_resume{ACTIVE_CKPT_SUFFIX}",
        ACTIVE_HISTORY_KEY,
        f"ssl_model_temporal_best{ACTIVE_CKPT_SUFFIX}",
        f"probe_results_partial{ACTIVE_CKPT_SUFFIX}",
        f"probe_results{ACTIVE_CKPT_SUFFIX}",
    )
    for stale_key in stale_keys:
        if qb.ObjectStore.ContainsKey(stale_key):
            qb.ObjectStore.Delete(stale_key)
            print(f"  Deleted stale ObjectStore key: {stale_key}")

    # Step 1: Load
    print("\n=== Step 1: Load SSL data ===")
    if "ssl_data" in _results:
        print("Reusing cached SSL data")
        ssl_data = _results["ssl_data"]
    else:
        ssl_data = load_ssl_days(qb, TRAIN_START, TEST_END)
        _results["ssl_data"] = ssl_data

    # Step 2: Split + normalize
    print("\n=== Step 2: Split and normalize ===")
    loaders = make_ssl_split_and_loaders(ssl_data)
    _results["ssl_loaders"] = loaders

    # Step 3: Train
    print("\n=== Step 3: SSL pretraining ===")
    model, recon_head, history = train_ssl(loaders)
    _results["ssl_model"] = model
    _results["ssl_recon_head"] = recon_head
    _results["ssl_history"] = history

    # Step 4: Save
    print("\n=== Step 4: Save artifacts ===")
    save_ssl_artifacts(qb, model, recon_head, loaders["z_stats"], history)

    # Step 5: Reconstruction baselines
    print("\n=== Step 5: Reconstruction baselines ===")
    run_ssl_baselines(loaders, model, recon_head)

    # Step 6: Per-group diagnostics
    print("\n=== Step 6: Per-group baseline diagnostics ===")
    run_ssl_baseline_diagnostics(loaders, model, recon_head)

    # Step 7 (v3 only): MQA.6 masking-artifact diagnostic — asserted gate.
    # No-op on v2 (no block-masked aggregates to probe).
    print("\n=== Step 7: Masking-artifact diagnostic ===")
    diag = compute_masking_artifact_diagnostic(
        model, recon_head, loaders, device,
        assert_pass=False,  # log-only in pipeline; caller can tighten
    )
    _results["masking_artifact_diagnostic"] = diag

    # Step 8 (v3 only): Concern #3 embedding-cosine shift.
    # Tests whether encoder is robust to aggregate absence (train/probe
    # distribution mismatch). Log-only on first run.
    print("\n=== Step 8: Embedding-cosine shift diagnostic ===")
    cosine_diag = compute_embedding_cosine_shift(
        model, loaders, device,
        assert_pass=False,
    )
    _results["embedding_cosine_diagnostic"] = cosine_diag

    # Step 9 (v3 only): Attention-entropy / capacity dilution detector.
    # MQA threat model A — v3's 18 extra tokens may steal attention
    # budget from useful raw-flow tokens without providing new signal.
    # Pure telemetry (no threshold asserts); explains probe deltas.
    print("\n=== Step 9: Attention-entropy diagnostic ===")
    attn_diag = compute_attention_entropy_diagnostic(
        model, loaders, device, max_batches=20,
    )
    _results["attention_entropy_diagnostic"] = attn_diag

    # Step 10: Collect and persist structured metrics
    print("\n=== Step 10: Persist structured metrics ===")
    n_params = sum(p_.numel() for p_ in model.parameters())
    tok = SSL_HYPERPARAMS.get("tokenizer", "linear")
    metrics = collect_training_metrics(history, SSL_HYPERPARAMS, n_params, tokenizer=tok)
    _results["training_metrics"] = metrics
    all_pass = metrics.get("sanity_gates", {}).get("all_pass", False)
    print(f"  Sanity gates: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    print("\nSSL pretraining complete!")
    return _results


def run_baselines_only():
    """Run reconstruction baselines on saved Run 2 checkpoint (no retraining).

    Loads the SSL model from ObjectStore, rebuilds the data loaders
    (using saved z-score stats), and evaluates all 6 baselines + transformer.
    Runtime: ~5-10 min (data loading + Ridge fitting + evaluation).
    """
    global _results
    qb = get_qb()

    print("=" * 60)
    print("D79 RECONSTRUCTION BASELINES (checkpoint mode)")
    print("=" * 60)

    # Step 1: Load data
    print("\n=== Step 1: Load SSL data ===")
    if "ssl_data" in _results:
        print("Reusing cached SSL data")
        ssl_data = _results["ssl_data"]
    else:
        ssl_data = load_ssl_days(qb, TRAIN_START, TEST_END)
        _results["ssl_data"] = ssl_data

    # Step 2: Split + normalize
    print("\n=== Step 2: Split and normalize ===")
    loaders = make_ssl_split_and_loaders(ssl_data)
    _results["ssl_loaders"] = loaders

    # Step 3: Load model from checkpoint
    print("\n=== Step 3: Load saved model ===")
    if not qb.ObjectStore.ContainsKey(ACTIVE_CKPT_KEY):
        raise RuntimeError(
            f"No SSL checkpoint found at '{ACTIVE_CKPT_KEY}'. "
            f"Run run_ssl_pipeline() first, or flip USE_V3_FEATURES to "
            f"match the checkpoint version.")

    raw = qb.ObjectStore.Read(ACTIVE_CKPT_KEY)
    buf = io.BytesIO(base64.b64decode(raw))
    ckpt = torch.load(buf, map_location=device, weights_only=False)

    p = ckpt.get("ssl_hyperparams", SSL_HYPERPARAMS)
    V = ckpt.get("n_ssl_features", ACTIVE_N_SSL_FEATURES)
    model = iTransformerEncoder(p, n_variates=V)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model = model.to(device)
    model.eval()

    # SSL-012: probe saved state_dict for FiLM keys to handle both legacy and
    # FiLM-equipped checkpoints. Avoids strict load_state_dict crashes when
    # the saved checkpoint contains film_scale/film_shift but the constructor
    # default would build a FiLM-less head.
    _recon_state = ckpt["recon_head_state_dict"]
    recon_head = _build_recon_head_from_hp(
        p, p["d_model"], p["seq_len"], V, ACTIVE_SSL_FEATURES)
    recon_head.load_state_dict(_recon_state)
    recon_head = recon_head.to(device)
    recon_head.eval()

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"  Loaded: {n_params:,} params (group_recon={recon_head.use_group_heads})")
    _results["ssl_model"] = model
    _results["ssl_recon_head"] = recon_head

    # Step 4: Run baselines
    print("\n=== Step 4: Reconstruction baselines ===")
    run_ssl_baselines(loaders, model, recon_head)

    # Step 5: Per-group diagnostics
    print("\n=== Step 5: Per-group baseline diagnostics ===")
    run_ssl_baseline_diagnostics(loaders, model, recon_head)

    print("\nBaseline evaluation complete!")
    return _results


def variate_ratio_sweep(
    ratios=(0.15, 0.20, 0.25, 0.30),
    num_epochs=10,
    seed=0,
):
    """Concern #1 — mask-rate inflation sweep on a short 10-epoch window.

    Runs SSL pretraining at each variate_ratio for num_epochs epochs on
    the same data/split/seed, then measures on each short-trained
    checkpoint:
      - Effective empirical mask rate on raw flow (v131-v139), roll3
        (v141-v149), roll15 (v150-v158), by sampling 50 mask draws from
        sample_ssl_mask at val-time statistics.
      - Final val_recon_mse.
      - MQA.6 shrinkage Spearman and MSE ratio on the short checkpoint.
      - Embedding-cosine on raw flow tokens.

    Emits a 4-row (N ratios) comparison table. Cheaper than running a
    full 100-epoch config four times — directional signal that tells
    you which variate_ratio is worth a full training run.

    IMPORTANT: each sweep trains from scratch. Total runtime is
    4 × ~10% of one full run ≈ 40% of one full training. Still cheaper
    than picking wrong and discovering after the fact.

    Args:
        ratios: iterable of variate_ratio values to test.
        num_epochs: epochs per sweep config (short — just directional).
        seed: torch.manual_seed value, same across all runs for
              apples-to-apples comparison.

    Returns:
        list of dicts, one per ratio, with the metrics above. Also
        prints a comparison table.
    """
    if not USE_V3_FEATURES:
        print("variate_ratio_sweep requires USE_V3_FEATURES=True — "
              "sweep is about mask-rate inflation on flow aggregates.")
        return []

    global _results
    qb = get_qb()
    original_hp = dict(SSL_HYPERPARAMS)   # shallow copy for restore
    original_ckpt_key = ACTIVE_CKPT_KEY
    sweep_results = []

    print("=" * 60)
    print(f"VARIATE_RATIO SWEEP ({len(ratios)} configs × {num_epochs} epochs)")
    print("=" * 60)

    # Load data once, reuse across ratios
    print("\n=== Loading SSL data (shared across sweep) ===")
    if "ssl_data" not in _results:
        _results["ssl_data"] = load_ssl_days(qb, TRAIN_START, TEST_END)
    ssl_data = _results["ssl_data"]
    loaders = make_ssl_split_and_loaders(ssl_data)

    for i, r in enumerate(ratios):
        print("\n" + "-" * 60)
        print(f"SWEEP {i+1}/{len(ratios)}: variate_ratio={r}")
        print("-" * 60)

        # Patch hyperparams for this run
        SSL_HYPERPARAMS["variate_ratio"] = r
        SSL_HYPERPARAMS["num_epochs"] = num_epochs
        torch.manual_seed(seed)

        # Fresh model + head each time (true sweep, not resume)
        model = iTransformerEncoder(SSL_HYPERPARAMS,
                                     n_variates=ACTIVE_N_SSL_FEATURES).to(device)
        recon_head = VariateReconstructionHead(
            SSL_HYPERPARAMS["d_model"], SSL_HYPERPARAMS["seq_len"],
            n_variates=ACTIVE_N_SSL_FEATURES,
            use_film=SSL_HYPERPARAMS.get("use_film_recon", False),
        ).to(device)

        # Short train (disable_checkpoints=True → in-memory only, does
        # NOT clobber the real ACTIVE_CKPT_KEY checkpoint from the
        # preceding full training run)
        model, recon_head, _hist = train_ssl(
            loaders, model=model, recon_head=recon_head,
            start_epoch=1, best_val_loss=float("inf"), history=None,
            disable_checkpoints=True,
        )

        # Measure empirical mask rates
        flow_positions = _build_flow_agg_position_map(ACTIVE_SSL_FEATURES)
        rv_positions = (_build_rv_position_map(ACTIVE_SSL_FEATURES)
                        if SSL_HYPERPARAMS.get("rv_block_mask", False) else None)
        eligible = torch.from_numpy(
            _build_eligible_mask(ACTIVE_SSL_FEATURES, ACTIVE_MASK_INELIGIBLE)
        ).to(device)
        T = SSL_HYPERPARAMS["seq_len"]
        V = ACTIVE_N_SSL_FEATURES
        B_test = 64
        n_draws = 50
        raw_rates = []
        r3_rates = []
        r15_rates = []
        gen = torch.Generator(device=device).manual_seed(42)
        with torch.no_grad():
            for _ in range(n_draws):
                mask, _ = sample_ssl_mask(
                    B_test, T, V, eligible,
                    r, SSL_HYPERPARAMS["cell_ratio"],
                    generator=gen,
                    flow_agg_positions=flow_positions,
                    feature_indices=ACTIVE_SSL_FEATURES,
                    rv_positions=rv_positions,
                )
                raw_positions = flow_positions["raw_positions_py"]
                r3_positions = flow_positions["roll3_positions_py"]
                r15_positions = flow_positions["roll15_positions_py"]
                raw_rates.append(mask[:, :, raw_positions].float().mean().item())
                r3_rates.append(mask[:, :, r3_positions].float().mean().item())
                r15_rates.append(mask[:, :, r15_positions].float().mean().item())

        # End-of-short-run diagnostics
        artifact = compute_masking_artifact_diagnostic(
            model, recon_head, loaders, device,
            assert_pass=False, max_batches=5,
        )
        cosine = compute_embedding_cosine_shift(
            model, loaders, device,
            assert_pass=False, max_batches=5,
        )

        result = {
            "variate_ratio": r,
            "num_epochs": num_epochs,
            "val_recon_mse_final": float(_hist["val_loss"][-1]) if _hist and _hist.get("val_loss") else float("nan"),
            "empirical_mask_rate": {
                "raw_flow": float(np.mean(raw_rates)),
                "roll3":    float(np.mean(r3_rates)),
                "roll15":   float(np.mean(r15_rates)),
            },
            "mqa6_global_spearman": artifact.get("global", {}).get("spearman"),
            "mqa6_global_pearson": artifact.get("global", {}).get("pearson"),
            "cosine_global_mean": cosine.get("global_mean_pool", {}).get("mean"),
        }
        # Flow-token per-variate cosine (avg across v131-v139)
        per_var = cosine.get("per_variate", {})
        flow_cosines = [per_var[v]["mean"] for v in range(131, 140)
                        if v in per_var and per_var[v].get("n", 0) > 0]
        if flow_cosines:
            result["cosine_flow_tokens_mean"] = float(np.mean(flow_cosines))
        sweep_results.append(result)

    # Restore original hyperparameters
    SSL_HYPERPARAMS.clear()
    SSL_HYPERPARAMS.update(original_hp)

    # Print comparison table
    print("\n" + "=" * 60)
    print("VARIATE_RATIO SWEEP RESULTS")
    print("=" * 60)
    print(f"  {'ratio':>6} {'val_mse':>9} {'raw%':>7} {'r3%':>7} {'r15%':>7} "
          f"{'mqa_sp':>8} {'cos_flw':>8}")
    for r in sweep_results:
        er = r["empirical_mask_rate"]
        print(f"  {r['variate_ratio']:>6.2f} "
              f"{r['val_recon_mse_final']:>9.4f} "
              f"{er['raw_flow']*100:>7.1f} "
              f"{er['roll3']*100:>7.1f} "
              f"{er['roll15']*100:>7.1f} "
              f"{r['mqa6_global_spearman'] or 0:>8.3f} "
              f"{r.get('cosine_flow_tokens_mean') or 0:>8.3f}")
    print("\n  raw%/r3%/r15% = empirical mask rate on raw flow / roll3 / roll15")
    print("  mqa_sp   = global Spearman (|true|, MSE) — shrinkage prior, lower is better")
    print("  cos_flw  = per-variate cosine on raw flow tokens — higher is better")

    _results["variate_ratio_sweep"] = sweep_results
    return sweep_results


def run_v3_diagnostics(assert_pass=False, max_batches=None):
    """Post-hoc v3 diagnostics against the saved ACTIVE_CKPT_KEY checkpoint.

    Runs the MQA.6 masking-artifact diagnostic (Spearman + quartile
    cohorts) and the Concern #3 embedding-cosine diagnostic (per-variate
    + per-group) on the saved encoder. Reuses loaders in _results if
    present, else rebuilds them from TRAIN_START..TEST_END. Prints
    results and stores them under _results["v3_diagnostics"].

    Use this to:
      - Re-run tightened diagnostics without re-training
      - Compare two checkpoints (flip USE_V3_FEATURES + re-run)
      - Calibrate thresholds before flipping assert_pass=True in the
        main pipeline

    Args:
        assert_pass: if True, raise on threshold violation. Default
            False (log-only) until v2 reference readings calibrate the
            thresholds.
        max_batches: cap val batches for speed during iteration. None
            (default) = full val pass.

    Returns dict with both diagnostic results.
    """
    global _results
    qb = get_qb()

    print("=" * 60)
    print(f"V3 POST-RUN DIAGNOSTICS — checkpoint: {ACTIVE_CKPT_KEY}")
    print("=" * 60)

    if not qb.ObjectStore.ContainsKey(ACTIVE_CKPT_KEY):
        raise RuntimeError(
            f"No checkpoint at '{ACTIVE_CKPT_KEY}'. Run run_ssl_pipeline() "
            f"first, or toggle USE_V3_FEATURES to match an existing "
            f"checkpoint.")

    # Load or reuse loaders
    if "ssl_loaders" in _results:
        print("Reusing cached loaders")
        loaders = _results["ssl_loaders"]
    else:
        if "ssl_data" in _results:
            ssl_data = _results["ssl_data"]
            print("Reusing cached SSL data")
        else:
            print("\n=== Loading SSL data ===")
            ssl_data = load_ssl_days(qb, TRAIN_START, TEST_END)
            _results["ssl_data"] = ssl_data
        print("\n=== Splitting and normalizing ===")
        loaders = make_ssl_split_and_loaders(ssl_data)
        _results["ssl_loaders"] = loaders

    # Load checkpoint
    print(f"\n=== Loading checkpoint ({ACTIVE_CKPT_KEY}) ===")
    raw = qb.ObjectStore.Read(ACTIVE_CKPT_KEY)
    buf = io.BytesIO(base64.b64decode(raw))
    ckpt = torch.load(buf, map_location=device, weights_only=False)
    p = ckpt.get("ssl_hyperparams", SSL_HYPERPARAMS)
    V = ckpt.get("n_ssl_features", ACTIVE_N_SSL_FEATURES)
    model = iTransformerEncoder(p, n_variates=V)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model = model.to(device).eval()
    _recon_state = ckpt["recon_head_state_dict"]
    recon_head = _build_recon_head_from_hp(
        p, p["d_model"], p["seq_len"], V, ACTIVE_SSL_FEATURES)
    recon_head.load_state_dict(_recon_state)
    recon_head = recon_head.to(device).eval()

    print(f"  Loaded: epoch={ckpt.get('epoch', '?')} "
          f"val_loss={ckpt.get('val_loss', '?')}  V={V}  "
          f"use_v3={ckpt.get('use_v3_features', '?')}")

    out = {"checkpoint_key": ACTIVE_CKPT_KEY, "epoch": ckpt.get("epoch")}

    print("\n=== MQA.6 masking-artifact diagnostic ===")
    out["masking_artifact"] = compute_masking_artifact_diagnostic(
        model, recon_head, loaders, device,
        assert_pass=assert_pass,
        max_batches=max_batches,
    )

    print("\n=== Concern #3 embedding-cosine shift ===")
    out["embedding_cosine"] = compute_embedding_cosine_shift(
        model, loaders, device,
        assert_pass=assert_pass,
        max_batches=max_batches,
    )

    print("\n=== Attention-entropy (capacity dilution) ===")
    attn_max_batches = max_batches if max_batches else 20
    out["attention_entropy"] = compute_attention_entropy_diagnostic(
        model, loaders, device,
        max_batches=attn_max_batches,
    )

    _results["v3_diagnostics"] = out
    print("\nv3 diagnostics complete.")
    return out


def _np_safe_default(o):
    """JSON encoder default for numpy types — catches the np.bool_ /
    np.float* / np.integer / np.ndarray cases that raw json.dumps rejects.

    Bug fix 2026-04-29 evening: the linear-arm Azure run printed
    `[checkpoint save failed: Object of type bool_ is not JSON serializable]`
    after every probe because `r["reportable"]` from the Bouthillier gate is
    a numpy.bool_ (from `np.isnan(...)` propagating through the boolean
    expression chain). The save was caught by a generic try/except that
    only printed a warning — net effect: probe_results_partial / probe_results
    JSON were NEVER written, only the stdout log. This default coerces
    numpy scalars to their Python equivalents.

    Why static review missed it: unit tests pass Python bools through
    fold_results fixtures; np.isnan return type only surfaces with real
    numpy fold-mean inputs. Round 2 verification agent traced logic, not
    runtime types. Mac smoke runs `--no-probes` so the save path was never
    exercised locally before the Azure run.
    """
    import numpy as _np
    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_probe_artifacts(qb, all_results):
    """Save probe results to ObjectStore for crash recovery.

    Bug fix 2026-04-29: now persists per-fold values via fold_results, NOT just
    summary stats. Without this, paired Wilcoxon cross-experiment testing —
    the project's pre-registered cross-experiment statistical method per the
    v15 pre-reg amendment lineage — is impossible. Per-fold values are
    extracted as primitive (int/float/str) only, dropping any tensor/array
    fields to keep the JSON small and serialisable.

    Bug fix 2026-04-29 evening: json.dumps now uses `_np_safe_default` to
    coerce numpy scalars (np.bool_, np.floating, np.integer, np.ndarray) so
    the file actually persists. Previously the save raised silently inside
    a generic try/except in `run_all_probes._save_probe_checkpoint`, leaving
    no JSON on disk.
    """
    serializable = {}
    for name, r in all_results.items():
        # Strip per-fold dicts to primitives. fold_results structure:
        # dict[input_type → list[dict_per_fold or None]] where each fold dict
        # contains primary_metric (R² or balanced_accuracy), n_train, n_test,
        # and optionally alpha (for ridge). We keep all primitive scalars.
        fold_results_serializable = {}
        for input_type, fold_list in r.get("fold_results", {}).items():
            fold_results_serializable[input_type] = [
                ({k: v for k, v in fold.items() if isinstance(v, (int, float, str, bool))}
                 if fold is not None else None)
                for fold in fold_list
            ]
        serializable[name] = dict(
            probe_name=r["probe_name"],
            probe_type=r["probe_type"],
            primary_metric=r["primary_metric"],
            summary=r["summary"],
            fold_results=fold_results_serializable,
            reportable=r["reportable"],
            is_expected_null=r.get("is_expected_null", False),
        )
    probe_key = f"probe_results{ACTIVE_CKPT_SUFFIX}"
    qb.ObjectStore.Save(probe_key, json.dumps(serializable, default=_np_safe_default))
    print(f"Saved {probe_key} to ObjectStore ({len(serializable)} probes, per-fold persisted)")


def run_all_probes(include_expected_null=True):
    """Run §11.3 linear probes on frozen encoder embeddings.

    Complete pipeline:
    1. Load probe data (SSL windows + forward targets + regime labels)
    2. Compute probe targets (log RV, validity masks)
    3. Encode windows through frozen SSL encoder (full, grid-only, shuffled)
    4. Generate walk-forward folds with embargo
    5. Run scoring probes FIRST (log_rv_15, log_rv_30, regime, spread_D) so
       the decision-metric results land in probe_results_partial before any
       crash-prone null probes run.
    6. (Optional) Run ret_{5,15,30} leakage-guard null probes last.
    7. Print comparison table and reportability verdicts.

    Args:
        include_expected_null: if True, also run ret_{5,15,30} expected-null
            probes after the scoring probes. Default False because these
            probes were burning memory on each run (n_classes × 5 folds ×
            3 seeds × 6 input types) and the D86 gates don't depend on
            them. To check for leakage, set True periodically — they run
            last so a mid-run crash doesn't kill scoring results.

    Usage (in QC Research notebook):
        run_all_probes()                            # scoring probes only
        run_all_probes(include_expected_null=True)  # also run leakage guards
        run_scoring_probes()                        # alias for default
    """
    global _results
    qb = get_qb()
    ph = PROBE_HYPERPARAMS
    seeds = list(range(ph["n_seeds"]))

    print("=" * 60)
    print("§11.3 LINEAR PROBES ON FROZEN ENCODER")
    print("=" * 60)
    # TRANSDUCTIVE DISCLOSURE (task #164, 2026-04-29):
    # This encoder was trained via SSL on the FULL corpus (all days, all
    # windows) — including windows that appear in the walk-forward probe
    # test folds. The encoder has processed test-fold inputs during
    # pretraining (not labels, but input distributions). Consequence:
    # - Probe R² measures representation quality under distributional
    # familiarity, NOT generalization to truly unseen data.
    # - For cross-arm tokenizer comparison (linear vs patch vs CNN),
    # this transductive exposure is UNIFORM across arms — paired
    # deltas cancel the bias.
    # - Absolute "encoder beats baseline" claims require this
    # disclosure in any dissertation chapter or publication.
    # - A future inductive-only re-run (train/test corpus split at
    # the SSL level) would validate generalization explicitly.
    print("\n  TRANSDUCTIVE DISCLOSURE: encoder was trained on the FULL")
    print("  corpus including test-fold windows. Probe R² reflects")
    print("  representation quality under distributional familiarity.")
    print("  Cross-arm paired deltas are unbiased (uniform exposure).")
    print("  Absolute claims require this caveat in write-up.\n")

    # Clear stale partial checkpoint so we always start fresh.
    # resume_probes() is the only consumer; a fresh run_all_probes()
    # should never inherit results from a previous (possibly different-model) run.
    partial_key = f"probe_results_partial{ACTIVE_CKPT_SUFFIX}"
    if qb.ObjectStore.ContainsKey(partial_key):
        qb.ObjectStore.Delete(partial_key)
        print(f"  Cleared stale {partial_key} checkpoint")

    # Step 1: Load probe data
    print("\n=== Step 1: Load probe data ===")
    if "probe_data" in _results:
        print("Reusing cached probe data")
        probe_data = _results["probe_data"]
    else:
        probe_data = load_probe_days(qb, TRAIN_START, TEST_END)
        _results["probe_data"] = probe_data

    # Step 2: Compute probe targets
    print("\n=== Step 2: Compute probe targets ===")
    probe_targets = compute_probe_targets(probe_data)
    _results["probe_targets"] = probe_targets

    # Step 3: Encode windows
    print("\n=== Step 3: Encode windows (frozen encoder) ===")
    if "probe_encodings" in _results:
        print("Reusing cached encodings")
        encodings = _results["probe_encodings"]
    else:
        encodings = encode_windows(qb, probe_data["X_windows"], seeds=seeds)
        _results["probe_encodings"] = encodings

    # Step 4: Walk-forward folds
    print("\n=== Step 4: Walk-forward folds ===")
    folds = make_probe_folds(probe_data)
    _results["probe_folds"] = folds

    # Step 5: Run all probes
    print("\n=== Step 5: Run probes ===")
    all_results = {}
    _probe_health = {"checkpoint_save_failures": [], "fold_fit_failures": 0}

    def _save_probe_checkpoint(results_so_far):
        """Incremental save after each probe — survives kernel crashes.

        N3 fix (2026-04-29): failures are now tracked in _probe_health and
        printed LOUDLY at end of run_all_probes. Previously the generic
        except silently swallowed errors (same blind-spot pattern as the
        np.bool_ bug that cost the linear-arm run's probe persistence).
        """
        try:
            serializable = {}
            for pname, pr in results_so_far.items():
                # Bug 3 fix (proactive hunt): include fold_results in
                # incremental checkpoint so crash recovery preserves per-fold
                # data for paired Wilcoxon (task #202).
                fr_ser = {}
                for inp_t, fl in pr.get("fold_results", {}).items():
                    fr_ser[inp_t] = [
                        ({k: v for k, v in f.items()
                          if isinstance(v, (int, float, str, bool))}
                         if f is not None else None)
                        for f in fl
                    ]
                serializable[pname] = {
                    "probe_name": pr["probe_name"],
                    "probe_type": pr["probe_type"],
                    "primary_metric": pr["primary_metric"],
                    "summary": pr["summary"],
                    "reportable": pr["reportable"],
                    "fold_results": fr_ser,
                    "is_expected_null": pr.get("is_expected_null", False),
                }
            qb.ObjectStore.Save(partial_key, json.dumps(serializable, default=_np_safe_default))
            print(f"    [checkpoint: {len(results_so_far)} probes saved to {partial_key}]")
        except Exception as e:
            _probe_health["checkpoint_save_failures"].append(
                f"{list(results_so_far.keys())[-1]}: {type(e).__name__}: {e}")
            print(f"    [CHECKPOINT SAVE FAILED: {e}]")

    # Run SCORING probes FIRST so a mid-run crash preserves decision-metric
    # results in probe_results_partial. Expected-null (ret_*) probes run
    # last and only if include_expected_null=True.

    # Probe B: forward realized vol (regression, 2 horizons) — gate
    for k in ph["probe_b_horizons"]:
        name = f"log_rv_{k}"
        r = run_single_probe(name, probe_data, probe_targets, folds, encodings, seeds)
        all_results[name] = r
        _save_probe_checkpoint(all_results)

    # Probe C: future regime (classification) — gate
    r = run_single_probe("regime", probe_data, probe_targets, folds, encodings, seeds)
    all_results["regime"] = r
    _save_probe_checkpoint(all_results)

    # Probe C2: concurrent regime (classification) — Objective 1 current-state test
    if "regime_concurrent" in probe_targets["targets"]:
        r = run_single_probe("regime_concurrent", probe_data, probe_targets, folds, encodings, seeds)
        all_results["regime_concurrent"] = r
        _save_probe_checkpoint(all_results)

    # Probe D: ATM spread delta (regression) — PRIMARY gate
    name = f"spread_{ph['probe_d_horizon']}"
    r = run_single_probe(name, probe_data, probe_targets, folds, encodings, seeds)
    all_results[name] = r
    _save_probe_checkpoint(all_results)

    # Probe A: forward return direction (classification, 3 horizons) —
    # expected null, leakage-guard only, not promoted by gates.
    if include_expected_null:
        for k in ph["probe_a_horizons"]:
            name = f"ret_{k}"
            r = run_single_probe(name, probe_data, probe_targets, folds, encodings, seeds)
            all_results[name] = r
            _save_probe_checkpoint(all_results)
    else:
        print("\n[skipped ret_{5,15,30} — run with include_expected_null=True to include them]")

    _results["probe_results"] = all_results

    # Step 6: Summary comparison table
    print("\n" + "=" * 70)
    print("PROBE RESULTS SUMMARY")
    print("=" * 70)
    header = (f"{'Probe':12s} {'Type':4s} {'emb_grp':>8s} {'emb_fine':>8s} "
              f"{'emb_full':>8s} {'emb_max':>8s} {'raw_36':>8s} {'raw_129':>8s} "
              f"{'shuffle':>8s} {'mlp_grp':>8s} {'Report':>7s}")
    print(header)
    print("-" * len(header))

    for name, r in all_results.items():
        short_type = "cls" if r["probe_type"] == "classification" else "reg"
        s = r["summary"]

        def _fmt(key):
            v = s.get(key, {}).get("mean", float("nan"))
            return f"{v:8.4f}" if not np.isnan(v) else "     N/A"

        if r["reportable"] is None:
            report = "NULL"
        elif r["reportable"]:
            report = "YES"
        else:
            report = "NO"
        print(f"{name:12s} {short_type:4s} "
              f"{_fmt('emb_group')} {_fmt('emb_fine')} "
              f"{_fmt('emb_full')} {_fmt('emb_max')} "
              f"{_fmt('raw_36')} {_fmt('raw_129')} "
              f"{_fmt('shuffle_combined')} {_fmt('mlp_group')} {report:>7s}")

    # Reportability: count only non-null probes
    scored = [r for r in all_results.values() if r["reportable"] is not None]
    n_reportable = sum(1 for r in scored if r["reportable"])
    n_scored = len(scored)
    n_null = sum(1 for r in all_results.values() if r["reportable"] is None)
    print(f"\nReportable: {n_reportable}/{n_scored} scored probes beat both baselines"
          f" ({n_null} probes excluded as expected null)")

    # N3 health report (2026-04-29): print LOUD summary of any silent failures
    # so the operator knows probe results may be incomplete.
    if _probe_health["checkpoint_save_failures"]:
        n_fail = len(_probe_health["checkpoint_save_failures"])
        print(f"\n{'!'*60}")
        print(f"  PROBE HEALTH WARNING: {n_fail} checkpoint save(s) FAILED")
        print(f"  Per-fold data may NOT be persisted on disk.")
        for _msg in _probe_health["checkpoint_save_failures"]:
            print(f"    - {_msg}")
        print(f"{'!'*60}")
    if _probe_health["fold_fit_failures"] > 0:
        print(f"  [health] {_probe_health['fold_fit_failures']} individual fold-fit "
              f"failures across all probes (logged per-fold above)")

    # Step 7: Save results to ObjectStore
    save_probe_artifacts(qb, all_results)

    # Step 8: PCA dimensionality diagnostic
    print("\n=== Step 8: PCA dimensionality diagnostic ===")
    _run_pca_spread_diagnostic(probe_data, probe_targets, folds)

    # Step 9: SKIPPED — 84-d transmission probe moved to scripts/run_pls_probe.py
    # The old run_pca_projected_probe() used a globally-fitted basis (target leakage).
    # Per-fold PLS probes are now run independently via:
    # python -m scripts.run_pls_probe --checkpoint <ckpt> --arm <arm>
    print("\n=== Step 9: SKIPPED (transmission probe → scripts/run_pls_probe.py) ===")

    print("\nProbe evaluation complete!")
    return _results


def run_scoring_probes():
    """Alias for run_all_probes(include_expected_null=False).

    Runs ONLY the decision-metric probes (log_rv_15, log_rv_30, regime,
    spread_5). Skips the ret_{5,15,30} expected-null leakage guards
    because they don't contribute to D86 promotability and have been
    the crash hotspot on QC Research (~40% of run-time, large memory
    footprint from high-seed classification probes).
    """
    return run_all_probes(include_expected_null=False)


def resume_probes(include_expected_null=True):
    """Resume probe evaluation from the last checkpoint.

    Loads probe_results_partial from ObjectStore, skips already-completed probes,
    re-encodes windows (unavoidable after kernel restart), and continues from
    the first incomplete probe. Scoring probes (log_rv/regime/spread) run
    BEFORE any expected-null probes, matching run_all_probes order.

    Args:
        include_expected_null: same semantics as run_all_probes. Default True
            (#222 fix 2026-04-29) — aligned with run_all_probes default so
            ret_{5,15,30} leakage-guard probes are NOT silently skipped on
            resume. Pass False explicitly to opt out.

    Usage (in QC Research notebook after kernel restart):
        resume_probes()
    """
    global _results
    qb = get_qb()
    ph = PROBE_HYPERPARAMS
    seeds = list(range(ph["n_seeds"]))

    print("=" * 60)
    print("RESUME PROBES FROM CHECKPOINT")
    print("=" * 60)

    # Load partial results
    partial_key = f"probe_results_partial{ACTIVE_CKPT_SUFFIX}"
    completed = {}
    if qb.ObjectStore.ContainsKey(partial_key):
        completed = json.loads(qb.ObjectStore.Read(partial_key))
        print(f"  Found {len(completed)} completed probes: {list(completed.keys())}")
    else:
        print("  No partial results found — starting from scratch")

    # Rebuild all probe names in order — scoring FIRST, expected-null LAST.
    # Matches run_all_probes ordering so mid-run crashes preserve scoring.
    all_probe_names = []
    for k in ph["probe_b_horizons"]:
        all_probe_names.append(f"log_rv_{k}")
    all_probe_names.append("regime")
    all_probe_names.append("regime_concurrent")
    all_probe_names.append(f"spread_{ph['probe_d_horizon']}")
    if include_expected_null:
        for k in ph["probe_a_horizons"]:
            all_probe_names.append(f"ret_{k}")

    remaining = [n for n in all_probe_names if n not in completed]
    print(f"  Remaining probes: {remaining}")

    if not remaining:
        print("  All probes already complete!")
        _results["probe_results"] = completed
        # Run PCA diagnostic + summary
        probe_data = _results.get("probe_data")
        if probe_data is None:
            probe_data = load_probe_days(qb, TRAIN_START, TEST_END)
            _results["probe_data"] = probe_data
        probe_targets = compute_probe_targets(probe_data)
        folds = make_probe_folds(probe_data)
        print("\n=== PCA dimensionality diagnostic ===")
        _run_pca_spread_diagnostic(probe_data, probe_targets, folds)
        return _results

    # Step 1: Load probe data
    print("\n=== Step 1: Load probe data ===")
    if "probe_data" in _results:
        print("Reusing cached probe data")
        probe_data = _results["probe_data"]
    else:
        probe_data = load_probe_days(qb, TRAIN_START, TEST_END)
        _results["probe_data"] = probe_data

    # Step 2: Compute targets
    print("\n=== Step 2: Compute probe targets ===")
    probe_targets = compute_probe_targets(probe_data)
    _results["probe_targets"] = probe_targets

    # Step 3: Encode windows
    print("\n=== Step 3: Encode windows (frozen encoder) ===")
    if "probe_encodings" in _results:
        print("Reusing cached encodings")
        encodings = _results["probe_encodings"]
    else:
        encodings = encode_windows(qb, probe_data["X_windows"], seeds=seeds)
        _results["probe_encodings"] = encodings

    # Step 4: Folds
    print("\n=== Step 4: Walk-forward folds ===")
    folds = make_probe_folds(probe_data)
    _results["probe_folds"] = folds

    # Step 5: Run remaining probes
    print(f"\n=== Step 5: Run remaining probes ({len(remaining)}) ===")
    all_results = dict(completed)  # start from checkpoint

    def _save_checkpoint(results_so_far):
        try:
            serializable = {}
            for pname, pr in results_so_far.items():
                serializable[pname] = {
                    "probe_name": pr["probe_name"],
                    "probe_type": pr["probe_type"],
                    "primary_metric": pr["primary_metric"],
                    "summary": pr["summary"],
                    "reportable": pr["reportable"],
                }
            qb.ObjectStore.Save(partial_key, json.dumps(serializable, default=_np_safe_default))
            print(f"    [checkpoint: {len(results_so_far)} probes saved to {partial_key}]")
        except Exception as e:
            print(f"    [checkpoint save failed: {e}]")

    for name in remaining:
        r = run_single_probe(name, probe_data, probe_targets, folds, encodings, seeds)
        all_results[name] = r
        _save_checkpoint(all_results)

    _results["probe_results"] = all_results

    # Step 6: Summary table
    print("\n" + "=" * 70)
    print("PROBE RESULTS SUMMARY")
    print("=" * 70)
    header = (f"{'Probe':12s} {'Type':4s} {'emb_grp':>8s} {'emb_fine':>8s} "
              f"{'emb_full':>8s} {'emb_max':>8s} {'raw_36':>8s} {'raw_129':>8s} "
              f"{'shuffle':>8s} {'mlp_grp':>8s} {'Report':>7s}")
    print(header)
    print("-" * len(header))

    for name in all_probe_names:
        if name not in all_results:
            continue
        r = all_results[name]
        short_type = "cls" if r["probe_type"] == "classification" else "reg"
        s = r["summary"]

        def _fmt(key):
            v = s.get(key, {}).get("mean", float("nan"))
            return f"{v:8.4f}" if not np.isnan(v) else "     N/A"

        if r["reportable"] is None:
            report = "NULL"
        elif r["reportable"]:
            report = "YES"
        else:
            report = "NO"
        print(f"{name:12s} {short_type:4s} "
              f"{_fmt('emb_group')} {_fmt('emb_fine')} "
              f"{_fmt('emb_full')} {_fmt('emb_max')} "
              f"{_fmt('raw_36')} {_fmt('raw_129')} "
              f"{_fmt('shuffle_combined')} {_fmt('mlp_group')} {report:>7s}")

    scored = [r for r in all_results.values() if r.get("reportable") is not None]
    n_reportable = sum(1 for r in scored if r["reportable"])
    print(f"\nReportable: {n_reportable}/{len(scored)} scored probes beat both baselines")

    # Step 7: Save final results
    save_probe_artifacts(qb, all_results)

    # Step 8: PCA diagnostic
    print("\n=== Step 8: PCA dimensionality diagnostic ===")
    _run_pca_spread_diagnostic(probe_data, probe_targets, folds)

    print("\nProbe evaluation complete!")
    return _results


def _run_pca_spread_diagnostic(probe_data, probe_targets, folds):
    """Diagnostic: can the spread signal survive dimensional compression?

    Tests three feature scopes to isolate WHERE the signal is lost:
    1. raw_36 (2,160-d) — the 36 leakage-free features that achieve R²=0.44
    2. full 129 variates (7,740-d) — tests whether options grid drowns spread in PCA
    3. spread-cluster (~360-d) — spread variate + nearby ATM microstructure only

    For each scope, compresses via PCA to d=128 and runs the spread probe.
    If raw_36 PCA@128 >> encoder R²=0.087, the signal CAN survive compression
    and the encoder/tokenizer is the bottleneck. If raw_36 PCA@128 ≈ 0.087,
    compression itself is the problem.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV

    X_raw = probe_data["X_windows"].copy()
    if _LOG1P_SSL_POS:
        X_raw[:, :, _LOG1P_SSL_POS] = (
            np.sign(X_raw[:, :, _LOG1P_SSL_POS])
            * np.log1p(np.abs(X_raw[:, :, _LOG1P_SSL_POS]))
        )
    N, T, V = X_raw.shape

    target = probe_targets["targets"]["spread_5"]
    valid = probe_targets["validity"]["spread_5"]

    # Define feature scopes
    # Spread-cluster: ATM grid point (v40-47: mid, iv, theta, gamma, qliq, qimb, spread, spread_chg)
    # + neighboring grid points' spread/spread_chg + strike aggregates
    spread_cluster_ssl = [40, 41, 42, 43, 44, 45, 46, 47,  # ATM (grid point 5)
                          38, 39,  # ATM-put spread, spread_chg
                          48, 49,  # ATM-call grid: mid, iv (context)
                          54, 55,  # ATM-call spread, spread_chg
                          ]

    scopes = {
        "raw_36": ("MODEL_FEATURES (36 variates × 60 bars)", _MF_SSL_POS),
        "spread_cluster": ("ATM spread cluster (14 variates × 60 bars)", spread_cluster_ssl),
        "full_129": ("All 129 variates × 60 bars", list(range(V))),
    }

    print(f"  PCA spread diagnostic: testing 3 feature scopes at d=128")
    print(f"  Encoder baseline: spread R² = 0.087 (emb_group)")
    print(f"  Raw baseline:     spread R² = 0.436 (raw_36 @ 2160-d, no compression)")
    print()

    all_results = {}

    for scope_name, (desc, variate_indices) in scopes.items():
        X_scope = X_raw[:, :, variate_indices].reshape(N, T * len(variate_indices))
        input_dim = X_scope.shape[1]

        scope_results = {}
        for d in [64, 128]:
            fold_r2s = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                train_valid = valid[train_idx]
                test_valid = valid[test_idx]

                X_tr = X_scope[train_idx][train_valid]
                X_te = X_scope[test_idx][test_valid]
                y_tr = target[train_idx][train_valid]
                y_te = target[test_idx][test_valid]

                if len(X_tr) < 100 or len(X_te) < 50:
                    continue

                # Dead zone
                train_std = float(np.std(y_tr[~np.isnan(y_tr)]))
                threshold = 0.1 * train_std
                tr_mask = np.abs(y_tr) >= threshold
                te_mask = np.abs(y_te) >= threshold
                X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
                X_te, y_te = X_te[te_mask], y_te[te_mask]

                if len(X_tr) < 50 or len(X_te) < 20:
                    continue

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                n_comp = min(d, X_tr_s.shape[0], X_tr_s.shape[1])
                pca = PCA(n_components=n_comp)
                X_tr_pca = pca.fit_transform(X_tr_s)
                X_te_pca = pca.transform(X_te_s)

                model = RidgeCV(alphas=np.logspace(-3, 5, 20))  # #216 unified grid (was logspace(-3,3,10), saturating ceiling)
                model.fit(X_tr_pca, y_tr)
                r2 = float(model.score(X_te_pca, y_te))
                fold_r2s.append(r2)

            if fold_r2s:
                mean_r2 = np.mean(fold_r2s)
                std_r2 = np.std(fold_r2s)
                scope_results[d] = {"mean": mean_r2, "std": std_r2, "n_folds": len(fold_r2s)}
                print(f"    {scope_name:15s} ({input_dim:>5d}-d) PCA@{d:>3d}: "
                      f"R² = {mean_r2:>7.4f} ± {std_r2:.4f}")

        all_results[scope_name] = scope_results

    # Interpretation
    print(f"\n  Interpretation:")
    r36_128 = all_results.get("raw_36", {}).get(128, {}).get("mean", float("nan"))
    cluster_128 = all_results.get("spread_cluster", {}).get(128, {}).get("mean", float("nan"))

    if not np.isnan(r36_128):
        if r36_128 >= 0.25:
            print(f"    raw_36 PCA@128 R²={r36_128:.3f} >> encoder R²=0.087")
            print(f"    → The spread signal SURVIVES compression to 128-d.")
            print(f"    → The encoder/tokenizer is the bottleneck, not dimensionality.")
            print(f"    → Tokenizer ablation IS worth pursuing.")
        elif r36_128 >= 0.10:
            print(f"    raw_36 PCA@128 R²={r36_128:.3f} — modest survival.")
            print(f"    → Some spread signal survives but compression hurts.")
            print(f"    → Tokenizer ablation may help marginally.")
        else:
            print(f"    raw_36 PCA@128 R²={r36_128:.3f} ≈ encoder R²=0.087")
            print(f"    → Spread cannot survive ANY compression to 128-d.")
            print(f"    → Tokenizer ablation will NOT help. Bypass is the only option.")

    if not np.isnan(cluster_128):
        if cluster_128 > r36_128 + 0.05 if not np.isnan(r36_128) else False:
            print(f"    Spread cluster PCA@128 R²={cluster_128:.3f} > raw_36 PCA@128")
            print(f"    → Focusing on spread-related variates preserves more signal.")

    _results["pca_spread_diagnostic"] = all_results


def run_pca_projected_probe(probe_data, probe_targets, folds, encodings):
    """DEPRECATED — use scripts/run_pls_probe.py instead.

    This function applies a single globally-fitted basis to all folds, leaking
    PLS target information into test folds. The standalone run_pls_probe.py
    does per-fold PLS fitting (no leakage) with proper baselines (random
    projection, supervised ceiling) and Wilcoxon significance tests.

    Retained for audit trail only. Results from this function should NOT be
    reported as transmission measurements.

    Original docstring:
    PRIMARY PROBE: 84-d PCA-projected scalars (GP-consumption-matched).
    Tests the EXACT representation the GP layer consumes: 7 typed embedding
    vectors projected through the per-group PCA bases (layer2/pca_bases.npz)
    with Mahalanobis normalization (divide by pc_std_ per group). This produces
    84 scalars per window — matching the EmbProj_*_k operators in the grammar.

    Dimensionality-matched baseline: project raw_36 (temporal stats) through
    PCA to 84-d so the comparison is fair w.r.t. feature count.

    Uses the same 8-fold expanding-window walk-forward protocol and Bouthillier
    2021 paired CI + Cohen d_z gates as the existing probe infrastructure.
    """
    import warnings
    warnings.warn(
        "run_pca_projected_probe() is DEPRECATED — uses globally-fitted basis "
        "(target leakage). Use `python -m scripts.run_pls_probe` for per-fold "
        "PLS transmission probes.",
        DeprecationWarning, stacklevel=2,
    )
    from pathlib import Path
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as SkPCA

    print("\n" + "=" * 70)
    print("=== DEPRECATED PROBE — use scripts/run_pls_probe.py instead ===")
    print("=" * 70)

    # --- Load PCA bases from layer2 ---
    bases_path = Path(__file__).resolve().parents[2] / "layer2" / "pca_bases.npz"
    manifest_path = Path(__file__).resolve().parents[2] / "layer2" / "pca_bases_manifest.json"

    if not bases_path.exists():
        print(f"  WARNING: {bases_path} not found. Skipping PCA-projected probe.")
        print(f"  (Re-fit via `python -m layer2.inference.pca_bases` to generate.)")
        return None

    # Import load_bases from layer2
    import sys
    _repo_root = str(Path(__file__).resolve().parents[1])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from layer2.inference.pca_bases import load_bases, project, PER_GROUP_K_V9, EMB_COLS

    try:
        bases = load_bases(path=bases_path, manifest_path=manifest_path)
    except Exception as e:
        print(f"  WARNING: Failed to load PCA bases: {e}")
        print(f"  Skipping PCA-projected probe.")
        return None

    # Verify total K matches 84
    total_k = sum(PER_GROUP_K_V9[col] for col in EMB_COLS)
    assert total_k == 84, f"Expected total_k=84, got {total_k}"
    print(f"  PCA bases loaded: {total_k} total scalars across {len(EMB_COLS)} groups")
    for col in EMB_COLS:
        k = PER_GROUP_K_V9[col]
        print(f"    {col:15s}: K={k}")

    # --- Typed vector SSL position ranges (v3) ---
    # Must match _TYPED_VECTOR_SSL_RANGES_V3 in batch_forecast.py
    typed_vector_ssl_ranges = {
        "emb_grid":     (0, 88),
        "emb_strike":   (88, 94),
        "emb_spx":      (94, 108),
        "emb_vix":      (108, 120),
        "emb_flow_raw": (120, 129),
        "emb_flow_agg": (129, 147),
    }
    # Map lowercase batch_forecast keys -> uppercase EMB_COLS keys
    bf_to_emb = {
        "emb_grid":     "EMB_GRID",
        "emb_strike":   "EMB_STRIKE",
        "emb_spx":      "EMB_SPX",
        "emb_vix":      "EMB_VIX",
        "emb_flow_raw": "EMB_FLOW_RAW",
        "emb_flow_agg": "EMB_FLOW_AGG",
    }

    # --- Extract per-variate tokens via the encoder ---
    # The encodings dict from encode_windows() only has pooled embeddings.
    # We need the raw per-variate tokens (B, V, D*L) to pool into 7 typed vectors.
    # Use the SAME model instance from encodings to guarantee checkpoint consistency.
    print(f"\n  Encoding windows for typed-vector extraction...")

    X_windows = probe_data["X_windows"]
    N = len(X_windows)

    # Reuse model + z-stats from encode_windows (BLOCKER fix: no independent reload)
    model = encodings["model"]
    z_mean = encodings["z_stats"]["mean"]
    z_std = encodings["z_stats"]["std"]
    log1p_cols = encodings["z_stats"].get("log1p_cols", [])
    hp = encodings.get("hp", SSL_HYPERPARAMS)
    n_variates = encodings.get("n_variates", N_SSL_FEATURES)
    d_model = hp["d_model"]
    n_layers = hp.get("n_layers", 3)
    ml_dim = d_model * n_layers  # 384 for multi-layer concat

    # Preprocess
    X = X_windows.copy()
    if log1p_cols:
        X[:, :, log1p_cols] = np.sign(X[:, :, log1p_cols]) * np.log1p(np.abs(X[:, :, log1p_cols]))
    X = ((X - z_mean) / z_std).astype(np.float32)
    X = np.clip(X, -5.0, 5.0)

    # Encode all windows → per-variate tokens (N, V, D*L)
    batch_size = 256
    V = n_variates
    per_variate = np.empty((N, V, ml_dim), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            x_batch = torch.from_numpy(X[s:e]).to(device)
            z_ml = model.encode_variates(x_batch, multi_layer=True)  # (B, V, D*L)
            per_variate[s:e] = z_ml.cpu().numpy()
    print(f"  Per-variate tokens extracted: ({N}, {V}, {ml_dim})")

    # --- Pool into 7 typed vectors (matching batch_forecast._pool_typed_vectors) ---
    typed_vectors = {}
    # EMB_SHARED = mean over ALL variates
    typed_vectors["EMB_SHARED"] = per_variate.mean(axis=1)  # (N, 384)
    # Other 6 groups: mean over their position range
    for bf_key, (gs, ge) in typed_vector_ssl_ranges.items():
        emb_key = bf_to_emb[bf_key]
        typed_vectors[emb_key] = per_variate[:, gs:ge, :].mean(axis=1)  # (N, 384)

    print(f"  Typed vectors pooled: {list(typed_vectors.keys())}")
    for k, v in typed_vectors.items():
        print(f"    {k:15s}: shape={v.shape}, norm=[{np.linalg.norm(v, axis=1).min():.3f}, "
              f"{np.linalg.norm(v, axis=1).max():.3f}]")

    # --- Project through PCA bases + Mahalanobis normalize → 84 scalars ---
    pca_84d = np.empty((N, total_k), dtype=np.float32)
    col_offset = 0
    for col in EMB_COLS:
        k = PER_GROUP_K_V9[col]
        group_bases = bases[col]
        group_vec = typed_vectors[col]  # (N, 384)
        # Project: (centered) @ components.T → (N, K)
        centered = group_vec - group_bases["mean_"]  # broadcast (N,384) - (384,)
        projected = centered @ group_bases["components"].T  # (N, K)
        # Mahalanobis normalization: divide by pc_std_ (same as evaluator)
        projected = projected / group_bases["pc_std_"]  # broadcast (N,K) / (K,)
        pca_84d[:, col_offset:col_offset + k] = projected
        col_offset += k

    assert col_offset == total_k
    print(f"\n  PCA-projected representation: ({N}, {total_k})")
    print(f"    Range: [{pca_84d.min():.3f}, {pca_84d.max():.3f}], "
          f"mean={pca_84d.mean():.4f}, std={pca_84d.std():.4f}")

    # --- Build dimensionality-matched baseline: raw_36 → PCA@84 ---
    # Use the same raw_36 preprocessing as run_single_probe
    X_raw = probe_data["X_windows"].copy()
    if _LOG1P_SSL_POS:
        X_raw[:, :, _LOG1P_SSL_POS] = (
            np.sign(X_raw[:, :, _LOG1P_SSL_POS])
            * np.log1p(np.abs(X_raw[:, :, _LOG1P_SSL_POS]))
        )
    np.nan_to_num(X_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    N_r, T_r, V_r = X_raw.shape
    X_raw_36 = X_raw[:, :, _MF_SSL_POS].reshape(N_r, T_r * len(_MF_SSL_POS))
    print(f"  raw_36 baseline (for PCA@84): ({N_r}, {X_raw_36.shape[1]})")

    # --- Run probes on scoring targets ---
    probe_names = ["spread_5", "log_rv_15", "log_rv_30", "regime"]
    results = {}

    for probe_name in probe_names:
        if probe_name not in probe_targets["targets"]:
            print(f"\n  [pca_84d] {probe_name}: target not available, skipping")
            continue

        targets = probe_targets["targets"][probe_name]
        valid = probe_targets["validity"][probe_name]

        is_classification = probe_name.startswith("regime")
        primary_metric = "balanced_accuracy" if is_classification else "r2"

        print(f"\n  --- {probe_name} ({primary_metric}) ---")

        needs_dead_zone = probe_name.startswith("spread_")
        pca84_fold_scores = []
        raw36_pca84_fold_scores = []

        for fi, (train_idx, test_idx) in enumerate(folds):
            # Apply validity + dead-zone (same protocol as run_single_probe)
            v_fold = valid.copy()

            if needs_dead_zone:
                train_mask_full = np.zeros(len(targets), dtype=bool)
                train_mask_full[train_idx] = True
                v_fold = apply_dead_zone(targets, v_fold, train_mask_full)

            tr_valid = v_fold[train_idx]
            te_valid = v_fold[test_idx]

            if is_classification:
                y_all = targets.astype(np.int64)
            else:
                y_all = targets.astype(np.float32)

            y_train = y_all[train_idx][tr_valid]
            y_test = y_all[test_idx][te_valid]

            n_tr = len(y_train)
            n_te = len(y_test)
            if n_tr < 50 or n_te < 10:
                pca84_fold_scores.append(None)
                raw36_pca84_fold_scores.append(None)
                continue

            if is_classification:
                n_cls = len(np.unique(y_train))
                if n_cls < 2:
                    pca84_fold_scores.append(None)
                    raw36_pca84_fold_scores.append(None)
                    continue

            # --- PCA-84d encoder probe ---
            X_tr_enc = pca_84d[train_idx][tr_valid]
            X_te_enc = pca_84d[test_idx][te_valid]
            n_p_ratio = len(X_tr_enc) / 84.0

            if is_classification:
                try:
                    m = fit_classification_probe(X_tr_enc, y_train, X_te_enc, y_test)
                    pca84_fold_scores.append(m[primary_metric])
                except Exception as e:
                    print(f"    fold {fi} pca_84d: FAILED ({e})")
                    pca84_fold_scores.append(None)
            else:
                try:
                    m = fit_regression_probe(X_tr_enc, y_train, X_te_enc, y_test)
                    pca84_fold_scores.append(m[primary_metric])
                except Exception as e:
                    print(f"    fold {fi} pca_84d: FAILED ({e})")
                    pca84_fold_scores.append(None)

            # --- raw_36 → PCA@84 baseline (dimensionality-matched) ---
            # Per-probe leakage exclusion for raw_36
            _hc_excl = HC_PROBE_LEAKAGE_EXCLUSIONS.get(probe_name, frozenset())
            if _hc_excl:
                _keep_pos = [SSL_FEATURES.index(v) for v in MODEL_FEATURES if v not in _hc_excl]
                X_raw_36_probe = X_raw[:, :, _keep_pos].reshape(N_r, T_r * len(_keep_pos))
            else:
                X_raw_36_probe = X_raw_36

            X_tr_raw = X_raw_36_probe[train_idx][tr_valid]
            X_te_raw = X_raw_36_probe[test_idx][te_valid]

            # Fit PCA on train → project to 84-d
            scaler = StandardScaler()
            X_tr_raw_s = scaler.fit_transform(X_tr_raw)
            X_te_raw_s = scaler.transform(X_te_raw)

            n_comp = min(84, X_tr_raw_s.shape[0], X_tr_raw_s.shape[1])
            pca_raw = SkPCA(n_components=n_comp)
            X_tr_raw_pca = pca_raw.fit_transform(X_tr_raw_s)
            X_te_raw_pca = pca_raw.transform(X_te_raw_s)

            if is_classification:
                try:
                    m_raw = fit_classification_probe(X_tr_raw_pca, y_train, X_te_raw_pca, y_test)
                    raw36_pca84_fold_scores.append(m_raw[primary_metric])
                except Exception as e:
                    print(f"    fold {fi} raw_36_pca84: FAILED ({e})")
                    raw36_pca84_fold_scores.append(None)
            else:
                try:
                    m_raw = fit_regression_probe(X_tr_raw_pca, y_train, X_te_raw_pca, y_test)
                    raw36_pca84_fold_scores.append(m_raw[primary_metric])
                except Exception as e:
                    print(f"    fold {fi} raw_36_pca84: FAILED ({e})")
                    raw36_pca84_fold_scores.append(None)

        # --- Summarize ---
        valid_pca84 = [s for s in pca84_fold_scores if s is not None]
        valid_raw36 = [s for s in raw36_pca84_fold_scores if s is not None]

        if valid_pca84:
            pca84_mean = float(np.mean(valid_pca84))
            # Compute min-fold N/p for reporting
            min_n_p = float("inf")
            for fi, (train_idx, _) in enumerate(folds):
                if pca84_fold_scores[fi] is not None:
                    v_fold = valid.copy()
                    if needs_dead_zone:
                        train_mask_full = np.zeros(len(targets), dtype=bool)
                        train_mask_full[train_idx] = True
                        v_fold = apply_dead_zone(targets, v_fold, train_mask_full)
                    n_valid_tr = int(v_fold[train_idx].sum())
                    ratio = n_valid_tr / 84.0
                    if ratio < min_n_p:
                        min_n_p = ratio
            print(f"  [pca_84d] {probe_name}: {primary_metric}={pca84_mean:.4f} "
                  f"({len(valid_pca84)} folds, N/p={min_n_p:.1f} at min fold)")
        else:
            pca84_mean = float("nan")
            print(f"  [pca_84d] {probe_name}: NO VALID FOLDS")

        if valid_raw36:
            raw36_mean = float(np.mean(valid_raw36))
            print(f"  [pca_84d] raw_36_pca84: {primary_metric}={raw36_mean:.4f} "
                  f"(baseline, dimensionality-matched)")
        else:
            raw36_mean = float("nan")
            print(f"  [pca_84d] raw_36_pca84: NO VALID FOLDS")

        # --- Paired delta + Bouthillier CI + Cohen d_z ---
        # Pair by fold index (same fix as Pass-A B1 in run_single_probe)
        paired_pairs = [
            (p, r)
            for p, r in zip(pca84_fold_scores, raw36_pca84_fold_scores)
            if p is not None and r is not None
        ]
        n_paired = len(paired_pairs)

        if n_paired >= 3:
            pca_arr = np.asarray([p[0] for p in paired_pairs], dtype=np.float64)
            raw_arr = np.asarray([p[1] for p in paired_pairs], dtype=np.float64)
            deltas = pca_arr - raw_arr
            delta_mean = float(deltas.mean())
            delta_std = float(deltas.std(ddof=1)) if n_paired > 1 else float("nan")
            # 80% CI (same as run_single_probe)
            try:
                from scipy import stats as _sp_stats
                t_crit = float(_sp_stats.t.ppf(0.90, df=n_paired - 1))
            except Exception:
                _T_CRIT_10_90 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476,
                                 6: 1.440, 7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372}
                t_crit = _T_CRIT_10_90.get(n_paired - 1, 1.372)
            ci_half = t_crit * delta_std / np.sqrt(n_paired) if n_paired > 1 else float("nan")
            ci_low = delta_mean - ci_half
            ci_high = delta_mean + ci_half
            d_z = delta_mean / delta_std if (delta_std and delta_std > 1e-12) else float("nan")
            ci_excludes_zero = (ci_low > 0.0) if not np.isnan(ci_low) else False
            d_z_medium = (not np.isnan(d_z)) and (d_z > 0.5)
            reportable = ci_excludes_zero and d_z_medium

            print(f"  Paired Δ(pca_84d − raw_36_pca84) mean={delta_mean:+.4f}, "
                  f"80% CI=[{ci_low:+.4f}, {ci_high:+.4f}], d_z={d_z:.3f}, n={n_paired}")
            print(f"  Reportable: {'YES' if reportable else 'NO'}")
        else:
            delta_mean = float("nan"); ci_low = float("nan"); ci_high = float("nan")
            d_z = float("nan"); reportable = False
            print(f"  Paired test: insufficient paired folds (n={n_paired} < 3)")
            print(f"  Reportable: NO (insufficient data)")

        results[probe_name] = {
            "pca_84d_mean": pca84_mean,
            "raw_36_pca84_mean": raw36_mean,
            "pca_84d_fold_scores": pca84_fold_scores,
            "raw_36_pca84_fold_scores": raw36_pca84_fold_scores,
            "n_paired": n_paired,
            "delta_mean": delta_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "d_z": d_z,
            "reportable": reportable,
        }

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("PCA-84d PRIMARY PROBE SUMMARY (GP-consumption-matched)")
    print("=" * 70)
    header = f"{'Probe':12s} {'pca_84d':>10s} {'raw36@84':>10s} {'delta':>8s} {'d_z':>6s} {'Report':>7s}"
    print(header)
    print("-" * len(header))
    for pname, r in results.items():
        pca_v = f"{r['pca_84d_mean']:.4f}" if not np.isnan(r['pca_84d_mean']) else "N/A"
        raw_v = f"{r['raw_36_pca84_mean']:.4f}" if not np.isnan(r['raw_36_pca84_mean']) else "N/A"
        d_v = f"{r['delta_mean']:+.4f}" if not np.isnan(r['delta_mean']) else "N/A"
        dz_v = f"{r['d_z']:.3f}" if not np.isnan(r['d_z']) else "N/A"
        rep_v = "YES" if r['reportable'] else "NO"
        print(f"  {pname:12s} {pca_v:>10s} {raw_v:>10s} {d_v:>8s} {dz_v:>6s} {rep_v:>7s}")

    n_reportable = sum(1 for r in results.values() if r["reportable"])
    print(f"\n  Reportable: {n_reportable}/{len(results)} probes (pca_84d > raw_36@84)")
    print(f"  NOTE: Existing emb_fine probes are DIAGNOSTIC (full-resolution, 3072-d).")
    print(f"        This probe tests the EXACT 84-d representation the GP consumes.")

    return results


# --- Completed diagnostics removed to stay under QC 256K notebook limit ---
# run_spread_pooling_controls() — results in EXPERIMENTS.md
# measure_fine_pca_spectrum() — results in (internal doc)
# run_pca_diagnostic_only() — results in EXPERIMENTS.md
# Code preserved in git history.


def resume_ssl_pipeline():
    """Resume SSL pretraining from the last periodic checkpoint.

    Loads the ssl_resume checkpoint from ObjectStore (saved every 25 epochs),
    rebuilds data loaders, and continues train_ssl() from the saved epoch.
    After training completes, saves final artifacts and runs baselines.

    Usage (in QC Research notebook):
        resume_ssl_pipeline()
    """
    global _results
    qb = get_qb()

    print("=" * 60)
    print("D71 SSL PRETRAINING — RESUME")
    print("=" * 60)

    # Step 0: Validate config before anything else
    _validate_config_vs_baseline(SSL_HYPERPARAMS)

    # Step 1: Load resume checkpoint (prefer ssl_resume, fall back to ssl_model)
    print("\n=== Step 1: Load resume checkpoint ===")
    resume_key = f"ssl_resume{ACTIVE_CKPT_SUFFIX}"
    has_resume = qb.ObjectStore.ContainsKey(resume_key)
    has_model = qb.ObjectStore.ContainsKey(ACTIVE_CKPT_KEY)
    if not has_resume and not has_model:
        raise RuntimeError(
            f"No checkpoint found for version {ACTIVE_CKPT_KEY}. "
            f"Run run_ssl_pipeline() first, or flip USE_V3_FEATURES to "
            f"match the intended checkpoint.")

    ckpt_key = resume_key if has_resume else ACTIVE_CKPT_KEY
    print(f"  Loading from '{ckpt_key}'...")
    raw = qb.ObjectStore.Read(ckpt_key)
    buf = io.BytesIO(base64.b64decode(raw))
    ckpt = torch.load(buf, map_location=device, weights_only=False)

    saved_epoch = ckpt["epoch"]
    hp = ckpt.get("ssl_hyperparams", SSL_HYPERPARAMS)
    V = ckpt.get("n_ssl_features", ACTIVE_N_SSL_FEATURES)

    # resume_key has separate best_val_loss + current state;
    # ACTIVE_CKPT_KEY only has the best state (encoder/recon are already best)
    if ckpt_key == resume_key:
        best_val_loss = ckpt["best_val_loss"]
        history = ckpt["history"]
    else:
        best_val_loss = ckpt.get("val_loss", float("inf"))
        # Load history from separate key if available
        if qb.ObjectStore.ContainsKey(ACTIVE_HISTORY_KEY):
            history = json.loads(qb.ObjectStore.Read(ACTIVE_HISTORY_KEY))
        else:
            history = None

    print(f"  Resuming from epoch {saved_epoch}, best_val_loss={best_val_loss:.4f}")
    if history:
        print(f"  History has {len(history.get('val_loss', []))} epochs recorded")

    # Step 2: Load data + build loaders
    print("\n=== Step 2: Load SSL data ===")
    if "ssl_data" in _results:
        print("Reusing cached SSL data")
        ssl_data = _results["ssl_data"]
    else:
        ssl_data = load_ssl_days(qb, TRAIN_START, TEST_END)
        _results["ssl_data"] = ssl_data

    print("\n=== Step 3: Split and normalize ===")
    loaders = make_ssl_split_and_loaders(ssl_data)
    _results["ssl_loaders"] = loaders

    # Step 4: Rebuild model + recon_head from current (not best) state
    print("\n=== Step 4: Rebuild model from checkpoint ===")
    model = iTransformerEncoder(hp, n_variates=V)
    model.load_state_dict(ckpt["encoder_state_dict"])

    _recon_state = ckpt["recon_head_state_dict"]
    recon_head = _build_recon_head_from_hp(
        hp, hp["d_model"], hp["seq_len"], V, ACTIVE_SSL_FEATURES)
    recon_head.load_state_dict(_recon_state)

    # Rebuild best_state: load from ssl_model (which stores best weights)
    best_state = None
    if ckpt_key == resume_key and ckpt.get("best_encoder_state") is not None:
        # Legacy format: best state was embedded in resume checkpoint
        best_state = {
            "encoder": ckpt["best_encoder_state"],
            "recon_head": ckpt["best_recon_state"],
        }
    elif ckpt_key == resume_key and qb.ObjectStore.ContainsKey(ACTIVE_CKPT_KEY):
        # New format: best state stored separately in ACTIVE_CKPT_KEY
        print(f"  Loading best state from {ACTIVE_CKPT_KEY}...")
        raw_best = qb.ObjectStore.Read(ACTIVE_CKPT_KEY)
        best_ckpt = torch.load(io.BytesIO(base64.b64decode(raw_best)),
                               map_location="cpu", weights_only=False)
        best_state = {
            "encoder": best_ckpt["encoder_state_dict"],
            "recon_head": best_ckpt["recon_head_state_dict"],
        }
    elif ckpt_key == ACTIVE_CKPT_KEY:
        # Fallback: ACTIVE_CKPT_KEY is the only checkpoint, its weights ARE the best
        best_state = {
            "encoder": ckpt["encoder_state_dict"],
            "recon_head": ckpt["recon_head_state_dict"],
        }

    # Restore temporal head if saved in checkpoint
    temporal_head_state = ckpt.get("temporal_head_state_dict")
    if temporal_head_state is not None:
        print(f"  Tokenizer: {hp.get('tokenizer', 'linear')}")
        print(f"  Temporal head state restored from checkpoint")

    # : restore adaptive spectral EMA weight
    spectral_ema_state = ckpt.get("spectral_ema_weight")
    if spectral_ema_state is not None:
        print(f"  Spectral EMA weight restored: {spectral_ema_state:.4f}")

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"  Loaded: {n_params:,} params, resuming epoch {saved_epoch + 1}")

    # Step 5: Resume training
    print(f"\n=== Step 5: Resume training (epoch {saved_epoch + 1} → {hp['num_epochs']}) ===")
    model, recon_head, history = train_ssl(
        loaders,
        start_epoch=saved_epoch + 1,
        history=history,
        best_val_loss=best_val_loss,
        best_state=best_state,
        model=model,
        recon_head=recon_head,
        opt_state=ckpt.get("optimizer_state_dict"),
        sched_state=ckpt.get("scheduler_state_dict"),
        temporal_head_state=temporal_head_state,
        spectral_ema_state=spectral_ema_state,
    )
    _results["ssl_model"] = model
    _results["ssl_recon_head"] = recon_head
    _results["ssl_history"] = history

    # Step 6: Save final artifacts
    print("\n=== Step 6: Save artifacts ===")
    save_ssl_artifacts(qb, model, recon_head, loaders["z_stats"], history)

    # Step 7: Reconstruction baselines
    print("\n=== Step 7: Reconstruction baselines ===")
    run_ssl_baselines(loaders, model, recon_head)

    # Step 8: Per-group diagnostics
    print("\n=== Step 8: Per-group baseline diagnostics ===")
    run_ssl_baseline_diagnostics(loaders, model, recon_head)

    # Step 9: Collect and persist structured metrics
    print("\n=== Step 9: Persist structured metrics ===")
    n_params_r = sum(p_.numel() for p_ in model.parameters())
    tok_r = hp.get("tokenizer", "linear")
    metrics_r = collect_training_metrics(history, hp, n_params_r, tokenizer=tok_r)
    _results["training_metrics"] = metrics_r
    all_pass_r = metrics_r.get("sanity_gates", {}).get("all_pass", False)
    print(f"  Sanity gates: {'ALL PASS' if all_pass_r else 'SOME FAILED'}")

    print("\nSSL resume + baselines complete!")
    return _results


# --- Auto-run ---
# Switch between pipeline stages by uncommenting ONE line:
# run_ssl_pipeline() # Full retrain from scratch
# resume_ssl_pipeline() # Resume from last checkpoint (after QC timeout)
# run_baselines_only() # Load checkpoint + baselines only (~5-10 min)
# run_all_probes() # §11.3 probes w/ walk-forward + PCA diagnostic
# resume_probes() # Resume probes from checkpoint
# run_handcrafted_probes() # Hand-crafted baseline (Run 0 — DONE)

# DO NOT auto-launch training here. IPython's %run -i sets
# __name__ == "__main__", so any guarded auto-launch fires on the first
# load of the part that contains it — before the operator has had a
# chance to choose resume_ssl_pipeline() vs run_ssl_pipeline(). The
# notebook stub cell explicitly calls the right entry point after %run
# has finished loading all three parts. See layer1/training/
# deploy_training.py::_stub_notebook for the operator-facing dispatch.
#
# If you want a local CLI (python pipeline.py ...), add it here as an
# argparse block that is opt-in, not default-launch.
