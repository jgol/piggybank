"""L2→L3 train-serve boundary (2026-06-01 holistic review, Batch 1).

B1: per-fold MINUTE normalization (fold_norm_stats) must survive the JSONL →
load_gp_strategies → codegen chain, so QC normalizes terminals with the SAME stats
the strategy was evolved under (not the frozen daily TERMINAL_NORM_STATS).

B2-int: probe/regime terminals (L1-encoder outputs QC cannot compute live) must
FAIL LOUD at codegen, never silently resolve to raw 0.0.
"""
import json
import pathlib

import pytest


# ----------------------------- B1: fold_norm_stats --------------------------- #

def test_fold_norm_stats_survives_loader(tmp_path):
    """load_gp_strategies must carry fold_norm_stats from the JSONL onto the dict
    the deploy path reads (regression: it was dropped → frozen-daily fallback)."""
    from layer3.pipeline_v2 import load_gp_strategies

    fold_dir = tmp_path / "fold_1"
    fold_dir.mkdir()
    rec = {
        "entry_tree": "GT(ATM_IV, EphReal(0.0))",
        "exit_tree": "LT(MinutesToClose, EphReal(0.0))",
        "size_tree": "EphReal(0.5)", "delta_tree": None,
        "val_sharpe": 0.9, "n_trades_val": 150,
        "fold_norm_stats": {"ATM_IV": [0.111, 0.051, "robust"],
                            "VIXSpot": [18.0, 2.5, "zscore"]},
    }
    (fold_dir / "strategies_iron_condor.jsonl").write_text(json.dumps(rec) + "\n")

    strategies = load_gp_strategies(str(tmp_path))
    assert strategies, "expected one loaded strategy"
    s = next(iter(strategies.values()))
    assert s.get("fold_norm_stats") == rec["fold_norm_stats"], \
        "fold_norm_stats must survive the loader (else P0-6 is dead on arrival)"


def test_fold_norm_stats_bakes_into_generated_code(tmp_path):
    """The per-fold stat must appear in the generated QC code's _NORM_STATS (proving
    the override reaches codegen end-to-end), NOT the frozen daily default."""
    from layer3.codegen import generate_qc_algorithm
    code = generate_qc_algorithm(
        strategy_id="t", template_name="iron_condor",
        entry_sexpr="GT(ATM_IV, EphReal(0.0))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)", start_date="2024-01-02", end_date="2024-02-01",
        norm_stats={"ATM_IV": (0.111, 0.051), "VIXSpot": (18.0, 2.5)},
    )
    assert "0.111" in code and "0.051" in code, \
        "per-fold ATM_IV normalization must be baked into the generated code"


# ----------------------------- B2-int: probe fail-loud ----------------------- #

# Only the WRITABLE probe/regime terminals (those from_sexpr accepts in a tree, i.e.
# the actual codegen-leak vectors). PredRegime / RegimeProb0..3 are internal-only
# (from_sexpr rejects them) so they cannot reach codegen; the codegen prefix-check
# still covers them as defense-in-depth.
@pytest.mark.parametrize("probe", [
    "PredRV15", "PredRV30", "PredSpread", "PredGammaAccel", "PredSmileConvexity",
    "PredJump", "PredFlowToxicity",
    "RegimeAboveLow", "RegimeIsHigh", "RegimeIsPremium",
    # EMB_* bare typed-vector terminals (campaign review HIGH-1): these parse to a
    # TermNode and so BYPASS the FuncNode EmbProj fail-loud — must also raise.
    "EMB_GRID", "EMB_SPX", "EMB_SHARED", "EMB_VIX", "EMB_VAR_ATM_IV",
])
def test_probe_terminal_fails_loud_at_codegen(probe):
    """Every WRITABLE L1-encoder terminal (Pred*/Regime*/EMB_*) must raise at codegen,
    never emit a silent s.get(name, 0.0) that resolves to raw 0.0 in QC."""
    from layer3.codegen import generate_qc_algorithm
    with pytest.raises(NotImplementedError):
        generate_qc_algorithm(
            strategy_id="t", template_name="iron_condor",
            entry_sexpr=f"GT({probe}, EphReal(0.0))",
            exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
            size_sexpr="EphReal(0.5)", start_date="2024-01-02", end_date="2024-02-01",
        )


def test_real_scalar_terminal_does_not_raise():
    """A scalar-only tree (real market terminals computed+normalized in QC) must
    still code-generate cleanly — the fail-loud is targeted at probe/regime only."""
    from layer3.codegen import generate_qc_algorithm, validate_generated_code
    code = generate_qc_algorithm(
        strategy_id="t", template_name="iron_condor",
        entry_sexpr="AND(GT(ATM_IV, EphReal(0.5)), LT(VIXSpot, EphReal(1.0)))",
        exit_sexpr="GT(MinutesToClose, EphReal(0.2))",
        size_sexpr="EphReal(0.5)", start_date="2024-01-02", end_date="2024-02-01",
    )
    ok, err = validate_generated_code(code)
    assert ok, f"scalar-only codegen must be valid: {err}"
