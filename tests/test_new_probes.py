"""Tests for L2.62 new probe targets: GammaAccel, SmileConvexity, Jump, FlowToxicity.

Validates:
  - New probe terminals exist in probes-only terminal set
  - New probe terminals are forbidden in scalar-only terminal set
  - New probes have normalization stats in terminal_stats
  - PROBE_SCALAR_COLUMNS_V2 includes all 8 probes
  - Grammar builds successfully with new terminals
"""
import pytest


class TestNewProbeTerminals:
    """Verify new probe terminals are correctly wired in the grammar."""

    def test_probes_only_has_new_terminals(self):
        """Condition B (probes-only) must include all 4 new probe terminals."""
        from layer2.grammar import build_probes_only_terminal_set
        terms = build_probes_only_terminal_set()
        term_names = {t.name for t in terms}
        for probe in ["PredGammaAccel", "PredSmileConvexity", "PredJump", "PredFlowToxicity"]:
            assert probe in term_names, f"{probe} missing from probes-only terminal set"

    def test_scalar_only_forbids_new_terminals(self):
        """Condition A (scalar-only) must forbid all 4 new probe terminals."""
        from layer2.grammar import _SCALAR_ONLY_TERMS_FORBIDDEN
        for probe in ["PredGammaAccel", "PredSmileConvexity", "PredJump", "PredFlowToxicity"]:
            assert probe in _SCALAR_ONLY_TERMS_FORBIDDEN, f"{probe} not forbidden in scalar-only"

    def test_normalization_stats_exist(self):
        """All 4 new probes must have normalization stats."""
        from layer2.terminal_stats import TERMINAL_NORM_STATS
        for probe in ["PredGammaAccel", "PredSmileConvexity", "PredJump", "PredFlowToxicity"]:
            assert probe in TERMINAL_NORM_STATS, f"{probe} missing from TERMINAL_NORM_STATS"
            center, scale, method = TERMINAL_NORM_STATS[probe]
            assert scale > 0, f"{probe} has non-positive scale: {scale}"
            assert method in ("standard", "robust"), f"{probe} has unknown method: {method}"

    def test_probe_columns_v2_has_all_8(self):
        """PROBE_SCALAR_COLUMNS_V2 must contain all 8 probe column names."""
        from layer2.io import PROBE_SCALAR_COLUMNS_V2
        assert len(PROBE_SCALAR_COLUMNS_V2) == 8
        expected = {"PredRV15", "PredRV30", "PredRegime", "PredSpread",
                    "PredGammaAccel", "PredSmileConvexity", "PredJump", "PredFlowToxicity"}
        assert set(PROBE_SCALAR_COLUMNS_V2) == expected

    def test_original_probe_columns_unchanged(self):
        """PROBE_SCALAR_COLUMNS (original 4) must not change."""
        from layer2.io import PROBE_SCALAR_COLUMNS
        assert len(PROBE_SCALAR_COLUMNS) == 4
        assert set(PROBE_SCALAR_COLUMNS) == {"PredRV15", "PredRV30", "PredRegime", "PredSpread"}

    def test_grammar_builds_with_probes_only(self):
        """Grammar with probes-only condition must build successfully."""
        from layer2.grammar import Grammar, build_probes_only_terminal_set
        terms = build_probes_only_terminal_set()
        g = Grammar(max_depth=3, max_nodes=8, terminals=terms)
        assert g is not None
        # Should be able to grow a BOOL tree
        from layer2.grammar import GType
        tree = g.grow(GType.BOOL)
        assert tree is not None

    def test_new_probes_not_in_emb_only(self):
        """Condition C (emb-only) must NOT include probe terminals."""
        from layer2.grammar import build_emb_only_terminal_set
        terms = build_emb_only_terminal_set()
        term_names = {t.name for t in terms}
        for probe in ["PredGammaAccel", "PredSmileConvexity", "PredJump", "PredFlowToxicity"]:
            assert probe not in term_names, f"{probe} should not be in emb-only"
        # Also verify original probes not in emb-only
        for probe in ["PredRV15", "PredRV30", "PredSpread"]:
            assert probe not in term_names, f"{probe} should not be in emb-only"
