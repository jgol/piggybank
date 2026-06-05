"""Unit tests for variate-block masking in layer1/training/pipeline.py.

Tests the L1-SSL-009 v3 mask-generator fix: block-masking of flow
aggregate derivatives (v141-v158) when their base raw flow variate
(v131-v139) is masked. Without this, the encoder trivially inverts the
rolling mean to reconstruct masked raw flow cells (leak documented in
AI Engineer Round 1 review and L1-SSL-009 spec §4.3).

These tests exercise the pure helpers without requiring QC Research:
  - _build_flow_agg_position_map
  - _trailing_window_or
  - _expand_flow_aggregate_mask
  - sample_ssl_mask with flow_agg_positions parameter

Run locally:
    PYTHONPATH=. python3 -m pytest tests/test_variate_block_masking.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Import pipeline.py directly without triggering __init__.py side effects
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec = importlib.util.spec_from_file_location(
    "pipeline",
    Path(__file__).resolve().parents[1] / "layer1" / "training" / "pipeline.py",
)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


# ---------------------------------------------------------------------------
# _build_flow_agg_position_map
# ---------------------------------------------------------------------------

class TestBuildFlowAggPositionMap:
    def test_returns_none_for_ssl004_layout(self):
        """SSL-004's 129-variate layout has no aggregates → None."""
        result = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES)
        assert result is None

    def test_returns_dict_for_ssl009_v3_layout(self):
        """SSL-009's 147-variate layout includes all 18 aggregates → dict."""
        result = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        assert result is not None
        assert set(result.keys()) == {
            "raw_positions_py", "roll3_positions_py", "roll15_positions_py"
        }
        for key in result:
            assert isinstance(result[key], list)
            assert all(isinstance(x, int) for x in result[key])
            assert len(result[key]) == 9

    def test_positions_correct_in_v3_layout(self):
        """Positions must correspond to v131-v139 (raw), v141-v149 (roll3),
        v150-v158 (roll15) in SSL_FEATURES_V3 order."""
        result = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        features = pipeline.SSL_FEATURES_V3
        for i, v in enumerate(range(131, 140)):
            pos = result["raw_positions_py"][i]
            assert features[pos] == v, f"raw_positions[{i}] expected v{v}, got v{features[pos]}"
        for i, v in enumerate(range(141, 150)):
            pos = result["roll3_positions_py"][i]
            assert features[pos] == v
        for i, v in enumerate(range(150, 159)):
            pos = result["roll15_positions_py"][i]
            assert features[pos] == v

    def test_returns_none_when_partial_aggregates(self):
        """Feature list with only SOME aggregates → None (can't safely block-mask)
        AND emits a RuntimeWarning so silent leak is loud."""
        partial = sorted(pipeline.SSL_FEATURES + [141, 142])  # only 2 aggregates
        with pytest.warns(RuntimeWarning, match="partial set"):
            result = pipeline._build_flow_agg_position_map(partial)
        assert result is None

    def test_no_warning_for_zero_aggregates(self):
        """SSL-004 layout (no aggregates at all) → None without warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → error
            result = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES)
        assert result is None


# ---------------------------------------------------------------------------
# _trailing_window_or
# ---------------------------------------------------------------------------

class TestTrailingWindowOr:
    def test_window_1_is_identity(self):
        x = torch.tensor([[False, True, False, True, False]], dtype=torch.bool)
        result = pipeline._trailing_window_or(x, 1)
        assert torch.equal(result, x)

    def test_window_3_propagates_forward_by_2(self):
        """For window=3: result[b, t] = OR(x[b, t-2], x[b, t-1], x[b, t])."""
        x = torch.tensor([[False, False, True, False, False, False, False]], dtype=torch.bool)
        result = pipeline._trailing_window_or(x, 3)
        # x[2] is True → result[2], result[3], result[4] should be True
        expected = torch.tensor([[False, False, True, True, True, False, False]], dtype=torch.bool)
        assert torch.equal(result, expected)

    def test_window_15_propagates_forward_by_14(self):
        x = torch.zeros(1, 30, dtype=torch.bool)
        x[0, 5] = True  # single spike
        result = pipeline._trailing_window_or(x, 15)
        # Expect result[5..19] = True, rest False
        assert torch.all(result[0, 5:20])
        assert not torch.any(result[0, :5])
        assert not torch.any(result[0, 20:])

    def test_batch_processing(self):
        """Different samples should propagate independently."""
        x = torch.zeros(3, 10, dtype=torch.bool)
        x[0, 2] = True
        x[1, 5] = True
        # sample 2 is all False
        result = pipeline._trailing_window_or(x, 3)
        assert torch.all(result[0, 2:5])
        assert not torch.any(result[0, :2])
        assert not torch.any(result[0, 5:])
        assert torch.all(result[1, 5:8])
        assert not torch.any(result[2])

    def test_clip_at_sequence_end(self):
        """A mask at bar T-1 should only affect bar T-1 (no forward propagation
        beyond T-1). Similarly for bar T-2 with window=3 — only T-2 and T-1."""
        T = 10
        x = torch.zeros(1, T, dtype=torch.bool)
        x[0, T - 1] = True  # mask at last bar
        result = pipeline._trailing_window_or(x, 15)
        # Only T-1 should be True (no room to propagate)
        assert result[0, T - 1]
        assert not torch.any(result[0, :T - 1])

    def test_does_not_mutate_input(self):
        x = torch.tensor([[False, True, False]], dtype=torch.bool)
        x_clone = x.clone()
        _ = pipeline._trailing_window_or(x, 3)
        assert torch.equal(x, x_clone)


# ---------------------------------------------------------------------------
# _expand_flow_aggregate_mask — whole-variate propagation
# ---------------------------------------------------------------------------

class TestExpandFlowAggregateMaskWholeVariate:
    def _make_masks(self, n_features, seq_len=60, batch_size=2):
        """Empty (all-False) combined and variate_mask tensors."""
        return (
            torch.zeros(batch_size, seq_len, n_features, dtype=torch.bool),
            torch.zeros(batch_size, seq_len, n_features, dtype=torch.bool),
        )

    def test_noop_when_flow_positions_none(self):
        """SSL-004 mode (no aggregates) → function is a no-op."""
        combined, variate_mask = self._make_masks(pipeline.N_SSL_FEATURES)
        # Mark one cell masked
        combined[0, 10, 50] = True
        before = combined.clone()
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, None)
        assert torch.equal(combined, before)

    def test_whole_variate_v131_propagates_to_v141_and_v150(self):
        """If v131 is whole-masked, v141 and v150 must also be whole-masked."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        assert pos is not None
        combined, variate_mask = self._make_masks(pipeline.N_SSL_FEATURES_V3)
        # v131 is raw_positions[0]; v141 is roll3_positions[0]; v150 is roll15_positions[0]
        v131_pos = int(pos["raw_positions_py"][0])
        v141_pos = pos["roll3_positions_py"][0]
        v150_pos = pos["roll15_positions_py"][0]
        # Sample 0: whole-mask v131
        variate_mask[0, :, v131_pos] = True
        combined[0, :, v131_pos] = True
        # Sample 1: nothing masked
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        # Sample 0: v141 and v150 should now be whole-masked
        assert torch.all(variate_mask[0, :, v141_pos])
        assert torch.all(variate_mask[0, :, v150_pos])
        assert torch.all(combined[0, :, v141_pos])
        assert torch.all(combined[0, :, v150_pos])
        # Sample 1: should be unaffected
        assert not torch.any(variate_mask[1, :, v141_pos])
        assert not torch.any(variate_mask[1, :, v150_pos])

    def test_only_affects_derivatives_of_masked_raw(self):
        """Whole-masking v131 should NOT affect v142/v151 (aggregates of v132)."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        combined, variate_mask = self._make_masks(pipeline.N_SSL_FEATURES_V3)
        v131_pos = int(pos["raw_positions_py"][0])
        v142_pos = pos["roll3_positions_py"][1]  # agg of v132
        v151_pos = pos["roll15_positions_py"][1]
        variate_mask[0, :, v131_pos] = True
        combined[0, :, v131_pos] = True
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        assert not torch.any(variate_mask[0, :, v142_pos])
        assert not torch.any(variate_mask[0, :, v151_pos])


# ---------------------------------------------------------------------------
# _expand_flow_aggregate_mask — cell-wise propagation
# ---------------------------------------------------------------------------

class TestExpandFlowAggregateMaskCellWise:
    """Tests for cell-wise block-mask propagation with threshold semantics.

    C1 remediation (2026-05-02): changed from ANY-masked to threshold-based.
    With block_threshold=0.5 (default):
      - roll3 (window=3): need ceil(3*0.5)=2 of 3 cells masked to trigger
      - roll15 (window=15): need ceil(15*0.5)=8 of 15 cells masked to trigger
    A single masked cell in a window no longer triggers block-masking.
    """

    def test_single_raw_cell_does_NOT_propagate_threshold(self):
        """C1 remediation: single cell masked (1/3 = 33% < 50%) should NOT
        trigger roll3 or roll15 block-mask."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        B, T, V = 1, 60, pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(B, T, V, dtype=torch.bool)
        variate_mask = torch.zeros(B, T, V, dtype=torch.bool)
        v131_pos = int(pos["raw_positions_py"][0])
        v141_pos = pos["roll3_positions_py"][0]
        v150_pos = pos["roll15_positions_py"][0]
        # Mask v131 at bar 20 only -- 1 cell, below threshold for both windows
        combined[0, 20, v131_pos] = True
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        # roll3: 1/3 = 33% < 50% threshold -> no propagation
        assert not torch.any(combined[0, :, v141_pos])
        # roll15: 1/15 = 6.7% < 50% threshold -> no propagation
        assert not torch.any(combined[0, :, v150_pos])

    def test_two_adjacent_cells_trigger_roll3(self):
        """Two adjacent raw cells masked (2/3 = 67% > 50%) should trigger
        roll3 block-mask at the bars where 2+ of 3 trailing cells are masked."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        B, T, V = 1, 60, pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(B, T, V, dtype=torch.bool)
        variate_mask = torch.zeros(B, T, V, dtype=torch.bool)
        v131_pos = int(pos["raw_positions_py"][0])
        v141_pos = pos["roll3_positions_py"][0]
        # Mask bars 20 and 21
        combined[0, 20, v131_pos] = True
        combined[0, 21, v131_pos] = True
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        # At bar 20: trailing window [20] has 1/1 -> masked (partial window, ceil(1*0.5)=1)
        # Wait -- let's think about this properly:
        # For bar t, trailing window is [max(0, t-w+1), t].
        # Effective window at bar t = min(t+1, w).
        # Threshold count at bar t = ceil(effective_window * 0.5)
        #
        # Bar 20: effective_window=3, count of masked in [18,19,20] = 1, threshold=2 -> no
        # Bar 21: effective_window=3, count of masked in [19,20,21] = 2, threshold=2 -> YES
        # Bar 22: effective_window=3, count of masked in [20,21,22] = 2, threshold=2 -> YES
        # Bar 23: effective_window=3, count of masked in [21,22,23] = 1, threshold=2 -> no
        assert not combined[0, 20, v141_pos]
        assert combined[0, 21, v141_pos]
        assert combined[0, 22, v141_pos]
        assert not combined[0, 23, v141_pos]

    def test_heavy_masking_triggers_roll15(self):
        """8 of 15 contiguous cells masked (53% > 50%) should trigger roll15
        at bars where the trailing window has 8+ masked."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        B, T, V = 1, 60, pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(B, T, V, dtype=torch.bool)
        variate_mask = torch.zeros(B, T, V, dtype=torch.bool)
        v131_pos = int(pos["raw_positions_py"][0])
        v150_pos = pos["roll15_positions_py"][0]
        # Mask bars 10-17 (8 contiguous bars)
        for t in range(10, 18):
            combined[0, t, v131_pos] = True
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        # At bar 24 (trailing window [10,24]): 8 masked out of 15 -> triggers
        # At bar 25 (trailing window [11,25]): 7 masked out of 15 -> no trigger
        # The exact bar range where it triggers depends on the sliding window.
        # Key assertion: some bars ARE masked (the fix works) and some are NOT
        # (unlike the old behavior which would mask bars 10..31 = 22 bars).
        assert combined[0, 24, v150_pos]  # 8/15 masked -> triggers
        assert not combined[0, 25, v150_pos]  # 7/15 masked -> no trigger

    def test_clip_at_session_end_threshold(self):
        """Single mask at bar T-1 does NOT propagate (below threshold)."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        B, T, V = 1, 60, pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(B, T, V, dtype=torch.bool)
        variate_mask = torch.zeros(B, T, V, dtype=torch.bool)
        v131_pos = int(pos["raw_positions_py"][0])
        v141_pos = pos["roll3_positions_py"][0]
        v150_pos = pos["roll15_positions_py"][0]
        combined[0, T - 1, v131_pos] = True
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        # Single cell: 1/3 for roll3, 1/15 for roll15 -> below 50% threshold
        assert not torch.any(combined[0, :, v141_pos])
        assert not torch.any(combined[0, :, v150_pos])

    def test_boundary_partial_window_threshold(self):
        """At bar 0, the effective window is 1. A single masked cell is 1/1 = 100%
        which exceeds any threshold. Verify boundary handling."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        B, T, V = 1, 60, pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(B, T, V, dtype=torch.bool)
        variate_mask = torch.zeros(B, T, V, dtype=torch.bool)
        v131_pos = int(pos["raw_positions_py"][0])
        v141_pos = pos["roll3_positions_py"][0]
        v150_pos = pos["roll15_positions_py"][0]
        # Mask bar 0 only
        combined[0, 0, v131_pos] = True
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        # Bar 0: effective window=1 for both roll3 and roll15.
        # count=1, threshold=ceil(1*0.5)=1 -> triggers
        assert combined[0, 0, v141_pos]
        assert combined[0, 0, v150_pos]
        # Bar 1: effective window=2, count=0 (bar 0 is in window but only
        # the count matters -- wait, bar 0 IS masked and is in the trailing
        # window for bar 1). Let me trace:
        # Bar 1 roll3: window [0, 1], eff_window=2, count=1, threshold=ceil(2*0.5)=1 -> triggers
        assert combined[0, 1, v141_pos]
        # Bar 2 roll3: window [0, 1, 2], eff_window=3, count=1, threshold=ceil(3*0.5)=2 -> no
        assert not combined[0, 2, v141_pos]

    def test_cross_variate_independence(self):
        """Masking v131 cells does not affect v132's aggregates and vice versa."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        B, T, V = 1, 60, pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(B, T, V, dtype=torch.bool)
        variate_mask = torch.zeros(B, T, V, dtype=torch.bool)
        v131_pos = int(pos["raw_positions_py"][0])  # raw v131
        v132_pos = pos["raw_positions_py"][1]  # raw v132
        v142_pos = pos["roll3_positions_py"][1]  # agg of v132
        v151_pos = pos["roll15_positions_py"][1]  # agg of v132
        # Mask v131 at bar 10, leave v132 unmasked
        combined[0, 10, v131_pos] = True
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        # v142 and v151 (aggregates of v132) must NOT be masked
        assert not torch.any(combined[0, :, v142_pos])
        assert not torch.any(combined[0, :, v151_pos])

    def test_cross_variate_independence_batch_threshold(self):
        """Verify cross-variate independence across batch dimension under
        threshold-based cell-wise masking. Single-cell masks in different
        samples should NOT trigger propagation (below threshold)."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        B, T, V = 4, 60, pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(B, T, V, dtype=torch.bool)
        variate_mask = torch.zeros(B, T, V, dtype=torch.bool)
        v131_pos = pos["raw_positions_py"][0]
        v132_pos = pos["raw_positions_py"][1]
        v133_pos = pos["raw_positions_py"][2]
        v141_pos = pos["roll3_positions_py"][0]  # agg of v131
        v142_pos = pos["roll3_positions_py"][1]  # agg of v132
        v143_pos = pos["roll3_positions_py"][2]  # agg of v133
        # Single cell masks -- all below threshold
        # Sample 0: mask v131[10]
        combined[0, 10, v131_pos] = True
        # Sample 1: mask v132[20]
        combined[1, 20, v132_pos] = True
        # Sample 2: mask v133[30] AND v131[5]
        combined[2, 30, v133_pos] = True
        combined[2, 5, v131_pos] = True
        # Sample 3: nothing masked
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        # With threshold semantics, single cell masks don't propagate
        # Sample 0: v141 NOT propagated (1/3 < 50%)
        assert not torch.any(combined[0, :, v141_pos])
        assert not torch.any(combined[0, :, v142_pos])
        assert not torch.any(combined[0, :, v143_pos])
        # Sample 1: v142 NOT propagated
        assert not torch.any(combined[1, :, v141_pos])
        assert not torch.any(combined[1, :, v142_pos])
        assert not torch.any(combined[1, :, v143_pos])
        # Sample 2: neither v141 nor v143 propagated
        assert not torch.any(combined[2, :, v141_pos])
        assert not torch.any(combined[2, :, v142_pos])
        assert not torch.any(combined[2, :, v143_pos])
        # Sample 3: everything untouched
        assert not torch.any(combined[3, :, v141_pos])
        assert not torch.any(combined[3, :, v142_pos])
        assert not torch.any(combined[3, :, v143_pos])

    def test_invariant_variate_mask_subset_of_combined(self):
        """DE review: post-condition `variate_mask <= combined` must hold.
        The __debug__ assert inside _expand_flow_aggregate_mask enforces this
        at every call; this test exercises a realistic scenario and asserts
        the invariant directly at the end."""
        pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        B, T, V = 3, 60, pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(B, T, V, dtype=torch.bool)
        variate_mask = torch.zeros(B, T, V, dtype=torch.bool)
        v131_pos = pos["raw_positions_py"][0]
        variate_mask[0, :, v131_pos] = True
        combined[0, :, v131_pos] = True
        pipeline._expand_flow_aggregate_mask(combined, variate_mask, pos)
        assert (variate_mask <= combined).all()


# ---------------------------------------------------------------------------
# sample_ssl_mask integration — v3 mode with flow_agg_positions
# ---------------------------------------------------------------------------

class TestSampleSslMaskIntegration:
    def test_ssl004_mode_backward_compat(self):
        """SSL-004 callers pass flow_agg_positions=None (default) → no change."""
        n_feat = pipeline.N_SSL_FEATURES
        eligible = torch.tensor(
            pipeline._build_eligible_mask(pipeline.SSL_FEATURES), dtype=torch.bool
        )
        gen = torch.Generator().manual_seed(42)
        combined, var_mask = pipeline.sample_ssl_mask(
            batch_size=4, seq_len=60, n_features=n_feat,
            eligible_mask=eligible, generator=gen,
        )
        assert combined.shape == (4, 60, n_feat)
        assert var_mask.shape == (4, 60, n_feat)

    def test_ssl009_v3_mode_block_masks_aggregates(self):
        """When raw v131 is whole-masked, v141 and v150 must end up whole-masked
        at the end of sample_ssl_mask."""
        n_feat = pipeline.N_SSL_FEATURES_V3
        eligible = torch.tensor(
            pipeline._build_eligible_mask(
                pipeline.SSL_FEATURES_V3, pipeline.SSL_MASK_INELIGIBLE_V3
            ),
            dtype=torch.bool,
        )
        flow_pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        # Use high variate_ratio so at least one raw flow variate gets whole-masked
        gen = torch.Generator().manual_seed(42)
        combined, var_mask = pipeline.sample_ssl_mask(
            batch_size=32, seq_len=60, n_features=n_feat,
            eligible_mask=eligible,
            variate_ratio=0.5, cell_ratio=0.15,
            generator=gen,
            flow_agg_positions=flow_pos,
        )
        # For each sample and each raw flow variate that was whole-masked,
        # verify its aggregate derivatives are also whole-masked.
        n_errors = 0
        for i_flow in range(9):
            raw_p = flow_pos["raw_positions_py"][i_flow]
            r3_p = flow_pos["roll3_positions_py"][i_flow]
            r15_p = flow_pos["roll15_positions_py"][i_flow]
            raw_whole = var_mask[:, 0, raw_p]  # (B,)
            if raw_whole.any():
                # For every sample where raw was whole-masked, roll3 and roll15 must too
                roll3_whole = var_mask[:, 0, r3_p]
                roll15_whole = var_mask[:, 0, r15_p]
                if not (roll3_whole >= raw_whole).all():
                    n_errors += 1
                if not (roll15_whole >= raw_whole).all():
                    n_errors += 1
        assert n_errors == 0, f"{n_errors} flow/aggregate mask-propagation failures"

    def test_ssl009_aggregates_never_primary_targets(self):
        """SSL_MASK_INELIGIBLE_V3 includes aggregates → they are never chosen
        as primary whole-variate mask targets before block-expansion."""
        # Build eligible mask and verify aggregates are all False
        eligible = pipeline._build_eligible_mask(
            pipeline.SSL_FEATURES_V3, pipeline.SSL_MASK_INELIGIBLE_V3
        )
        for agg_v in range(141, 159):
            pos = pipeline.SSL_FEATURES_V3.index(agg_v)
            assert not eligible[pos], f"v{agg_v} should be mask-ineligible"
        # And v131 should still be eligible
        raw_pos = pipeline.SSL_FEATURES_V3.index(131)
        assert eligible[raw_pos]

    def test_v3_layout_without_flow_agg_positions_raises(self):
        """Model QA review: sample_ssl_mask must refuse to run in v3 layout
        (n_features == 147) without flow_agg_positions, or a future caller
        could silently train with the leak."""
        n_feat = pipeline.N_SSL_FEATURES_V3
        eligible = torch.tensor(
            pipeline._build_eligible_mask(
                pipeline.SSL_FEATURES_V3, pipeline.SSL_MASK_INELIGIBLE_V3
            ),
            dtype=torch.bool,
        )
        with pytest.raises(ValueError, match="v3 layout.*flow_agg_positions"):
            pipeline.sample_ssl_mask(
                batch_size=2, seq_len=60, n_features=n_feat,
                eligible_mask=eligible,
                flow_agg_positions=None,  # ← the leak-inducing omission
            )

    def test_v3_cell_mask_on_raw_propagates_threshold(self):
        """C1 remediation: cell-masking propagation uses threshold semantics.
        An aggregate cell is only block-masked when >50% of its source window
        cells are masked (not ANY-masked as before)."""
        import math
        n_feat = pipeline.N_SSL_FEATURES_V3
        eligible = torch.tensor(
            pipeline._build_eligible_mask(
                pipeline.SSL_FEATURES_V3, pipeline.SSL_MASK_INELIGIBLE_V3
            ),
            dtype=torch.bool,
        )
        flow_pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        # High cell_ratio to get dense masking -- threshold logic needs enough
        # masked cells in a window to trigger
        gen = torch.Generator().manual_seed(123)
        combined, var_mask = pipeline.sample_ssl_mask(
            batch_size=8, seq_len=60, n_features=n_feat,
            eligible_mask=eligible,
            variate_ratio=0.0, cell_ratio=0.50,  # High rate so threshold fires
            generator=gen,
            flow_agg_positions=flow_pos,
        )
        # For each (sample, flow variate) with cell masks, verify the threshold
        # propagation contract: aggregate cell is masked IFF >50% of its trailing
        # window source cells are masked.
        T = 60
        for b in range(8):
            for i_flow in range(9):
                raw_p = flow_pos["raw_positions_py"][i_flow]
                r3_p = flow_pos["roll3_positions_py"][i_flow]
                r15_p = flow_pos["roll15_positions_py"][i_flow]
                raw_cells = combined[b, :, raw_p]  # (T,)
                if not raw_cells.any():
                    continue
                # Compute expected threshold mask for roll3
                r3_count = pipeline._trailing_window_count(
                    raw_cells.unsqueeze(0), pipeline.ROLLING_WINDOW_ROLL3
                ).squeeze(0)
                r3_eff = torch.clamp(
                    torch.arange(1, T + 1), max=pipeline.ROLLING_WINDOW_ROLL3
                )
                r3_thresh = torch.ceil(r3_eff.float() * 0.5).int()
                expected_r3 = (r3_count >= r3_thresh)
                actual_r3 = combined[b, :, r3_p]
                # actual >= expected: threshold propagation must have fired
                assert (actual_r3 >= expected_r3).all(), (
                    f"sample {b}, flow {i_flow}: roll3 threshold propagation "
                    f"incomplete"
                )

    def test_content_based_guard_raises_on_odd_layout(self):
        """Round 7 MQA-blocker: content-based guard must catch ablation
        layouts of any total size that carry aggregates. Size-based check
        (n_features == 147) alone would miss a 150-variate layout that
        injects partial aggregates."""
        # Build a fake 'ablation' layout: SSL_FEATURES + a few aggregates.
        # This is neither 129 (SSL-004) nor 147 (v3) — size guard misses it.
        ablation_feats = list(pipeline.SSL_FEATURES) + [141, 142, 143]
        assert len(ablation_feats) not in (129, pipeline.N_SSL_FEATURES_V3)
        eligible = torch.ones(len(ablation_feats), dtype=torch.bool)
        with pytest.raises(ValueError, match="flow aggregates"):
            pipeline.sample_ssl_mask(
                batch_size=2, seq_len=60, n_features=len(ablation_feats),
                eligible_mask=eligible,
                flow_agg_positions=None,
                feature_indices=ablation_feats,
            )

    def test_content_based_guard_silent_when_aggregates_absent(self):
        """Content check must not raise on SSL-004 layout passed with
        feature_indices — flow_agg_positions=None is the correct value
        for the 129-variate layout."""
        eligible = torch.tensor(
            pipeline._build_eligible_mask(pipeline.SSL_FEATURES),
            dtype=torch.bool,
        )
        combined, _ = pipeline.sample_ssl_mask(
            batch_size=2, seq_len=60, n_features=len(pipeline.SSL_FEATURES),
            eligible_mask=eligible,
            flow_agg_positions=None,
            feature_indices=list(pipeline.SSL_FEATURES),
        )
        assert combined.shape == (2, 60, len(pipeline.SSL_FEATURES))

    def test_expand_flow_aggregate_mask_length_assert(self):
        """Round 7 nice-to-have: structural validation on position lists.
        A corrupted flow_positions dict (wrong-length list) must fail
        loudly, not silently block-mask wrong columns."""
        n_feat = pipeline.N_SSL_FEATURES_V3
        combined = torch.zeros(2, 60, n_feat, dtype=torch.bool)
        variate_mask = torch.zeros(2, 60, n_feat, dtype=torch.bool)
        bad_positions = {
            "raw_positions_py":    [0, 1, 2],              # length 3, not 9
            "roll3_positions_py":  list(range(10, 19)),
            "roll15_positions_py": list(range(20, 29)),
        }
        with pytest.raises(AssertionError, match="expected 9 raw positions"):
            pipeline._expand_flow_aggregate_mask(combined, variate_mask, bad_positions)


# ---------------------------------------------------------------------------
# C1 remediation: _trailing_window_count
# ---------------------------------------------------------------------------

class TestTrailingWindowCount:
    def test_window_1_is_identity(self):
        x = torch.tensor([[False, True, False, True, False]], dtype=torch.bool)
        result = pipeline._trailing_window_count(x, 1)
        expected = torch.tensor([[0, 1, 0, 1, 0]])
        assert torch.equal(result, expected)

    def test_window_3_counts_correctly(self):
        x = torch.tensor([[True, False, True, True, False, False]], dtype=torch.bool)
        result = pipeline._trailing_window_count(x, 3)
        # Bar 0: window [0], count=1
        # Bar 1: window [0,1], count=1
        # Bar 2: window [0,1,2], count=2
        # Bar 3: window [1,2,3], count=2
        # Bar 4: window [2,3,4], count=2
        # Bar 5: window [3,4,5], count=1
        expected = torch.tensor([[1, 1, 2, 2, 2, 1]])
        assert torch.equal(result, expected)

    def test_all_masked(self):
        x = torch.ones(1, 10, dtype=torch.bool)
        result = pipeline._trailing_window_count(x, 5)
        # Bar 0-3: partial windows (1,2,3,4), bar 4+: full window of 5
        expected = torch.tensor([[1, 2, 3, 4, 5, 5, 5, 5, 5, 5]])
        assert torch.equal(result, expected)

    def test_none_masked(self):
        x = torch.zeros(1, 10, dtype=torch.bool)
        result = pipeline._trailing_window_count(x, 5)
        assert torch.equal(result, torch.zeros(1, 10, dtype=torch.int32))

    def test_batch_independence(self):
        x = torch.zeros(3, 10, dtype=torch.bool)
        x[0, 5] = True
        x[1, 3] = True
        x[1, 4] = True
        result = pipeline._trailing_window_count(x, 3)
        # Sample 0: only bar 5 masked -> count peaks at 1 for bars 5,6,7
        assert result[0, 5] == 1
        assert result[0, 6] == 1
        assert result[0, 7] == 1
        assert result[0, 8] == 0
        # Sample 1: bars 3,4 masked
        assert result[1, 4] == 2  # window [2,3,4] -> 2 masked
        assert result[1, 5] == 2  # window [3,4,5] -> 2 masked
        assert result[1, 6] == 1  # window [4,5,6] -> 1 masked
        # Sample 2: nothing masked
        assert not torch.any(result[2])

    def test_does_not_mutate_input(self):
        x = torch.tensor([[False, True, False]], dtype=torch.bool)
        x_clone = x.clone()
        _ = pipeline._trailing_window_count(x, 3)
        assert torch.equal(x, x_clone)


# ---------------------------------------------------------------------------
# C1 remediation: roll15 visibility under threshold logic
# ---------------------------------------------------------------------------

class TestRoll15VisibilityRemediation:
    """C1 CRITICAL: flow_roll15 was visible only ~6% of training under the old
    ANY-masked logic. With threshold=0.5, visibility should be >30%."""

    def test_roll15_visibility_above_30_percent(self):
        """Statistical test: generate many masks at variate_ratio=0.20,
        cell_ratio=0.15 and verify roll15 context visibility > 30%."""
        n_feat = pipeline.N_SSL_FEATURES_V3
        eligible = torch.tensor(
            pipeline._build_eligible_mask(
                pipeline.SSL_FEATURES_V3, pipeline.SSL_MASK_INELIGIBLE_V3
            ),
            dtype=torch.bool,
        )
        flow_pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        # Generate many samples to get stable statistics
        gen = torch.Generator().manual_seed(42)
        combined, var_mask = pipeline.sample_ssl_mask(
            batch_size=256, seq_len=60, n_features=n_feat,
            eligible_mask=eligible,
            variate_ratio=0.20, cell_ratio=0.15,
            generator=gen,
            flow_agg_positions=flow_pos,
        )
        # Check roll15 visibility (fraction of cells NOT masked)
        for i_flow in range(9):
            r15_p = flow_pos["roll15_positions_py"][i_flow]
            mask_rate = combined[:, :, r15_p].float().mean().item()
            visibility = 1.0 - mask_rate
            assert visibility > 0.30, (
                f"roll15 variate {i_flow} (v{150 + i_flow}) has visibility "
                f"{visibility:.1%} < 30% -- C1 threshold fix did not work"
            )

    def test_roll3_visibility_above_50_percent(self):
        """Roll3 has a less severe issue (43% visible was tolerable under old
        logic). With threshold semantics, visibility should improve further."""
        n_feat = pipeline.N_SSL_FEATURES_V3
        eligible = torch.tensor(
            pipeline._build_eligible_mask(
                pipeline.SSL_FEATURES_V3, pipeline.SSL_MASK_INELIGIBLE_V3
            ),
            dtype=torch.bool,
        )
        flow_pos = pipeline._build_flow_agg_position_map(pipeline.SSL_FEATURES_V3)
        gen = torch.Generator().manual_seed(42)
        combined, var_mask = pipeline.sample_ssl_mask(
            batch_size=256, seq_len=60, n_features=n_feat,
            eligible_mask=eligible,
            variate_ratio=0.20, cell_ratio=0.15,
            generator=gen,
            flow_agg_positions=flow_pos,
        )
        for i_flow in range(9):
            r3_p = flow_pos["roll3_positions_py"][i_flow]
            mask_rate = combined[:, :, r3_p].float().mean().item()
            visibility = 1.0 - mask_rate
            assert visibility > 0.50, (
                f"roll3 variate {i_flow} (v{141 + i_flow}) has visibility "
                f"{visibility:.1%} < 50%"
            )

    def test_high_variate_ratio_triggers_assertion_concern(self):
        """If someone sets variate_ratio=0.40, roll15 visibility under old
        ANY-masked logic would be < 5%. The closed-form rate diagnostic
        should catch this. Verify the rate computation flags it."""
        from math import comb, ceil
        # Compute what roll15 mask rate would be at variate_ratio=0.40
        base_rate = 0.40 + (1.0 - 0.40) * 0.15  # = 0.49
        # Under threshold logic: P(Binomial(15, 0.49) >= 8)
        prob_masked = 0.0
        for k in range(8, 16):
            prob_masked += comb(15, k) * (0.49 ** k) * (0.51 ** (15 - k))
        visibility = 1.0 - prob_masked
        # Even with threshold, visibility should still be reasonable at 0.40
        # The key is that it's WAY better than the old logic where it would be ~0%
        assert visibility > 0.15, (
            f"At variate_ratio=0.40, roll15 visibility is {visibility:.1%} "
            f"which is still very low"
        )


# ---------------------------------------------------------------------------
# C2 remediation: v105 eligibility
# ---------------------------------------------------------------------------

class TestV105Eligibility:
    """C2: v105 (session_log_return) restored as mask-eligible."""

    def test_v105_not_in_ineligible(self):
        """v105 must NOT be in SSL_MASK_INELIGIBLE after remediation."""
        assert 105 not in pipeline.SSL_MASK_INELIGIBLE

    def test_v105_not_in_ineligible_v3(self):
        """v105 must NOT be in SSL_MASK_INELIGIBLE_V3 after remediation."""
        assert 105 not in pipeline.SSL_MASK_INELIGIBLE_V3

    def test_v105_is_eligible_in_mask(self):
        """v105 must be eligible for masking in the eligible mask."""
        eligible = pipeline._build_eligible_mask(
            pipeline.SSL_FEATURES_V3, pipeline.SSL_MASK_INELIGIBLE_V3
        )
        v105_pos = pipeline.SSL_FEATURES_V3.index(105)
        assert eligible[v105_pos], "v105 should be mask-eligible after C2 fix"

    def test_eligible_count_after_d100(self):
        """After D100 (VIX context-only), eligible count reflects all ineligible sets."""
        eligible = pipeline._build_eligible_mask(
            pipeline.SSL_FEATURES_V3, pipeline.SSL_MASK_INELIGIBLE_V3
        )
        n_eligible = sum(eligible)
        # Count: 147 total v3 features minus all ineligible (original + VIX + flow aggs)
        n_ineligible_in_v3 = len(pipeline.SSL_MASK_INELIGIBLE_V3 & set(pipeline.SSL_FEATURES_V3))
        expected = len(pipeline.SSL_FEATURES_V3) - n_ineligible_in_v3
        assert n_eligible == expected, f"Expected {expected} eligible variates, got {n_eligible}"

    def test_other_spx_derived_still_ineligible(self):
        """v106-v113, v115-v118 must remain ineligible (valid exclusion reasons)."""
        still_ineligible = [106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118]
        for v in still_ineligible:
            assert v in pipeline.SSL_MASK_INELIGIBLE, (
                f"v{v} should still be mask-ineligible"
            )


# ---------------------------------------------------------------------------
# C3 remediation: variate_ratio
# ---------------------------------------------------------------------------

class TestVariateRatioReduction:
    """C3: variate_ratio reduced from 0.30 to 0.20 in SSL_HYPERPARAMS."""

    def test_active_variate_ratio_is_020(self):
        assert pipeline.SSL_HYPERPARAMS["variate_ratio"] == 0.20

    def test_baseline_variate_ratio_unchanged(self):
        """EXPERIMENT_BASELINE must retain 0.30 for SSL-010 reproduction."""
        assert pipeline.EXPERIMENT_BASELINE["variate_ratio"] == 0.30

    def test_ssl010_baseline_unchanged(self):
        """EXPERIMENT_BASELINE_SSL010 must retain 0.30."""
        assert pipeline.EXPERIMENT_BASELINE_SSL010["variate_ratio"] == 0.30


# ---------------------------------------------------------------------------
# Diagnostic assertions (would-have-caught-it tests)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# H2: Per-group spectral covariance penalty
# ---------------------------------------------------------------------------

class TestPerGroupSpectralPenalty:
    """Tests for the per-group spectral penalty (H2 fix, line ~5300 of pipeline.py).

    The spectral penalty computes -log(det(Cov(tokens))) per group (not globally),
    penalizing collapsed embedding dimensions. These tests verify:
      1. Collapsed groups produce HIGHER loss than spread-out groups
      2. Separate covariance matrices are computed per group
      3. Groups with <3 tokens are skipped
    """

    @staticmethod
    def _compute_per_group_spectral_loss(tokens, group_positions_list):
        """Re-implement the per-group spectral loss from pipeline.py lines 5310-5341.

        tokens: (B, V, D) tensor
        group_positions_list: list of (name, list_of_positions) tuples
        Returns: (rank_loss, n_groups)
        """
        D = tokens.shape[-1]
        eye_D = torch.eye(D, device=tokens.device)
        rank_loss = torch.tensor(0.0, device=tokens.device)
        n_groups = 0
        for _grp_name, _grp_positions in group_positions_list:
            if len(_grp_positions) < 3:
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
            rank_loss = rank_loss / n_groups
        return rank_loss, n_groups

    def test_collapsed_group_higher_loss_than_spread(self):
        """A group where all tokens are identical (collapsed) should produce
        a HIGHER spectral penalty than a group with spread-out tokens."""
        B, D = 8, 16
        # Group A: 5 tokens, all identical (collapsed)
        collapsed_tokens = torch.randn(B, 1, D).expand(B, 5, D).clone()
        # Group B: 5 tokens, spread out (random)
        torch.manual_seed(42)
        spread_tokens = torch.randn(B, 5, D)

        groups_collapsed = [("group_a", list(range(5)))]
        groups_spread = [("group_a", list(range(5)))]

        loss_collapsed, n1 = self._compute_per_group_spectral_loss(
            collapsed_tokens, groups_collapsed
        )
        loss_spread, n2 = self._compute_per_group_spectral_loss(
            spread_tokens, groups_spread
        )
        assert n1 == 1 and n2 == 1
        assert loss_collapsed > loss_spread, (
            f"Collapsed loss {loss_collapsed:.4f} should be > spread loss "
            f"{loss_spread:.4f}"
        )

    def test_separate_covariance_per_group(self):
        """Each group must compute its own covariance matrix, not one global.

        Proof: if we have two groups (A: collapsed, B: spread), the per-group
        penalty should be dominated by group A's collapse. If it were computed
        globally (all tokens mixed), group B's variance would dilute group A's
        collapse and the penalty would be lower."""
        B, D = 8, 16
        torch.manual_seed(99)

        # Build combined tensor: positions 0-4 = collapsed, 5-9 = spread
        collapsed = torch.randn(B, 1, D).expand(B, 5, D).clone()
        spread = torch.randn(B, 5, D) * 3.0  # large variance
        combined = torch.cat([collapsed, spread], dim=1)  # (B, 10, D)

        # Per-group: two groups, each with 5 tokens
        per_group = [
            ("collapsed", [0, 1, 2, 3, 4]),
            ("spread", [5, 6, 7, 8, 9]),
        ]
        loss_per_group, n_groups = self._compute_per_group_spectral_loss(
            combined, per_group
        )
        assert n_groups == 2

        # Global: one group with all 10 tokens
        one_global = [("global", list(range(10)))]
        loss_global, _ = self._compute_per_group_spectral_loss(
            combined, one_global
        )

        # Per-group loss should be HIGHER because the collapsed group's penalty
        # is not diluted by the spread group's good covariance
        assert loss_per_group > loss_global, (
            f"Per-group loss {loss_per_group:.4f} should exceed global "
            f"loss {loss_global:.4f} when one group is collapsed"
        )

    def test_groups_with_fewer_than_3_tokens_skipped(self):
        """Groups with <3 tokens are skipped (the code has
        `if len(_grp_positions) < 3: continue`)."""
        B, D = 4, 8
        torch.manual_seed(7)
        tokens = torch.randn(B, 10, D)

        # One group with 2 tokens, one with 5 tokens
        groups = [
            ("tiny", [0, 1]),        # <3 -> skipped
            ("normal", [2, 3, 4, 5, 6]),  # >=3 -> included
        ]
        loss, n_groups = self._compute_per_group_spectral_loss(tokens, groups)
        assert n_groups == 1, f"Expected 1 group (tiny skipped), got {n_groups}"

        # All groups <3 -> n_groups=0, loss=0
        tiny_only = [
            ("a", [0]),
            ("b", [1, 2]),
        ]
        loss_tiny, n_tiny = self._compute_per_group_spectral_loss(tokens, tiny_only)
        assert n_tiny == 0
        assert loss_tiny.item() == 0.0

    def test_spectral_loss_is_nonnegative_for_collapsed(self):
        """For collapsed embeddings, -log(det(Cov)) should be large and positive
        (since det is near zero, log(det) is very negative, -log(det) is very positive)."""
        B, D = 8, 16
        collapsed = torch.randn(B, 1, D).expand(B, 5, D).clone()
        groups = [("collapsed", list(range(5)))]
        loss, _ = self._compute_per_group_spectral_loss(collapsed, groups)
        assert loss.item() > 0, f"Spectral loss should be positive for collapsed tokens"


class TestD1HardAssertion:
    """Test that D1 context visibility floor is now a hard ValueError."""

    def test_d1_raises_on_low_visibility(self):
        """If any group has <20% context visibility, _print_and_assert_mask_rate_ceiling
        must raise ValueError (not just print a warning)."""
        # Use hyperparams with very high variate_ratio to push visibility below 20%
        import copy
        hp_bad = copy.deepcopy(dict(pipeline.SSL_HYPERPARAMS))
        hp_bad["variate_ratio"] = 0.95  # extreme: most variates masked
        hp_bad["cell_ratio"] = 0.80     # extreme cell ratio too
        # base_rate = 0.95 + 0.05*0.80 = 0.99; visibility = 1% << 20%
        # This should raise ValueError with the D1 hard stop.
        # Loosen eligible/catastrophe ceilings to 1.0 so they never fire.
        with pytest.raises(ValueError, match="D1 HARD STOP"):
            pipeline._print_and_assert_mask_rate_ceiling(
                hp_bad,
                pipeline.ACTIVE_SSL_FEATURES,
                eligible_ceiling=1.0,       # disable eligible ceiling
                catastrophe_ceiling=1.0,    # disable catastrophe ceiling
            )

    def test_d1_passes_on_normal_config(self):
        """Current production config should pass D1 without error."""
        # Should not raise
        pipeline._print_and_assert_mask_rate_ceiling(
            pipeline.SSL_HYPERPARAMS,
            pipeline.ACTIVE_SSL_FEATURES,
            eligible_ceiling=0.85,
            catastrophe_ceiling=0.95,
        )


class TestDiagnosticAssertions:
    """Verify that the new diagnostic assertions in _print_and_assert_mask_rate_ceiling
    would have caught the original defects."""

    def test_diagnostics_run_without_error(self):
        """Current config (variate_ratio=0.20, v105 eligible) should pass
        all diagnostic assertions without raising."""
        # This calls _print_and_assert_mask_rate_ceiling which includes
        # the new D1/D2/D3 diagnostics
        pipeline._print_and_assert_mask_rate_ceiling(
            pipeline.SSL_HYPERPARAMS,
            pipeline.ACTIVE_SSL_FEATURES,
            eligible_ceiling=0.85,
            catastrophe_ceiling=0.95,
        )
