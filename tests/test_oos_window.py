"""Per-fold out-of-sample QC backtest window (P1: D5-D8 OOS-overlap fix).

A walk-forward strategy must be validated AFTER its fold's train_end. The fixed
DEFAULT_BACKTEST window overlapped fold-3/4 training, so those strategies were
validated partly on data they trained on. oos_window_for_fold maps each fold to
a genuinely out-of-sample window.
"""
from layer2.experiment import WALK_FORWARD_FOLDS
from layer3.config import DEFAULT_BACKTEST_START, DEFAULT_BACKTEST_END
from layer3.pipeline_v2 import oos_window_for_fold


def test_oos_window_starts_at_train_end_per_fold():
    for f in WALK_FORWARD_FOLDS:
        start, end = oos_window_for_fold(f["fold_id"])
        assert start == f["train_end"], f"fold {f['fold_id']} OOS must start at train_end"
        if f["test_end"] is None:        # holdout fold
            assert end == DEFAULT_BACKTEST_END
        else:
            assert end == f["test_end"]


def test_oos_window_excludes_fold4_training():
    """The bug: fold 4 trained through 2025-09-30 but was backtested from the fixed
    2024-10-01 start, i.e. ON its own training data. The fix starts at train_end."""
    f4 = next(f for f in WALK_FORWARD_FOLDS if f["fold_id"] == 4)
    start, _ = oos_window_for_fold(4)
    assert start == f4["train_end"]
    assert start > DEFAULT_BACKTEST_START  # no longer reaches back into training


def test_oos_window_accepts_str_and_int_fold_id():
    assert oos_window_for_fold("2") == oos_window_for_fold(2)


def test_oos_window_unknown_fold_falls_back_to_default():
    assert oos_window_for_fold("0") == (DEFAULT_BACKTEST_START, DEFAULT_BACKTEST_END)
    assert oos_window_for_fold(None) == (DEFAULT_BACKTEST_START, DEFAULT_BACKTEST_END)
    assert oos_window_for_fold("garbage") == (DEFAULT_BACKTEST_START, DEFAULT_BACKTEST_END)
