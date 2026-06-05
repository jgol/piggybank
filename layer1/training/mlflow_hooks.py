"""
MLflow in-loop logging hooks for pipeline.py local training runs.

Design goals:

- **Safe import.** If `mlflow` isn't installed (e.g. QC Research), every
  function here no-ops. Pipeline.py can call these unconditionally without
  breaking the QC-deployed notebook.
- **Safe with no active run.** If no `mlflow.start_run()` is open, every
  function no-ops. Pipeline.py doesn't need to know whether it's running
  inside an MLflow context.
- **Minimal surface.** Three functions: `log_epoch`, `log_checkpoint_artifact`,
  `log_diagnostics`. That's all the hook points train_ssl needs.
- **Unified storage.** Uses the same SQLite-backed tracking URI the existing
  `experiments/log_experiment.py` uses, so local runs appear alongside
  registered experiments in the same MLflow UI.

Integration with the existing flow:

    Existing (QC):
        qb.ObjectStore.Save(history) → enrich_mlflow.py backfills per-epoch later
        log_experiment.py writes registry + creates git tag at the end

    New (local):
        mlflow_hooks.log_epoch() writes per-epoch LIVE during training
        log_experiment.py still writes registry + creates git tag at the end
        enrich_mlflow.py becomes redundant for local runs (already live)
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Resilient imports
# ---------------------------------------------------------------------------

_HAS_MLFLOW = False
try:
    import mlflow  # noqa: F401
    _HAS_MLFLOW = True
except ImportError:
    pass


# Use the same SQLite backend the existing tooling uses.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}"


# ---------------------------------------------------------------------------
# Setup helpers (called by the launcher, not by pipeline.py)
# ---------------------------------------------------------------------------

def configure(tracking_uri: Optional[str] = None,
              experiment: str = "layer1") -> bool:
    """Idempotent setup. Returns True if MLflow is usable, False otherwise."""
    if not _HAS_MLFLOW:
        print("[mlflow_hooks] mlflow not installed — all hooks will no-op")
        return False
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    (PROJECT_ROOT / "mlruns").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    # H2 pre-reg v4: for sqlite backends, flip the journal to WAL so a
    # running pilot can log metrics concurrently with a dashboard or a
    # second experiment reading from the same DB without "database is
    # locked" errors. PRAGMA is persisted in the file header; once set
    # it survives across connections.
    _enable_sqlite_wal_if_applicable(uri)
    return True


def _enable_sqlite_wal_if_applicable(uri: str) -> None:
    """Best-effort: if `uri` points at a local sqlite file, enable WAL mode.

    No-op for non-sqlite URIs (PostgreSQL, MySQL, etc.) and swallows all
    failures so an unusual permission or disk-layout setup can never kill
    an otherwise-usable training run.
    """
    if not uri.startswith("sqlite:///"):
        return
    try:
        import sqlite3 as _sqlite3
        db_path = uri.replace("sqlite:///", "", 1)
        with _sqlite3.connect(db_path, timeout=5.0) as _conn:
            _conn.execute("PRAGMA journal_mode=WAL;")
            _conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception as _exc:  # noqa: BLE001
        print(f"[mlflow_hooks] WAL enable skipped: {_exc}")


def is_active() -> bool:
    """True iff mlflow is installed AND there's a currently active run."""
    if not _HAS_MLFLOW:
        return False
    return mlflow.active_run() is not None


def enable_pytorch_autolog():
    """Enable torch autolog (optimizer params, module hierarchy)."""
    if not _HAS_MLFLOW:
        return
    try:
        mlflow.pytorch.autolog(log_models=False, silent=True)
    except Exception as e:
        print(f"[mlflow_hooks] autolog setup failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# In-loop hooks (these are the ones pipeline.py calls)
# ---------------------------------------------------------------------------

def log_epoch(epoch: int,
              train_loss: float,
              val_loss: float,
              best_val_loss: Optional[float] = None,
              lr: Optional[float] = None,
              grad_norm: Optional[float] = None,
              epoch_wall_seconds: Optional[float] = None,
              n_effective: Optional[float] = None,
              extras: Optional[Dict[str, float]] = None) -> None:
    """Log per-epoch scalar metrics. Safe to call unconditionally.

    The entire body is wrapped in try/except so a logging failure (network
    blip, SQLite lock, unexpected extras type) can never propagate out and
    kill the training loop — losing an hour of epoch work to a metric
    logging hiccup would be unacceptable.
    """
    try:
        if not is_active():
            return
        _safe_log("train_loss", train_loss, step=epoch)
        _safe_log("val_loss", val_loss, step=epoch)
        if best_val_loss is not None:
            _safe_log("best_val_loss", best_val_loss, step=epoch)
        if lr is not None:
            _safe_log("learning_rate", lr, step=epoch)
        if grad_norm is not None:
            _safe_log("grad_norm", grad_norm, step=epoch)
        if epoch_wall_seconds is not None:
            _safe_log("epoch_wall_seconds", epoch_wall_seconds, step=epoch)
        if n_effective is not None:
            _safe_log("n_effective", n_effective, step=epoch)
        if extras:
            for k, v in extras.items():
                if v is None:
                    continue
                try:
                    _safe_log(k, float(v), step=epoch)
                except (TypeError, ValueError):
                    pass  # non-numeric — skip
    except Exception as e:
        print(f"[mlflow_hooks] log_epoch crashed (non-fatal): {type(e).__name__}: {e}")


def log_probe_metrics(epoch: int, probe_results: Dict[str, Any]) -> None:
    """Probe R² / accuracy values, namespaced under probe.<key>."""
    if not is_active():
        return
    for key, val in probe_results.items():
        if val is None:
            continue
        if isinstance(val, dict):
            for sub_k, sub_v in val.items():
                try:
                    _safe_log(f"probe.{key}.{sub_k}", float(sub_v), step=epoch)
                except (TypeError, ValueError):
                    pass
        else:
            try:
                _safe_log(f"probe.{key}", float(val), step=epoch)
            except (TypeError, ValueError):
                pass


def log_pipeline_probe_results(probe_results: Dict[str, Any]) -> None:
    """Walk the pipeline._results['probe_results'] structure and push each
    probe × emb_type {mean, std} to MLflow.

    Expected input shape (produced by pipeline.run_all_probes):
        {
          "log_rv_15": {
              "probe_type": "regression",
              "reportable": True|False|None,
              "summary": {
                  "emb_group": {"mean": 0.52, "std": 0.04, ...},
                  "emb_fine":  {"mean": 0.53, "std": 0.03, ...},
                  "emb_full":  {...}, "emb_max": {...},
                  "raw_36":    {...}, "raw_129": {...},
                  "shuffle_combined": {...}, "mlp_group": {...},
              },
          },
          "regime": {...}, "spread_5": {...}, ...
        }

    Logs each as probe.<name>.<emb_type>.{mean,std}, plus a tag
    probes.<name>.reportable (1/0/null) for at-a-glance filtering in the UI.
    Also logs an aggregate `probes_reportable` metric = count of reportable
    scoring probes, matching the gate language in project_experiment_state.md.
    """
    if not is_active():
        return
    n_reportable = 0
    n_scored = 0
    try:
        for probe_name, r in probe_results.items():
            if not isinstance(r, dict):
                continue
            summary = r.get("summary", {}) or {}
            reportable = r.get("reportable", None)
            probe_type = r.get("probe_type", "")

            for emb_type, stats in summary.items():
                if not isinstance(stats, dict):
                    continue
                mean = stats.get("mean")
                std = stats.get("std")
                if mean is not None:
                    try:
                        _safe_log(f"probe.{probe_name}.{emb_type}.mean", float(mean))
                    except (TypeError, ValueError):
                        pass
                if std is not None:
                    try:
                        _safe_log(f"probe.{probe_name}.{emb_type}.std", float(std))
                    except (TypeError, ValueError):
                        pass

            # Reportability as numeric metric for easy filtering
            if reportable is True:
                n_reportable += 1
                n_scored += 1
                _safe_log(f"probe.{probe_name}.reportable", 1.0)
            elif reportable is False:
                n_scored += 1
                _safe_log(f"probe.{probe_name}.reportable", 0.0)
            # None = expected null (e.g. ret_*), don't count

            # Also tag probe type (regression/classification) for the UI
            try:
                import mlflow
                mlflow.set_tag(f"probe.{probe_name}.type", probe_type)
            except Exception:
                pass

        # Aggregate reportability — matches the HOLD/PASS gate language
        _safe_log("probes_reportable_count", float(n_reportable))
        _safe_log("probes_scored_count", float(n_scored))
        if n_scored > 0:
            _safe_log("probes_reportable_frac", n_reportable / n_scored)

        # Log the full structure as a JSON artifact for audit
        log_json_artifact(_serialize_probe_results(probe_results),
                          filename="probe_results.json",
                          artifact_path="probes")
    except Exception as e:
        print(f"[mlflow_hooks] log_pipeline_probe_results failed (non-fatal): "
              f"{type(e).__name__}: {e}")


def _serialize_probe_results(probe_results: Dict[str, Any]) -> Dict[str, Any]:
    """Convert nested probe results to a JSON-safe dict. Drops fold-level
    arrays (too large) and keeps only means/stds + reportability flag."""
    out = {}
    for probe_name, r in probe_results.items():
        if not isinstance(r, dict):
            out[probe_name] = str(r)
            continue
        slim = {
            "probe_type": r.get("probe_type", ""),
            "reportable": r.get("reportable", None),
            "summary": {},
        }
        for emb_type, stats in (r.get("summary") or {}).items():
            if isinstance(stats, dict):
                slim["summary"][emb_type] = {
                    k: (float(v) if isinstance(v, (int, float)) else str(v))
                    for k, v in stats.items()
                    if k in ("mean", "std", "n")
                }
        out[probe_name] = slim
    return out


def _disk_artifact_path(filename: str, artifact_path: str = "json") -> Path:
    """Deterministic disk path for an artifact, INDEPENDENT of MLflow state.

    Azure-deployment retrievability fix (2026-04-29): prior to this, every
    `log_json_artifact` / `log_checkpoint_artifact` call wrote to a temp dir
    that was deleted as soon as the MLflow upload finished. If MLflow was
    inactive (Pass-A H1 from Round 2 audit — bracket_reportability.json was
    silently dropped because no run was active when it was logged), the
    artifact was lost entirely.

    Now every artifact also lands on a durable disk path under
    `raw_data/local_store/probe_artifacts/<run_prefix>/<artifact_path>/`.
    The MLflow log is layered on top as a redundant copy. Result: model,
    training history, and probe/diagnostic JSON artifacts are now all
    INDEPENDENTLY retrievable from filesystem alone, without any MLflow
    dependency.

    `<run_prefix>` is `<run_name>_<run_id8>` when MLflow is active, else
    `no_mlflow_run` (best-effort fallback so the call still persists).
    """
    if _HAS_MLFLOW:
        active = mlflow.active_run()
    else:
        active = None
    if active is not None:
        run_name = (active.data.tags.get("mlflow.runName") if active.data and active.data.tags else None) or "active_run"
        run_id = active.info.run_id[:8] if active.info and active.info.run_id else "no_id"
        prefix = f"{run_name}_{run_id}"
    else:
        prefix = "no_mlflow_run"
    out_dir = PROJECT_ROOT / "raw_data" / "local_store" / "probe_artifacts" / prefix / artifact_path
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def log_checkpoint_artifact(ckpt_bytes: bytes,
                             filename: str,
                             artifact_path: str = "checkpoints") -> None:
    """Persist a checkpoint blob to disk + MLflow.

    Always writes to a durable disk path under `raw_data/local_store/
    probe_artifacts/...` so the artifact is independently retrievable
    regardless of MLflow state.
    """
    disk_path = _disk_artifact_path(filename, artifact_path)
    disk_path.write_bytes(ckpt_bytes)
    if not is_active():
        return
    try:
        mlflow.log_artifact(str(disk_path), artifact_path=artifact_path)
    except Exception as e:
        print(f"[mlflow_hooks] checkpoint MLflow log failed: {e} "
              f"(disk copy intact at {disk_path})")


def log_json_artifact(payload: Any, filename: str,
                       artifact_path: str = "json") -> None:
    """Serialize payload to JSON to disk + MLflow.

    Always writes to a durable disk path under `raw_data/local_store/
    probe_artifacts/...` so the artifact is independently retrievable
    regardless of MLflow state. MLflow is then a redundant copy.
    """
    disk_path = _disk_artifact_path(filename, artifact_path)
    disk_path.write_text(json.dumps(payload, indent=2, default=str))
    if not is_active():
        return
    try:
        mlflow.log_artifact(str(disk_path), artifact_path=artifact_path)
    except Exception as e:
        print(f"[mlflow_hooks] json MLflow log failed: {e} "
              f"(disk copy intact at {disk_path})")


def log_diagnostics(diagnostics: Dict[str, Any]) -> None:
    """End-of-training diagnostics (MQA.6, cosine shift, attention entropy).

    Expects a dict possibly nested 1-level deep. Scalars become metrics,
    lists/dicts become a diagnostics.json artifact.
    """
    if not is_active():
        return
    scalars: Dict[str, float] = {}
    non_scalars: Dict[str, Any] = {}
    for key, val in diagnostics.items():
        if isinstance(val, (int, float)):
            scalars[f"diag.{key}"] = float(val)
        elif isinstance(val, dict):
            for sub_k, sub_v in val.items():
                if isinstance(sub_v, (int, float)):
                    scalars[f"diag.{key}.{sub_k}"] = float(sub_v)
                else:
                    non_scalars.setdefault(key, {})[sub_k] = sub_v
        else:
            non_scalars[key] = val
    for k, v in scalars.items():
        _safe_log(k, v)
    if non_scalars:
        log_json_artifact(non_scalars, "diagnostics.json", artifact_path="diagnostics")


def log_params(params: Dict[str, Any]) -> None:
    """Log hyperparameters. Flattens one level of nesting."""
    if not is_active():
        return
    for key, val in params.items():
        if isinstance(val, dict):
            for sub_k, sub_v in val.items():
                _safe_log_param(f"{key}.{sub_k}", sub_v)
        else:
            _safe_log_param(key, val)


def log_tags(tags: Dict[str, Any]) -> None:
    """Set MLflow run tags."""
    if not is_active():
        return
    for key, val in tags.items():
        try:
            mlflow.set_tag(key, str(val))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal helpers — isolate all mlflow.* calls here so failures are noisy
# but non-fatal.
# ---------------------------------------------------------------------------

def _safe_log(key: str, value: float, step: Optional[int] = None) -> None:
    try:
        mlflow.log_metric(key, value, step=step)
    except Exception as e:
        print(f"[mlflow_hooks] log_metric({key!r}) failed: {e}")


def _safe_log_param(key: str, value: Any) -> None:
    try:
        # MLflow params have a 500-char string limit; truncate defensively.
        v_str = str(value)
        if len(v_str) > 500:
            v_str = v_str[:497] + "..."
        mlflow.log_param(key, v_str)
    except Exception as e:
        print(f"[mlflow_hooks] log_param({key!r}) failed: {e}")


__all__ = [
    "configure", "is_active", "enable_pytorch_autolog",
    "log_epoch", "log_probe_metrics", "log_pipeline_probe_results",
    "log_checkpoint_artifact", "log_json_artifact",
    "log_diagnostics", "log_params", "log_tags",
]
