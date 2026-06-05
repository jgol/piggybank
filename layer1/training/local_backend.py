"""
Local-disk shim for qb.ObjectStore, used when training pipeline.py outside QC.

Provides a LocalQuantBook object whose .ObjectStore attribute mimics the exact
surface used by pipeline.py:

    Read(key) -> str                   (UTF-8 decoded)
    ReadBytes(key) -> bytes
    Save(key, str_or_bytes) -> None
    SaveBytes(key, bytes) -> None
    ContainsKey(key) -> bool
    Delete(key) -> None
    Keys -> list[str]                  (property, enumerates all known keys)

Two sources for values:

1) VIRTUAL KEYS (served from the reassembled corpus bundle on first access).
   For each date D in the bundle, the key `layer1_feat_v3/D.npz` returns the
   same JSON-envelope string that backfill_flow_aggregates.py writes into QC:
       {"meta": {...}, "data": "<base64-of-compressed-NPZ>"}
   This matches what pipeline.py:_load_tensor expects, so pipeline.py needs
   no special handling — it calls Read() and parses as usual.

2) REAL KEYS (disk-backed under raw_data/local_store/).
   Every non-virtual key maps to a file; value is raw bytes.
   Keys with "/" in them become nested directories.

Entry point:
    from layer1.training.local_backend import get_local_qb
    qb = get_local_qb()   # idempotent — creates the shim on first call

The LOCAL_MODE env var toggles routing in pipeline.py:get_qb (see patch).
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
from pathlib import Path
from typing import Iterable, Optional, Union


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DEFAULT = PROJECT_ROOT / "raw_data" / "layer1_corpus_v3_bundle.npz"
STORE_DEFAULT = PROJECT_ROOT / "raw_data" / "local_store"

# Virtual-key prefix for corpus features — matches the QC ObjectStore layout.
CORPUS_PREFIX = "layer1_feat_v3/"


# ---------------------------------------------------------------------------
# Lazy bundle loader
# ---------------------------------------------------------------------------

class _BundleView:
    """Lazy wrapper around the reassembled corpus NPZ. Synthesizes the same
    JSON envelope pipeline.py expects from ObjectStore, on demand."""

    def __init__(self, bundle_path: Path):
        self._bundle_path = bundle_path
        self._lock = threading.Lock()
        self._npz = None                  # np.load handle; opened lazily
        self._dates = None                # list[str], set on first access
        self._envelope_cache = {}         # date_str -> JSON envelope string

    def _ensure_loaded(self):
        if self._npz is not None:
            return
        with self._lock:
            if self._npz is not None:
                return
            if not self._bundle_path.exists():
                raise FileNotFoundError(
                    f"Corpus bundle not found at {self._bundle_path}. "
                    f"Run `layer1.data.assemble_corpus_chunks` first."
                )
            import numpy as np
            self._np = np
            self._npz = np.load(self._bundle_path, allow_pickle=False)
            manifest_bytes = bytes(self._npz["_manifest"])
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            self._dates = list(manifest["dates"])

    def dates(self) -> list[str]:
        self._ensure_loaded()
        return list(self._dates)

    def has_date(self, date_str: str) -> bool:
        self._ensure_loaded()
        return date_str in self._dates

    def envelope_for(self, date_str: str) -> str:
        """Return the JSON-envelope string identical to what ObjectStore.Read
        would return from QC: `{"meta": {...}, "data": "<base64-npz>"}`."""
        self._ensure_loaded()
        if date_str not in self._dates:
            raise KeyError(date_str)
        cached = self._envelope_cache.get(date_str)
        if cached is not None:
            return cached

        X = self._npz[f"d_{date_str}"]
        # Re-encode the day as a standalone compressed NPZ to match collector
        # output: single "X" array, same dtype pipeline.py expects.
        buf = io.BytesIO()
        self._np.savez_compressed(buf, X=X.astype(self._np.float32))
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        envelope = json.dumps({
            "meta": {
                "date": date_str,
                "shape": list(X.shape),
                "dtype": "float32",
                "source": "local_backend/corpus_bundle",
            },
            "data": payload,
        }, separators=(",", ":"))
        # Cache sparingly — each envelope is a few hundred KB. We'll cache up
        # to N entries (FIFO) so the working set during a training epoch
        # (which walks dates sequentially) doesn't thrash.
        if len(self._envelope_cache) >= 64:
            # Drop oldest (dict insertion order)
            try:
                self._envelope_cache.pop(next(iter(self._envelope_cache)))
            except StopIteration:
                pass
        self._envelope_cache[date_str] = envelope
        return envelope


# ---------------------------------------------------------------------------
# Disk-backed key/value store for non-corpus keys (checkpoints, history, etc.)
# ---------------------------------------------------------------------------

class _DiskStore:
    """Simple filesystem-backed KV for ObjectStore keys that aren't corpus days.

    Keys may contain slashes (e.g. 'ssl_resume_v3' — but also nested patterns).
    We mirror the key as a path under `root`, storing raw bytes. Strings are
    UTF-8 encoded on save, decoded on Read().
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Defensive: refuse absolute paths / ".." traversal in keys
        if os.path.isabs(key) or ".." in Path(key).parts:
            raise ValueError(f"Invalid ObjectStore key (traversal not allowed): {key!r}")
        return self.root / key

    def contains(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_bytes(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise KeyError(key)
        return p.read_bytes()

    def read_str(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def save(self, key: str, value: Union[str, bytes, bytearray]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            data = value.encode("utf-8")
        elif isinstance(value, (bytes, bytearray)):
            data = bytes(value)
        else:
            raise TypeError(f"ObjectStore.Save expects str or bytes, got {type(value).__name__}")
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.is_file():
            p.unlink()

    def keys(self) -> Iterable[str]:
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix != ".tmp":
                yield str(p.relative_to(self.root))


# ---------------------------------------------------------------------------
# LocalObjectStore — orchestrates virtual corpus reads + disk-backed KV
# ---------------------------------------------------------------------------

class LocalObjectStore:
    def __init__(self, bundle_view: _BundleView, disk: _DiskStore):
        self._bundle = bundle_view
        self._disk = disk

    # ---- Reads ----

    def _is_corpus_key(self, key: str) -> tuple[bool, Optional[str]]:
        if not key.startswith(CORPUS_PREFIX):
            return False, None
        tail = key[len(CORPUS_PREFIX):]
        if not tail.endswith(".npz"):
            return False, None
        return True, tail[:-len(".npz")]

    def ContainsKey(self, key: str) -> bool:  # noqa: N802 — C# surface
        is_corpus, date_str = self._is_corpus_key(key)
        if is_corpus:
            return self._bundle.has_date(date_str)
        return self._disk.contains(key)

    def Read(self, key: str) -> str:          # noqa: N802
        is_corpus, date_str = self._is_corpus_key(key)
        if is_corpus:
            return self._bundle.envelope_for(date_str)
        return self._disk.read_str(key)

    def ReadBytes(self, key: str) -> bytes:   # noqa: N802
        is_corpus, date_str = self._is_corpus_key(key)
        if is_corpus:
            return self._bundle.envelope_for(date_str).encode("utf-8")
        return self._disk.read_bytes(key)

    # ---- Writes (corpus keys are read-only in local mode) ----

    def Save(self, key: str, value) -> None:  # noqa: N802
        is_corpus, _ = self._is_corpus_key(key)
        if is_corpus:
            raise PermissionError(
                f"Cannot Save to corpus key {key!r} in local mode — bundle is read-only."
            )
        self._disk.save(key, value)

    def SaveBytes(self, key: str, value: bytes) -> None:  # noqa: N802
        self.Save(key, bytes(value))

    def Delete(self, key: str) -> None:       # noqa: N802
        is_corpus, _ = self._is_corpus_key(key)
        if is_corpus:
            raise PermissionError(
                f"Cannot Delete corpus key {key!r} in local mode — bundle is read-only."
            )
        self._disk.delete(key)

    @property
    def Keys(self):                           # noqa: N802
        out = [f"{CORPUS_PREFIX}{d}.npz" for d in self._bundle.dates()]
        out.extend(self._disk.keys())
        return out


# ---------------------------------------------------------------------------
# Top-level: LocalQuantBook shim and get_local_qb() factory
# ---------------------------------------------------------------------------

class LocalQuantBook:
    """Minimal surface compatible with the subset of QuantBook pipeline.py
    actually calls — i.e. `.ObjectStore.*`. Other QC APIs (Securities, History,
    AddData, ...) are out of scope for SSL training."""

    def __init__(self, bundle_path: Path, store_root: Path):
        self.ObjectStore = LocalObjectStore(  # noqa: N815 — C# surface
            bundle_view=_BundleView(bundle_path),
            disk=_DiskStore(store_root),
        )

    def __repr__(self):
        return f"LocalQuantBook(bundle={self.ObjectStore._bundle._bundle_path}, store={self.ObjectStore._disk.root})"


_local_qb: Optional[LocalQuantBook] = None
_init_lock = threading.Lock()


def get_local_qb(bundle_path: Optional[Path] = None,
                 store_root: Optional[Path] = None) -> LocalQuantBook:
    """Return a singleton LocalQuantBook. Idempotent; safe under threads."""
    global _local_qb
    if _local_qb is not None:
        return _local_qb
    with _init_lock:
        if _local_qb is None:
            bp = Path(bundle_path) if bundle_path else Path(
                os.environ.get("LOCAL_CORPUS_BUNDLE", str(BUNDLE_DEFAULT))
            )
            sr = Path(store_root) if store_root else Path(
                os.environ.get("LOCAL_STORE_ROOT", str(STORE_DEFAULT))
            )
            _local_qb = LocalQuantBook(bp, sr)
    return _local_qb


__all__ = ["LocalQuantBook", "LocalObjectStore", "get_local_qb",
           "BUNDLE_DEFAULT", "STORE_DEFAULT", "CORPUS_PREFIX"]
