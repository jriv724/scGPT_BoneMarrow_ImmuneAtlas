#!/usr/bin/env python3
"""Resumable, read-only scGPT embedding of a large AnnData atlas.

The source .h5ad is always opened in backed read-only mode. Raw counts are
copied from the requested layer into temporary in-memory AnnData chunks. The
resulting embeddings are stored separately as a NumPy .npy memmap.

Designed for scGPT 0.2.x and long-running tmux jobs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from scgpt.tasks.cell_emb import embed_data

# scGPT 0.2.1 expects scipy sparse matrices to expose the historical `.A`
# property. Newer SciPy releases removed it.
if not hasattr(sp.spmatrix, "A"):
    sp.spmatrix.A = property(lambda self: self.toarray())


SCRIPT_VERSION = "1.0.0"
EMBEDDING_FILENAME = "X_scGPT.npy"
PROGRESS_FILENAME = "progress.json"
MANIFEST_FILENAME = "manifest.json"
LOG_FILENAME = "run.log"
OBS_NAMES_FILENAME = "obs_names.txt.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate resumable scGPT cell embeddings from an AnnData count layer."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Source .h5ad",
    )
    parser.add_argument(
        "--layer", default="counts", help="Raw-count layer to use (default: counts)"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing args.json, best_model.pt, and vocab.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1_200)
    parser.add_argument(
        "--device",
        default="cuda",
        help="scGPT device argument, normally cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Technical pilot: process only the first N cells",
    )
    parser.add_argument(
        "--start-cell",
        type=int,
        default=0,
        help="First source row to process (default: 0)",
    )
    parser.add_argument(
        "--skip-count-validation",
        action="store_true",
        help="Skip per-chunk checks for finite, nonnegative, integer-like counts",
    )
    parser.add_argument(
        "--no-write-obs-names",
        action="store_true",
        help="Do not save the processed source obs_names as a gzipped text file",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Start over in an existing compatible output directory",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("scgpt_atlas")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / LOG_FILENAME, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def hash_strings(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def require_checkpoint_files(model_dir: Path) -> dict[str, str]:
    required = ["args.json", "best_model.pt", "vocab.json"]
    result: dict[str, str] = {}
    for name in required:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
        result[name] = str(path.resolve())
    return result


def checkpoint_fingerprint(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ["args.json", "best_model.pt", "vocab.json"]:
        path = model_dir / name
        stat = path.stat()
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        if name != "best_model.pt":
            with path.open("rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def validate_counts(matrix: sp.csr_matrix, start: int, end: int) -> None:
    values = matrix.data
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite count encountered in source rows [{start}, {end})")
    if np.any(values < 0):
        raise ValueError(f"Negative count encountered in source rows [{start}, {end})")
    if not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6):
        raise ValueError(f"Fractional count encountered in source rows [{start}, {end})")


def gpu_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        summary.update(
            {
                "device_index": device_index,
                "device_name": torch.cuda.get_device_name(device_index),
                "free_gib": round(free_bytes / 1024**3, 3),
                "total_gib": round(total_bytes / 1024**3, 3),
            }
        )
    return summary


def call_embed_data(
    chunk: ad.AnnData,
    model_dir: Path,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    """Call the installed scGPT 0.2.x API using only supported parameters."""
    signature = inspect.signature(embed_data)
    candidate_kwargs: dict[str, Any] = {
        "model_dir": str(model_dir),
        "gene_col": "feature_name",
        "max_length": max_length,
        "batch_size": batch_size,
        "device": device,
        "return_new_adata": False,
        "use_fast_transformer": False,
    }
    kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in signature.parameters
    }
    result = embed_data(chunk, **kwargs)
    target = result if result is not None else chunk

    if "X_scGPT" in target.obsm:
        embedding = np.asarray(target.obsm["X_scGPT"], dtype=np.float32)
    elif target is not chunk and target.n_obs == chunk.n_obs:
        embedding = np.asarray(target.X, dtype=np.float32)
    else:
        raise RuntimeError(
            "scGPT completed but no X_scGPT embedding was found in returned AnnData.obsm"
        )

    if embedding.ndim != 2 or embedding.shape[0] != chunk.n_obs:
        raise RuntimeError(
            f"Unexpected embedding shape {embedding.shape}; expected ({chunk.n_obs}, D)"
        )
    if not np.all(np.isfinite(embedding)):
        raise RuntimeError("scGPT produced nonfinite embedding values")
    return embedding


def write_obs_names(path: Path, obs_names, start: int, stop: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for name in obs_names[start:stop]:
            handle.write(str(name))
            handle.write("\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("chunk-size, batch-size, and max-length must be positive")
    if args.start_cell < 0:
        raise ValueError("start-cell must be nonnegative")
    if args.max_cells is not None and args.max_cells <= 0:
        raise ValueError("max-cells must be positive when provided")

    input_path = args.input.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    checkpoint_files = require_checkpoint_files(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("Starting run_scgpt_atlas.py version %s", SCRIPT_VERSION)
    logger.info("Source atlas: %s", input_path)
    logger.info("Raw-count layer: %s", args.layer)
    logger.info("Checkpoint: %s", model_dir)
    logger.info("Output: %s", output_dir)
    logger.info("GPU state: %s", json.dumps(gpu_summary(), sort_keys=True))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    source = ad.read_h5ad(input_path, backed="r")
    try:
        if args.layer not in source.layers:
            raise KeyError(
                f"Layer {args.layer!r} not found; available layers: {list(source.layers.keys())}"
            )
        if args.start_cell >= source.n_obs:
            raise ValueError(
                f"start-cell {args.start_cell} is outside atlas with {source.n_obs} cells"
            )

        source_stop = source.n_obs
        if args.max_cells is not None:
            source_stop = min(source_stop, args.start_cell + args.max_cells)
        output_n_obs = source_stop - args.start_cell

        obs_hash = hash_strings(source.obs_names[args.start_cell:source_stop])
        var_hash = hash_strings(source.var_names)
        stat = input_path.stat()
        checkpoint_hash = checkpoint_fingerprint(model_dir)
        run_identity = {
            "script_version": SCRIPT_VERSION,
            "input_path": str(input_path),
            "input_size_bytes": stat.st_size,
            "input_mtime_ns": stat.st_mtime_ns,
            "source_n_obs": source.n_obs,
            "source_n_vars": source.n_vars,
            "source_start_cell": args.start_cell,
            "source_stop_cell": source_stop,
            "output_n_obs": output_n_obs,
            "layer": args.layer,
            "obs_names_sha256": obs_hash,
            "var_names_sha256": var_hash,
            "checkpoint_dir": str(model_dir),
            "checkpoint_fingerprint": checkpoint_hash,
            "chunk_size": args.chunk_size,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        }
        identity_hash = hashlib.sha256(
            json.dumps(run_identity, sort_keys=True).encode("utf-8")
        ).hexdigest()

        progress_path = output_dir / PROGRESS_FILENAME
        embedding_path = output_dir / EMBEDDING_FILENAME
        manifest_path = output_dir / MANIFEST_FILENAME
        obs_names_path = output_dir / OBS_NAMES_FILENAME

        if args.overwrite_output:
            logger.warning("overwrite-output requested; previous progress will be ignored")
            completed_chunks: list[int] = []
            existing_progress = None
        elif progress_path.exists():
            with progress_path.open("r", encoding="utf-8") as handle:
                existing_progress = json.load(handle)
            if existing_progress.get("identity_hash") != identity_hash:
                raise RuntimeError(
                    "Existing output was created with different inputs or parameters. "
                    "Use a new output directory or --overwrite-output."
                )
            completed_chunks = [int(x) for x in existing_progress["completed_chunks"]]
            logger.info("Resuming with %d completed chunks", len(completed_chunks))
        else:
            existing_progress = None
            completed_chunks = []

        if not args.no_write_obs_names and (
            args.overwrite_output or not obs_names_path.exists()
        ):
            logger.info("Writing processed obs_names to %s", obs_names_path)
            write_obs_names(obs_names_path, source.obs_names, args.start_cell, source_stop)

        total_chunks = (output_n_obs + args.chunk_size - 1) // args.chunk_size
        embedding_store = None
        embedding_dim = None

        if embedding_path.exists() and not args.overwrite_output:
            embedding_store = np.lib.format.open_memmap(embedding_path, mode="r+")
            if embedding_store.shape[0] != output_n_obs:
                raise RuntimeError(
                    f"Existing embedding shape {embedding_store.shape} is incompatible"
                )
            embedding_dim = int(embedding_store.shape[1])

        started_at = existing_progress.get("started_at") if existing_progress else utc_now()
        completed_set = set(completed_chunks)

        for chunk_index in range(total_chunks):
            if chunk_index in completed_set:
                logger.info("Skipping completed chunk %d/%d", chunk_index + 1, total_chunks)
                continue

            output_start = chunk_index * args.chunk_size
            output_end = min(output_start + args.chunk_size, output_n_obs)
            source_start = args.start_cell + output_start
            source_end = args.start_cell + output_end
            chunk_started = time.monotonic()

            logger.info(
                "Processing chunk %d/%d: source rows [%d, %d)",
                chunk_index + 1,
                total_chunks,
                source_start,
                source_end,
            )

            counts = source.layers[args.layer][source_start:source_end, :]
            if not sp.issparse(counts):
                counts = sp.csr_matrix(counts)
            else:
                counts = counts.tocsr()
            counts = counts.astype(np.float32, copy=False)
            counts.eliminate_zeros()

            if not args.skip_count_validation:
                validate_counts(counts, source_start, source_end)

            obs = source.obs.iloc[source_start:source_end].copy()
            var = source.var.copy()
            var["feature_name"] = source.var_names.astype(str)
            chunk = ad.AnnData(X=counts, obs=obs, var=var)

            embedding = call_embed_data(
                chunk=chunk,
                model_dir=model_dir,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=args.device,
            )

            if embedding_store is None:
                embedding_dim = int(embedding.shape[1])
                embedding_store = np.lib.format.open_memmap(
                    embedding_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(output_n_obs, embedding_dim),
                )
                logger.info(
                    "Created embedding store with shape %s (~%.3f GiB)",
                    embedding_store.shape,
                    embedding_store.nbytes / 1024**3,
                )
            elif embedding.shape[1] != embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension changed from {embedding_dim} to {embedding.shape[1]}"
                )

            embedding_store[output_start:output_end, :] = embedding
            embedding_store.flush()

            completed_set.add(chunk_index)
            completed_chunks = sorted(completed_set)
            progress = {
                "identity_hash": identity_hash,
                "run_identity": run_identity,
                "started_at": started_at,
                "updated_at": utc_now(),
                "completed_chunks": completed_chunks,
                "total_chunks": total_chunks,
                "embedding_dim": embedding_dim,
                "embedding_path": str(embedding_path),
                "status": "running",
            }
            atomic_json_write(progress_path, progress)

            elapsed = time.monotonic() - chunk_started
            logger.info(
                "Completed chunk %d/%d in %.1f seconds (%.1f cells/s); GPU=%s",
                chunk_index + 1,
                total_chunks,
                elapsed,
                (source_end - source_start) / elapsed,
                json.dumps(gpu_summary(), sort_keys=True),
            )

            del chunk, counts, embedding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if embedding_store is None or embedding_dim is None:
            raise RuntimeError("No embedding data were produced or opened")
        embedding_store.flush()

        manifest = {
            "status": "complete",
            "completed_at": utc_now(),
            "identity_hash": identity_hash,
            "run_identity": run_identity,
            "embedding": {
                "path": str(embedding_path),
                "format": "numpy_npy_memmap",
                "dtype": "float32",
                "shape": [output_n_obs, embedding_dim],
                "source_row_interval": [args.start_cell, source_stop],
                "row_order": "identical_to_source_interval",
            },
            "checkpoint_files": checkpoint_files,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "scgpt": package_version("scgpt"),
                "anndata": package_version("anndata"),
                "scanpy": package_version("scanpy"),
                "numpy": np.__version__,
            },
            "gpu": gpu_summary(),
        }
        atomic_json_write(manifest_path, manifest)
        atomic_json_write(
            progress_path,
            {
                "identity_hash": identity_hash,
                "run_identity": run_identity,
                "started_at": started_at,
                "updated_at": utc_now(),
                "completed_chunks": list(range(total_chunks)),
                "total_chunks": total_chunks,
                "embedding_dim": embedding_dim,
                "embedding_path": str(embedding_path),
                "status": "complete",
            },
        )
        logger.info("Run complete: %s", embedding_path)
        logger.info("Manifest: %s", manifest_path)
        return 0
    finally:
        if getattr(source, "file", None) is not None:
            source.file.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
