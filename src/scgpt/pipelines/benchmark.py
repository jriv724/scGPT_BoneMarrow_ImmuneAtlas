#!/usr/bin/env python3
"""Resumable continual-vs-blood scGPT checkpoint benchmark.

Creates one immutable stratified pilot, generates embeddings for both official
checkpoints, calculates biological/batch metrics, and writes visual audits.
The source atlas is opened read-only and is never modified.
"""

from __future__ import annotations

import argparse
import gc
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
# Compatibility for scGPT 0.2.1 with newer SciPy sparse objects
if not hasattr(sp.spmatrix, "A"):
    sp.spmatrix.A = property(lambda self: self.toarray())
import seaborn as sns
import torch
from scgpt.tasks.cell_emb import embed_data
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


SCRIPT_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Atlas H5AD used to create/load the pilot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Benchmark output directory",
    )
    parser.add_argument(
        "--continual-checkpoint-dir",
        type=Path,
        required=True,
        help="Path to continual checkpoint directory",
    )
    parser.add_argument(
        "--blood-checkpoint-dir",
        type=Path,
        required=True,
        help="Path to blood checkpoint directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Embedding device passed to scGPT (default: cuda)",
    )
    parser.add_argument("--target-cells", type=int, default=50_000)
    parser.add_argument("--minimum-per-stratum", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1_200)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--silhouette-cells", type=int, default=5_000)
    parser.add_argument("--force-pilot", action="store_true")
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--force-umap", action="store_true")
    parser.add_argument("--skip-umap", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("scgpt_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "benchmark.log", mode="a"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def sha256_strings(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def gpu_state() -> dict:
    result = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "free_GiB": round(free / 1024**3, 3),
                "total_GiB": round(total / 1024**3, 3),
            }
        )
    return result


def validate_checkpoints(checkpoints: dict[str, Path], output_dir: Path) -> None:
    rows = []
    for model_name, model_dir in checkpoints.items():
        required = {name: model_dir / name for name in ("args.json", "best_model.pt", "vocab.json")}
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{model_name} checkpoint is incomplete: {missing}")
        with required["args.json"].open() as handle:
            config = json.load(handle)
        with required["vocab.json"].open() as handle:
            vocab = json.load(handle)
        rows.append(
            {
                "model": model_name,
                "directory": str(model_dir),
                "vocab_size": len(vocab),
                "weights_MiB": required["best_model.pt"].stat().st_size / 1024**2,
                "embsize": config.get("embsize"),
                "layers": config.get("nlayers"),
                "heads": config.get("nheads"),
                "max_seq_len": config.get("max_seq_len"),
                "n_bins": config.get("n_bins"),
                "input_style": config.get("input_style"),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "checkpoint_inventory.csv", index=False)


def create_or_load_pilot(
    source_path: Path,
    pilot_path: Path,
    target_cells: int,
    minimum_per_stratum: int,
    seed: int,
    force: bool,
    logger: logging.Logger,
) -> ad.AnnData:
    if pilot_path.exists() and not force:
        logger.info("Reusing pilot: %s", pilot_path)
        pilot = ad.read_h5ad(pilot_path)
        expected = pilot.uns["pilot_provenance"]["obs_names_sha256"]
        if sha256_strings(pilot.obs_names) != expected:
            raise RuntimeError("Existing pilot cell-order hash does not match its manifest")
        return pilot

    logger.info("Creating stratified %s-cell pilot", f"{target_cells:,}")
    source = ad.read_h5ad(source_path, backed="r")
    try:
        strata = (
            source.obs["dataset"].astype(str)
            + "___"
            + source.obs["preserved"].astype(str)
        )
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        positions = pd.Series(np.arange(source.n_obs), index=strata.to_numpy())
        for _, group in positions.groupby(level=0, sort=True):
            values = group.to_numpy()
            n = min(minimum_per_stratum, len(values))
            selected.extend(rng.choice(values, n, replace=False))
        selected_array = np.unique(np.asarray(selected, dtype=np.int64))
        if len(selected_array) < target_cells:
            available = np.setdiff1d(
                np.arange(source.n_obs), selected_array, assume_unique=True
            )
            selected_array = np.concatenate(
                [
                    selected_array,
                    rng.choice(
                        available,
                        target_cells - len(selected_array),
                        replace=False,
                    ),
                ]
            )
        elif len(selected_array) > target_cells:
            selected_array = rng.choice(selected_array, target_cells, replace=False)
        selected_array = np.sort(selected_array)

        counts = source.layers["counts"][selected_array, :]
        counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
        counts = counts.astype(np.float32, copy=False)
        if not np.all(np.isfinite(counts.data)) or np.any(counts.data < 0):
            raise ValueError("Pilot contains nonfinite or negative counts")
        if not np.allclose(counts.data, np.rint(counts.data), rtol=0, atol=1e-6):
            raise ValueError("Pilot contains fractional counts")

        pilot = ad.AnnData(
            X=counts.copy(),
            obs=source.obs.iloc[selected_array].copy(),
            var=source.var.copy(),
        )
        pilot.layers["counts"] = counts.copy()
        pilot.var["feature_name"] = pilot.var_names.astype(str)
        pilot.obs["source_atlas_row"] = selected_array
        pilot.uns["pilot_provenance"] = {
            "source": str(source_path),
            "seed": seed,
            "target_cells": target_cells,
            "minimum_per_dataset_celltype_stratum": minimum_per_stratum,
            "obs_names_sha256": sha256_strings(pilot.obs_names),
            "created_at": utc_now(),
        }

        temporary = pilot_path.with_name(pilot_path.stem + ".tmp.h5ad")
        pilot.write_h5ad(temporary)
        os.replace(temporary, pilot_path)
        logger.info("Pilot saved: %s", pilot_path)
        return pilot
    finally:
        source.file.close()


def vocabulary_coverage(
    pilot: ad.AnnData, checkpoints: dict[str, Path], output_dir: Path
) -> None:
    atlas_genes = set(pilot.var_names.astype(str))
    rows = []
    for name, model_dir in checkpoints.items():
        with (model_dir / "vocab.json").open() as handle:
            vocab = json.load(handle)
        overlap = atlas_genes.intersection(vocab)
        rows.append(
            {
                "model": name,
                "atlas_genes": len(atlas_genes),
                "overlap_genes": len(overlap),
                "atlas_gene_coverage": len(overlap) / len(atlas_genes),
                "vocab_size": len(vocab),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "gene_vocabulary_coverage.csv", index=False)


def run_embedding(
    model_name: str,
    model_dir: Path,
    pilot_path: Path,
    pilot_hash: str,
    output_dir: Path,
    batch_size: int,
    max_length: int,
    device: str,
    force: bool,
    logger: logging.Logger,
) -> Path:
    model_output = output_dir / model_name
    model_output.mkdir(exist_ok=True)
    embedding_path = model_output / "X_scGPT.npy"
    manifest_path = model_output / "manifest.json"

    if embedding_path.exists() and manifest_path.exists() and not force:
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        if manifest.get("pilot_obs_names_sha256") != pilot_hash:
            raise RuntimeError(f"{model_name} cached embedding belongs to another pilot")
        cached = np.load(embedding_path, mmap_mode="r")
        if tuple(cached.shape) != tuple(manifest["shape"]):
            raise RuntimeError(f"{model_name} cached embedding shape is inconsistent")
        logger.info("Reusing %s embedding: %s", model_name, embedding_path)
        return embedding_path

    logger.info("Embedding pilot with %s; GPU=%s", model_name, gpu_state())
    work = ad.read_h5ad(pilot_path)
    work.X = work.layers["counts"].copy()
    work.var["feature_name"] = work.var_names.astype(str)
    signature = inspect.signature(embed_data)
    candidates = {
        "model_dir": str(model_dir),
        "gene_col": "feature_name",
        "max_length": max_length,
        "batch_size": batch_size,
        "device": device,
        "return_new_adata": False,
        "use_fast_transformer": False,
    }
    kwargs = {key: value for key, value in candidates.items() if key in signature.parameters}
    started = time.time()
    returned = embed_data(work, **kwargs)
    target = returned if returned is not None else work
    if "X_scGPT" in target.obsm:
        embedding = np.asarray(target.obsm["X_scGPT"], dtype=np.float32)
    elif returned is not None and returned.n_obs == work.n_obs:
        embedding = np.asarray(returned.X, dtype=np.float32)
    else:
        raise RuntimeError(f"{model_name}: scGPT returned no recognizable embedding")
    if embedding.shape[0] != work.n_obs or not np.all(np.isfinite(embedding)):
        raise RuntimeError(f"{model_name}: invalid embedding shape or values")

    atomic_npy(embedding_path, embedding)
    atomic_json(
        manifest_path,
        {
            "model": model_name,
            "checkpoint": str(model_dir),
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
            "pilot_obs_names_sha256": pilot_hash,
            "elapsed_seconds": time.time() - started,
            "batch_size": batch_size,
            "max_length": max_length,
            "created_at": utc_now(),
            "torch": torch.__version__,
            "python": sys.version,
        },
    )
    logger.info("Saved %s embedding: %s", model_name, embedding_path)
    del work, returned, target, embedding
    gc.collect()
    torch.cuda.empty_cache()
    return embedding_path


def median_lisi(labels: np.ndarray, neighbors: np.ndarray) -> float:
    codes, _ = pd.factorize(labels, sort=True)
    values = np.empty(len(neighbors), dtype=np.float32)
    for index, row in enumerate(neighbors):
        counts = np.bincount(codes[row])
        probabilities = counts[counts > 0] / len(row)
        values[index] = 1.0 / np.sum(probabilities**2)
    return float(np.median(values))


def calculate_metrics(
    pilot: ad.AnnData,
    embedding_paths: dict[str, Path],
    output_dir: Path,
    k: int,
    silhouette_cells: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    macro = pilot.obs["macro_cell_type_v2"].astype(str).to_numpy()
    celltype = pilot.obs["preserved"].astype(str).to_numpy()
    dataset = pilot.obs["dataset"].astype(str).to_numpy()
    stage = pilot.obs["stage_model_v2"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    subset = rng.choice(pilot.n_obs, min(silhouette_cells, pilot.n_obs), replace=False)
    rows = []
    neighbor_indices = {}
    for model_name, path in embedding_paths.items():
        embedding = np.load(path, mmap_mode="r")
        logger.info("Calculating metrics for %s", model_name)
        nearest = NearestNeighbors(
            n_neighbors=k + 1, metric="cosine", n_jobs=-1
        ).fit(embedding)
        neighbors = nearest.kneighbors(return_distance=False)[:, 1:]
        neighbor_indices[model_name] = neighbors
        rows.append(
            {
                "model": model_name,
                "macro_knn_purity": np.mean(macro[neighbors] == macro[:, None]),
                "fine_celltype_knn_purity": np.mean(
                    celltype[neighbors] == celltype[:, None]
                ),
                "same_dataset_neighbor_fraction": np.mean(
                    dataset[neighbors] == dataset[:, None]
                ),
                "macro_LISI": median_lisi(macro, neighbors),
                "dataset_LISI": median_lisi(dataset, neighbors),
                "macro_silhouette": silhouette_score(
                    embedding[subset], macro[subset], metric="cosine"
                ),
                "dataset_silhouette": silhouette_score(
                    embedding[subset], dataset[subset], metric="cosine"
                ),
                "stage_silhouette_descriptive": silhouette_score(
                    embedding[subset], stage[subset], metric="cosine"
                ),
                "embedding_norm_mean": np.linalg.norm(embedding, axis=1).mean(),
                "embedding_nonfinite": (~np.isfinite(embedding)).sum(),
            }
        )
    metrics = pd.DataFrame(rows).set_index("model")
    metrics.to_csv(output_dir / "checkpoint_metrics.csv")
    names = list(neighbor_indices)
    if len(names) == 2:
        first, second = (neighbor_indices[name] for name in names)
        jaccard = np.asarray(
            [
                len(set(a).intersection(b)) / len(set(a).union(b))
                for a, b in zip(first, second)
            ]
        )
        pd.Series(jaccard, name="neighbor_jaccard").to_csv(
            output_dir / "cross_model_neighbor_jaccard.csv", index=False
        )
        logger.info("Median cross-model neighbor Jaccard: %.4f", np.median(jaccard))
    return metrics, neighbor_indices


def metric_dashboard(metrics: pd.DataFrame, output_dir: Path) -> None:
    specifications = [
        ("macro_knn_purity", "Macro-type neighbor purity ↑"),
        ("fine_celltype_knn_purity", "Fine cell-state neighbor purity ↑"),
        ("dataset_LISI", "Dataset neighborhood diversity ↑"),
        ("same_dataset_neighbor_fraction", "Same-dataset neighbors ↓"),
        ("macro_silhouette", "Macro-type silhouette ↑"),
        ("dataset_silhouette", "Dataset silhouette (closer to 0)"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    frame = metrics.reset_index()
    for axis, (column, title) in zip(axes.flat, specifications):
        sns.barplot(
            data=frame,
            x="model",
            y=column,
            hue="model",
            legend=False,
            ax=axis,
            palette="Set2",
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", padding=3)
    figure.suptitle("scGPT checkpoint benchmark", fontsize=16, y=1.01)
    figure.tight_layout()
    figure.savefig(
        output_dir / "checkpoint_metric_dashboard.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def matched_umaps(
    pilot: ad.AnnData,
    embedding_paths: dict[str, Path],
    output_dir: Path,
    seed: int,
    force: bool,
    logger: logging.Logger,
) -> None:
    umaps = {}
    for model_name, embedding_path in embedding_paths.items():
        cache = output_dir / model_name / "umap.npy"
        if cache.exists() and not force:
            coordinates = np.load(cache)
        else:
            logger.info("Calculating UMAP for %s", model_name)
            embedding = np.load(embedding_path, mmap_mode="r")
            coordinates = UMAP(
                n_neighbors=30,
                min_dist=0.3,
                metric="cosine",
                random_state=seed,
                n_jobs=1,
            ).fit_transform(embedding)
            atomic_npy(cache, coordinates)
        umaps[model_name] = coordinates

    colorings = [
        ("macro_cell_type_v2", pilot.obs["macro_cell_type_v2"].astype(str).to_numpy()),
        ("dataset", pilot.obs["dataset"].astype(str).to_numpy()),
        ("stage_model_v2", pilot.obs["stage_model_v2"].astype(str).to_numpy()),
    ]
    figure, axes = plt.subplots(
        len(colorings), len(embedding_paths), figsize=(18, 21), squeeze=False
    )
    for column, (model_name, coordinates) in enumerate(umaps.items()):
        for row, (label_name, labels) in enumerate(colorings):
            axis = axes[row, column]
            frame = pd.DataFrame(
                {"UMAP1": coordinates[:, 0], "UMAP2": coordinates[:, 1], "label": labels}
            )
            sns.scatterplot(
                data=frame,
                x="UMAP1",
                y="UMAP2",
                hue="label",
                s=4,
                linewidth=0,
                alpha=0.65,
                ax=axis,
                palette="tab20",
                rasterized=True,
            )
            axis.set_title(f"{model_name}: {label_name}")
            axis.set(xticks=[], yticks=[], xlabel="", ylabel="")
            axis.legend(
                title="",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                markerscale=3,
                fontsize=7,
                frameon=False,
            )
    figure.tight_layout()
    figure.savefig(output_dir / "matched_umap_audit.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def transition_heatmaps(
    pilot: ad.AnnData,
    neighbor_indices: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    macro = pilot.obs["macro_cell_type_v2"].astype(str).to_numpy()
    categories = sorted(pd.unique(macro))
    codes = {label: index for index, label in enumerate(categories)}
    figure, axes = plt.subplots(
        1, len(neighbor_indices), figsize=(18, 7), sharex=True, sharey=True, squeeze=False
    )
    for axis, (model_name, neighbors) in zip(axes.flat, neighbor_indices.items()):
        matrix = np.zeros((len(categories), len(categories)), dtype=float)
        for index, row in enumerate(neighbors):
            source_code = codes[macro[index]]
            neighbor_codes = np.fromiter((codes[label] for label in macro[row]), dtype=int)
            matrix[source_code] += np.bincount(neighbor_codes, minlength=len(categories))
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        sns.heatmap(
            matrix,
            xticklabels=categories,
            yticklabels=categories,
            cmap="mako",
            vmin=0,
            vmax=1,
            ax=axis,
            cbar_kws={"label": "neighbor fraction"},
        )
        axis.set_title(model_name)
        axis.set_xlabel("Neighbor macro type")
        axis.set_ylabel("Cell macro type")
    figure.tight_layout()
    figure.savefig(
        output_dir / "macro_neighbor_transition_heatmaps.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.source = args.source.resolve()
    args.output_dir = args.output_dir.resolve()
    args.continual_checkpoint_dir = args.continual_checkpoint_dir.resolve()
    args.blood_checkpoint_dir = args.blood_checkpoint_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    checkpoints = {
        "continual": args.continual_checkpoint_dir,
        "blood": args.blood_checkpoint_dir,
    }
    pilot_path = args.output_dir / f"scgpt_pilot_{args.target_cells // 1000}k.h5ad"
    logger.info("Starting benchmark version %s", SCRIPT_VERSION)
    logger.info("GPU state: %s", gpu_state())
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False"
        )
    validate_checkpoints(checkpoints, args.output_dir)
    pilot = create_or_load_pilot(
        args.source,
        pilot_path,
        args.target_cells,
        args.minimum_per_stratum,
        args.seed,
        args.force_pilot,
        logger,
    )
    pilot_hash = sha256_strings(pilot.obs_names)
    vocabulary_coverage(pilot, checkpoints, args.output_dir)
    embedding_paths = {
        name: run_embedding(
            name,
            model_dir,
            pilot_path,
            pilot_hash,
            args.output_dir,
            args.batch_size,
            args.max_length,
            args.device,
            args.force_embed,
            logger,
        )
        for name, model_dir in checkpoints.items()
    }
    metrics, neighbors = calculate_metrics(
        pilot,
        embedding_paths,
        args.output_dir,
        args.neighbors,
        args.silhouette_cells,
        args.seed,
        logger,
    )
    metric_dashboard(metrics, args.output_dir)
    transition_heatmaps(pilot, neighbors, args.output_dir)
    if not args.skip_umap:
        matched_umaps(
            pilot,
            embedding_paths,
            args.output_dir,
            args.seed,
            args.force_umap,
            logger,
        )
    atomic_json(
        args.output_dir / "benchmark_manifest.json",
        {
            "status": "complete",
            "completed_at": utc_now(),
            "script_version": SCRIPT_VERSION,
            "source": str(args.source),
            "pilot": str(pilot_path),
            "pilot_obs_names_sha256": pilot_hash,
            "checkpoints": {key: str(value) for key, value in checkpoints.items()},
            "parameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "anndata": ad.__version__ if hasattr(ad, "__version__") else None,
                "numpy": np.__version__,
            },
        },
    )
    logger.info("Benchmark complete. Outputs: %s", args.output_dir)
    logger.info("Metrics:\n%s", metrics.to_string())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume cached stages.", file=sys.stderr)
        raise SystemExit(130)
