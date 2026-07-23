#!/usr/bin/env python3
"""Audit scGPT-based cell-type labeling against existing atlas annotations.

This script trains a multinomial classifier head on frozen scGPT cell embeddings.
Cells are split by biological sample (never randomly by cell), so test predictions
come from samples unseen during classifier training. The source atlas is never
opened for writing.

Expected inputs are produced by run_checkpoint_benchmark.py:
  scgpt_pilot_50k.h5ad
  <model>/X_scGPT.npy
  <model>/umap.npy                 (optional)

Outputs include per-cell predictions/confidences, metrics, confusion matrices,
per-class scores, disagreement audits, fitted classifier heads, and a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "2.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--enriched-h5ad",
        type=Path,
        default=None,
        help="New H5AD containing embeddings, predictions, metrics, and original pilot counts",
    )
    parser.add_argument(
        "--full-source-atlas",
        type=Path,
        default=None,
        help="Path to the full source atlas used for provenance metadata",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["continual", "blood"],
        help="Subdirectories containing X_scGPT.npy",
    )
    parser.add_argument("--label-column", default="preserved")
    parser.add_argument("--macro-column", default="macro_cell_type_v2")
    parser.add_argument("--group-column", default="biological_sample_id")
    parser.add_argument("--fallback-group-column", default="sample")
    parser.add_argument("--dataset-column", default="dataset")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--minimum-class-cells", type=int, default=30)
    parser.add_argument("--minimum-class-groups", type=int, default=2)
    parser.add_argument("--max-split-attempts", type=int, default=250)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(out: Path) -> logging.Logger:
    logger = logging.getLogger("scgpt_annotation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(out / "annotation.log", mode="a"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def clean_series(series: pd.Series) -> pd.Series:
    result = series.astype("string").fillna("__missing__").str.strip()
    return result.replace({"": "__missing__", "nan": "__missing__", "None": "__missing__"})


def canonicalize_formatting_aliases(labels: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Merge labels differing only by case, whitespace, punctuation, or separators."""
    original = clean_series(labels)
    keys = original.str.lower().map(lambda value: re.sub(r"[^a-z0-9]+", "", value))
    counts = (
        pd.DataFrame({"key": keys, "original": original})
        .groupby(["key", "original"], observed=True)
        .size()
        .rename("cells")
        .reset_index()
        .sort_values(["key", "cells", "original"], ascending=[True, False, True])
    )
    representatives = counts.drop_duplicates("key").set_index("key")["original"]
    canonical = keys.map(representatives).astype("string")
    alias_map = counts.copy()
    alias_map["canonical"] = alias_map["key"].map(representatives)
    alias_map["changed"] = alias_map["original"].ne(alias_map["canonical"])
    return canonical, alias_map[["key", "original", "canonical", "cells", "changed"]]


def choose_groups(obs: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    if primary in obs:
        groups = clean_series(obs[primary])
        missing = groups.eq("__missing__")
        if missing.any() and fallback in obs:
            groups.loc[missing] = "fallback::" + clean_series(obs.loc[missing, fallback])
        if groups.nunique() >= 3:
            return groups
    if fallback not in obs:
        raise KeyError(f"Neither usable group column {primary!r} nor {fallback!r} exists")
    groups = clean_series(obs[fallback])
    if groups.nunique() < 3:
        raise ValueError("At least three biological/sample groups are required")
    return groups


def eligible_mask(labels: pd.Series, groups: pd.Series, min_cells: int, min_groups: int) -> np.ndarray:
    table = pd.DataFrame({"label": labels, "group": groups})
    cells = table.groupby("label", observed=True).size()
    group_counts = table.groupby("label", observed=True)["group"].nunique()
    keep = cells.index[(cells >= min_cells) & (group_counts >= min_groups)]
    return labels.isin(keep).to_numpy()


def split_score(y: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple:
    all_classes = set(np.unique(y))
    train_classes = set(np.unique(y[train]))
    test_classes = set(np.unique(y[test]))
    missing_train = len(all_classes - train_classes)
    missing_test = len(all_classes - test_classes)
    size_error = abs(len(test) / len(y) - 0.20)
    return missing_train, missing_test, size_error


def make_within_dataset_splits(
    groups: np.ndarray,
    datasets: np.ndarray,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign whole samples within every dataset; require train/val/test presence."""
    split = np.full(len(groups), "excluded", dtype=object)
    rows = []
    rng = np.random.default_rng(seed)
    for dataset in sorted(np.unique(datasets)):
        positions = np.flatnonzero(datasets == dataset)
        unique_groups = np.unique(groups[positions])
        n_groups = len(unique_groups)
        if n_groups < 3:
            rows.append(
                {
                    "dataset": dataset,
                    "groups": n_groups,
                    "cells": len(positions),
                    "status": "excluded_fewer_than_3_groups",
                    "train_groups": 0,
                    "validation_groups": 0,
                    "test_groups": 0,
                }
            )
            continue
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(n_groups * test_size)))
        n_validation = max(1, int(round(n_groups * validation_size)))
        while n_test + n_validation > n_groups - 1:
            if n_test >= n_validation and n_test > 1:
                n_test -= 1
            elif n_validation > 1:
                n_validation -= 1
            else:
                break
        test_groups = set(shuffled[:n_test])
        validation_groups = set(shuffled[n_test : n_test + n_validation])
        train_groups = set(shuffled[n_test + n_validation :])
        local_groups = groups[positions]
        split[positions[np.isin(local_groups, list(train_groups))]] = "train"
        split[positions[np.isin(local_groups, list(validation_groups))]] = "validation"
        split[positions[np.isin(local_groups, list(test_groups))]] = "test"
        rows.append(
            {
                "dataset": dataset,
                "groups": n_groups,
                "cells": len(positions),
                "status": "included",
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "test_groups": len(test_groups),
            }
        )
    return split, pd.DataFrame(rows)


def metrics_row(y_true: np.ndarray, y_pred: np.ndarray, model: str, level: str) -> dict:
    return {
        "model": model,
        "level": level,
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    size = max(10, min(28, 0.42 * len(labels) + 5))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        matrix,
        cmap="mako",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        ax=ax,
        cbar_kws={"label": "Fraction of true class"},
    )
    ax.set(title=title, xlabel="scGPT classifier prediction", ylabel="Existing atlas label")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_f1(report: pd.DataFrame, path: Path, title: str) -> None:
    shown = report.loc[~report.index.isin(["accuracy", "macro avg", "weighted avg"])]
    shown = shown.sort_values("f1-score")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.25 * len(shown) + 2)))
    ax.barh(shown.index, shown["f1-score"], color="#3977a8")
    ax.set(xlim=(0, 1), xlabel="Held-out F1", ylabel="Cell type", title=title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confidence(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for correct, color, label in [(True, "#2a9d8f", "correct"), (False, "#e76f51", "discordant")]:
        values = frame.loc[frame["correct"] == correct, "confidence"]
        if len(values):
            ax.hist(values, bins=np.linspace(0, 1, 41), alpha=0.58, density=True, color=color, label=f"{label} (n={len(values):,})")
    ax.set(xlabel="Maximum predicted probability", ylabel="Density", title=title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_umap_audit(umap: np.ndarray, frame: pd.DataFrame, path: Path, title: str) -> None:
    test = frame["split"].eq("test").to_numpy()
    discordant = test & ~frame["correct"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(umap[:, 0], umap[:, 1], s=0.7, c="#d8d8d8", rasterized=True)
    axes[0].scatter(umap[test, 0], umap[test, 1], s=1.2, c="#3977a8", rasterized=True)
    axes[0].set_title("Held-out test cells")
    axes[1].scatter(umap[:, 0], umap[:, 1], s=0.7, c="#d8d8d8", rasterized=True)
    axes[1].scatter(umap[discordant, 0], umap[discordant, 1], s=2, c="#d1495b", rasterized=True)
    axes[1].set_title(f"Discordant predictions (n={discordant.sum():,})")
    for ax in axes:
        ax.set(xticks=[], yticks=[])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_dashboard(metrics: pd.DataFrame, path: Path) -> None:
    measures = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]
    long = metrics.melt(id_vars=["model", "level"], value_vars=measures, var_name="metric", value_name="value")
    chart = sns.catplot(
        data=long,
        x="metric",
        y="value",
        hue="model",
        col="level",
        kind="bar",
        height=5,
        aspect=1.25,
        sharey=True,
    )
    chart.set(ylim=(-0.05, 1.0), xlabel="", ylabel="Held-out score")
    chart.set_xticklabels(rotation=35, ha="right")
    chart.fig.subplots_adjust(top=0.82)
    chart.fig.suptitle("scGPT annotation agreement with existing atlas labels")
    chart.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(chart.fig)


def main() -> int:
    args = parse_args()
    if args.enriched_h5ad is None:
        args.enriched_h5ad = args.output_dir / "scgpt_pilot_annotated.h5ad"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    logger = setup_logging(args.output_dir)
    logger.info("Starting scGPT annotation audit version %s", SCRIPT_VERSION)

    if not args.pilot.is_file():
        raise FileNotFoundError(args.pilot)
    pilot = ad.read_h5ad(args.pilot, backed="r")
    required = [args.label_column, args.macro_column, args.dataset_column]
    missing = [column for column in required if column not in pilot.obs]
    if missing:
        raise KeyError(f"Pilot is missing obs columns: {missing}")

    obs = pilot.obs.copy()
    obs.index = obs.index.astype(str)
    fine_original = clean_series(obs[args.label_column])
    fine, alias_map = canonicalize_formatting_aliases(fine_original)
    alias_map.to_csv(args.output_dir / "fine_label_alias_map.csv", index=False)
    logger.info(
        "Canonicalized %s formatting-only aliases across %s label keys",
        f"{int(alias_map['changed'].sum()):,}",
        f"{alias_map['key'].nunique():,}",
    )
    macro = clean_series(obs[args.macro_column])
    datasets = clean_series(obs[args.dataset_column])
    raw_groups = choose_groups(obs, args.group_column, args.fallback_group_column)
    groups = datasets + "::" + raw_groups
    eligible = eligible_mask(
        fine, groups, args.minimum_class_cells, args.minimum_class_groups
    )
    if eligible.sum() < 100:
        raise ValueError("Too few eligible labeled cells after class filtering")

    eligible_indices = np.flatnonzero(eligible)
    eligible_split, dataset_split_audit = make_within_dataset_splits(
        groups.to_numpy()[eligible],
        datasets.to_numpy()[eligible],
        args.test_size,
        args.validation_size,
        args.seed,
    )
    dataset_split_audit.to_csv(args.output_dir / "dataset_split_audit.csv", index=False)
    split = np.full(pilot.n_obs, "excluded", dtype=object)
    split[eligible_indices] = eligible_split

    # A label cannot be evaluated if no training cell carries that label.
    train_labels = set(fine.to_numpy()[split == "train"])
    labels_without_training = sorted(set(fine.to_numpy()[split != "excluded"]) - train_labels)
    if labels_without_training:
        logger.warning("Excluding labels absent from training: %s", labels_without_training)
        split[np.isin(fine.to_numpy(), labels_without_training)] = "excluded"
    pd.DataFrame({"label_absent_from_training": labels_without_training}).to_csv(
        args.output_dir / "labels_excluded_no_training.csv", index=False
    )
    split_table = pd.DataFrame(
        {
            "cell": obs.index,
            "split": split,
            "fine_label_original": fine_original.to_numpy(),
            "fine_label": fine.to_numpy(),
            "macro_label": macro.to_numpy(),
            "group": groups.to_numpy(),
            "dataset": datasets.to_numpy(),
        }
    )
    split_table.to_csv(args.output_dir / "sample_level_split.csv.gz", index=False)
    logger.info("Split counts: %s", pd.Series(split).value_counts().to_dict())
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = set(groups[split == left]) & set(groups[split == right])
        if overlap:
            raise RuntimeError(f"Group leakage between {left} and {right}: {sorted(overlap)[:5]}")

    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    fit = train | validation
    fine_to_macro = (
        pd.DataFrame({"fine": fine[train], "macro": macro[train]})
        .groupby("fine", observed=True)["macro"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )

    metric_rows = []
    model_predictions = []
    for model_name in args.models:
        model_out = args.output_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        predictions_path = model_out / "predictions.csv.gz"
        if predictions_path.exists() and not args.force:
            logger.info("Reusing completed predictions for %s", model_name)
            cached = pd.read_csv(predictions_path)
            model_predictions.append(cached)
            cached_test = cached["split"].eq("test")
            metric_rows.append(metrics_row(cached.loc[cached_test, "atlas_fine"], cached.loc[cached_test, "predicted_fine"], model_name, "fine"))
            metric_rows.append(metrics_row(cached.loc[cached_test, "atlas_macro"], cached.loc[cached_test, "predicted_macro"], model_name, "macro"))
            continue

        embedding_path = args.benchmark_dir / model_name / "X_scGPT.npy"
        if not embedding_path.is_file():
            raise FileNotFoundError(
                f"Missing {embedding_path}. Run the checkpoint embedding benchmark first."
            )
        embedding = np.load(embedding_path, mmap_mode="r")
        if embedding.shape[0] != pilot.n_obs:
            raise ValueError(f"{model_name} embedding rows {embedding.shape[0]} != pilot rows {pilot.n_obs}")
        if not np.isfinite(embedding).all():
            raise ValueError(f"{model_name} embedding contains nonfinite values")

        logger.info("Fitting %s classifier on %s cells", model_name, f"{fit.sum():,}")
        classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=args.regularization_c,
                        class_weight="balanced",
                        max_iter=args.max_iter,
                        solver="lbfgs",
                        multi_class="auto",
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        classifier.fit(np.asarray(embedding[fit]), fine.to_numpy()[fit])
        joblib.dump(classifier, model_out / "classifier_head.joblib", compress=3)

        predicted = np.full(pilot.n_obs, "__excluded__", dtype=object)
        confidence = np.full(pilot.n_obs, np.nan, dtype=np.float32)
        evaluate = eligible
        probabilities = classifier.predict_proba(np.asarray(embedding[evaluate]))
        local_prediction = classifier.classes_[np.argmax(probabilities, axis=1)]
        predicted[evaluate] = local_prediction
        confidence[evaluate] = probabilities.max(axis=1).astype(np.float32)
        predicted_macro = np.asarray(
            [fine_to_macro.get(value, "__unmapped__") for value in predicted], dtype=object
        )

        frame = pd.DataFrame(
            {
                "cell": obs.index,
                "model": model_name,
                "split": split,
                "group": groups.to_numpy(),
                "dataset": datasets.to_numpy(),
                "atlas_fine_original": fine_original.to_numpy(),
                "atlas_fine": fine.to_numpy(),
                "predicted_fine": predicted,
                "confidence": confidence,
                "atlas_macro": macro.to_numpy(),
                "predicted_macro": predicted_macro,
            }
        )
        frame["correct"] = frame["atlas_fine"].eq(frame["predicted_fine"])
        frame.to_csv(predictions_path, index=False, compression="gzip")
        model_predictions.append(frame)

        fine_report = pd.DataFrame(
            classification_report(
                fine.to_numpy()[test], predicted[test], output_dict=True, zero_division=0
            )
        ).T
        macro_report = pd.DataFrame(
            classification_report(
                macro.to_numpy()[test], predicted_macro[test], output_dict=True, zero_division=0
            )
        ).T
        fine_report.to_csv(model_out / "fine_classification_report.csv")
        macro_report.to_csv(model_out / "macro_classification_report.csv")
        metric_rows.append(metrics_row(fine.to_numpy()[test], predicted[test], model_name, "fine"))
        metric_rows.append(metrics_row(macro.to_numpy()[test], predicted_macro[test], model_name, "macro"))

        plot_confusion(fine.to_numpy()[test], predicted[test], model_out / "fine_confusion_matrix.png", f"{model_name}: fine labels, held-out samples")
        plot_confusion(macro.to_numpy()[test], predicted_macro[test], model_out / "macro_confusion_matrix.png", f"{model_name}: macro labels, held-out samples")
        plot_class_f1(fine_report, model_out / "fine_per_class_f1.png", f"{model_name}: held-out fine-label F1")
        plot_confidence(frame.loc[test], model_out / "confidence_audit.png", f"{model_name}: prediction confidence")

        disagreement = (
            frame.loc[test & ~frame["correct"]]
            .groupby(["atlas_fine", "predicted_fine"], observed=True)
            .size()
            .sort_values(ascending=False)
            .rename("cells")
            .reset_index()
        )
        disagreement.to_csv(model_out / "top_disagreements.csv", index=False)
        by_dataset = (
            frame.loc[test]
            .groupby("dataset", observed=True)["correct"]
            .agg(cells="size", agreement="mean")
            .sort_values("agreement")
        )
        by_dataset.to_csv(model_out / "agreement_by_dataset.csv")

        umap_path = args.benchmark_dir / model_name / "umap.npy"
        if umap_path.is_file():
            umap = np.load(umap_path)
            if umap.shape == (pilot.n_obs, 2):
                plot_umap_audit(umap, frame, model_out / "disagreement_umap.png", f"{model_name}: held-out annotation audit")

        atomic_json(
            model_out / "manifest.json",
            {
                "completed_utc": utc_now(),
                "model": model_name,
                "embedding": str(embedding_path),
                "embedding_sha256": file_sha256(embedding_path),
                "n_features": int(embedding.shape[1]),
                "n_classes": int(len(classifier.classes_)),
                "classes": classifier.classes_.tolist(),
            },
        )

    metrics = pd.DataFrame(metric_rows).sort_values(["level", "macro_f1"], ascending=[True, False])
    metrics.to_csv(args.output_dir / "annotation_metrics.csv", index=False)
    plot_metric_dashboard(metrics, args.output_dir / "annotation_metric_dashboard.png")

    combined = pd.concat(model_predictions, ignore_index=True)
    test_combined = combined[combined["split"].eq("test")]
    pairwise = []
    for i, left in enumerate(args.models):
        left_frame = test_combined[test_combined["model"].eq(left)].set_index("cell")
        for right in args.models[i + 1 :]:
            right_frame = test_combined[test_combined["model"].eq(right)].set_index("cell")
            common = left_frame.index.intersection(right_frame.index)
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "n_test": len(common),
                    "prediction_agreement": float(
                        np.mean(
                            left_frame.loc[common, "predicted_fine"].to_numpy()
                            == right_frame.loc[common, "predicted_fine"].to_numpy()
                        )
                    ),
                }
            )
    pd.DataFrame(pairwise).to_csv(args.output_dir / "cross_model_prediction_agreement.csv", index=False)

    # Assemble one self-contained audit object. This is always a NEW file; the
    # pilot and full source atlas are never opened for writing.
    logger.info("Assembling enriched H5AD: %s", args.enriched_h5ad)
    enriched = ad.read_h5ad(args.pilot)
    if "counts" not in enriched.layers:
        raise RuntimeError("Pilot has no counts layer; refusing to write an incomplete deliverable")
    enriched.obs["scgpt_annotation_split"] = pd.Categorical(split)
    enriched.obs["preserved_canonical"] = pd.Categorical(fine.to_numpy())
    for model_name, frame in zip(args.models, model_predictions):
        aligned = frame.set_index("cell").reindex(enriched.obs_names.astype(str))
        if aligned.index.has_duplicates or aligned["model"].isna().any():
            raise RuntimeError(f"Could not align {model_name} predictions to pilot cells")
        embedding_path = args.benchmark_dir / model_name / "X_scGPT.npy"
        embedding = np.load(embedding_path)
        if embedding.shape[0] != enriched.n_obs:
            raise RuntimeError(f"Could not align {model_name} embedding to pilot cells")
        enriched.obsm[f"X_scGPT_{model_name}"] = embedding.astype(np.float32, copy=False)
        enriched.obs[f"scgpt_{model_name}_predicted_preserved"] = pd.Categorical(
            aligned["predicted_fine"].to_numpy()
        )
        enriched.obs[f"scgpt_{model_name}_prediction_confidence"] = aligned[
            "confidence"
        ].to_numpy(dtype=np.float32)
        enriched.obs[f"scgpt_{model_name}_predicted_macro"] = pd.Categorical(
            aligned["predicted_macro"].to_numpy()
        )
        enriched.obs[f"scgpt_{model_name}_agrees_preserved"] = aligned[
            "correct"
        ].to_numpy(dtype=bool)
        umap_path = args.benchmark_dir / model_name / "umap.npy"
        if umap_path.is_file():
            umap = np.load(umap_path)
            if umap.shape == (enriched.n_obs, 2):
                enriched.obsm[f"X_umap_scGPT_{model_name}"] = umap.astype(
                    np.float32, copy=False
                )
    audit_metadata = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
        "models": list(args.models),
        "reference_fine_label": args.label_column,
        "reference_macro_label": args.macro_column,
        "split_group": args.group_column,
        "source_pilot": str(args.pilot),
        "source_atlas_modified": False,
    }
    if args.full_source_atlas is not None:
        audit_metadata["full_source_atlas"] = str(args.full_source_atlas)
    enriched.uns["scgpt_annotation_audit"] = audit_metadata
    enriched.uns["scgpt_annotation_metrics"] = metrics.reset_index(drop=True)
    args.enriched_h5ad.parent.mkdir(parents=True, exist_ok=True)
    temporary_h5ad = args.enriched_h5ad.with_suffix(".tmp.h5ad")
    enriched.write_h5ad(temporary_h5ad, compression="gzip")
    os.replace(temporary_h5ad, args.enriched_h5ad)
    logger.info("Enriched H5AD saved: %s", args.enriched_h5ad)

    atomic_json(
        args.output_dir / "annotation_manifest.json",
        {
            "script_version": SCRIPT_VERSION,
            "completed_utc": utc_now(),
            "pilot": str(args.pilot),
            "pilot_sha256": file_sha256(args.pilot),
            "models": args.models,
            "label_column": args.label_column,
            "canonicalization": "lowercase and remove non-alphanumeric characters; retain most frequent display label",
            "macro_column": args.macro_column,
            "group_column_requested": args.group_column,
            "fallback_group_column": args.fallback_group_column,
            "split_protocol": "biological-sample-disjoint train/validation/test within each dataset; datasets with fewer than 3 groups excluded",
            "seed": args.seed,
            "split_counts": pd.Series(split).value_counts().to_dict(),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "source_h5ad_modified": False,
            "enriched_h5ad": str(args.enriched_h5ad),
        },
    )
    logger.info("Annotation audit complete: %s", args.output_dir)
    logger.info("Metrics:\n%s", metrics.to_string(index=False))
    pilot.file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
