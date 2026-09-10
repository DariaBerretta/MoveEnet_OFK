#!/usr/bin/env python3
"""Aggregate repeated energy runs and join them to per-sample accuracy.

Expected accuracy CSV columns:
    dataset,sample_id,model,pck,mpjpe

Energy input is the existing ExpEnergy/measure_energy.py output.
The script first averages repetitions within each sample, preserving the sample
as the statistical unit, then joins energy and accuracy and exports dataset/model
summaries plus the four Energy-vs-Accuracy scatter plots used in the paper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--energy", required=True, type=Path)
    p.add_argument("--accuracy", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def normalize_dataset(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"h36m", "eh36m", "e-h36m"}:
        return "h36m"
    if v == "dhp19":
        return "dhp19"
    return v


def scatter(summary: pd.DataFrame, dataset: str, metric: str, out: Path) -> None:
    sub = summary[summary["dataset"] == dataset].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(sub["energy_mean_j"], sub[f"{metric}_mean"])
    for _, row in sub.iterrows():
        ax.annotate(
            row["model"],
            (row["energy_mean_j"], row[f"{metric}_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Measured compute energy [J]")
    ax.set_ylabel("PCK" if metric == "pck" else "MPJPE [px]")
    ax.set_title(f"{dataset}: Energy vs {'PCK' if metric == 'pck' else 'MPJPE'}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    energy = pd.read_csv(args.energy)
    accuracy = pd.read_csv(args.accuracy)

    required_energy = {"dataset", "sample_id", "model", "status", "measured_compute_energy_j"}
    required_accuracy = {"dataset", "sample_id", "model", "pck", "mpjpe"}
    missing_e = required_energy - set(energy.columns)
    missing_a = required_accuracy - set(accuracy.columns)
    if missing_e:
        raise SystemExit(f"Energy CSV missing columns: {sorted(missing_e)}")
    if missing_a:
        raise SystemExit(f"Accuracy CSV missing columns: {sorted(missing_a)}")

    energy = energy[energy["status"] == "OK"].copy()
    energy["dataset"] = energy["dataset"].map(normalize_dataset)
    accuracy["dataset"] = accuracy["dataset"].map(normalize_dataset)
    energy["measured_compute_energy_j"] = pd.to_numeric(
        energy["measured_compute_energy_j"], errors="coerce"
    )
    accuracy["pck"] = pd.to_numeric(accuracy["pck"], errors="coerce")
    accuracy["mpjpe"] = pd.to_numeric(accuracy["mpjpe"], errors="coerce")

    # Repetitions are technical repeats; collapse them before any dataset-level
    # inference so the sequence/sample remains the experimental unit.
    sample_energy = (
        energy.groupby(["dataset", "sample_id", "model"], as_index=False)
        .agg(
            energy_mean_j=("measured_compute_energy_j", "mean"),
            energy_sd_j=("measured_compute_energy_j", "std"),
            energy_median_j=("measured_compute_energy_j", "median"),
            repetitions=("measured_compute_energy_j", "count"),
        )
    )

    sample_accuracy = (
        accuracy.groupby(["dataset", "sample_id", "model"], as_index=False)
        .agg(pck=("pck", "mean"), mpjpe=("mpjpe", "mean"))
    )

    merged = sample_energy.merge(
        sample_accuracy,
        on=["dataset", "sample_id", "model"],
        how="inner",
        validate="one_to_one",
    )
    merged.to_csv(args.out / "energy_accuracy_by_sample.csv", index=False)

    summary = (
        merged.groupby(["dataset", "model"], as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            energy_mean_j=("energy_mean_j", "mean"),
            energy_sd_across_samples_j=("energy_mean_j", "std"),
            pck_mean=("pck", "mean"),
            pck_sd=("pck", "std"),
            mpjpe_mean=("mpjpe", "mean"),
            mpjpe_sd=("mpjpe", "std"),
        )
    )
    summary.to_csv(args.out / "energy_accuracy_summary.csv", index=False)

    for dataset in ("h36m", "dhp19"):
        scatter(summary, dataset, "pck", args.out / f"{dataset}_energy_vs_pck.png")
        scatter(summary, dataset, "mpjpe", args.out / f"{dataset}_energy_vs_mpjpe.png")

    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
