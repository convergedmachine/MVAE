#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare VAE vs MVAE across metrics by dataset & latent dimension.

- Walks dataset folders (e.g., CIFAR10/, MNIST_BASIC/, …)
- Reads files named like: <dataset>_<model>_d<latent>_s<seed>_metrics.json
  where model ∈ {vae, mvae}
- Aggregates metrics and writes:
    * per-dataset JSON summary
    * per-dataset Markdown table (bold best, arrows for higher/lower-better)
    * per-dataset CSV table

Usage:
    python compare_vae_mvae.py --root . --outdir aggregates
"""

import os
import re
import json
import argparse
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Any

# -------------------------
# Configuration
# -------------------------

FILENAME_RE = re.compile(r"""
    ^(?P<dataset>.+?)_
    (?P<model>m?vae)_
    d(?P<latent>\d+)_
    s(?P<seed>\d+)
    _metrics\.json$
""", re.VERBOSE | re.IGNORECASE)

# Metrics to compare and their "direction"
# True  -> higher is better (↑)
# False -> lower  is better (↓)
METRICS_DIR = OrderedDict([
    ("mse/test",          False),
    ("probe/acc",         True),
    ("probe/nll",         False),
    ("probe/brier",       False),
    ("probe/ece",         False),
    ("cluster/nmi",       True),
    ("cluster/ari",       True),
])

# Nice arrow markers for headers
ARROW = {True: "↑", False: "↓"}

# -------------------------
# Helpers
# -------------------------

def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Failed to read {path}: {e}")
        return {}

def bold(s: str) -> str:
    return f"**{s}**"

def fmt_float(x: Any) -> str:
    if x is None:
        return "—"
    try:
        # compact but readable
        return f"{x:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def percent_change(new: float, base: float, higher_is_better: bool) -> str:
    """
    Relative improvement of 'new' vs 'base' in percent.
    For higher-is-better:  (new - base) / |base|
    For lower-is-better:   (base - new) / |base|
    """
    if base is None or new is None:
        return "—"
    if base == 0:
        return "—"
    try:
        if higher_is_better:
            rel = (new - base) / abs(base)
        else:
            rel = (base - new) / abs(base)
        return f"{(100.0 * rel):.2f}%"
    except Exception:
        return "—"

def pick_best(vae_v: Any, mvae_v: Any, higher_is_better: bool) -> Tuple[str, Any]:
    """
    Returns (winner, best_value) among {'vae','mvae'} for given metric direction.
    If a value is missing, picks the other if available; if both missing, returns ('—', None).
    """
    if vae_v is None and mvae_v is None:
        return "—", None
    if vae_v is None:
        return "mvae", mvae_v
    if mvae_v is None:
        return "vae", vae_v
    try:
        if higher_is_better:
            return ("mvae", mvae_v) if mvae_v > vae_v else ("vae", vae_v)
        else:
            return ("mvae", mvae_v) if mvae_v < vae_v else ("vae", vae_v)
    except Exception:
        return "—", None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------------
# Core logic
# -------------------------

def collect_runs(root: Path) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
    """
    Traverse root and collect metrics into:
    data[dataset][latent]['vae'|'mvae'][metric] = value
    """
    data: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Walk immediate children directories (each is a dataset folder)
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.lower() == "curves":
            continue

        # Inside each dataset folder, read *_metrics.json
        for f in entry.glob("*.json"):
            m = FILENAME_RE.match(f.name)
            if not m:
                # allow non-matching files (e.g., curves/*.json) to be ignored
                # print(f"[skip] {f}")
                continue

            dataset = m.group("dataset")
            model   = m.group("model").lower()
            latent  = int(m.group("latent"))
            # seed  = int(m.group("seed"))  # not currently used in aggregation

            j = safe_read_json(f)
            if not j:
                continue

            # Map raw JSON keys to our METRICS_DIR keys
            record: Dict[str, float] = {}
            for k in METRICS_DIR.keys():
                # For "mse/test" we expect exact key; sometimes users log as "mse/test" or "recon/mse_per_pixel"
                if k in j:
                    record[k] = j[k]
                elif k == "mse/test" and "recon/mse_per_pixel" in j:
                    record[k] = j["recon/mse_per_pixel"]
                else:
                    record[k] = None  # mark missing, handled later

            data[dataset][latent][model] = record

    return data

def write_dataset_json(outdir: Path, dataset: str, table_rows: List[Dict[str, Any]]) -> None:
    """
    table_rows: list of dict rows containing the comparison per latent and per metric.
    """
    ensure_dir(outdir)
    out_json = outdir / f"{dataset}_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, ensure_ascii=False, indent=2)
    print(f"[ok] Wrote {out_json}")

def write_dataset_markdown(outdir: Path, dataset: str, latents_sorted: List[int],
                           matrix: Dict[int, Dict[str, Dict[str, Any]]]) -> None:
    """
    matrix[latent][metric] -> dict with keys:
        vae, mvae, winner, best_value, delta_abs, delta_rel
    Writes a compact Markdown table (bold best values, arrows in headers).
    """
    ensure_dir(outdir)
    out_md = outdir / f"{dataset}_comparison.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# {dataset} — VAE vs MVAE\n\n")
        f.write("**Bold** indicates the better value per metric ({} = higher is better, {} = lower is better).\n\n".format(
            ARROW[True], ARROW[False]
        ))

        # One table per metric for readability
        for metric, higher_is_better in METRICS_DIR.items():
            f.write(f"## {metric} {ARROW[higher_is_better]}\n\n")
            f.write("| latent | VAE | MVAE | winner | Δabs (best−other) | Δrel |\n")
            f.write("|:-----:|:----:|:----:|:------:|:-----------------:|:----:|\n")
            for d in latents_sorted:
                cell = matrix.get(d, {}).get(metric, {})
                vae_v  = cell.get("vae")
                mvae_v = cell.get("mvae")
                winner = cell.get("winner", "—")
                best_v = cell.get("best_value")
                delta_abs = cell.get("delta_abs")
                delta_rel = cell.get("delta_rel", "—")

                # Bold best
                vae_txt  = fmt_float(vae_v)
                mvae_txt = fmt_float(mvae_v)
                if winner == "vae":
                    vae_txt = bold(vae_txt)
                elif winner == "mvae":
                    mvae_txt = bold(mvae_txt)

                f.write("| {} | {} | {} | {} | {} | {} |\n".format(
                    d,
                    vae_txt,
                    mvae_txt,
                    winner,
                    fmt_float(delta_abs),
                    delta_rel
                ))
            f.write("\n")

    print(f"[ok] Wrote {out_md}")

def write_dataset_csv(outdir: Path, dataset: str, latents_sorted: List[int],
                      matrix: Dict[int, Dict[str, Dict[str, Any]]]) -> None:
    """
    Wide CSV: columns per metric include VAE, MVAE, winner, delta_abs, delta_rel.
    """
    import csv
    ensure_dir(outdir)
    out_csv = outdir / f"{dataset}_comparison.csv"

    # Build header
    header = ["latent"]
    for metric in METRICS_DIR.keys():
        header.extend([
            f"{metric}::vae",
            f"{metric}::mvae",
            f"{metric}::winner",
            f"{metric}::delta_abs",
            f"{metric}::delta_rel",
        ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for d in latents_sorted:
            row = [d]
            for metric in METRICS_DIR.keys():
                cell = matrix.get(d, {}).get(metric, {})
                row.extend([
                    cell.get("vae"),
                    cell.get("mvae"),
                    cell.get("winner"),
                    cell.get("delta_abs"),
                    cell.get("delta_rel"),
                ])
            w.writerow(row)

    print(f"[ok] Wrote {out_csv}")

def build_comparisons(data: Dict[str, Dict[int, Dict[str, Dict[str, float]]]]) -> None:
    """
    For each dataset, create per-latent comparisons and write outputs.
    """
    # Create output root
    parser_args = getattr(build_comparisons, "_args", None)
    outdir_root: Path = parser_args.outdir if parser_args else Path("aggregates")
    ensure_dir(outdir_root)

    for dataset, by_latent in sorted(data.items()):
        latents_sorted = sorted(by_latent.keys())

        # Build a matrix: [latent][metric] -> comparison dict
        matrix: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        json_rows: List[Dict[str, Any]] = []

        for d in latents_sorted:
            vae_metrics  = by_latent[d].get("vae", {})
            mvae_metrics = by_latent[d].get("mvae", {})

            row = {"dataset": dataset, "latent": d, "metrics": {}}

            for metric, higher_is_better in METRICS_DIR.items():
                vae_v  = vae_metrics.get(metric)
                mvae_v = mvae_metrics.get(metric)

                winner, best_v = pick_best(vae_v, mvae_v, higher_is_better)

                # Absolute difference (best − other) in the *direction of better*
                delta_abs = None
                if winner == "vae" and mvae_v is not None and vae_v is not None:
                    delta_abs = vae_v - mvae_v if higher_is_better else mvae_v - vae_v
                elif winner == "mvae" and mvae_v is not None and vae_v is not None:
                    delta_abs = mvae_v - vae_v if higher_is_better else vae_v - mvae_v

                delta_rel = percent_change(
                    new=mvae_v, base=vae_v, higher_is_better=higher_is_better
                ) if (vae_v is not None and mvae_v is not None) else "—"

                comp = {
                    "vae": vae_v,
                    "mvae": mvae_v,
                    "winner": winner,          # 'vae' | 'mvae' | '—'
                    "best_value": best_v,
                    "delta_abs": delta_abs,    # in favor of the winner
                    "delta_rel": delta_rel,    # % improvement of MVAE over VAE in the correct direction
                    "higher_is_better": higher_is_better,
                }
                matrix[d][metric] = comp
                row["metrics"][metric] = comp

            json_rows.append(row)

        # Write files
        ds_outdir = outdir_root
        write_dataset_json(ds_outdir, dataset, json_rows)
        write_dataset_markdown(ds_outdir, dataset, latents_sorted, matrix)
        write_dataset_csv(ds_outdir, dataset, latents_sorted, matrix)

# -------------------------
# CLI
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Compare VAE vs MVAE metrics by dataset & latent dimension.")
    p.add_argument("--root", type=str, default="./results", help="Root folder containing dataset subfolders.")
    p.add_argument("--outdir", type=str, default="aggregates", help="Output directory for summaries and tables.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()

    # Attach args so writers know where to place files
    build_comparisons._args = argparse.Namespace(outdir=outdir)

    data = collect_runs(root)
    if not data:
        print("[warn] No matching metric files found. Check your --root and filenames.")
        return

    build_comparisons(data)
    print("[done] Comparisons generated.")

if __name__ == "__main__":
    main()
