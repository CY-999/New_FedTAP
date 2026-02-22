#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Table-3 style summary (mean±std over last K rounds) from:
  runs/<dataset>/<attack>/<defense>/results.json

Example:
  python process_data.py --runs_dir runs --dataset cifar10 --attack lga \
    --defenses all --asr_key asr_full --last_k 10 --out_tex lga.tex
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


# ------------------ name helpers ------------------
def norm_dir_name(name: str) -> str:
    # matches your runner: defense.lower().replace('_','')
    return str(name).lower().replace("_", "").replace(" ", "")

def pretty_defense(name: str) -> str:
    mapping = {
        "fedavg": "FedAvg",
        "fltrust": "FLTrust",
        "rfa": "RFA",
        "trimmedmean": "TrimmedMean",
        "coordinatemedian": "CoordinateMedian",
        "flshield": "FLShield",
        "flame": "FLAME",
        "foolsgold": "FoolsGold",
        "fedtap": "FedTAP",
        "geomedian": "GeoMedian",
        "noisyaggregation": "NoisyAggregation",
        "fedbn": "FedBN",
        "fedprox": "FedProx",
        "decare": "DeCARE",
        "decaregeom": "DeCARE_GEOM",
    }
    key = norm_dir_name(name)
    return mapping.get(key, name)


# ------------------ IO helpers ------------------
def scan_defenses_with_results(base_dir: str) -> List[str]:
    """Return defense directory names that contain results.json."""
    if not os.path.isdir(base_dir):
        return []
    out = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "results.json")):
            out.append(name)
    return out

def load_results_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def maybe_truncate_by_end_round(d: Dict[str, Any], end_round: Optional[int]) -> Dict[str, Any]:
    """If end_round is set, truncate any per-round list fields aligned to `rounds`."""
    if end_round is None or "rounds" not in d:
        return d
    rounds = np.asarray(d["rounds"], dtype=int)
    mask = rounds <= int(end_round)
    out = dict(d)
    for k, v in d.items():
        if isinstance(v, list) and len(v) == len(rounds):
            out[k] = list(np.asarray(v)[mask])
    return out


# ------------------ stats helpers ------------------
def _to_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)

def mean_std_last_k(y: List[float], last_k: int) -> Tuple[float, float]:
    """mean±std over the last_k rounds; ignores NaN/inf; std uses ddof=0."""
    arr = _to_float_array(y)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    k = max(1, min(int(last_k), int(arr.size)))
    tail = arr[-k:]
    return float(np.mean(tail)), float(np.std(tail, ddof=0))

def fmt_mean_pm_std(mu: float, sd: float, digits: int) -> str:
    if (mu is None or sd is None or
        (isinstance(mu, float) and (np.isnan(mu) or np.isinf(mu))) or
        (isinstance(sd, float) and (np.isnan(sd) or np.isinf(sd)))):
        return "-"
    return f"{mu:.{digits}f}$\\pm${sd:.{digits}f}"


# ------------------ LaTeX ------------------
def build_latex_table(rows: List[Tuple[str, str, str]],
                      caption: str,
                      label: str) -> str:
    """
    rows: (Defense, ACC_str, ASR_str)
    Requires \\usepackage{booktabs} in your preamble.
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Defense & ACC ($\mu\pm\sigma$) & ASR ($\mu\pm\sigma$) \\")
    lines.append(r"\midrule")
    for d, acc, asr in rows:
        lines.append(f"{d} & {acc} & {asr} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--attack", required=True, help="e.g., lga or a_m")
    ap.add_argument("--defenses", default="all",
                    help="comma-separated defense names or 'all' to auto-scan directories")

    ap.add_argument("--acc_key", default="acc_clean", help="ACC key in results.json (default: acc_clean)")
    ap.add_argument("--asr_key", default="asr_full", help="ASR key in results.json (e.g., asr_full)")
    ap.add_argument("--end_round", type=int, default=None, help="Optional: truncate curves to <= end_round")
    ap.add_argument("--last_k", type=int, default=50, help="Tail window length K for mean±std (default: 50)")
    ap.add_argument("--digits", type=int, default=4, help="Decimal digits")
    ap.add_argument("--out_tex", required=True, help="Write LaTeX table snippet to this path")

    ap.add_argument("--caption", default=None, help="LaTeX caption (optional)")
    ap.add_argument("--label", default=None, help="LaTeX label (optional)")
    args = ap.parse_args()

    base = os.path.join(args.runs_dir, args.dataset, args.attack)

    # Decide defenses
    if args.defenses.strip().lower() == "all":
        defense_dirs = scan_defenses_with_results(base)
        if len(defense_dirs) == 0:
            raise RuntimeError(f"No results.json found under: {base}")
        defense_list = defense_dirs
    else:
        defense_list = [norm_dir_name(x.strip()) for x in args.defenses.split(",") if x.strip()]

    # Caption/label defaults
    caption = args.caption if args.caption is not None else f"{args.attack.upper()} attack on {args.dataset.upper()}."
    label = args.label if args.label is not None else f"tab:{args.attack.lower()}_{args.dataset.lower()}"

    rows: List[Tuple[str, str, str]] = []

    for d_dir in defense_list:
        path = os.path.join(base, d_dir, "results.json")
        if not os.path.exists(path):
            print(f"[WARN] missing: {path} (skip)")
            continue

        d = load_results_json(path)
        d = maybe_truncate_by_end_round(d, args.end_round)

        acc_curve = d.get(args.acc_key, None)
        asr_curve = d.get(args.asr_key, None)

        if not isinstance(acc_curve, list):
            acc_mu, acc_sd = float("nan"), float("nan")
        else:
            acc_mu, acc_sd = mean_std_last_k(acc_curve, args.last_k)

        if not isinstance(asr_curve, list):
            asr_mu, asr_sd = float("nan"), float("nan")
        else:
            asr_mu, asr_sd = mean_std_last_k(asr_curve, args.last_k)

        acc_s = fmt_mean_pm_std(acc_mu, acc_sd, args.digits)
        asr_s = fmt_mean_pm_std(asr_mu, asr_sd, args.digits)

        rows.append((pretty_defense(d_dir), acc_s, asr_s))

    if len(rows) == 0:
        raise RuntimeError(f"No valid rows produced. Check: {base}")

    tex = build_latex_table(rows, caption=caption, label=label)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex + "\n")

    print(f"[OK] wrote LaTeX table to: {args.out_tex}")
    print(f"[INFO] base_dir: {base}")
    print(f"[INFO] last_k: {args.last_k}, acc_key: {args.acc_key}, asr_key: {args.asr_key}")


if __name__ == "__main__":
    main()