import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ================= Style (same as your reference) =================
font_family = "DejaVu Sans"
font1 = {"family": font_family, "weight": "normal", "size": 80}
font2 = {"family": font_family, "weight": "normal", "size": 65}
font_legend = {"family": font_family, "weight": "normal", "size": 42}
Label_Size = 50

bwith = 1.7
lwidth = 5
msize = 10
marker_alpha = 0.85

rcParams["axes.unicode_minus"] = False
rcParams["font.family"] = font_family

# enough colors/markers for many defenses
COLS = [
    "darkorange", "limegreen", "c", "y", "m",
    "tab:blue", "tab:red", "tab:purple", "tab:brown", "tab:pink",
    "tab:gray", "tab:olive", "tab:cyan"
]
MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*", "<", ">", "h", "8", "p"]


# ===== defense name normalization / pretty label =====
def norm_dir_name(name: str) -> str:
    # your run_experiment uses defense.lower().replace('_','')
    return str(name).lower().replace("_", "").replace(" ", "")

def pretty_defense(name: str) -> str:
    # match your defense_factory names
    mapping = {
        "fedavg": "FedAvg",
        "fedbn": "FedBN",
        "fedprox": "FedProx",
        "coordinatemedian": "CoordinateMedian",
        "trimmedmean": "TrimmedMean",
        "geomedian": "GeoMedian",
        "fltrust": "FLTrust",
        "foolsgold": "FoolsGold",
        "flame": "FLAME",
        "noisyaggregation": "NoisyAggregation",
        "rfa": "RFA",
        "flshield": "FLShield",
        "fedtap": "FedTAP",
    }
    key = norm_dir_name(name)
    return mapping.get(key, name)


def is_backdoor_attack(attack_name: str) -> bool:
    a = str(attack_name).lower()
    return any(k in a for k in ["a_m", "a_s", "dba", "tail", "attack_of_the_tails", "semantic"])


# ================= Load results.json =================
def load_results(path: str, asr_key: str | None = None):
    with open(path, "r") as f:
        d = json.load(f)

    rounds = np.array(d["rounds"], dtype=int)
    acc = np.array(d["acc_clean"], dtype=float)

    asr = None
    used_key = None

    if asr_key is not None:
        if asr_key in d:
            asr = np.array(d[asr_key], dtype=float)
            used_key = asr_key
        else:
            raise KeyError(f"ASR key '{asr_key}' not found in {path}. Available keys: {list(d.keys())}")
    else:
        # paper-friendly: prefer asr_full (non-target->target), then asr_overall
        for k in ["asr_full", "asr_overall", "asr_partial_23", "asr_shift", "asr_scale", "asr_target"]:
            if k in d:
                asr = np.array(d[k], dtype=float)
                used_key = k
                break

    return rounds, acc, asr, used_key


def build_union_rounds(series_dict):
    # series_dict: {name: (rounds, y)}
    all_rounds = set()
    for r, _ in series_dict.values():
        all_rounds |= set(map(int, r.tolist()))
    return np.array(sorted(all_rounds), dtype=int)


def align_on_union_rounds(all_rounds, series_dict):
    # return aligned {name: y_aligned} with NaN for missing rounds
    aligned = {}
    for name, (r, y) in series_dict.items():
        idx = {int(rr): i for i, rr in enumerate(r.tolist())}
        yy = np.full((len(all_rounds),), np.nan, dtype=float)
        for i, rr in enumerate(all_rounds):
            if int(rr) in idx:
                yy[i] = float(y[idx[int(rr)]])
        aligned[name] = yy
    return aligned


def plot_multi_curves(rounds, curves, ylabel, title, out_path):
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)

    if len(rounds) == 0:
        print(f"[WARN] empty rounds: {out_path}")
        plt.close(fig)
        return

    max_round = int(rounds[-1])

    # keep ~18 markers per curve
    markevery = max(1, int(np.ceil(len(rounds) / 18)))

    for i, (label, y) in enumerate(curves):
        color = COLS[i % len(COLS)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(
            rounds, y,
            color=color,
            linewidth=lwidth,
            linestyle="-",
            marker=marker,
            markersize=msize,
            markevery=markevery,
            alpha=marker_alpha,
            label=label,
        )

    ax.set_xlim(1, max_round)
    ax.set_ylim(0.0, 1.0)

    if max_round >= 300:
        step = 50
    elif max_round >= 120:
        step = 20
    else:
        step = 10
    ax.set_xticks(np.arange(0, max_round + 1, step))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    ax.tick_params(axis="both", labelsize=Label_Size)
    ax.grid(True)
    for s in ["bottom", "left", "top", "right"]:
        ax.spines[s].set_linewidth(bwith)

    ax.set_xlabel("Round", font2)
    ax.set_ylabel(ylabel, font1)
    ax.set_title(title, font2)

    ax.legend(loc="lower right", prop=font_legend, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def scan_defenses_with_results(base_dir: str):
    """Return defense directory names that contain results.json."""
    if not os.path.isdir(base_dir):
        return []
    ds = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "results.json")):
            ds.append(name)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--attack", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--end_round", type=int, default=None)
    ap.add_argument(
        "--defenses",
        default="fedtap,fltrust",
        help="comma-separated defenses (e.g., 'fedavg,fltrust,fedtap') or 'all' to plot all available defenses"
    )

    ap.add_argument("--plot_asr", action=argparse.BooleanOptionalAction, default=None,
                    help="Plot ASR curve (auto True for backdoor attacks). Use --no-plot_asr to disable.")
    ap.add_argument("--asr_key", default=None,
                    help="Force ASR key (recommended for paper: asr_full). If None, auto-pick.")

    args = ap.parse_args()

    if args.plot_asr is None:
        args.plot_asr = is_backdoor_attack(args.attack)

    base = os.path.join(args.runs_dir, args.dataset, args.attack)
    out = os.path.join(args.out_dir, args.dataset, args.attack)
    os.makedirs(out, exist_ok=True)

    # ---- decide defenses to plot ----
    defenses_arg = args.defenses.strip()
    if defenses_arg.lower() == "all":
        defense_dirs = scan_defenses_with_results(base)
        if len(defense_dirs) == 0:
            raise RuntimeError(f"No defenses with results.json found under: {base}")
        # use the directory names directly
        defense_list = defense_dirs
    else:
        # user given names; normalize to directory names used by your runner
        defense_list = [norm_dir_name(x.strip()) for x in defenses_arg.split(",") if x.strip()]

    # ---- load series ----
    acc_series = {}
    asr_series = {}
    chosen_asr_key = None

    for d_dir in defense_list:
        path = os.path.join(base, d_dir, "results.json")
        if not os.path.exists(path):
            print(f"[WARN] missing: {path} (skip)")
            continue

        r, acc, asr, used_key = load_results(path, asr_key=args.asr_key)

        if args.end_round is not None:
            m = r <= args.end_round
            r, acc = r[m], acc[m]
            if asr is not None:
                asr = asr[m]

        acc_series[d_dir] = (r, acc)
        if asr is not None:
            asr_series[d_dir] = (r, asr)
            if chosen_asr_key is None:
                chosen_asr_key = used_key

    if len(acc_series) == 0:
        raise RuntimeError("No valid results.json loaded. Check runs_dir/dataset/attack/defense layout.")
    if len(acc_series) == 1:
        print("[WARN] Only 1 defense found. Will still plot single curve.")

    # ---- align by union rounds ----
    rounds = build_union_rounds(acc_series)
    acc_aligned = align_on_union_rounds(rounds, acc_series)
    curves_acc = [(pretty_defense(k), acc_aligned[k]) for k in acc_aligned.keys()]

    plot_multi_curves(
        rounds, curves_acc,
        ylabel="Benign Accuracy",
        title=f"{args.dataset.upper()} / {args.attack.upper()}",
        out_path=os.path.join(out, "acc_clean.pdf"),
    )

    # ---- ASR ----
    if args.plot_asr:
        if len(asr_series) <= 0:
            print("[WARN] plot_asr enabled but no ASR keys found. Skip ASR plot.")
            return

        rounds2 = build_union_rounds(asr_series)
        asr_aligned = align_on_union_rounds(rounds2, asr_series)
        curves_asr = [(pretty_defense(k), asr_aligned[k]) for k in asr_aligned.keys()]

        use_key = args.asr_key or chosen_asr_key or "asr"
        plot_multi_curves(
            rounds2, curves_asr,
            ylabel="Attack Success Rate",
            title=f"{args.dataset.upper()} / {args.attack.upper()} ({use_key})",
            out_path=os.path.join(out, f"asr_{use_key}.pdf"),
        )


if __name__ == "__main__":
    main()
