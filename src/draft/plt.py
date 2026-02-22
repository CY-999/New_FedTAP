import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ===== Fig.1-like poster style =====
font_family = "DejaVu Sans"

# 轴标题（大 + 粗）
font1 = {"family": font_family, "weight": "bold", "size": 34}  # y-label
font2 = {"family": font_family, "weight": "bold", "size": 34}  # x-label / title

# 刻度 & legend
Label_Size = 22
font_legend = {"family": font_family, "weight": "bold", "size": 18}

# 线条/边框
bwith = 1.8
lwidth = 3.0
msize = 10          # marker size base
marker_alpha = 0.90

rcParams["axes.unicode_minus"] = False
rcParams["font.family"] = font_family

# enough colors/markers for many defenses
COLS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
    "#dede00",  # yellow
    "#00bcd4",  # cyan
    "#00c853",  # vivid green
    "#ff4081",  # vivid magenta
    "#651fff",  # vivid indigo
]
MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*", "<", ">", "h", "8", "p"]


# ===== defense name normalization / pretty label =====
def norm_dir_name(name: str) -> str:
    # your run_experiment uses defense.lower().replace('_','')
    return str(name).lower().replace("_", "").replace(" ", "")

def pretty_defense(name: str) -> str:
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
        "decare": "DeCARE",
        "decaregeom": "DeCARE_GEOM",
    }
    key = norm_dir_name(name)
    return mapping.get(key, name)


def is_backdoor_attack(attack_name: str) -> bool:
    a = str(attack_name).lower()
    return any(k in a for k in ["a_m", "a_s", "dba", "tail", "attack_of_the_tails", "semantic", "lga"])


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
    all_rounds = set()
    for r, _ in series_dict.values():
        all_rounds |= set(map(int, r.tolist()))
    return np.array(sorted(all_rounds), dtype=int)


def align_on_union_rounds(all_rounds, series_dict):
    aligned = {}
    for name, (r, y) in series_dict.items():
        idx = {int(rr): i for i, rr in enumerate(r.tolist())}
        yy = np.full((len(all_rounds),), np.nan, dtype=float)
        for i, rr in enumerate(all_rounds):
            if int(rr) in idx:
                yy[i] = float(y[idx[int(rr)]])
        aligned[name] = yy
    return aligned


def gaussian_smooth_nan(y: np.ndarray, sigma: float, window: int) -> np.ndarray:
    """
    Gaussian smoothing that ignores NaNs.
    - y: 1D array (may contain NaNs)
    - sigma: Gaussian std; <=0 means no smoothing
    - window: kernel size (odd). If <=1 means no smoothing
    """
    y = np.asarray(y, dtype=float)
    if sigma is None or sigma <= 0 or window is None or window <= 1:
        return y

    if window % 2 == 0:
        window += 1

    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= np.sum(k)

    mask = np.isfinite(y).astype(float)
    y0 = np.where(np.isfinite(y), y, 0.0)

    num = np.convolve(y0, k, mode="same")
    den = np.convolve(mask, k, mode="same")

    out = np.full_like(y0, np.nan, dtype=float)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def _pick_xtick_step(max_round: int) -> int:
    # match Fig.1: roughly 0,100,200,300,400...
    if max_round >= 400:
        return 100
    if max_round >= 300:
        return 50
    if max_round >= 150:
        return 25
    return 10


def plot_multi_curves(rounds, curves, ylabel, title, out_path, legend_loc="upper left"):
    fig = plt.figure(figsize=(10.5, 7.0))
    ax = fig.gca()

    if len(rounds) == 0:
        print(f"[WARN] empty rounds: {out_path}")
        plt.close(fig)
        return

    max_round = int(rounds[-1])

    # --- grid like Fig.1 ---
    ax.grid(True, linestyle="--", linewidth=1.2, alpha=0.35)

    # --- markers only on key rounds (Fig.1 style) ---
    xt_step = _pick_xtick_step(max_round)
    key_rounds = list(range(0, max_round + 1, xt_step))
    round_to_idx = {int(r): i for i, r in enumerate(rounds.tolist())}
    key_idx = [round_to_idx[r] for r in key_rounds if r in round_to_idx]

    for i, (label, y) in enumerate(curves):
        color = COLS[i % len(COLS)]
        marker = MARKERS[i % len(MARKERS)]

        # main line (NO marker on every point)
        ax.plot(
            rounds, y,
            color=color,
            linewidth=lwidth,
            linestyle="-",
            alpha=1.0,
            label=label,
        )

        # overlay sparse markers at key rounds only
        if len(key_idx) > 0:
            ax.scatter(
                rounds[key_idx],
                np.asarray(y)[key_idx],
                marker=marker,
                s=(msize * 1.3) ** 2,
                color=color,
                alpha=marker_alpha,
                zorder=4,
            )

    # axis limits/ticks like Fig.1
    ax.set_xlim(0, max_round)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(np.arange(0, max_round + 1, xt_step))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    ax.tick_params(axis="both", labelsize=Label_Size, width=bwith, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(bwith)

    ax.set_xlabel("Training Round", font2)
    ax.set_ylabel(ylabel, font1)
    ax.set_title(title, fontdict={"family": font_family, "weight": "bold", "size": 28})

    ax.legend(
        loc=legend_loc,
        prop=font_legend,
        frameon=True,
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.5,
        handlelength=2.2,
    )

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
        help="comma-separated defenses (e.g., 'fedavg,fltrust,fedtap') or 'all' to plot all available defenses",
    )

    ap.add_argument(
        "--plot_asr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Plot ASR curve (auto True for backdoor attacks). Use --no-plot_asr to disable.",
    )
    ap.add_argument(
        "--asr_key",
        default=None,
        help="Force ASR key (recommended for paper: asr_full). If None, auto-pick.",
    )
    ap.add_argument("--smooth_sigma", type=float, default=0.0, help="Gaussian smoothing sigma. 0 disables smoothing.")
    ap.add_argument("--smooth_window", type=int, default=0, help="Gaussian kernel window size (odd). 0 disables smoothing.")

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
        defense_list = defense_dirs
    else:
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

    curves_acc = []
    for k in acc_aligned.keys():
        y = acc_aligned[k]
        y = gaussian_smooth_nan(y, args.smooth_sigma, args.smooth_window)
        curves_acc.append((pretty_defense(k), y))

    plot_multi_curves(
        rounds,
        curves_acc,
        ylabel="Accuracy",
        title=f"Clean Accuracy",
        out_path=os.path.join(out, "acc_clean.pdf"),
        legend_loc="lower right",
    )

    # ---- ASR ----
    if args.plot_asr:
        if len(asr_series) <= 0:
            print("[WARN] plot_asr enabled but no ASR keys found. Skip ASR plot.")
            return

        rounds2 = build_union_rounds(asr_series)
        asr_aligned = align_on_union_rounds(rounds2, asr_series)

        curves_asr = []
        for k in asr_aligned.keys():
            y = asr_aligned[k]
            y = gaussian_smooth_nan(y, args.smooth_sigma, args.smooth_window)
            curves_asr.append((pretty_defense(k), y))

        use_key = args.asr_key or chosen_asr_key or "asr"

        # Fig.1-like title for full-trigger curve
        if use_key == "asr_full":
            title = "Attack Success Rate - Full Trigger"
        if use_key == "asr_partial_23":
            title = "Attack Success Rate - 2/3 Trigger"
        if use_key == "asr_shift":
            title = "Attack Success Rate - Shifted Trigger"

        plot_multi_curves(
            rounds2,
            curves_asr,
            ylabel="Attack Success Rate",
            title=title,
            out_path=os.path.join(out, f"asr_{use_key}.pdf"),
        )


if __name__ == "__main__":
    main()