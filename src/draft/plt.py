# plt.py  (FIXED: legend in-figure + readable markers)
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
font_legend = {"family": font_family, "weight": "normal", "size": 55}
Label_Size = 50

bwith = 1.7
lwidth = 5
msize = 10            # ↓ 关键：marker 变小
marker_alpha = 0.8

Cols = ["darkorange", "limegreen", "c", "y"]
BASELINE_MARKER = "^"
SPECIAL_MARKER = "o"

rcParams["axes.unicode_minus"] = False
rcParams["font.family"] = font_family


# ================= Load results.json =================
def load_results(path):
    with open(path, "r") as f:
        d = json.load(f)

    rounds = np.array(d["rounds"])
    acc = np.array(d["acc_clean"])

    if "asr_overall" in d:
        asr = np.array(d["asr_overall"])
        asr_key = "asr_overall"
    elif "asr_full" in d:
        asr = np.array(d["asr_full"])
        asr_key = "asr_full"
    else:
        raise KeyError("No ASR key found")

    return rounds, acc, asr, asr_key


def pretty(name):
    return {"fedtap": "FedTAP", "fltrust": "FLTrust"}.get(name.lower(), name)


# ================= Plot core =================
def plot_two_curves(
    x1, y1, label1,
    x2, y2, label2,
    ylabel, title, out_path
):
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)

    max_round = max(x1[-1], x2[-1])

    def draw(x, y, label, color, marker):
        ax.plot(
            x, y,
            color=color,
            linewidth=lwidth,
            linestyle="-",
            marker=marker,
            markersize=msize,
            markevery=20,      
            alpha=marker_alpha,
            label=label,
        )

    draw(x1, y1, label1, Cols[0], SPECIAL_MARKER)
    draw(x2, y2, label2, Cols[1], BASELINE_MARKER)

    # ---- axes ----
    ax.set_xlim(1, max_round)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(np.arange(0, max_round + 1, 50))   
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    ax.tick_params(axis="both", labelsize=Label_Size)
    ax.grid(True)

    for s in ["bottom", "left", "top", "right"]:
        ax.spines[s].set_linewidth(bwith)

    ax.set_xlabel("Round", font2)
    ax.set_ylabel(ylabel, font1)
    ax.set_title(title, font2)

    ax.legend(
        loc="lower right",
        prop=font_legend,
        frameon=True,
        framealpha=0.9
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--attack", default="alie")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    base = os.path.join(args.runs_dir, args.dataset, args.attack)

    r1, acc1, asr1, asr_key = load_results(os.path.join(base, "fedtap", "results.json"))
    r2, acc2, asr2, _ = load_results(os.path.join(base, "fltrust", "results.json"))

    out = os.path.join(args.out_dir, args.dataset, args.attack)
    os.makedirs(out, exist_ok=True)

    plot_two_curves(
        r1, acc1, "FedTAP",
        r2, acc2, "FLTrust",
        ylabel="Benign Accuracy",
        title=f"{args.dataset.upper()} / {args.attack.upper()}",
        out_path=os.path.join(out, "acc_clean.pdf"),
    )

    plot_two_curves(
        r1, asr1, "FedTAP",
        r2, asr2, "FLTrust",
        ylabel="Attack Success Rate",
        title=f"{args.dataset.upper()} / {args.attack.upper()} ({asr_key})",
        out_path=os.path.join(out, f"asr_{asr_key}.pdf"),
    )


if __name__ == "__main__":
    main()
