#!/usr/bin/env python
"""Delivery-over-time plots: boxes delivered vs sim time, mitigation vs baseline.

Reads the capture .npz pairs and emits a clean per-scenario figure plus a combined
3-panel figure for the quantitative slide. Time axis is real seconds (tick/50).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_common as V

MIT = "#2e6f4e"      # green = with mitigation
BASE = "#b03030"     # red = no mitigation (baseline)


def curve(npz):
    d = np.load(npz)
    drate = d["drate"]
    tps = int(d["ticks_per_sec"])
    t = np.arange(len(drate)) / tps
    return t, drate, int(d["fault_type"]), int(d["num_faults"]), d["bx"].shape[1]


def title_for(ft, nf):
    if ft == 0 or nf == 0:
        return "No fault"
    return f"{V.FAULT_CODE.get(ft, 'T'+str(ft))}: {V.FAULT_TECH.get(ft, ft)}  ({nf}/10 faulty)"


def plot_one(ax, base_npz, mit_npz):
    tb, db, ftb, nfb, nbox = curve(base_npz)
    tm, dm, ft, nf, _ = curve(mit_npz)
    ax.step(tm, dm, where="post", color=MIT, lw=2.6, label="With mitigation")
    ax.step(tb, db, where="post", color=BASE, lw=2.6, label="No mitigation (baseline)")
    ax.set_title(title_for(ft, nf), fontsize=13, fontweight="bold")
    ax.set_xlabel("Sim time (s)", fontsize=11)
    ax.set_ylabel(f"Boxes delivered (/ {nbox})", fontsize=11)
    ax.set_ylim(-0.3, nbox + 0.3)
    ax.set_yticks(range(0, nbox + 1, 2))
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # final-value annotations
    ax.annotate(f"{dm[-1]:.0f}", (tm[-1], dm[-1]), color=MIT, fontsize=11,
                fontweight="bold", xytext=(4, 2), textcoords="offset points")
    ax.annotate(f"{db[-1]:.0f}", (tb[-1], db[-1]), color=BASE, fontsize=11,
                fontweight="bold", xytext=(4, -12), textcoords="offset points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="name:baseline.npz:mitigation.npz triples")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    pairs = []
    for p in a.pairs:
        name, base, mit = p.split(":")
        pairs.append((name, base, mit))

    # per-scenario figures
    for name, base, mit in pairs:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        plot_one(ax, base, mit)
        ax.legend(fontsize=10, loc="lower right", frameon=False)
        fig.tight_layout()
        out = f"{a.outdir}/delivery_{name}.png"
        fig.savefig(out, facecolor="white")
        plt.close(fig)
        print(f"[plot] {out}")

    # combined panel
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), dpi=150)
    if n == 1:
        axes = [axes]
    for ax, (name, base, mit) in zip(axes, pairs):
        plot_one(ax, base, mit)
    axes[-1].legend(fontsize=10, loc="lower right", frameon=False)
    fig.suptitle("Box delivery over time: mitigation vs baseline (ST-SE)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{a.outdir}/delivery_combined.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
