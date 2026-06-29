"""
Correct full (N=0..10) mitigation-power heatmap for STNF-SE.

The pre-existing u-test/stnf-se_mitigation_power_heatmap.png is sign-inverted: it
reports the no-fault-trained policy as far below the random-walk baseline, which
contradicts STNF-SE/performance_comparison_results.csv and summary_statistics.csv
(mitigated clearly beats baseline). That file was built with the mitigated/baseline
roles swapped and STNF-SE has no without_mitigation/ folder for u-test.py to read.

This rebuilds it from performance_comparison_results.csv (P_Mitigated vs the paired
shared baseline P_Baseline), using the same N-CBDS metric and mitigation-power
formula as the rest of the project. Output is written under a *_full name so the
incorrect original is left in place for comparison.
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
BUCKET = "STNF-SE"

FAULT_NAMES = {
    0: "NO_FAULT", 3: "F1: ALL_WHEEL_V0", 4: "F2: ALL_WHEEL_V10",
    5: "F3: ALL_WHEEL_V50", 8: "F4: PICKUP",
}


def mitigation_power(d_mit, d_none):
    d_mit, d_none = list(d_mit), list(d_none)
    if len(set(d_mit + d_none)) == 1:
        return 0.0
    U, _ = stats.mannwhitneyu(d_mit, d_none, alternative="two-sided")
    n1, n2 = len(d_mit), len(d_none)
    ranks = stats.rankdata(np.concatenate([d_mit, d_none]))
    r_mit, r_none = ranks[:n1].sum(), ranks[n1:].sum()
    h = 1 if r_mit > r_none else (-1 if r_mit < r_none else 0)
    return 2 * h * abs(U / (n1 * n2) - 0.5)


def main():
    c = pd.read_csv(os.path.join(TESTS_DIR, BUCKET, "performance_comparison_results.csv"))
    rows = []
    for (n, t), g in c.groupby(["Fault_Number", "Fault_Type"]):
        rows.append({
            "Fault_Number": n, "Fault_Type": t,
            "Fault_Name": "DEFAULT" if t == 0 else FAULT_NAMES.get(t, f"T{t}"),
            "Mitigation_Power": mitigation_power(g["P_Mitigated"], g["P_Baseline"]),
        })
    res = pd.DataFrame(rows)
    csv_path = os.path.join(HERE, "stnf-se_mitigation_power_analysis_full.csv")
    res.to_csv(csv_path, index=False)
    print("wrote", os.path.relpath(csv_path, HERE))

    pivot = res.pivot(index="Fault_Number", columns="Fault_Name", values="Mitigation_Power")
    if 0 in pivot.index and "DEFAULT" in pivot.columns:
        pivot.loc[0] = pivot.loc[0].fillna(pivot.loc[0, "DEFAULT"])
    pivot = pivot.drop("DEFAULT", axis=1, errors="ignore")
    cols = [FAULT_NAMES[k] for k in sorted(FAULT_NAMES) if FAULT_NAMES.get(k) in pivot.columns]
    pivot = pivot.reindex(columns=cols).sort_index()

    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(pivot, annot=True, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                     fmt=".2f", annot_kws={"size": 20})
    plt.title("Mitigation Power Heatmap (STNF-SE, full range N=0-10)", fontsize=22)
    plt.ylabel("Number of Faulty Robots", fontsize=20)
    plt.xlabel("Fault Type", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Mitigation Power", fontsize=20)
    plt.tight_layout()
    png_path = os.path.join(HERE, "stnf-se_mitigation_power_heatmap_full.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote", os.path.relpath(png_path, HERE))


if __name__ == "__main__":
    main()
