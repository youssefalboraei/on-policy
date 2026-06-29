"""
Per-fault-type box plot comparing Baseline, STNF-SE and ST-SE at two fault-count
ranges (up to 3 vs up to 10 faulty robots).

x-axis : fault type (DEFAULT, F1, F2, F3, F4)
per fault type, 6 boxes:
    Baseline / STNF-SE / ST-SE  at <=3 faulty robots   (light shades)
    Baseline / STNF-SE / ST-SE  at <=10 faulty robots  (dark shades)
= 5 fault types x 6 = 30 boxes.

Metric: per-run N-CBDS. Baseline = ST-SE P_Baseline (shared random-walk baseline);
STNF-SE / ST-SE = their own P_Mitigated. DEFAULT has only N=0 data, so its <=3 and
<=10 boxes are identical by construction.
"""
import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.abspath(os.path.join(HERE, ".."))

FAULT_NAMES = {0: "DEFAULT", 3: "F1: ALL_WHEEL_V0", 4: "F2: ALL_WHEEL_V10",
               5: "F3: ALL_WHEEL_V50", 8: "F4: PICKUP"}
X_ORDER = [FAULT_NAMES[t] for t in [0, 3, 4, 5, 8]]
CONDS = ["Baseline", "STNF-SE", "ST-SE"]
RANGES = [("≤3", 3), ("≤10", 10)]

# 3 conditions x 2 ranges; ColorBrewer Greys/Blues/Oranges, light = <=3, dark = <=10.
# Gray = Baseline (reference), Blue = STNF-SE, Orange = ST-SE (colorblind-safe trio).
PALETTE = {
    "Baseline (≤3)": "#BDBDBD", "STNF-SE (≤3)": "#9ECAE1", "ST-SE (≤3)": "#FDAE6B",
    "Baseline (≤10)": "#636363", "STNF-SE (≤10)": "#3182BD", "ST-SE (≤10)": "#E6550D",
}
HUE_ORDER = [f"Baseline ({RANGES[0][0]})", f"STNF-SE ({RANGES[0][0]})", f"ST-SE ({RANGES[0][0]})",
             f"Baseline ({RANGES[1][0]})", f"STNF-SE ({RANGES[1][0]})", f"ST-SE ({RANGES[1][0]})"]


def comp(bucket):
    c = pd.read_csv(os.path.join(TESTS_DIR, bucket, "performance_comparison_results.csv"))
    c["Fault_Name"] = c["Fault_Type"].map(FAULT_NAMES)
    return c


def main():
    st = comp("ST-SE")
    stnf = comp("STNF-SE")

    parts = []
    for rtag, cap in RANGES:
        sources = [
            ("Baseline", st, "P_Baseline"),
            ("STNF-SE", stnf, "P_Mitigated"),
            ("ST-SE", st, "P_Mitigated"),
        ]
        for cond, src, col in sources:
            sub = src.loc[src.Fault_Number <= cap, ["Fault_Name", col]].rename(columns={col: "Performance"})
            sub["Group"] = f"{cond} ({rtag})"
            parts.append(sub)
    df = pd.concat(parts, ignore_index=True)

    plt.figure(figsize=(24, 10))
    ax = sns.boxplot(
        x="Fault_Name", y="Performance", hue="Group", data=df,
        order=X_ORDER, hue_order=HUE_ORDER, palette=PALETTE,
        width=0.8, linewidth=1.6, fliersize=2,
    )
    plt.title("Performance by fault type: up to 3 vs up to 10 faulty robots (Standard Environment)", fontsize=20)
    plt.ylabel("Performance (N-CBDS)", fontsize=18)
    plt.xlabel("Fault Type", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=13)
    plt.legend(title="Condition (fault-count range)", bbox_to_anchor=(1.01, 1), loc="upper left",
               fontsize=12, title_fontsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    # light separators between fault-type clusters
    for x in range(len(X_ORDER) - 1):
        ax.axvline(x + 0.5, color="0.85", linewidth=1)
    plt.tight_layout()
    out = os.path.join(HERE, "boxplot_per_fault_n3_vs_n10.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote", os.path.relpath(out, HERE))

    summ = (df.groupby(["Fault_Name", "Group"])["Performance"].agg(["count", "mean", "median", "std"])
            .reindex(pd.MultiIndex.from_product([X_ORDER, HUE_ORDER], names=["Fault_Name", "Group"])))
    csv = os.path.join(HERE, "boxplot_per_fault_n3_vs_n10_summary.csv")
    summ.to_csv(csv)
    print("wrote", os.path.relpath(csv, HERE))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
