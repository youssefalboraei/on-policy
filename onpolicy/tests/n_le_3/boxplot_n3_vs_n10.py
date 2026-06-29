"""
Box plot comparing Baseline, STNF-SE and ST-SE under two fault-count ranges:
up to 3 faulty robots (N<=3) vs up to 10 faulty robots (N<=10).

3 conditions x 2 ranges = 6 boxes. Each box is the distribution of per-run N-CBDS
performance pooled over all fault types within that range.

Data (Standard Environment, SE):
  Baseline  ST-SE/performance_comparison_results.csv  P_Baseline  (shared random-walk baseline)
  STNF-SE   STNF-SE/performance_comparison_results.csv P_Mitigated (no-fault-trained policy)
  ST-SE     ST-SE/performance_comparison_results.csv   P_Mitigated (fault-trained policy)
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

RANGES = [("Up to 3 faulty robots", 3), ("Up to 10 faulty robots", 10)]
COND_ORDER = ["Baseline", "STNF-SE", "ST-SE"]


def comp(bucket):
    return pd.read_csv(os.path.join(TESTS_DIR, bucket, "performance_comparison_results.csv"))


def main():
    st = comp("ST-SE")
    stnf = comp("STNF-SE")

    rows = []
    for label, cap in RANGES:
        rows.append(st.loc[st.Fault_Number <= cap, ["P_Baseline"]]
                    .rename(columns={"P_Baseline": "Performance"}).assign(Condition="Baseline", Range=label))
        rows.append(stnf.loc[stnf.Fault_Number <= cap, ["P_Mitigated"]]
                    .rename(columns={"P_Mitigated": "Performance"}).assign(Condition="STNF-SE", Range=label))
        rows.append(st.loc[st.Fault_Number <= cap, ["P_Mitigated"]]
                    .rename(columns={"P_Mitigated": "Performance"}).assign(Condition="ST-SE", Range=label))
    df = pd.concat(rows, ignore_index=True)

    plt.figure(figsize=(13, 8))
    ax = sns.boxplot(
        x="Condition", y="Performance", hue="Range", data=df,
        order=COND_ORDER, hue_order=[r[0] for r in RANGES],
        width=0.6, palette=sns.color_palette("pastel"), linewidth=2.2,
    )
    plt.title("Performance: up to 3 vs up to 10 faulty robots (Standard Environment)", fontsize=18)
    plt.ylabel("Performance (N-CBDS)", fontsize=16)
    plt.xlabel("")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=13)
    plt.legend(title="Fault-count range", fontsize=13, title_fontsize=13, loc="upper right")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    plt.tight_layout()
    out = os.path.join(HERE, "boxplot_n3_vs_n10.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote", os.path.relpath(out, HERE))

    summ = df.groupby(["Condition", "Range"])["Performance"].agg(["count", "mean", "median", "std"]).reindex(
        pd.MultiIndex.from_product([COND_ORDER, [r[0] for r in RANGES]], names=["Condition", "Range"]))
    csv = os.path.join(HERE, "boxplot_n3_vs_n10_summary.csv")
    summ.to_csv(csv)
    print("wrote", os.path.relpath(csv, HERE))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
