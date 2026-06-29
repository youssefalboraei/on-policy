"""
Regenerate the standard-environment (SE) result figures capped at <=3 faulty
robots, for the two standard-training policies:

  STNF-SE  Standard-Training, No-Fault (trained without faults), Standard-Execution
  ST-SE    Standard-Training (trained with faults),             Standard-Execution

Figure families produced per bucket (N = number of faulty robots, kept <= 3):
  1. Performance box plot  (N-CBDS, Baseline vs MARL)        from performance_comparison_results.csv
  2. Mitigation-power heatmap (Mann-Whitney U, rows N=0..3)  from performance_comparison_results.csv
  3. Action-frequency by fault type (heatmap + grouped bar)  from with_mitigation step CSVs
  4. Action-frequency faulty vs non-faulty (rows N=0..3)     from with_mitigation step CSVs

Plus a combined performance box plot (Baseline vs MARL(STNF-SE) vs MARL(ST-SE)).

STNF-SE has no without_mitigation/ folder because the no-mitigation baseline is the
same random-walk swarm regardless of the trained policy; its P_Baseline column in
performance_comparison_results.csv already carries that shared baseline.

Outputs land in this directory (n_le_3/), per bucket subfolders. Nothing existing is
overwritten.
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass
sns.set_palette("Set2")
plt.rcParams["font.family"] = "sans-serif"

N_CAP = 3
TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BUCKETS = ["STNF-SE", "ST-SE"]
TOTAL_ROBOTS = 10  # SE

FAULT_NAMES = {
    0: "NO_FAULT", 1: "SINGLE_WHEEL", 2: "DOUBLE_WHEEL",
    3: "F1: ALL_WHEEL_V0", 4: "F2: ALL_WHEEL_V10", 5: "F3: ALL_WHEEL_V50",
    6: "RADIAL_CAM_4", 7: "UPFACING_CAM", 8: "F4: PICKUP",
    9: "DROPOFF", 10: "LASER_16", 11: "R_COMMS", 12: "S_COMMS",
}
ACTION_NAMES = [
    "A1: NO_ACTION", "A2: DECREASE_SPEED_50", "A3: STOP_MOVING", "A4: BIAS_TO_NEAREST_ROBOT",
    "A5: BIAS_TO_NEAREST_BOX", "A6: BIAS_TO_NEAREST_WALL", "A7: BIAS_LEFT",
    "A8: BIAS_FROM_NEAREST_ROBOT", "A9: BIAS_FROM_NEAREST_BOX", "A10: BIAS_FROM_NEAREST_WALL",
    "A11: ATTRACT_NEIGHBOUR", "A12: REPEL_NEIGHBOUR", "A13: DROP_BOX",
]

MANIFEST = []


def save(fig_path):
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    MANIFEST.append(fig_path)
    print("  wrote", os.path.relpath(fig_path, OUT_DIR))


def present_fault_order(names):
    ordered = [FAULT_NAMES[i] for i in sorted(FAULT_NAMES)]
    return [n for n in ordered if n in set(names)]


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------
def load_comparison(bucket):
    """Per-run N-CBDS for mitigated and baseline, filtered to N <= N_CAP."""
    c = pd.read_csv(os.path.join(TESTS_DIR, bucket, "performance_comparison_results.csv"))
    c = c[c["Fault_Number"] <= N_CAP].copy()
    c["Fault_Name"] = c["Fault_Type"].map(FAULT_NAMES)
    return c


def long_performance(c, marl_label):
    """Melt a comparison frame into Condition rows (Baseline / MARL)."""
    base = c[["Fault_Number", "Fault_Type", "Fault_Name", "P_Baseline"]].rename(
        columns={"P_Baseline": "Performance"})
    base["Condition"] = "Baseline"
    mit = c[["Fault_Number", "Fault_Type", "Fault_Name", "P_Mitigated"]].rename(
        columns={"P_Mitigated": "Performance"})
    mit["Condition"] = marl_label
    return pd.concat([base, mit], ignore_index=True)


def action_records(bucket):
    """Per-file, per-robot-group action frequencies from with_mitigation step CSVs (N<=N_CAP)."""
    folder = os.path.join(TESTS_DIR, bucket, "with_mitigation")
    by_fault, faulty, nonfaulty = [], [], []
    for path in glob.glob(os.path.join(folder, "simulation_data_run*.csv")):
        fname = os.path.basename(path)
        n = int(fname.split("_N")[1].split("T")[0])
        t = int(fname.split("_N")[1].split("T")[1].split(".")[0])
        if n > N_CAP:
            continue
        df = pd.read_csv(path)
        cols = [col for col in df.columns if col.startswith("Robot_") and col.endswith("_Action")]

        def freqs(columns):
            arr = df[columns].values.flatten().astype(int)
            vc = pd.Series(arr).value_counts()
            return vc / len(arr)

        for a, f in freqs(cols).items():
            by_fault.append({"Fault_Name": FAULT_NAMES[t], "Action": ACTION_NAMES[a], "Frequency": f})

        f_cols = cols[:n]
        nf_cols = cols[n:]
        if f_cols:
            for a, f in freqs(f_cols).items():
                faulty.append({"N": n, "Action": ACTION_NAMES[a], "Frequency": f})
        if nf_cols:
            for a, f in freqs(nf_cols).items():
                nonfaulty.append({"N": n, "Action": ACTION_NAMES[a], "Frequency": f})

    return pd.DataFrame(by_fault), pd.DataFrame(faulty), pd.DataFrame(nonfaulty)


# ---------------------------------------------------------------------------
# figure 1: performance box plot
# ---------------------------------------------------------------------------
def boxplot(df_long, title, out_path, condition_order):
    plt.figure(figsize=(16, 9))
    order = present_fault_order(df_long["Fault_Name"].unique())
    ax = sns.boxplot(x="Fault_Name", y="Performance", hue="Condition", data=df_long,
                     width=0.55, palette=sns.color_palette("pastel"),
                     order=order, hue_order=condition_order, linewidth=2.5)
    plt.title(title, fontsize=20)
    plt.ylabel("Performance (N-CBDS)", fontsize=18)
    plt.xlabel("Fault Type", fontsize=18)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=0, ha="center", fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0., fontsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")
    plt.tight_layout()
    save(out_path)


# ---------------------------------------------------------------------------
# figure 2: mitigation-power heatmap
# ---------------------------------------------------------------------------
def mitigation_power(d_mit, d_none):
    d_mit = list(d_mit)
    d_none = list(d_none)
    if len(set(d_mit + d_none)) == 1:
        return 0.0
    U, _ = stats.mannwhitneyu(d_mit, d_none, alternative="two-sided")
    n1, n2 = len(d_mit), len(d_none)
    ranks = stats.rankdata(np.concatenate([d_mit, d_none]))
    r_mit, r_none = np.sum(ranks[:n1]), np.sum(ranks[n1:])
    h = 1 if r_mit > r_none else (-1 if r_mit < r_none else 0)
    return 2 * h * abs(U / (n1 * n2) - 0.5)


def heatmap_mitigation(c, bucket, out_png, out_csv):
    rows = []
    for (n, t), g in c.groupby(["Fault_Number", "Fault_Type"]):
        rows.append({"Fault_Number": n, "Fault_Type": t,
                     "Fault_Name": FAULT_NAMES.get(t, "DEFAULT") if t != 0 else "DEFAULT",
                     "Mitigation_Power": mitigation_power(g["P_Mitigated"], g["P_Baseline"])})
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    MANIFEST.append(out_csv)

    pivot = res.pivot(index="Fault_Number", columns="Fault_Name", values="Mitigation_Power")
    # row N=0 only has the DEFAULT (no-fault) cell; spread it across the fault columns
    if 0 in pivot.index and "DEFAULT" in pivot.columns:
        pivot.loc[0] = pivot.loc[0].fillna(pivot.loc[0, "DEFAULT"])
    pivot = pivot.drop("DEFAULT", axis=1, errors="ignore")
    fault_cols = [FAULT_NAMES[i] for i in sorted(FAULT_NAMES) if FAULT_NAMES[i] in pivot.columns]
    pivot = pivot.reindex(columns=fault_cols).sort_index()

    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(pivot, annot=True, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                     fmt=".2f", annot_kws={"size": 16})
    plt.title(f"Mitigation Power Heatmap ({bucket}, up to {N_CAP} faulty robots)", fontsize=18)
    plt.ylabel("Number of Faulty Robots", fontsize=16)
    plt.xlabel("Fault Type", fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Mitigation Power", fontsize=15)
    plt.tight_layout()
    save(out_png)


# ---------------------------------------------------------------------------
# figure 3: action frequency by fault type
# ---------------------------------------------------------------------------
def action_by_fault(df, bucket, out_heatmap, out_bar):
    agg = df.groupby(["Fault_Name", "Action"])["Frequency"].mean().reset_index()
    fault_order = present_fault_order(agg["Fault_Name"].unique())
    pivot = agg.pivot(index="Fault_Name", columns="Action", values="Frequency")
    pivot = pivot.reindex(index=fault_order, columns=ACTION_NAMES).fillna(0)

    plt.figure(figsize=(20, 7))
    ax = sns.heatmap(pivot, annot=True, cmap="Blues", vmin=0, fmt=".2f", annot_kws={"size": 12})
    cbar = ax.collections[0].colorbar
    cbar.set_label("Frequency", fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    plt.title(f"Action Frequencies by Fault Type ({bucket}, up to {N_CAP} faulty robots)", fontsize=18)
    plt.xlabel("Action", fontsize=15)
    plt.ylabel("Fault Type", fontsize=15)
    plt.xticks(fontsize=11, rotation=45, ha="right")
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    save(out_heatmap)

    plt.figure(figsize=(20, 9))
    pivot.plot(kind="bar", ax=plt.gca(), width=0.85,
               colormap="tab20", edgecolor="black", linewidth=0.3)
    plt.title(f"Action Frequencies by Fault Type ({bucket}, up to {N_CAP} faulty robots)", fontsize=18)
    plt.xlabel("Fault Type", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.xticks(rotation=0, ha="center", fontsize=12)
    plt.legend(title="Action", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    save(out_bar)


# ---------------------------------------------------------------------------
# figure 4: action frequency faulty vs non-faulty (rows = number of faults)
# ---------------------------------------------------------------------------
def action_faulty_split(df_faulty, df_nonfaulty, bucket, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(26, 7))
    for ax, df, label in [(axes[0], df_faulty, "Faulty robots"),
                          (axes[1], df_nonfaulty, "Non-faulty robots")]:
        if df.empty:
            ax.set_visible(False)
            continue
        agg = df.groupby(["N", "Action"])["Frequency"].mean().reset_index()
        pivot = agg.pivot(index="N", columns="Action", values="Frequency")
        pivot = pivot.reindex(columns=ACTION_NAMES).fillna(0).sort_index()
        sns.heatmap(pivot, annot=True, cmap="Blues", vmin=0, fmt=".2f",
                    annot_kws={"size": 11}, ax=ax, cbar_kws={"label": "Frequency"})
        ax.set_title(f"{label}", fontsize=16)
        ax.set_xlabel("Action", fontsize=14)
        ax.set_ylabel("Number of Faulty Robots", fontsize=14)
        ax.tick_params(axis="x", labelrotation=45, labelsize=10)
        ax.tick_params(axis="y", labelrotation=0, labelsize=12)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.suptitle(f"Action Frequencies vs Number of Faulty Robots ({bucket}, up to {N_CAP})", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(out_png)


# ---------------------------------------------------------------------------
def main():
    print(f"tests dir: {TESTS_DIR}")
    print(f"output dir: {OUT_DIR}\n")

    combined_long = []
    shared_baseline = None

    for bucket in BUCKETS:
        print(f"== {bucket} ==")
        bdir = os.path.join(OUT_DIR, bucket.lower())
        os.makedirs(bdir, exist_ok=True)

        c = load_comparison(bucket)

        # summary statistics
        summ = c.groupby("Fault_Name")[["P_Mitigated", "P_Baseline", "P_Difference"]].agg(
            ["mean", "median", "std"])
        summ_path = os.path.join(bdir, f"summary_statistics_n{N_CAP}.csv")
        summ.to_csv(summ_path)
        MANIFEST.append(summ_path)

        # 1. per-bucket performance box plot
        marl_label = f"MARL ({bucket})"
        dl = long_performance(c, marl_label)
        boxplot(dl, f"Performance in the Standard Environment\n{bucket}, up to {N_CAP} faulty robots",
                os.path.join(bdir, f"performance_boxplot_n{N_CAP}.png"),
                ["Baseline", marl_label])
        combined_long.append(c.assign(_bucket=bucket))
        if bucket == "ST-SE":
            shared_baseline = c[["Fault_Number", "Fault_Type", "Fault_Name", "P_Baseline"]].copy()

        # 2. mitigation-power heatmap
        heatmap_mitigation(c, bucket,
                           os.path.join(bdir, f"mitigation_power_heatmap_n{N_CAP}.png"),
                           os.path.join(bdir, f"mitigation_power_analysis_n{N_CAP}.csv"))

        # 3 & 4. action frequency
        df_fault, df_faulty, df_nonfaulty = action_records(bucket)
        action_by_fault(df_fault, bucket,
                        os.path.join(bdir, f"action_freq_by_fault_heatmap_n{N_CAP}.png"),
                        os.path.join(bdir, f"action_freq_by_fault_bar_n{N_CAP}.png"))
        action_faulty_split(df_faulty, df_nonfaulty, bucket,
                            os.path.join(bdir, f"action_freq_faulty_vs_nonfaulty_n{N_CAP}.png"))
        print()

    # combined performance box plot: shared baseline + MARL per bucket
    print("== combined ==")
    parts = [shared_baseline.assign(Performance=shared_baseline["P_Baseline"], Condition="Baseline")
             [["Fault_Number", "Fault_Type", "Fault_Name", "Performance", "Condition"]]]
    cond_order = ["Baseline"]
    for c in combined_long:
        bucket = c["_bucket"].iloc[0]
        label = f"MARL ({bucket})"
        cond_order.append(label)
        m = c[["Fault_Number", "Fault_Type", "Fault_Name", "P_Mitigated"]].rename(
            columns={"P_Mitigated": "Performance"})
        m["Condition"] = label
        parts.append(m)
    dl = pd.concat(parts, ignore_index=True)
    boxplot(dl, f"Performance in the Standard Environment (up to {N_CAP} faulty robots)\n"
                f"Baseline vs no-fault-trained (STNF) vs fault-trained (ST)",
            os.path.join(OUT_DIR, f"combined_performance_boxplot_n{N_CAP}.png"), cond_order)

    print(f"\nDone. {len(MANIFEST)} files written under {OUT_DIR}")


if __name__ == "__main__":
    main()
