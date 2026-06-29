"""
Action-frequency analysis for the Standard-env (SE) eval buckets.

Reproduces the five action-frequency analyses already present in this repo and
applies them to the no-fault Standard-env baseline so its action signatures sit
alongside the existing RT-SE / ST-SE results:

  1. action_frequency/            per-fault action-frequency heatmap, barplot, stacked bar
  2. action_frequency_analysis/   faulty vs non-faulty robot heatmaps
  3. action_frequency_signature/  combined faulty/non-faulty signature heatmap + CSV
  4. action_frequency_signature_change/  per-fault heatmap of frequency vs number of faults
  5. action_frequency_signature_n/       per-N combined signature heatmaps

Outputs land in the same five directories as the existing results, with the
bucket suffix appended (e.g. action_frequencies_heatmap_stnf-se.png), matching
the existing rt-se / st-se filenames.

Run:  python action_frequency_se.py                 # default: STNF-SE
      python action_frequency_se.py STNF-SE RTNF-SE
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TESTS = os.path.dirname(os.path.abspath(__file__))

FAULT_NAMES = {
    0: "NO_FAULT", 1: "SINGLE_WHEEL", 2: "DOUBLE_WHEEL",
    3: "F1: ALL_WHEEL_V0", 4: "F2: ALL_WHEEL_V10", 5: "F3: ALL_WHEEL_V50",
    6: "RADIAL_CAM_4", 7: "UPFACING_CAM", 8: "F4: PICKUP",
    9: "DROPOFF", 10: "LASER_16", 11: "R_COMMS", 12: "S_COMMS",
}
FAULT_NICE = {0: "No fault", 3: "F1: 0% speed", 4: "F2: 10% speed",
              5: "F3: 50% speed", 8: "F4: Pickup"}
FAULT_DESC = {0: "No fault     ", 3: "F1: 0% speed     ", 4: "F2: 10% speed     ",
              5: "F3: 50% speed    ", 8: "F4: Can't pickup     "}

ACTION_NAMES = [
    "A1: NO_ACTION", "A2: DECREASE_SPEED_50", "A3: STOP_MOVING", "A4: BIAS_TO_NEAREST_ROBOT",
    "A5: BIAS_TO_NEAREST_BOX", "A6: BIAS_TO_NEAREST_WALL", "A7: BIAS_LEFT",
    "A8: BIAS_FROM_NEAREST_ROBOT", "A9: BIAS_FROM_NEAREST_BOX", "A10: BIAS_FROM_NEAREST_WALL",
    "A11: ATTRACT_NEIGHBOUR", "A12: REPEL_NEIGHBOUR", "A13: DROP_BOX",
]
ACTION_NICE = [
    "A1: No action", "A2: Decrease speed 50%", "A3: Stop moving", "A4: Bias to nearest robot",
    "A5: Bias to nearest box", "A6: Bias to nearest wall", "A7: Bias left",
    "A8: Bias from nearest robot", "A9: Bias from nearest box", "A10: Bias from nearest wall",
    "A11: Attract neighbour", "A12: Repel neighbour", "A13: Drop box",
]


def parse_nt(filename):
    tail = filename.split("_N")[1]
    return int(tail.split("T")[0]), int(tail.split("T")[1].split(".")[0])


def load_bucket(bucket):
    """Return (n_robots, [(n_faults, fault_type, action_matrix), ...]).

    action_matrix is an int array of shape (steps, n_robots); column j is robot j.
    Faults are assigned to the first n_faults robots, so columns [:n] are faulty.
    """
    folder = os.path.join(TESTS, bucket, "with_mitigation")
    runs = []
    n_robots = None
    for fn in sorted(os.listdir(folder)):
        if not fn.startswith("simulation_data_run"):
            continue
        n, t = parse_nt(fn)
        df = pd.read_csv(os.path.join(folder, fn))
        cols = [c for c in df.columns if c.startswith("Robot_") and c.endswith("_Action")]
        cols = sorted(cols, key=lambda c: int(c.split("_")[1]))
        n_robots = len(cols)
        runs.append((n, t, df[cols].to_numpy().astype(int)))
    return n_robots, runs


def freq_series(values):
    """Normalised action-frequency Series indexed by action id."""
    counts = pd.Series(values.flatten()).value_counts()
    return counts / counts.sum()


# ---------------------------------------------------------------- analysis 1
def analysis_1(bucket, runs, out_dir):
    rows = []
    for n, t, mat in runs:
        for action, freq in freq_series(mat).items():
            rows.append({"Scenario": bucket, "Fault_Type": FAULT_NAMES[t],
                         "Action": ACTION_NAMES[action], "Frequency": freq})
    df = (pd.DataFrame(rows)
          .groupby(["Scenario", "Fault_Type", "Action"])["Frequency"].mean().reset_index())
    sc = bucket.lower()

    # heatmap
    plt.figure(figsize=(20, 10))
    pivot = df.pivot(index="Fault_Type", columns="Action", values="Frequency").fillna(0)
    faults = pivot.loc[(pivot.sum(axis=1) != 0)].index.tolist()
    pivot = pivot.reindex(index=faults, columns=ACTION_NAMES).fillna(0)
    ax = sns.heatmap(pivot, annot=True, cmap="Blues", vmin=0, fmt=".2f", annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Frequency", fontsize=18)
    plt.title(f"Action Frequencies by Fault Type - {bucket}", fontsize=18)
    plt.xlabel("Action", fontsize=18)
    plt.ylabel("Fault Type", fontsize=18)
    plt.xticks(fontsize=16, rotation=45, ha="right")
    plt.yticks(fontsize=16, rotation=0, va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"action_frequencies_heatmap_{sc}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # barplot
    plt.figure(figsize=(20, 12))
    sns.barplot(x="Fault_Type", y="Frequency", hue="Action", data=df)
    plt.title(f"Action Frequencies by Fault Type - {bucket}", fontsize=16)
    plt.xlabel("Fault Type", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Action", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"action_frequencies_{sc}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # stacked bar
    plt.figure(figsize=(20, 12))
    df.pivot(index="Fault_Type", columns="Action", values="Frequency").plot(kind="bar", stacked=True)
    plt.title(f"Action Frequencies by Fault Type - {bucket}", fontsize=20)
    plt.xlabel("Fault Type", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.legend(title="Action", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"action_frequencies_stacked_bar_{sc}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    return 3


# ---------------------------------------------------------------- analysis 2
def analysis_2(bucket, n_robots, runs, out_dir):
    sc = bucket.lower()
    made = 0
    for group, label in (("faulty", "faulty"), ("non-faulty", "non-faulty")):
        rows = []
        for n, t, mat in runs:
            sub = mat[:, :n] if group == "faulty" else mat[:, n:]
            if sub.shape[1] == 0:
                continue
            for action, freq in freq_series(sub).items():
                rows.append({"Fault_Type": FAULT_NAMES[t],
                             "Action": ACTION_NAMES[action], "Frequency": freq})
        if not rows:
            continue
        df = pd.DataFrame(rows).groupby(["Fault_Type", "Action"])["Frequency"].mean().reset_index()
        plt.figure(figsize=(20, 10))
        pivot = df.pivot(index="Fault_Type", columns="Action", values="Frequency").fillna(0)
        faults = pivot.loc[(pivot.sum(axis=1) != 0)].index.tolist()
        pivot = pivot.reindex(index=faults, columns=ACTION_NAMES).fillna(0)
        ax = sns.heatmap(pivot, annot=True, cmap="Blues", vmin=0, fmt=".2f", annot_kws={"size": 20})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label("Frequency", fontsize=18)
        title_grp = "Faulty" if group == "faulty" else "Non-Faulty"
        plt.title(f"Action Frequencies by Fault Type - {bucket}\n{title_grp} Robots", fontsize=18)
        plt.xlabel("Action", fontsize=18)
        plt.ylabel("Fault Type", fontsize=18)
        plt.xticks(fontsize=16, rotation=45, ha="right")
        plt.yticks(fontsize=16, rotation=0, va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"action_frequencies_heatmap_{sc}_{label}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        made += 1
    return made


def signature_rows(n_robots, runs):
    """Per-run faulty/non-faulty action percentages, tagged with N and fault type."""
    rows = []
    for n, t, mat in runs:
        faulty = mat[:, :n]
        nonfaulty = mat[:, n:]
        if faulty.shape[1] > 0:
            for action, freq in freq_series(faulty).items():
                rows.append({"N": n, "Robot_State": f"F T{t}",
                             "Action": ACTION_NICE[action], "Frequency": freq * 100})
        if nonfaulty.shape[1] > 0:
            for action, freq in freq_series(nonfaulty).items():
                rows.append({"N": n, "Robot_State": f"NF T{t}",
                             "Action": ACTION_NICE[action], "Frequency": freq * 100})
    return pd.DataFrame(rows)


def build_combined(pivot):
    """Build the spaced faulty/non-faulty row layout shared by analyses 3 and 5."""
    fault_types = sorted({int(idx.split("T")[1]) for idx in pivot.index})
    new_rows, row_data = [], []
    if 0 in fault_types and "NF T0" in pivot.index:
        new_rows.append("No fault")
        row_data.append(pivot.loc["NF T0"].values)
        new_rows.append("")
        row_data.append([np.nan] * len(ACTION_NICE))
    others = [ft for ft in fault_types if ft != 0]
    for i, ft in enumerate(others):
        if f"F T{ft}" in pivot.index:
            new_rows.append(f"{FAULT_DESC.get(ft, f'Fault type {ft}')} F")
            row_data.append(pivot.loc[f"F T{ft}"].values)
        if f"NF T{ft}" in pivot.index:
            new_rows.append("NF")
            row_data.append(pivot.loc[f"NF T{ft}"].values)
        if i < len(others) - 1:
            new_rows.append("")
            row_data.append([np.nan] * len(ACTION_NICE))
    return pd.DataFrame(row_data, index=new_rows, columns=pivot.columns)


def draw_combined(new_df, title, path):
    plt.figure(figsize=(22, 14))
    ax = sns.heatmap(new_df, annot=True, cmap="Blues", vmin=0, vmax=100, fmt=".2f",
                     annot_kws={"size": 25}, cbar_kws={"label": "Frequency (%)"},
                     mask=new_df.isna(), linewidths=2, linecolor="white")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label("Frequency (%)", fontsize=25)
    plt.title(title, fontsize=25, pad=20)
    plt.xlabel("Action", fontsize=25, labelpad=15)
    plt.xticks(fontsize=23, rotation=45, ha="right")
    ax.set_yticklabels(new_df.index, rotation=0, fontsize=25)
    for label in ax.get_yticklabels():
        if label.get_text() == "":
            label.set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()


# ---------------------------------------------------------------- analysis 3
def analysis_3(bucket, n_robots, runs, out_dir):
    sc = bucket.lower()
    df = signature_rows(n_robots, runs)
    agg = df.groupby(["Robot_State", "Action"])["Frequency"].mean().reset_index()
    agg.to_csv(os.path.join(out_dir, f"frequency_data_{sc}.csv"), index=False)
    pivot = agg.pivot(index="Robot_State", columns="Action", values="Frequency")
    pivot = pivot.reindex(columns=ACTION_NICE).fillna(0)
    new_df = build_combined(pivot)
    draw_combined(new_df, bucket, os.path.join(out_dir, f"action_frequencies_combined_{sc}.png"))
    return 2


# ---------------------------------------------------------------- analysis 4
def analysis_4(bucket, n_robots, runs, out_dir):
    sc = bucket.lower()
    df = signature_rows(n_robots, runs)
    agg = df.groupby(["N", "Robot_State", "Action"])["Frequency"].mean().reset_index()
    made = 0
    for ft in (3, 4, 5, 8):
        for state in ("F", "NF"):
            sel = agg[agg["Robot_State"] == f"{state} T{ft}"]
            if sel.empty:
                continue
            pivot = sel.pivot(index="N", columns="Action", values="Frequency")
            pivot = pivot.reindex(columns=ACTION_NICE).fillna(0).sort_index()
            plt.figure(figsize=(22, 12))
            ax = sns.heatmap(pivot, annot=True, cmap="Blues", vmin=0, vmax=100, fmt=".2f",
                             annot_kws={"size": 16}, cbar_kws={"label": "Frequency (%)"},
                             linewidths=1, linecolor="white")
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label("Frequency (%)", fontsize=16)
            grp = "Faulty" if state == "F" else "Non-Faulty"
            plt.title(f"Action Frequencies vs Number of Faulty Robots - {bucket}\n"
                      f"{FAULT_NICE[ft]} - {grp} Robots", fontsize=18, pad=20)
            plt.xlabel("Action", fontsize=16, labelpad=15)
            plt.ylabel("Number of Faulty Robots", fontsize=16)
            plt.xticks(fontsize=12, rotation=45, ha="right")
            plt.yticks(fontsize=12, rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"action_frequencies_3d_{sc}_T{ft}_{state}.png"),
                        dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()
            made += 1
    return made


# ---------------------------------------------------------------- analysis 5
def analysis_5(bucket, n_robots, runs, out_dir):
    sc = bucket.lower()
    per_n = {}
    for n, t, mat in runs:
        faulty = mat[:, :n]
        nonfaulty = mat[:, n:]
        bag = per_n.setdefault(n, [])
        if faulty.shape[1] > 0:
            counts = pd.Series(faulty.flatten()).value_counts()
            for action, c in counts.items():
                bag.append({"Robot_State": f"F T{t}", "Action": ACTION_NICE[action],
                            "Frequency": float(c)})
        if nonfaulty.shape[1] > 0:
            counts = pd.Series(nonfaulty.flatten()).value_counts()
            for action, c in counts.items():
                bag.append({"Robot_State": f"NF T{t}", "Action": ACTION_NICE[action],
                            "Frequency": float(c)})
    made = 0
    for n in sorted(per_n):
        df = pd.DataFrame(per_n[n])
        if df.empty:
            continue
        df = df.groupby(["Robot_State", "Action"])["Frequency"].sum().reset_index()
        df["Frequency"] = df.groupby("Robot_State")["Frequency"].transform(lambda x: x / x.sum() * 100)
        pivot = df.pivot(index="Robot_State", columns="Action", values="Frequency")
        pivot = pivot.reindex(columns=ACTION_NICE).fillna(0)
        new_df = build_combined(pivot)
        title = f"{bucket} - {n} Faulty Robot{'s' if n != 1 else ''}"
        draw_combined(new_df, title, os.path.join(out_dir, f"action_frequencies_N{n}_{sc}.png"))
        made += 1
    return made


def main():
    buckets = sys.argv[1:] or ["STNF-SE"]
    dirs = {
        "1": os.path.join(TESTS, "action_frequency"),
        "2": os.path.join(TESTS, "action_frequency_analysis"),
        "3": os.path.join(TESTS, "action_frequency_signature"),
        "4": os.path.join(TESTS, "action_frequency_signature_change"),
        "5": os.path.join(TESTS, "action_frequency_signature_n"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    for bucket in buckets:
        folder = os.path.join(TESTS, bucket, "with_mitigation")
        if not os.path.isdir(folder):
            print(f"[{bucket}] SKIP - {folder} missing")
            continue
        print(f"[{bucket}] loading runs ...", flush=True)
        n_robots, runs = load_bucket(bucket)
        print(f"[{bucket}] {len(runs)} runs, {n_robots} robots/run", flush=True)

        c1 = analysis_1(bucket, runs, dirs["1"])
        print(f"[{bucket}] analysis 1 (action_frequency)            -> {c1} plots", flush=True)
        c2 = analysis_2(bucket, n_robots, runs, dirs["2"])
        print(f"[{bucket}] analysis 2 (faulty vs non-faulty)        -> {c2} plots", flush=True)
        c3 = analysis_3(bucket, n_robots, runs, dirs["3"])
        print(f"[{bucket}] analysis 3 (combined signature)          -> {c3} files", flush=True)
        c4 = analysis_4(bucket, n_robots, runs, dirs["4"])
        print(f"[{bucket}] analysis 4 (frequency vs number faulty)  -> {c4} plots", flush=True)
        c5 = analysis_5(bucket, n_robots, runs, dirs["5"])
        print(f"[{bucket}] analysis 5 (per-N signature)             -> {c5} plots", flush=True)
        print(f"[{bucket}] DONE", flush=True)


if __name__ == "__main__":
    main()
