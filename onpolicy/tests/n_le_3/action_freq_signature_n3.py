"""
Action-frequency "signature" heatmaps in the established F/NF-subrow format,
capped at <=3 faulty robots, for STNF-SE and ST-SE.

Layout matches action_frequency_signature/action_freq_sign.py
(plot_combined_heatmap): one heatmap whose rows are grouped by fault type, each
group split into an  F  subrow (faulty robots) and an  NF  subrow (non-faulty
robots), with a blank spacer row between groups. NO_FAULT shows only an NF row.
Columns are the 13 actions; cell values are mean action frequency (%).

Produces, per bucket:
  action_freq_signature_combined_n3.png   aggregated over N in {0,1,2,3}
  action_freq_signature_N{n}_n3.png       one per N in {0,1,2,3}
"""
import os
import glob
import warnings
import numpy as np
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

N_CAP = 3
TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BUCKETS = ["STNF-SE", "ST-SE"]
TOTAL_ROBOTS = 10  # SE

ACTION_NAMES = [
    "A1: No action", "A2: Decrease speed 50%", "A3: Stop moving", "A4: Bias to nearest robot",
    "A5: Bias to nearest box", "A6: Bias to nearest wall", "A7: Bias left",
    "A8: Bias from nearest robot", "A9: Bias from nearest box", "A10: Bias from nearest wall",
    "A11: Attract neighbour", "A12: Repel neighbour", "A13: Drop box",
]
FAULT_DESCRIPTIONS = {
    0: "No fault     ",
    3: "F1: 0% speed     ",
    4: "F2: 10% speed     ",
    5: "F3: 50% speed    ",
    8: "F4: Can't pickup     ",
}

MANIFEST = []


def file_freqs(columns, df):
    arr = df[columns].values.flatten().astype(int)
    vc = pd.Series(arr).value_counts()
    return (vc / len(arr)) * 100.0  # per-file percentage, sums to 100


def collect(bucket):
    """Per-file F/NF action frequencies, keyed by N then aggregated to mean per (Robot_State, Action)."""
    folder = os.path.join(TESTS_DIR, bucket, "with_mitigation")
    by_n = {}            # n -> list of record dicts
    all_records = []     # combined over N<=N_CAP
    for path in glob.glob(os.path.join(folder, "simulation_data_run*.csv")):
        fname = os.path.basename(path)
        n = int(fname.split("_N")[1].split("T")[0])
        t = int(fname.split("_N")[1].split("T")[1].split(".")[0])
        if n > N_CAP:
            continue
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c.startswith("Robot_") and c.endswith("_Action")]
        groups = []
        if n > 0:
            groups.append((f"F T{t}", cols[:n]))
        if n < TOTAL_ROBOTS:
            groups.append((f"NF T{t}", cols[n:]))
        for state, gcols in groups:
            for a, f in file_freqs(gcols, df).items():
                rec = {"Robot_State": state, "Action": ACTION_NAMES[a], "Frequency": f}
                all_records.append(rec)
                by_n.setdefault(n, []).append(dict(rec))

    def agg(records):
        d = pd.DataFrame(records)
        if d.empty:
            return d
        return d.groupby(["Robot_State", "Action"])["Frequency"].mean().reset_index()

    return agg(all_records), {n: agg(recs) for n, recs in sorted(by_n.items())}


def signature_heatmap(df, title, out_path):
    """Reproduce the F/NF-subrow + spacer layout from action_freq_sign.py."""
    if df is None or df.empty:
        print("  (no data)", out_path)
        return
    pivot = df.pivot(index="Robot_State", columns="Action", values="Frequency")
    pivot = pivot.reindex(columns=ACTION_NAMES).fillna(0)
    fault_types = sorted(set(int(idx.split("T")[1]) for idx in pivot.index))

    new_rows, row_data = [], []
    # NO_FAULT: NF row only
    if 0 in fault_types:
        new_rows.append("No fault")
        if "NF T0" in pivot.index:
            row_data.append(pivot.loc["NF T0"].values)
        else:
            row_data.append([np.nan] * len(ACTION_NAMES))
        new_rows.append("")
        row_data.append([np.nan] * len(ACTION_NAMES))

    other = [ft for ft in fault_types if ft != 0]
    for i, ft in enumerate(other):
        f_idx = f"F T{ft}"
        if f_idx in pivot.index:
            new_rows.append(f"{FAULT_DESCRIPTIONS.get(ft, f'Fault type {ft}')} F")
            row_data.append(pivot.loc[f_idx].values)
        nf_idx = f"NF T{ft}"
        if nf_idx in pivot.index:
            new_rows.append("NF")
            row_data.append(pivot.loc[nf_idx].values)
        if i < len(other) - 1:
            new_rows.append("")
            row_data.append([np.nan] * len(ACTION_NAMES))

    new_df = pd.DataFrame(row_data, index=new_rows, columns=pivot.columns)

    plt.figure(figsize=(22, 14))
    ax = sns.heatmap(
        new_df, annot=True, cmap="Blues", vmin=0, vmax=100, fmt=".2f",
        annot_kws={"size": 25}, cbar_kws={"label": "Frequency (%)"},
        mask=new_df.isna(), linewidths=2, linecolor="white",
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label("Frequency (%)", fontsize=25)
    plt.title(title, fontsize=25, pad=20)
    plt.xlabel("Action", fontsize=25, labelpad=15)
    plt.ylabel("")
    plt.xticks(fontsize=23, rotation=45, ha="right")
    ax.set_yticklabels(new_rows, rotation=0, fontsize=25)
    for label in ax.get_yticklabels():
        if label.get_text() == "":
            label.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    MANIFEST.append(out_path)
    print("  wrote", os.path.relpath(out_path, OUT_DIR))


def main():
    for bucket in BUCKETS:
        print(f"== {bucket} ==")
        bdir = os.path.join(OUT_DIR, bucket.lower())
        os.makedirs(bdir, exist_ok=True)

        # remove the superseded two-panel faulty/non-faulty figure from the earlier run
        old = os.path.join(bdir, f"action_freq_faulty_vs_nonfaulty_n{N_CAP}.png")
        if os.path.exists(old):
            os.remove(old)
            print("  removed superseded", os.path.relpath(old, OUT_DIR))

        combined, by_n = collect(bucket)
        signature_heatmap(
            combined, f"{bucket} (up to {N_CAP} faulty robots)",
            os.path.join(bdir, f"action_freq_signature_combined_n{N_CAP}.png"))
        for n, dfn in by_n.items():
            signature_heatmap(
                dfn, f"{bucket} - {n} Faulty Robot{'s' if n != 1 else ''}",
                os.path.join(bdir, f"action_freq_signature_N{n}_n{N_CAP}.png"))
        print()

    print(f"Done. {len(MANIFEST)} signature figures written.")


if __name__ == "__main__":
    main()
