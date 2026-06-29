# `n_le_3/` — SE figures capped at ≤3 faulty robots (STNF-SE vs ST-SE)

Figures comparing the no-fault-trained (`STNF-SE`) and fault-trained (`ST-SE`) standard
policies against the shared random-walk baseline, for the Standard Environment, with the
fault count capped at ≤3 faulty robots. Run any script with `~/.venv/marl/bin/python`.
See `../../../HANDOFF.md` section 3-4 for context and the bucket-naming key.

Data sources (read-only): `../STNF-SE/` and `../ST-SE/`
`performance_comparison_results.csv` (per-run N-CBDS: P_Mitigated vs shared P_Baseline)
and `with_mitigation/simulation_data_run*.csv` (per-step robot actions).

## Scripts -> outputs

| Script | Produces |
| --- | --- |
| `make_n3_figures.py` | per bucket (`stnf-se/`, `st-se/`): `performance_boxplot_n3.png`, `mitigation_power_heatmap_n3.png` (+ `_analysis_n3.csv`), `action_freq_by_fault_heatmap_n3.png`, `action_freq_by_fault_bar_n3.png`, `summary_statistics_n3.csv`; plus `combined_performance_boxplot_n3.png` |
| `action_freq_signature_n3.py` | per bucket: `action_freq_signature_combined_n3.png` and `action_freq_signature_N{0..3}_n3.png` — F/NF-subrow signature heatmaps (fault-type groups, each split into Faulty / Non-faulty rows) |
| `boxplot_n3_vs_n10.py` | `boxplot_n3_vs_n10.png` — 6 boxes: Baseline / STNF-SE / ST-SE at ≤3 vs ≤10 (pooled over fault types) |
| `boxplot_per_fault_n3_vs_n10.py` | `boxplot_per_fault_n3_vs_n10.png` — x = fault type (DEFAULT, F1-F4); 6 boxes per fault type (3 conditions x {≤3, ≤10}) = 30 boxes |

## Notes
- N-CBDS = `Delivery_Rate.sum()/10/50`; `P_Mitigated`/`P_Baseline` already equal this.
- DEFAULT (no-fault) has only N=0 data, so its ≤3 and ≤10 boxes are identical by construction.
- Mitigation power = `2*sign(R_mit - R_none)*|U/(n1*n2) - 0.5|` (Mann-Whitney, two-sided).
- These regenerate from committed CSVs alone — no simulator/`marl_sim` needed for the
  boxplots/heatmaps; the action-frequency scripts need the committed `with_mitigation/` CSVs.
