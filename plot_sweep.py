"""Step 5 summary figure from the dense sweeps: change in the twin's strict push limit from each filter, per
controller and push duration, mean over 4 starts x 2 directions, with the worst case as a marker.
Writes results/filter_sweep.png and results/filter_sweep_summary.csv."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"0.06 s": "filter_sweep_steps3.csv", "0.1 s (design)": "filter_sweep.csv", "0.2 s": "filter_sweep_steps10.csv"}
FILTERS = [("always", "ankle recovery\nalways on"), ("a0.3_eps0.3", "persistent push\ncritic, eps 0.3"),
           ("imp2.0_eps0.3", "impulse critic\neps 0.3"), ("qimp2.0_eps0.1", "impulse critic,\nQ filter, eps 0.1")]
COLORS = ["#9ecae1", "#3182bd", "#08519c"]

rows = []
for dur, f in FILES.items():
    d = pd.read_csv(os.path.join(HERE, "data", f))
    base = d[d["filter"] == "none"].set_index(["controller", "start", "direction"]).strict_G1_N
    d = d[d["filter"].isin([k for k, _ in FILTERS])].copy()
    d["gain"] = [r.strict_G1_N - base[(r.controller, r.start, r.direction)] for r in d.itertuples()]
    for (c, flt), g in d.groupby(["controller", "filter"]):
        rows.append(dict(duration=dur, controller=c, filter=flt, mean_gain=g.gain.mean(), worst_gain=g.gain.min(),
                         n_worse=int((g.gain < -1).sum()), baseline_mean=base.loc[c].mean()))
S = pd.DataFrame(rows)
S.round(1).to_csv(os.path.join(HERE, "results", "filter_sweep_summary.csv"), index=False)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, c in zip(axs, ["stiff", "omni_jointrange"]):
    x = np.arange(len(FILTERS)); w = 0.26
    for k, dur in enumerate(FILES):
        sub = S[(S.controller == c) & (S.duration == dur)].set_index("filter").reindex([f for f, _ in FILTERS])
        ax.bar(x + (k - 1) * w, sub.mean_gain, w, color=COLORS[k], label=f"push {dur}")
        ax.scatter(x + (k - 1) * w, sub.worst_gain, marker="v", color="#d62728", s=28, zorder=5,
                   label="worst of the 8 cases" if k == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [l for _, l in FILTERS], fontsize=8.5)
    ax.set_title(f"task controller: {c}", fontsize=11)
axs[0].set_ylabel("change in strict push limit vs no filter\n(N, G1 equivalent; mean of 4 starts x 2 directions)")
axs[0].legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "filter_sweep.png"), dpi=140)
print(S.round(1).to_string())
