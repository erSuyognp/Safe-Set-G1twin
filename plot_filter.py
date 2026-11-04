"""Step 5 figure: twin push limit with and without the critic safety filter. Writes results/filter_limits.png."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW = [("none", "task controller only", "k"), ("a0.0_eps0.3", "filter: no push critic, eps 0.3", "#1f77b4"),
        ("a0.3_eps0.3", "filter: robust critic, eps 0.3", "#2ca02c"),
        ("always", "ankle recovery always on (= eps >= 0.5)", "#d62728")]

d = pd.read_csv(os.path.join(HERE, "data", "filter_limits.csv"))
pv = d.pivot_table(index=["controller", "start"], columns="filter", values="thr_G1_N")
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, cname in zip(axs, ["stiff", "omni_jointrange"]):
    sub = pv.loc[cname].reindex(["still", "fwd_small", "fwd_large", "back"])
    x = np.arange(len(sub)); w = 0.2
    for k, (col, label, c) in enumerate(SHOW):
        ax.bar(x + (k - 1.5) * w, sub[col], w, color=c, label=label)
    ax.set_xticks(x, sub.index); ax.set_title(f"task controller: {cname}", fontsize=11)
    ax.set_ylabel("forward push limit (N, G1 equivalent)")
axs[0].legend(fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "filter_limits.png"), dpi=140)
pv.round(1).to_csv(os.path.join(HERE, "results", "filter_limits.csv"))
