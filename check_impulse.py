"""Step 5, impulse critic: compare the learned impulse critic with the exact impulse HJ set and an analytic
check (constant worst case push in either direction, best constant CoP once the direction is known).
Writes results/impulse_sets.png and results/impulse_summary.json."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lip import LIP, HERE
import critic

C_AN, C_HJ, C_LR = "#1f77b4", "#2ca02c", "#d62728"


def analytic_impulse_safe(lip, X, Vv, A, n_p=41, n_t=20):
    """Safe if, for a constant push of +A and of -A over the push duration, some constant CoP keeps x in the
    foot during the push and leaves the state in the capture region."""
    ts = np.linspace(0, lip.T_push, n_t)
    ok = np.ones(X.shape, bool)
    for a in (A, -A):
        any_p = np.zeros(X.shape, bool)
        for p in np.linspace(lip.heel, lip.toe, n_p):
            inside = np.ones(X.shape, bool)
            for t in ts:
                x, v = lip.flow(X, Vv, p, a, t)
                inside &= (x >= lip.heel) & (x <= lip.toe)
            any_p |= inside & lip.analytic_safe(x, v)
        ok &= any_p
    return ok


def main():
    lip = LIP()
    thr = pd.read_csv(os.path.join(HERE, "data", "twin_thresholds.csv"))
    G1 = lip.P["G1_N_per_I"] / lip.P["F_per_I"]
    summary = {}
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, at in zip(axs, (1.5, 2.0)):
        z = np.load(os.path.join(HERE, "data", f"hj_imp_a{at:.1f}.npz"))
        xs, vs, Vhj = z["xs"], z["vs"], z["V"]                # Vhj[k]: k / K of the push still to come
        K = len(Vhj) - 1
        X, Vv = np.meshgrid(xs, vs, indexing="ij")
        an = analytic_impulse_safe(lip, X, Vv, z["A"].item())
        path = os.path.join(HERE, "data", f"critic_imp_a{at:.1f}.pt")
        row = {}
        for r in (1.0, 0.5, 0.0):
            hj = Vhj[int(round(r * K))] >= 0
            lr = critic.value(path, xs / lip.toe, vs / (lip.w * lip.toe), r=r) >= 0
            row[f"r{r}"] = dict(cells_hj=int(hj.sum()), cells_learned=int(lr.sum()),
                                iou_learned_hj=float((hj & lr).sum() / max(1, (hj | lr).sum())),
                                learned_false_safe=float((lr & ~hj).sum() / max(1, hj.sum())),
                                learned_conservative=float((hj & ~lr).sum() / max(1, hj.sum())))
            if r == 1.0:
                row["iou_hj_analytic"] = float((hj & an).sum() / max(1, (hj | an).sum()))
                xm = xs * 1e3
                ax.contourf(xm, vs, hj.T.astype(float), levels=[0.5, 1.5], colors=[C_HJ], alpha=0.15)
                ax.contour(xm, vs, an.T.astype(float), levels=[0.5], colors=[C_AN], linewidths=2.5, linestyles="--")
                ax.contour(xm, vs, hj.T.astype(float), levels=[0.5], colors=[C_HJ], linewidths=2)
                ax.contour(xm, vs, lr.T.astype(float), levels=[0.5], colors=[C_LR], linewidths=1.6)
                nopush = np.load(os.path.join(HERE, "data", "hj_a0.0.npz"))["V"] >= 0
                ax.contour(xm, vs, nopush.T.astype(float), levels=[0.5], colors=["0.6"], linewidths=1, linestyles=":")
        summary[f"a{at}"] = dict(push_G1_N=at * lip.w ** 2 * lip.toe * lip.m * G1, **row)
        for _, t in thr.iterrows():
            mk = "o" if t.controller == "stiff" else "s"
            ax.scatter(t.x0 * 1e3, t.v0, marker=mk, s=40, facecolor="none", edgecolor="k", zorder=5)
            ax.annotate(t.start, (t.x0 * 1e3, t.v0), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.axvline(lip.heel * 1e3, color="0.5", lw=0.8); ax.axvline(lip.toe * 1e3, color="0.5", lw=0.8)
        ax.set_xlim(-32, 32); ax.set_ylim(-0.35, 0.35)
        ax.set_xlabel("CoM position over the foot x (mm)   heel -  toe +")
        ax.set_title(f"Safe against one push of up to {summary[f'a{at}']['push_G1_N']:.0f} N G1 for 0.1 s G1 time,\n"
                     f"starting now (IoU learned vs HJ {row['r1.0']['iou_learned_hj']:.2f})", fontsize=11)
    axs[0].set_ylabel("CoM velocity v (m/s)")
    axs[0].legend(handles=[Line2D([], [], color=C_AN, lw=2.5, ls="--", label="analytic (constant worst push)"),
                           Line2D([], [], color=C_HJ, lw=2, label="exact HJ (grid)"),
                           Line2D([], [], color=C_LR, lw=1.6, label="learned impulse critic"),
                           Line2D([], [], color="0.6", lw=1, ls=":", label="no push safe set"),
                           Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="twin pre push, stiff"),
                           Line2D([], [], marker="s", ls="", mfc="none", mec="k", label="twin pre push, omni_jointrange")],
                  fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "impulse_sets.png"), dpi=140)
    json.dump(summary, open(os.path.join(HERE, "results", "impulse_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
