"""Step 4: one figure comparing the three safe sets with the twin's measured recover / fall points, plus the
numbers that answer the three questions. Writes results/safe_sets.png and results/summary.json.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

from lip import LIP
import critic

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
C_AN, C_HJ, C_LR = "#1f77b4", "#2ca02c", "#d62728"


def load_sets(lip, a_tilde):
    z = np.load(os.path.join(DATA, f"hj_a{a_tilde:.1f}.npz"))
    xs, vs, V_hj = z["xs"], z["vs"], z["V"]
    h, w = lip.toe, lip.w
    V_lr = critic.value(os.path.join(DATA, f"critic_a{a_tilde:.1f}.pt"), xs / h, vs / (w * h)) * h
    X, Vv = np.meshgrid(xs, vs, indexing="ij")
    an = lip.analytic_safe(X, Vv, z["a_max"].item())
    return dict(xs=xs, vs=vs, X=X, Vv=Vv, hj=V_hj, lr=V_lr, an=an, a_max=z["a_max"].item())


def set_stats(S, lip):
    hj, lr, an = S["hj"] >= 0, S["lr"] >= 0, S["an"]
    inside_foot = (S["X"] >= lip.heel) & (S["X"] <= lip.toe)
    xi = S["X"] + S["Vv"] / lip.w
    miss_out = lr & ~hj          # learned says safe, HJ says unsafe (dangerous)
    miss_in = hj & ~lr           # learned too conservative
    def where(mask):
        if not mask.any(): return None
        return dict(x_mm=[float(S["X"][mask].min() * 1e3), float(S["X"][mask].max() * 1e3)],
                    xi_minus_toe_mm=[float((xi[mask] - lip.toe).min() * 1e3), float((xi[mask] - lip.toe).max() * 1e3)])
    return dict(cells_hj=int(hj.sum()), cells_learned=int(lr.sum()), cells_analytic=int(an.sum()),
                iou_learned_hj=float((hj & lr).sum() / max(1, (hj | lr).sum())),
                iou_hj_analytic=float((hj & an).sum() / max(1, (hj | an).sum())),
                learned_unsafe_false_positive=int(miss_out.sum()), learned_conservative=int(miss_in.sum()),
                false_positive_where=where(miss_out), conservative_where=where(miss_in),
                frac_false_positive=float(miss_out.sum() / max(1, hj.sum())),
                frac_conservative=float(miss_in.sum() / max(1, hj.sum())))


def predicted_limits(lip, S0, thr):
    """Push limit of the LIP from each measured pre push state, by the three set descriptions (d = 0 after the push)."""
    hj = RegularGridInterpolator((S0["xs"], S0["vs"]), S0["hj"], bounds_error=False, fill_value=-1)
    lr = RegularGridInterpolator((S0["xs"], S0["vs"]), S0["lr"], bounds_error=False, fill_value=-1)
    out = []
    for _, r in thr.iterrows():
        lim = lambda fn: lip.predicted_push_limit(r.x0, r.v0, safe=lambda x, v: fn([[x, v]])[0] >= 0)
        out.append(dict(controller=r.controller, start=r.start, x0_mm=r.x0 * 1e3, v0=r.v0,
                        measured_N=r.thr_N, analytic_N=lip.predicted_push_limit(r.x0, r.v0),
                        hj_N=lim(hj), learned_N=lim(lr)))
    return pd.DataFrame(out)


def main():
    os.makedirs(RES, exist_ok=True)
    lip = LIP()
    S0, S3 = load_sets(lip, 0.0), load_sets(lip, 0.3)
    pushes = pd.read_csv(os.path.join(DATA, "twin_pushes.csv"))
    thr = pd.read_csv(os.path.join(DATA, "twin_thresholds.csv"))
    G1_per_twin = lip.P["G1_N_per_I"] / lip.P["F_per_I"]
    lim = predicted_limits(lip, S0, thr)
    for c in ("measured_N", "analytic_N", "hj_N", "learned_N"):
        lim[c.replace("_N", "_G1N")] = lim[c] * G1_per_twin
    lim["measured_over_analytic"] = lim.measured_N / lim.analytic_N

    # post push states of the twin, classified by the LIP capture region (d = 0 after the push)
    pushes["xi1"] = pushes.x1 + pushes.v1 / lip.w
    pushes["lip_safe"] = lip.analytic_safe(pushes.x1.values, pushes.v1.values)
    pushes["outcome"] = np.where(pushes.fell, "fell", "recovered")
    conf = pushes.groupby(["controller", "lip_safe", "outcome"]).size().unstack(fill_value=0)
    beyond = pushes[(~pushes.fell) & (~pushes.lip_safe)]
    within = pushes[(~pushes.fell) & (pushes.lip_safe)]

    # --- figure ------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw=dict(width_ratios=[1, 1.25, 0.9]))
    for ax, S, title in ((axs[0], S3, f"Robust safe set vs a persistent worst case push\n"
                          f"|F| <= {S3['a_max']*lip.m:.2f} N twin ({S3['a_max']*lip.m*G1_per_twin:.0f} N G1 equivalent)"),
                         (axs[1], S0, "Safe set once the push has ended, with the twin's\nstate at the end of each push")):
        xm, vm = S["xs"] * 1e3, S["vs"]
        ax.contourf(xm, vm, (S["hj"] >= 0).T.astype(float), levels=[0.5, 1.5], colors=[C_HJ], alpha=0.15)
        ax.contour(xm, vm, S["an"].T.astype(float), levels=[0.5], colors=[C_AN], linewidths=2.5, linestyles="--")
        ax.contour(xm, vm, S["hj"].T, levels=[0], colors=[C_HJ], linewidths=2)
        ax.contour(xm, vm, S["lr"].T, levels=[0], colors=[C_LR], linewidths=1.6)
        ax.axvline(lip.heel * 1e3, color="0.5", lw=0.8); ax.axvline(lip.toe * 1e3, color="0.5", lw=0.8)
        ax.set_xlabel("CoM position over the foot x (mm)   heel -  toe +"); ax.set_ylabel("CoM velocity v (m/s)")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xm[0], xm[-1]); ax.set_ylim(-0.45, 0.45)
    ax = axs[1]
    mk = {"stiff": "o", "omni_jointrange": "s"}
    for cname, g in pushes.groupby("controller"):
        rec, fell = g[~g.fell], g[g.fell]
        ax.scatter(rec.x1 * 1e3, rec.v1, marker=mk[cname], s=22, facecolor="none", edgecolor="k", lw=0.9, zorder=5)
        ax.scatter(fell.x1 * 1e3, fell.v1, marker=mk[cname], s=22, color="#ff7f0e", zorder=5)
    ax.set_ylim(-0.1, 0.7)
    handles = [Line2D([], [], color=C_AN, lw=2.5, ls="--", label="analytic (capture point)"),
               Line2D([], [], color=C_HJ, lw=2, label="exact HJ (grid)"),
               Line2D([], [], color=C_LR, lw=1.6, label="learned adversarial critic"),
               Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="twin, stiff PD: recovered"),
               Line2D([], [], marker="o", ls="", color="#ff7f0e", label="twin, stiff PD: fell"),
               Line2D([], [], marker="s", ls="", mfc="none", mec="k", label="twin, omni_jointrange: recovered"),
               Line2D([], [], marker="s", ls="", color="#ff7f0e", label="twin, omni_jointrange: fell")]
    axs[0].legend(handles=handles[:3], loc="lower left", fontsize=8.5)
    axs[1].legend(handles=handles[3:], loc="upper right", fontsize=8.5)
    # panel 3: push limits
    ax = axs[2]
    labels = [f"{r.controller.replace('omni_jointrange','omni_jr')}\n{r.start}" for _, r in lim.iterrows()]
    y = np.arange(len(lim))
    ax.barh(y + 0.27, lim.measured_G1N, 0.25, color="k", label="twin, measured")
    ax.barh(y, lim.analytic_G1N, 0.25, color=C_AN, label="LIP, analytic")
    ax.barh(y - 0.27, lim.learned_G1N, 0.25, color=C_LR, label="LIP, learned critic")
    ax.scatter(lim.hj_G1N, y, marker="|", s=120, color=C_HJ, zorder=5, label="LIP, exact HJ")
    ax.set_yticks(y, labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("forward push limit, 0.1 s push (N, G1 equivalent)")
    ax.set_title("Push limit: twin vs the reduced model\nfrom the same pre push state", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_xlim(0, 230)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "safe_sets.png"), dpi=140)

    summary = dict(
        lip=dict(omega=lip.w, com_height_mm=lip.zc * 1e3, foot_mm=[lip.heel * 1e3, lip.toe * 1e3], mass=lip.m,
                 push_duration_s=lip.T_push, G1_N_per_twin_N=G1_per_twin),
        robust_a0_3=set_stats(S3, lip), no_push=set_stats(S0, lip),
        push_limits=lim.round(3).to_dict(orient="records"),
        confusion=conf.reset_index().to_dict(orient="records"),
        survived_outside_capture_region=dict(
            n=int(len(beyond)), n_survived_total=int((~pushes.fell).sum()),
            max_xi_beyond_toe_mm=float((beyond.xi1 - lip.toe).max() * 1e3) if len(beyond) else 0.0,
            max_com_x_beyond_toe_mm=float((pushes[~pushes.fell].max_com_x - lip.toe).max() * 1e3),
            foot_pitch_deg_beyond=float(np.degrees(beyond.max_foot_pitch).mean()) if len(beyond) else None,
            foot_pitch_deg_within=float(np.degrees(within.max_foot_pitch).mean()) if len(within) else None,
            pelvis_pitch_deg_beyond=float(np.degrees(beyond.max_pelvis_pitch).mean()) if len(beyond) else None,
            pelvis_pitch_deg_within=float(np.degrees(within.max_pelvis_pitch).mean()) if len(within) else None,
            by_controller=beyond.groupby("controller").size().to_dict()),
        fell_inside_capture_region=int((pushes.fell & pushes.lip_safe).sum()),
    )
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
    lim.round(2).to_csv(os.path.join(RES, "push_limits.csv"), index=False)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
