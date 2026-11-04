"""Step 5: the learned critic as a safety filter on the twin.

Each control step the filter reads the twin's CoM state over the foot (privileged, straight from the sim),
evaluates the critic V(s) = min Q(s, pi_c(s), pi_d(s)) in dimensionless LIP units and
    V >= eps          -> task controller (stiff PD or omni_jointrange) unchanged
    V <  eps          -> recovery: the task action, with both ankle pitch actions replaced by the CoP the
                         critic's controller actor asks for (ankle strategy); latched until V > 2 eps.
CoP to ankle action map from probe_ankle.py: CoP ~ -6 mm + 230 mm x action, saturating at the heel / toe.
Reference "always": the recovery ankle command at every step, no switching.

For each controller x start x filter: forward push limit by bisection, same protocol as twin_push.py.
Run with the omni_brain_starter venv. Output: data/filter_limits.csv. Pass filter names to run only those and
merge them into the existing csv.
"""
import math, os, sys, time
import numpy as np
import torch

import twin_push as tp
from twin_push import oe, ev
import critic

HERE = os.path.dirname(os.path.abspath(__file__))
ANKLE = [3, 8]
P_GAIN, P_OFFSET = 0.13, 0.03            # ankle action = P_GAIN * p~ + P_OFFSET
FILTERS = {"none": None, "always": ("a0.0", None),
           "a0.0_eps0.1": ("a0.0", 0.1), "a0.0_eps0.3": ("a0.0", 0.3), "a0.0_eps0.5": ("a0.0", 0.5), "a0.0_eps0.7": ("a0.0", 0.7),
           "a0.3_eps0.1": ("a0.3", 0.1), "a0.3_eps0.3": ("a0.3", 0.3),
           # impulse critics: one push of up to a~ for the twin's push duration; evaluated at "a full push could start now"
           "imp1.5_eps0.1": ("imp_a1.5", 0.1), "imp1.5_eps0.3": ("imp_a1.5", 0.3),
           "imp2.0_eps0.1": ("imp_a2.0", 0.1), "imp2.0_eps0.3": ("imp_a2.0", 0.3),
           # Q filter: judge the task controller's own CoP, Q(s, p~_task, pi_d(s)), not the state's best case
           "qimp1.5_eps0.1": ("imp_a1.5", 0.1, "q"), "qimp1.5_eps0.3": ("imp_a1.5", 0.3, "q"),
           "qimp2.0_eps0.1": ("imp_a2.0", 0.1, "q"), "qimp2.0_eps0.3": ("imp_a2.0", 0.3, "q")}


class Filter:
    def __init__(self, task, which, eps, lip, mode="v"):
        self.task, self.eps, self.mode = task, eps, mode
        _, self.Q, self.pc, self.pd = critic.load(os.path.join(HERE, "data", f"critic_{which}.pt"))
        self.h, self.w = lip["toe"], math.sqrt(lip["g"] / lip["com_height"])
        self.extra = [1.0] if self.Q.q1[0].in_features == 5 else []      # remaining push time = a full push
        self.active = False; self.n_active = 0; self.n = 0

    @torch.no_grad()
    def __call__(self, obs, x, v):
        a = np.array(self.task(obs), dtype=np.float64)
        s = torch.tensor([[x / self.h, v / (self.w * self.h)] + self.extra], dtype=torch.float32)
        u = self.pc(s)
        if self.eps is not None and self.mode == "q":
            # the task's CoP from its ankle actions, through the inverse of the probe_ankle map
            p_task = float(np.clip((a[ANKLE].mean() - P_OFFSET) / P_GAIN, -1, 1))
            V = torch.min(*self.Q(s, torch.tensor([[p_task]]), self.pd(s))).item()
            self.active = V < self.eps                  # no latch: judged afresh every step
        elif self.eps is not None:
            V = torch.min(*self.Q(s, u, self.pd(s))).item()
            self.active = V < (2 * self.eps if self.active else self.eps)
        else:
            self.active = True
        self.n += 1
        if self.active:
            self.n_active += 1
            a[ANKLE] = P_GAIN * u.item() + P_OFFSET
        return a


def trial(env, ctrl, I_main, I_pre, foot_mid, angle=0.0):
    pushes = [(tp.MAIN_STEP, float(I_main), angle)]
    if I_pre: pushes.append((tp.PRE_STEP, abs(I_pre), 0.0 if I_pre > 0 else math.pi))
    obs, _ = env.reset(seed=0, options={"pushes": pushes})
    done = False
    while not done:
        c, v = tp.com_state(env)
        obs, r, term, trunc, info = env.step(ctrl(obs, c[0] - foot_mid, v[0]))
        done = term or trunc
    return bool(info["fell"])


def limit(env, ctrl, I_pre, foot_mid):
    if trial(env, ctrl, 0.0, I_pre, foot_mid): return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if trial(env, ctrl, mid, I_pre, foot_mid) else (mid, hi)
    return 0.5 * (lo + hi)


def main():
    import json, pandas as pd
    lip = json.load(open(os.path.join(HERE, "data", "twin_lip.json")))
    only = sys.argv[1:]
    rows = []
    for cname, path, hist in ev.parse_controllers(tp.CONTROLLERS):
        policy = ev.load_policy(path)
        env = oe.BalanceEnv("twin", history=hist)
        env.reset(seed=0, options={"pushes": []})
        xs = np.array([env.d.geom_xpos[i][0] for i in tp.foot_spheres(env.m)]); foot_mid = 0.5 * (xs.min() + xs.max())
        for sname, I_pre in tp.STARTS.items():
            for fname, spec in FILTERS.items():
                if only and fname not in only: continue
                t0 = time.time()
                if spec is None:
                    ctrl = lambda obs, x, v: policy(obs); f = None
                else:
                    f = ctrl = Filter(policy, *spec[:2], lip, *spec[2:])
                thr = limit(env, ctrl, I_pre, foot_mid)
                frac = None
                if f is not None:            # time share in recovery mode during a push at 90% of the limit
                    f.n = f.n_active = 0; trial(env, f, 0.9 * thr, I_pre, foot_mid); frac = f.n_active / max(1, f.n)
                rows.append(dict(controller=cname, start=sname, filter=fname, thr_I=thr, thr_N=thr * lip["F_per_I"],
                                 thr_G1_N=thr * lip["G1_N_per_I"], recovery_share=frac))
                print(f"{cname:16s} {sname:10s} {fname:12s} {thr * lip['G1_N_per_I']:6.1f} N G1  "
                      f"recovery share {frac if frac is None else round(frac, 2)}  {time.time()-t0:.0f}s", flush=True)
    out = os.path.join(HERE, "data", "filter_limits.csv")
    new = pd.DataFrame(rows)
    if only and os.path.exists(out):
        old = pd.read_csv(out)
        new = pd.concat([old[~old["filter"].isin(only)], new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    main()
