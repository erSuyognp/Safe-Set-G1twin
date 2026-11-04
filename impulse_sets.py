"""Exact HJ safe sets against one bounded push of the twin's duration (step 5, impulse critic).
Writes data/hj_imp_a{A~}.npz with the value at every remaining push time, and prints the value of the
measured pre push states of the twin at "a full push could start now"."""
import os, sys
import numpy as np, pandas as pd
from scipy.interpolate import RegularGridInterpolator
from lip import LIP, hj_impulse_set, HERE

lip = LIP()
z = np.load(os.path.join(HERE, "data", "hj_a0.0.npz"))
xs, vs, V0 = z["xs"], z["vs"], z["V"]
thr = pd.read_csv(os.path.join(HERE, "data", "twin_thresholds.csv"))
unit = lip.w ** 2 * lip.toe                       # a~ = a / (w^2 h)
G1 = lip.P["G1_N_per_I"] / lip.P["F_per_I"]
for at in [float(a) for a in sys.argv[1:]] or [1.0, 1.5, 2.0, 2.5, 3.0]:
    Vs = hj_impulse_set(lip, at * unit, lip.T_push, V0, xs, vs)
    np.savez(os.path.join(HERE, "data", f"hj_imp_a{at:.1f}.npz"), xs=xs, vs=vs, V=np.stack(Vs), A=at * unit, T=lip.T_push)
    f = RegularGridInterpolator((xs, vs), Vs[-1], bounds_error=False, fill_value=-1)
    vals = [f([[r.x0, r.v0]])[0] / lip.toe for _, r in thr.iterrows()]
    print(f"a~ {at:.1f} = {at*unit*lip.m*G1:5.0f} N G1: safe cells {(Vs[-1]>=0).sum():6d} (no push {(V0>=0).sum()}) "
          f"pre push V~: " + " ".join(f"{r.controller[:4]}/{r.start}:{v:+.2f}" for (_, r), v in zip(thr.iterrows(), vals)))
