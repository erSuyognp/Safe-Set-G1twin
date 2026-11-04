"""Step 5 check: dense push sweeps instead of bisection. With a switching filter, survival need not be monotone
in push size, so bisection can land on a lucky or unlucky island. For each controller x start x filter x
direction (forward, backward), push sizes on a fixed grid; reports
    strict limit = largest push such that every push up to it was survived
    survival     = share of grid pushes survived (0 to 290 N G1)
Run with the omni_brain_starter venv:  python sweep_filter.py [filters...]   -> data/filter_sweep.csv
Environment variable PUSH_STEPS (default 5 = 0.1 s G1 time) changes the push duration at the same force
grid; the output is then data/filter_sweep_steps<N>.csv (a held out check: the critics assume 5)."""
import math, os, sys, json, time
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
PUSH_STEPS = int(os.environ.get("PUSH_STEPS", 5))
I_GRID = np.linspace(0.0, 0.35, 36)
DEFAULT = ["none", "always", "a0.3_eps0.3", "imp1.5_eps0.1", "imp1.5_eps0.3", "imp2.0_eps0.1", "imp2.0_eps0.3"]
DIRS = {"fwd": 0.0, "back": math.pi}


def job(args):
    cname, sname, fname = args
    import safety_filter as sf
    tp, oe, ev = sf.tp, sf.oe, sf.ev
    oe.PUSH_STEPS = PUSH_STEPS                         # force per unit I is unchanged, so I_GRID stays a force grid
    lip = json.load(open(os.path.join(HERE, "data", "twin_lip.json")))
    (cn, path, hist), = [c for c in ev.parse_controllers(tp.CONTROLLERS) if c[0] == cname]
    policy = ev.load_policy(path)
    env = oe.BalanceEnv("twin", history=hist)
    env.reset(seed=0, options={"pushes": []})
    xs = np.array([env.d.geom_xpos[i][0] for i in tp.foot_spheres(env.m)]); foot_mid = 0.5 * (xs.min() + xs.max())
    spec = sf.FILTERS[fname]
    ctrl = (lambda obs, x, v: policy(obs)) if spec is None else sf.Filter(policy, *spec[:2], lip, *spec[2:])
    rows = []
    for dname, ang in DIRS.items():
        fell = np.array([sf.trial(env, ctrl, I, tp.STARTS[sname], foot_mid, ang) for I in I_GRID])
        first = int(np.argmax(fell)) if fell.any() else len(I_GRID)
        strict = I_GRID[first - 1] if first > 0 else 0.0
        rows.append(dict(controller=cname, start=sname, filter=fname, direction=dname,
                         strict_G1_N=strict * lip["G1_N_per_I"], survival=float(1 - fell.mean()),
                         non_monotone=int(fell[first:].size - fell[first:].sum()),
                         outcomes="".join("x" if f else "." for f in fell)))
    return rows


def main():
    import pandas as pd
    import safety_filter as sf
    filters = sys.argv[1:] or DEFAULT
    jobs = [(c, s, f) for c in ("stiff", "omni_jointrange") for s in sf.tp.STARTS for f in filters]
    t0 = time.time()
    with mp.Pool(min(12, os.cpu_count())) as pool:
        rows = [r for rs in pool.imap_unordered(job, jobs) for r in rs]
    out = os.path.join(HERE, "data", "filter_sweep.csv" if PUSH_STEPS == 5 else f"filter_sweep_steps{PUSH_STEPS}.csv")
    new = pd.DataFrame(rows)
    if sys.argv[1:] and os.path.exists(out):
        old = pd.read_csv(out)
        new = pd.concat([old[~old["filter"].isin(filters)], new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"{len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
