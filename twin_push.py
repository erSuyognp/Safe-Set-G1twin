"""Step 1: push the G1 twin forward and record where its center of mass is when it recovers or falls.

Run with the omni_brain_starter venv (it has stable_baselines3 for the PPO policies):
    "D:/rookie robotics/omni_brain_starter/omni_brain_starter/.venv/Scripts/python" twin_push.py

Uses BalanceEnv("twin") from omni_env.py unchanged: 1/3 scale G1 twin, ideal PD at the joints, push at the
pelvis for 5 control steps (0.1 s of G1 time), fall = pelvis below 0.65 z0 or torso tilt above 60 deg.

Starting conditions: standing still, or already moving (a pre push 10 control steps before the main push).
For each (controller, starting condition):
  * the fall threshold of the main push by bisection
  * a sweep of main pushes from 0 to 1.6 x threshold, each recorded as
    CoM state over the foot at push start and push end, outcome, and how much the twin used
    its feet (heel or toe lift) and hips (pelvis pitch) to recover.
Everything is in twin SI units. Output: data/twin_pushes.csv, data/twin_thresholds.csv, data/twin_lip.json.
"""
import json, math, os, sys
import numpy as np
import mujoco

TWIN_DIR = r"D:\rookie robotics\omni_brain_starter\omni_brain_starter"
sys.path.insert(0, TWIN_DIR)
os.chdir(TWIN_DIR)
import omni_env as oe
import evaluate as ev

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAIN_STEP = ev.PUSH_STEP                     # 40: 0.8 s of G1 time
PRE_STEP = MAIN_STEP - 10
STARTS = {"still": 0.0, "fwd_small": 0.04, "fwd_large": 0.08, "back": -0.05}   # pre push normalized impulse
CONTROLLERS = ["stiff", "omni_jointrange"]
N_SWEEP = 24


def foot_spheres(m):
    return [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_contype[i]]


def com_state(env):
    m, d = env.m, env.d
    mujoco.mj_subtreeVel(m, d)
    return d.subtree_com[0].copy(), d.subtree_linvel[0].copy()


def foot_body_pitch(env, feet):
    """Largest absolute pitch of the foot bodies (heel or toe lift)."""
    out = 0.0
    for b in {env.m.geom_bodyid[i] for i in feet}:
        R = env.d.xmat[b].reshape(3, 3)
        out = max(out, abs(math.atan2(-R[2, 0], R[0, 0])))
    return out


def trial(env, policy, I_main, I_pre, feet, foot_mid, record=True):
    pushes = [(MAIN_STEP, float(I_main), 0.0)]
    if I_pre: pushes.append((PRE_STEP, abs(I_pre), 0.0 if I_pre > 0 else math.pi))
    obs, _ = env.reset(seed=0, options={"pushes": pushes})
    rec = dict(fell=False)
    max_foot, max_pitch, max_cx = 0.0, 0.0, -1e9
    done = False
    while not done:
        if env.k == MAIN_STEP:
            c, v = com_state(env); rec.update(x0=c[0] - foot_mid, v0=v[0], z0=c[2])
        if env.k == MAIN_STEP + oe.PUSH_STEPS:
            c, v = com_state(env); rec.update(x1=c[0] - foot_mid, v1=v[0], z1=c[2])
        obs, r, term, trunc, info = env.step(policy(obs))
        done = term or trunc
        if env.k > MAIN_STEP:
            max_foot = max(max_foot, foot_body_pitch(env, feet))
            Rm = oe.quat_to_mat(env.d.qpos[3:7])
            max_pitch = max(max_pitch, abs(math.atan2(-Rm[2, 0], math.hypot(Rm[0, 0], Rm[1, 0]))))
            max_cx = max(max_cx, com_state(env)[0][0] - foot_mid)
    rec.update(fell=bool(info["fell"]), max_foot_pitch=max_foot, max_pelvis_pitch=max_pitch, max_com_x=max_cx)
    return rec


def main():
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    env = oe.BalanceEnv("twin", history=16)
    env.reset(seed=0, options={"pushes": []})
    m, d = env.m, env.d
    feet = foot_spheres(m)
    xs = np.array([d.geom_xpos[i][0] for i in feet])
    heel, toe = xs.min(), xs.max()
    foot_mid = 0.5 * (heel + toe)
    c, _ = com_state(env)
    # force per unit normalized impulse for the twin, and the push duration (BalanceEnv._start)
    dur = oe.PUSH_STEPS * 0.02 * env.tau
    F_per_I = env.mass * env.v_unit / (0.1 * env.tau)
    lip = dict(mass=env.mass, com_height=float(c[2]), heel=float(heel - foot_mid), toe=float(toe - foot_mid),
               com_x_rest=float(c[0] - foot_mid), push_duration=dur, F_per_I=F_per_I, g=oe.G,
               G1_N_per_I=oe.N_PER_I, L=env.L, tau=env.tau, dt_control=0.02 * env.tau,
               ankle_torque_limit=float(env.P["tmax"][[i for i in range(m.nu) if "ankle_pitch" in m.actuator(i).name]].sum()))
    print(json.dumps(lip, indent=1))
    json.dump(lip, open(os.path.join(OUT, "twin_lip.json"), "w"), indent=1)

    rows, thr_rows = [], []
    for cname, path, hist in ev.parse_controllers(CONTROLLERS):
        policy = ev.load_policy(path)
        env = oe.BalanceEnv("twin", history=hist)
        for sname, I_pre in STARTS.items():
            falls = lambda I: trial(env, policy, I, I_pre, feet, foot_mid)["fell"]
            if falls(0.0):
                thr = 0.0
            else:
                lo, hi = 0.0, 1.0
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (lo, mid) if falls(mid) else (mid, hi)
                thr = 0.5 * (lo + hi)
            base = trial(env, policy, 0.0, I_pre, feet, foot_mid)
            thr_rows.append(dict(controller=cname, start=sname, I_pre=I_pre, thr_I=thr, thr_N=thr * F_per_I,
                                 thr_G1_N=thr * oe.N_PER_I, x0=base["x0"], v0=base["v0"]))
            print(f"{cname:16s} {sname:10s} x0 {base['x0']*1000:+6.1f} mm v0 {base['v0']:+.3f} m/s  "
                  f"threshold {thr:.3f} = {thr*F_per_I:.2f} N twin = {thr*oe.N_PER_I:.0f} N G1")
            for I in np.linspace(0, 1.6 * max(thr, 0.05), N_SWEEP):
                r = trial(env, policy, I, I_pre, feet, foot_mid)
                r.update(controller=cname, start=sname, I_pre=I_pre, I=I, F=I * F_per_I, F_G1=I * oe.N_PER_I)
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "twin_pushes.csv"), index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(OUT, "twin_thresholds.csv"), index=False)


if __name__ == "__main__":
    main()
