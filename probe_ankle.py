"""How far a constant ankle pitch action moves the twin's center of pressure (sign and gain for the recovery
controller of step 5). Run with the omni_brain_starter venv."""
import numpy as np
import twin_push as tp
from twin_push import oe
import mujoco

ANKLE = [3, 8]          # ankle_pitch entries in omni_env.LEG_JOINTS (left, right)


def cop_x(env, feet):
    m, d = env.m, env.d
    f6 = np.zeros(6); fz, mx = 0.0, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in feet or c.geom2 in feet:
            mujoco.mj_contactForce(m, d, i, f6)
            fz += f6[0]; mx += f6[0] * c.pos[0]
    return mx / fz if fz > 0 else np.nan


if __name__ == "__main__":
    env = oe.BalanceEnv("twin", history=16)
    env.reset(seed=0, options={"pushes": []})
    feet = set(tp.foot_spheres(env.m))
    xs = np.array([env.d.geom_xpos[i][0] for i in feet]); mid = 0.5 * (xs.min() + xs.max())
    for a in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
        env.reset(seed=0, options={"pushes": []})
        act = np.zeros(10); act[ANKLE] = a
        cops = []
        for k in range(8):                   # 8 control steps = 0.09 s twin time, before the body tips far
            env.step(act); cops.append(cop_x(env, feet) - mid)
        c, v = tp.com_state(env)
        print(f"ankle action {a:+.2f}: CoP after 1/3/8 steps {cops[0]*1e3:+6.1f} {cops[2]*1e3:+6.1f} {cops[7]*1e3:+6.1f} mm, "
              f"CoM v {v[0]:+.3f} m/s")
