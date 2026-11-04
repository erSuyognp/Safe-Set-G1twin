"""Steps 2 and 3a/3b: the Linear Inverted Pendulum fitted to the twin, its analytic safe set, and an
exact HJ (Hamilton-Jacobi-Isaacs) safe set computed on a grid.

State s = (x, v): CoM position over the foot (0 = foot middle) and velocity, twin SI units.
Control p in [heel, toe]: center of pressure (ankle strategy).
Disturbance a in [-a_max, a_max]: push acceleration F / m, applied for as long as it likes (worst case).
    x' = v,   v' = w^2 (x - p) + a,   w = sqrt(g / z_c)
Failure: x outside [heel, toe]. Safety margin g(s) = min(x - heel, toe - x).
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class LIP:
    def __init__(self, path=os.path.join(HERE, "data", "twin_lip.json")):
        P = json.load(open(path))
        self.P = P
        self.m, self.zc, self.g0 = P["mass"], P["com_height"], P["g"]
        self.heel, self.toe = P["heel"], P["toe"]
        self.w = math.sqrt(self.g0 / self.zc)
        self.T_push = P["push_duration"]

    def f(self, x, v, p, a):
        return v, self.w ** 2 * (x - p) + a

    def flow(self, x, v, p, a, t):
        """Exact solution after time t with p and a held constant: e = x - (p - a / w^2) obeys e'' = w^2 e."""
        w = self.w; c = p - a / w ** 2
        e = x - c; ch, sh = np.cosh(w * t), np.sinh(w * t)
        return c + e * ch + v / w * sh, e * w * sh + v * ch

    def margin(self, x, v=None):
        return np.minimum(x - self.heel, self.toe - x)

    # --- 3a: analytic -------------------------------------------------------------------
    def analytic_safe(self, x, v, a_max=0.0):
        """Capture point xi = x + v / w must stay in the foot, shrunk by a_max / w^2 on each side
        (the CoP offset needed to cancel the worst persistent push), and x itself must be in the foot."""
        xi = x + v / self.w
        d = a_max / self.w ** 2
        return (x >= self.heel) & (x <= self.toe) & (xi >= self.heel + d) & (xi <= self.toe - d)

    def analytic_v_bounds(self, x, a_max=0.0):
        d = a_max / self.w ** 2
        return (self.heel + d - x) * self.w, (self.toe - d - x) * self.w

    def predicted_push_limit(self, x0, v0, T=None, safe=None):
        """Largest constant forward push force (N) for T seconds after which the twin's LIP can still recover
        with ankle torque alone, starting from (x0, v0). During the push the CoP is held at the best constant
        value in [heel, toe]; recovery needs x(t) in the foot throughout and the post push state in the safe
        set `safe(x, v)` (default: the analytic capture region). Bisection on the exact ODE solution."""
        T = self.T_push if T is None else T
        safe = safe or (lambda x, v: bool(self.analytic_safe(x, v)))
        ts = np.linspace(0, T, 100)
        ps = np.linspace(self.heel, self.toe, 41)

        def ok(F):
            for p in ps:
                x, v = self.flow(x0, v0, p, F / self.m, ts)
                if x.min() >= self.heel and x.max() <= self.toe and safe(x[-1], v[-1]): return True
            return False
        if not ok(0.0): return 0.0
        lo, hi = 0.0, 100.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo


# --- 3b: exact HJ on a grid ------------------------------------------------------------------
def _successors(lip, xs, vs, ps, As, dt):
    """Bilinear gather indices and weights of every successor s' = flow(s, p, a, dt) on the grid, for each
    (p, a). They do not depend on V. A successor outside the grid points at a padded entry holding -1
    (outside the velocity range is unsafe; outside x is unsafe anyway)."""
    nx, nv = len(xs), len(vs)
    X, Vv = np.meshgrid(xs, vs, indexing="ij")
    succ = []
    for p in ps:
        row = []
        for a in As:
            xn, vn = lip.flow(X, Vv, p, a, dt)
            fx = (xn - xs[0]) / (xs[1] - xs[0]); fv = (vn - vs[0]) / (vs[1] - vs[0])
            out = ((fx < 0) | (fx > nx - 1) | (fv < 0) | (fv > nv - 1)).ravel()
            i0 = np.clip(np.floor(fx), 0, nx - 2).astype(int); j0 = np.clip(np.floor(fv), 0, nv - 2).astype(int)
            tx = np.clip(fx - i0, 0, 1).ravel(); tv = np.clip(fv - j0, 0, 1).ravel()
            i0, j0 = i0.ravel(), j0.ravel()
            idx = np.stack([i0 * nv + j0, (i0 + 1) * nv + j0, i0 * nv + j0 + 1, (i0 + 1) * nv + j0 + 1])
            idx[:, out] = nx * nv
            w = np.stack([(1 - tx) * (1 - tv), tx * (1 - tv), (1 - tx) * tv, tx * tv])
            row.append((idx, w))
        succ.append(row)
    return succ


def _backup(V, G, succ):
    """One HJI safety backup min(g, max_p min_a V(s')) on the flattened grid."""
    Vpad = np.append(V, -1.0)
    best = np.full(V.shape, -np.inf)
    for row in succ:
        worst = np.full(V.shape, np.inf)
        for idx, w in row:
            worst = np.minimum(worst, (Vpad[idx] * w).sum(0))
        best = np.maximum(best, worst)
    return np.minimum(G, best)


def hj_impulse_set(lip, A, T, V0, xs, vs, K=10, n_p=5):
    """Safe set against one push of up to A (m/s^2, either direction, any shape) lasting T seconds, after
    which there is no push. V0 is the no push HJ value on the grid (xs, vs). The remaining push time is a
    deterministic countdown, so this is a finite horizon recursion of K backups with dt = T / K:
        V_{k+1} = min(g, max_p min_a V_k(flow(s, p, a, T/K))),   a in {-A, 0, A}
    Returns the list [V_0, ..., V_K]: V_k is the value with k/K of the push still to come."""
    ps = np.linspace(lip.heel, lip.toe, n_p)
    succ = _successors(lip, xs, vs, ps, np.array([-A, 0.0, A]), T / K)
    X = np.meshgrid(xs, vs, indexing="ij")[0]
    G = lip.margin(X).ravel()
    Vs = [V0.ravel()]
    for _ in range(K):
        Vs.append(_backup(Vs[-1], G, succ))
    return [v.reshape(V0.shape) for v in Vs]


def hj_safe_set(lip, a_max, nx=401, nv=401, v_lim=None, dt=0.03, n_p=5, n_a=3, tol=1e-9, max_iter=3000):
    """Robust safe set by value iteration of the discrete-time HJI safety Bellman equation
        V(s) = min( g(s), max_p min_a V(s + dt f(s, p, a)) )
    on a regular grid with bilinear interpolation (a semi-Lagrangian scheme for the HJI variational
    inequality). The successor is the exact flow of the linear ODE with p and a held for dt (a large dt keeps
    interpolation diffusion small). The dynamics are affine in p and a, so a few values including the extremes
    suffice.
    The zero superlevel set of the fixed point is the robust viability kernel.
    Returns (xs, vs, V, iterations) with V indexed [ix, iv]."""
    if v_lim is None: v_lim = 1.5 * lip.w * (lip.toe - lip.heel)
    pad = 0.05 * (lip.toe - lip.heel)
    xs = np.linspace(lip.heel - pad, lip.toe + pad, nx)
    vs = np.linspace(-v_lim, v_lim, nv)
    X = np.meshgrid(xs, vs, indexing="ij")[0]
    G = lip.margin(X).ravel()
    ps = np.linspace(lip.heel, lip.toe, n_p)
    As = np.linspace(-a_max, a_max, n_a) if a_max > 0 else np.zeros(1)
    succ = _successors(lip, xs, vs, ps, As, dt)
    V = G.copy()
    for it in range(max_iter):
        Vn = _backup(V, G, succ)
        delta = np.abs(Vn - V).max()
        V = Vn
        if delta < tol: break
    return xs, vs, V.reshape(nx, nv), it


if __name__ == "__main__":
    lip = LIP()
    print(f"w = {lip.w:.3f} rad/s, foot [{lip.heel*1000:.1f}, {lip.toe*1000:.1f}] mm, z_c = {lip.zc*1000:.1f} mm")
    for a_tilde in (0.0, 0.3):                      # push acceleration in units of w^2 h (see critic.py)
        a_max = a_tilde * lip.w ** 2 * lip.toe
        xs, vs, V, it = hj_safe_set(lip, a_max)
        np.savez(os.path.join(HERE, "data", f"hj_a{a_tilde:.1f}.npz"), xs=xs, vs=vs, V=V, a_max=a_max, iters=it)
        X, Vv = np.meshgrid(xs, vs, indexing="ij")
        hj, an = V >= 0, lip.analytic_safe(X, Vv, a_max)
        inner = hj & ~an
        print(f"a_max {a_max:.3f}: {it} iterations, HJ cells {hj.sum()}, analytic cells {an.sum()}, "
              f"disagree {(hj != an).sum()} ({(hj != an).sum()/an.sum()*100:.2f}% of analytic set; HJ-only {inner.sum()})")
