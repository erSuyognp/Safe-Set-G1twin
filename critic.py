"""Step 3c: learned adversarial safety critic for the twin's LIP (ISAACS / Q-CBF style).

One critic Q(s, u, d), a controller actor u = pi_c(s) that maximizes it and a disturbance actor d = pi_d(s)
that minimizes it, trained off policy from a replay buffer with the safety Bellman backup
    Q(s, u, d) = (1 - gamma) g(s') + gamma min(g(s'), Q'(s', pi_c(s'), pi_d(s')))
(target = g(s') once s' has failed). The disturbance actor learns faster than the controller actor, gamma is
annealed toward 1, and the critic is a twin (clipped double Q, min of the two: conservative for safety).

Everything runs in dimensionless LIP units so one script serves any foot: h = half foot length,
    x~ = x / h (foot is [-1, 1]),  v~ = v / (w h),  t~ = w t,  p~ = p / h,  a~ = a / (w^2 h)
    x~' = v~,  v~' = x~ - p~ + a~,  g~ = 1 - |x~|
Impulse mode (step 5): the push lasts at most push_T~ = w T. The state gains the remaining push time r~
(a deterministic countdown, r~' = max(r~ - DT, 0)); the disturbance acts only while r~ > 0. Networks see
(x~, v~, r~ / push_T~). At run time the filter asks "could a full push starting now be survived": r~ = push_T~.

Run:  python critic.py <a_max~> <out.pt> [iters] [push_T~]     (a_max~ = 0 for no disturbance)
"""
import math, sys, time
import numpy as np
import torch
import torch.nn as nn

DT = 0.05            # in units of 1/w
SUB = 5              # Euler substeps per control step
HORIZON = 120        # control steps per episode (6 time constants)
V_BOX = 2.5          # resets sample |v~| up to this


def mlp(i, o, h=256, out_act=None):
    layers = [nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o)]
    if out_act: layers.append(out_act)
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, sdim=2):
        super().__init__()
        self.q1, self.q2 = mlp(sdim + 2, 1), mlp(sdim + 2, 1)

    def forward(self, s, u, d):
        z = torch.cat([s, u, d], -1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)


def step(s, u, d, a_max, push_T=None):
    """s = (x~, v~) or, in impulse mode, (x~, v~, r~ / push_T~)."""
    x, v = s[:, 0].clone(), s[:, 1].clone()
    p, a = u[:, 0], d[:, 0] * a_max
    if push_T is not None:
        r = s[:, 2] * push_T
        a = a * (r.clamp(max=DT) / DT)                 # the push ends partway through this step
    h = DT / SUB
    for _ in range(SUB):
        x, v = x + h * v, v + h * (x - p + a)
    if push_T is None:
        return torch.stack([x, v], -1)
    return torch.stack([x, v, (r - DT).clamp(min=0) / push_T], -1)


def margin(s):
    return 1.0 - s[:, 0].abs()


def sample_states(n, dev, push_T=None):
    s = [torch.rand(n, device=dev) * 2 - 1, (torch.rand(n, device=dev) * 2 - 1) * V_BOX]
    if push_T is not None:                             # 60% somewhere in a push (a third of those at its start), 40% none
        u = torch.rand(n, device=dev)
        s.append(torch.where(u < 0.2, torch.ones_like(u), torch.where(u < 0.6, torch.rand(n, device=dev), torch.zeros_like(u))))
    return torch.stack(s, -1)


def train(a_max, out, iters=60000, n_env=1024, batch=2048, seed=0, dev=None, push_T=None):
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    sd = 2 if push_T is None else 3
    Q, Qt = Critic(sd).to(dev), Critic(sd).to(dev)
    Qt.load_state_dict(Q.state_dict())
    pc, pd = mlp(sd, 1, out_act=nn.Tanh()).to(dev), mlp(sd, 1, out_act=nn.Tanh()).to(dev)
    pc_t, pd_t = mlp(sd, 1, out_act=nn.Tanh()).to(dev), mlp(sd, 1, out_act=nn.Tanh()).to(dev)
    pc_t.load_state_dict(pc.state_dict()); pd_t.load_state_dict(pd.state_dict())
    oq = torch.optim.Adam(Q.parameters(), 1e-3)
    oc = torch.optim.Adam(pc.parameters(), 1e-4)       # controller: slower
    od = torch.optim.Adam(pd.parameters(), 5e-4)       # disturbance: faster, as in the paper
    cap = 1_000_000
    B = dict(s=torch.zeros(cap, sd, device=dev), u=torch.zeros(cap, 1, device=dev), d=torch.zeros(cap, 1, device=dev),
             s2=torch.zeros(cap, sd, device=dev))
    n_buf, ptr = 0, 0
    s = sample_states(n_env, dev, push_T); t = torch.zeros(n_env, device=dev)
    t0 = time.time()
    for it in range(iters):
        frac = it / iters
        gamma = 1 - 0.1 * (0.0005 / 0.1) ** min(1.0, frac / 0.8)     # 0.9 -> 0.9995, log schedule
        eps = max(0.05, 0.5 * (1 - frac / 0.5))                          # exploration noise and random action rate
        with torch.no_grad():
            u = (pc(s) + eps * torch.randn(n_env, 1, device=dev)).clamp(-1, 1)
            d = (pd(s) + eps * torch.randn(n_env, 1, device=dev)).clamp(-1, 1)
            rnd = torch.rand(n_env, device=dev) < eps * 0.5
            u[rnd] = torch.rand(int(rnd.sum()), 1, device=dev) * 2 - 1
            d[rnd] = torch.rand(int(rnd.sum()), 1, device=dev) * 2 - 1
            s2 = step(s, u, d, a_max, push_T)
            idx = (torch.arange(n_env, device=dev) + ptr) % cap
            B["s"][idx], B["u"][idx], B["d"][idx], B["s2"][idx] = s, u, d, s2
            ptr = (ptr + n_env) % cap; n_buf = min(cap, n_buf + n_env)
            t += 1
            reset = (margin(s2) < 0) | (t >= HORIZON)
            s = torch.where(reset[:, None], sample_states(n_env, dev, push_T), s2); t[reset] = 0
        # critic update
        j = torch.randint(0, n_buf, (batch,), device=dev)
        bs, bu, bd, bs2 = B["s"][j], B["u"][j], B["d"][j], B["s2"][j]
        with torch.no_grad():
            g2 = margin(bs2)
            q1, q2 = Qt(bs2, pc_t(bs2), pd_t(bs2))
            q2n = torch.min(q1, q2)
            y = (1 - gamma) * g2 + gamma * torch.min(g2, q2n)
            y = torch.where(g2 < 0, g2, y)
        c1, c2 = Q(bs, bu, bd)
        lq = ((c1 - y) ** 2 + (c2 - y) ** 2).mean()
        oq.zero_grad(); lq.backward(); oq.step()
        # actor updates: controller maximizes, disturbance minimizes
        if it % 2 == 0:
            q1, q2 = Q(bs, pc(bs), pd(bs).detach())
            lc = -torch.min(q1, q2).mean()
            oc.zero_grad(); lc.backward(); oc.step()
            if a_max > 0:
                q1, q2 = Q(bs, pc(bs).detach(), pd(bs))
                ld = torch.min(q1, q2).mean()
                od.zero_grad(); ld.backward(); od.step()
        with torch.no_grad():
            for net, tgt in ((Q, Qt), (pc, pc_t), (pd, pd_t)):
                for p_, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(0.995).add_(0.005 * p_)
        if it % 5000 == 0 or it == iters - 1:
            print(f"it {it:6d} gamma {gamma:.4f} eps {eps:.2f} critic loss {lq.item():.2e} "
                  f"buffer {n_buf} {time.time()-t0:.0f}s", flush=True)
    torch.save(dict(a_max=a_max, push_T=push_T, Q=Q.state_dict(), pc=pc.state_dict(), pd=pd.state_dict()), out)


def load(path, dev="cpu"):
    ck = torch.load(path, map_location=dev)
    sd = 2 if ck.get("push_T") is None else 3
    Q = Critic(sd); Q.load_state_dict(ck["Q"])
    pc, pd = mlp(sd, 1, out_act=nn.Tanh()), mlp(sd, 1, out_act=nn.Tanh())
    pc.load_state_dict(ck["pc"]); pd.load_state_dict(ck["pd"])
    return ck["a_max"], Q.eval(), pc.eval(), pd.eval()


@torch.no_grad()
def value(path, xs, vs, r=1.0):
    """Critic value at the actors' actions, on a grid of dimensionless states. Returns V[ix, iv].
    r: remaining push time as a fraction of push_T (impulse critics only)."""
    a_max, Q, pc, pd = load(path)
    X, Vv = np.meshgrid(xs, vs, indexing="ij")
    cols = [X.ravel(), Vv.ravel()] + ([np.full(X.size, r)] if Q.q1[0].in_features == 5 else [])
    s = torch.tensor(np.stack(cols, -1), dtype=torch.float32)
    q1, q2 = Q(s, pc(s), pd(s))
    return torch.min(q1, q2).numpy().reshape(X.shape)


if __name__ == "__main__":
    train(float(sys.argv[1]), sys.argv[2], iters=int(sys.argv[3]) if len(sys.argv) > 3 else 80000,
          push_T=float(sys.argv[4]) if len(sys.argv) > 4 else None)
