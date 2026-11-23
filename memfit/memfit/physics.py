"""
Liénard–Wiechert fields of a point charge whose velocity follows

    v(t) = v_in + H(t / T) (v_out - v_in)                     (paper Eq. 30)

Reference: de Aguiar Alves & Arderucio Costa, Am. J. Phys. 94, 973 (2026).

Units used throughout the package:
    time ns, length m, velocity m/ns, charge C,
    vector potential A in V s/m, scalar potential V in V, electric field E in V/m.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

C = 0.299792458          # speed of light, m/ns
MU0_4PI = 1e-7           # mu0 / 4 pi (SI)
K_E = 8.9875517923e9     # 1 / (4 pi eps0) (SI)


class Source:
    """Point charge Q with velocity profile v(t) = v_in + H(t/T)(v_out - v_in), X(0) = x0."""

    def __init__(self, H, T=1.0, v_in=(0, 0, 0), v_out=(0, 0.5 * C, 0), Q=1.0,
                 x0=(0, 0, 0), label="H"):
        self.H = H
        self.T = float(T)
        self.v_in = np.asarray(v_in, float)
        self.v_out = np.asarray(v_out, float)
        self.Q = float(Q)
        self.x0 = np.asarray(x0, float)
        self.label = label

    def velocity(self, t):
        h = np.asarray(self.H(np.asarray(t, float) / self.T), float)
        return self.v_in[None, :] + h[:, None] * (self.v_out - self.v_in)[None, :]

    def kinematics(self, t):
        """Position, velocity, acceleration on a sorted grid of times t (ns)."""
        t = np.asarray(t, float)
        v = self.velocity(t)
        X = cumulative_trapezoid(v, t, axis=0, initial=0.0)
        if t[0] <= 0.0 <= t[-1]:
            X0 = np.array([np.interp(0.0, t, X[:, k]) for k in range(3)])
        else:  # integrate from 0 to t[0] separately
            s = np.linspace(0.0, t[0], 4001)
            X0 = -trapezoid(self.velocity(s), s, axis=0)
        X = X - X0[None, :] + self.x0[None, :]
        a = np.gradient(v, t, axis=0)
        return X, v, a

    def check_speed(self, t):
        vmax = np.linalg.norm(self.velocity(t), axis=1).max()
        if vmax >= C:
            raise ValueError(f"The source exceeds the speed of light (max |v| = {vmax / C:.3f} c). "
                             "Check H, v_in and v_out.")
        return vmax / C


def lw_E(Q, P, X, v, a):
    """Liénard–Wiechert E at points P from retarded position/velocity/acceleration (row-wise)."""
    R = P - X
    Rn = np.linalg.norm(R, axis=1)
    n = R / Rn[:, None]
    beta = v / C
    bdot = a / C                                   # 1/ns
    kappa = 1.0 - (n * beta).sum(1)
    nb = n - beta
    k3 = kappa ** 3
    E_vel = nb * ((1.0 - (beta ** 2).sum(1)) / (k3 * Rn ** 2))[:, None]
    E_rad = np.cross(n, np.cross(nb, bdot)) / (C * k3 * Rn)[:, None]
    return K_E * Q * (E_vel + E_rad), K_E * Q * E_vel, K_E * Q * E_rad


def lw_at_observer(src, x_obs, t_ret):
    """All fields seen by a fixed observer, parametrised by retarded time (no root-finding).
    Returns dict with t_obs, A, V, E, E_vel, E_rad, X, v."""
    X, v, a = src.kinematics(t_ret)
    P = np.broadcast_to(np.asarray(x_obs, float), X.shape)
    R = P - X
    Rn = np.linalg.norm(R, axis=1)
    n = R / Rn[:, None]
    kappa = 1.0 - (n * v).sum(1) / C
    A = MU0_4PI * src.Q * (v * 1e9) / (kappa * Rn)[:, None]
    V = K_E * src.Q / (kappa * Rn)
    E, E_vel, E_rad = lw_E(src.Q, P, X, v, a)
    return dict(t_obs=t_ret + Rn / C, A=A, V=V, E=E, E_vel=E_vel, E_rad=E_rad, X=X, v=v,
                t_ret=t_ret)


def retarded_grid(T, u0, u1, r, extra=0.0, max_points=400_000):
    """Fine retarded-time grid covering observation window u in [u0, u1] (u = t - r/c)."""
    span = u1 - u0
    pad = span + 2.0 + extra
    dt = min(T / 200.0, span / 4000.0)
    n = int(min(max_points, (span + 2 * pad) / dt + 1))
    return np.linspace(u0 - pad, u1 + pad, n)


def observe(src, x_obs, u_window=(-8.0, 8.0), n=2000):
    """Clean A and E sampled at n uniformly spaced observation times.
    u_window is in units of ns of retarded time u = t - |x_obs|/c."""
    x_obs = np.asarray(x_obs, float)
    r = np.linalg.norm(x_obs)
    u0, u1 = u_window
    tr = retarded_grid(src.T, u0, u1, r)
    src.check_speed(tr)
    f = lw_at_observer(src, x_obs, tr)
    t = np.linspace(u0, u1, n) + r / C
    if t[0] < f["t_obs"][0] or t[-1] > f["t_obs"][-1]:
        raise RuntimeError("Retarded grid does not cover the observation window.")
    interp = lambda Y: np.stack([np.interp(t, f["t_obs"], Y[:, k]) for k in range(3)], 1)
    return dict(t=t, A=interp(f["A"]), E=interp(f["E"]), fine=f)


def memory_prediction(Q, v_in, v_out, x_obs):
    """Leading-order memory (large r at fixed u).
    dA:    jump in A, paper Eq. 33 (single charge).
    Delta: memory vector  int E dt = -[dA]_transverse, paper Eq. 38 (test-charge kick per q/m)."""
    x_obs = np.asarray(x_obs, float)
    r = np.linalg.norm(x_obs)
    rh = x_obs / r
    f = lambda v: MU0_4PI * Q * (np.asarray(v, float) * 1e9) / (r * (1.0 - rh @ v / C))
    dA = f(v_out) - f(v_in)
    Delta = -(dA - rh * (rh @ dA))
    return dA, Delta
