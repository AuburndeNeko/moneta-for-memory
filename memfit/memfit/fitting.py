"""
Exact-physics maximum-likelihood inversion: noisy A(t) or E(t) at one observer ->
source velocities, transition, memory, and a detection statistic.

Model: the Liénard–Wiechert solution itself (Maxwell's equations hold exactly), with
    H(t) = sigmoid((t - t0)/tau) + sum_k a_k sin(k pi u)   on a window around t0
(the sum vanishes at the window edges, so memory is set by v_in and v_out alone).
K = 0 modes is the pure sigmoid, which sits on the Cramér–Rao bound for smooth steps.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.special import expit, stdtr

from .physics import C, MU0_4PI, lw_E, memory_prediction
from .hfunc import transition_metrics


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, t, Y, observable, x_obs, Q=1.0, sigma=None):
        self.t = np.asarray(t, float)
        self.Y = np.asarray(Y, float)
        self.observable = observable.upper()
        if self.observable not in ("A", "E"):
            raise ValueError("observable must be 'A' or 'E'")
        self.x_obs = np.asarray(x_obs, float)
        self.Q = float(Q)
        if sigma is None:
            sigma = estimate_noise(self.Y)
        self.sigma = np.broadcast_to(np.asarray(sigma, float), (3,)).copy()
        self.sigma[self.sigma <= 0] = np.inf    # ignore noiseless/constant components
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("Times must be strictly increasing.")


def estimate_noise(Y):
    """Robust white-noise estimate per component from first differences."""
    d = np.diff(Y, axis=0)
    mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0)
    return mad / 0.6744897501960817 / np.sqrt(2.0)


def to_w(v):
    b = np.asarray(v, float) / C
    nb = np.linalg.norm(b)
    if nb > 0.95:
        b = b * 0.95 / nb
    return b / np.sqrt(1.0 - b @ b)


def to_v(w):
    return C * w / np.sqrt(1.0 + w @ w)


# ----------------------------------------------------------------------------- step shape
# The base step is a Student-t CDF with fitted tail weight nu (theta[8] = log(nu - 0.5)):
# nu ~ 9 behaves like the logistic sigmoid, nu -> infinity is the Gaussian (erf) step, small nu
# has heavy tails. Fitting nu matters for E data: E follows the acceleration, and a wrong tail
# weight shifts part of the velocity change into (or out of) the low-level tails, biasing v_in,
# v_out and hence the memory by several percent.
Q_LOGISTIC = float(np.log(9.0 - 0.5))
Q_HEAVY = float(np.log(2.5 - 0.5))
Q_GAUSS = float(np.log(2000.0))
NULL_TAIL = np.array([0.0, 0.0, Q_LOGISTIC])


def tail_nu(q):
    return 0.5 + np.exp(np.clip(q, -3.0, 9.0))


def step_base(z, q):
    return stdtr(tail_nu(q), z)


# ----------------------------------------------------------------------------- model
class Model:
    def __init__(self, data, n_modes=0, window=1.0, centre=0.0):
        self.d = data
        self.K = int(n_modes)
        self.W = float(window)
        self.wc = float(centre)          # window centre (source time, ns), fixed during a fit
        t = data.t
        r = np.linalg.norm(data.x_obs)
        span = t[-1] - t[0]
        dt = min(np.median(np.diff(t)) / 2.0, 0.01)
        lo, hi = t[0] - r / C - span - 2.0, t[-1] - r / C + span + 2.0
        n = int(min(60_000, (hi - lo) / dt + 1))
        self.tg = np.linspace(lo, hi, n)
        self.i0 = np.argmin(np.abs(self.tg))
        self.N = data.Y.size

    # theta = [w_in(3), w_out(3), t0, log tau, log(nu - 1/2), a_1..a_K]
    def g(self, th):
        t0, tau = th[6], np.exp(np.clip(th[7], -12, 6))
        g = step_base((self.tg - t0) / tau, th[8])
        if self.K:
            i0, i1 = np.searchsorted(self.tg, [self.wc - self.W, self.wc + self.W])
            u = (self.tg[i0:i1] - self.wc + self.W) / (2 * self.W)
            k = np.arange(1, self.K + 1)
            g[i0:i1] += np.sin(np.pi * np.outer(u, k)) @ th[9:9 + self.K]
        return g

    def velocity(self, th, null=False):
        v_in = to_v(th[0:3])
        if null:
            return np.broadcast_to(v_in, (len(self.tg), 3))
        v_out = to_v(th[3:6])
        return v_in[None] + self.g(th)[:, None] * (v_out - v_in)[None]

    def predict(self, th, null=False, t=None):
        t = self.d.t if t is None else t
        v = self.velocity(th, null)
        X = cumulative_trapezoid(v, self.tg, axis=0, initial=0.0)
        X = X - X[self.i0]
        R = self.d.x_obs[None] - X
        Rn = np.linalg.norm(R, axis=1)
        if self.d.observable == "A":
            kappa = 1.0 - (R * v).sum(1) / (Rn * C)
            Y = MU0_4PI * self.d.Q * (v * 1e9) / (kappa * Rn)[:, None]
        else:
            a = np.gradient(v, self.tg, axis=0)
            Y = lw_E(self.d.Q, np.broadcast_to(self.d.x_obs, X.shape), X, v, a)[0]
        t_obs = self.tg + Rn / C
        return np.stack([np.interp(t, t_obs, Y[:, k]) for k in range(3)], 1), v

    def residuals(self, th, null=False):
        Yp, v = self.predict(th, null)
        res = ((Yp - self.d.Y) / self.d.sigma[None]).ravel()
        over = max(0.0, np.linalg.norm(v, axis=1).max() / C - 0.999)
        return np.append(res, 1e4 * over)


# ----------------------------------------------------------------------------- fitting
def _initial(data):
    """Velocities from the early/late levels, t0 from a matched-filter scan."""
    t, Y, Q = data.t, data.Y, data.Q
    r = np.linalg.norm(data.x_obs)
    if data.observable == "A":
        Ap = Y
    else:   # integrated E ~ -(A - A_early) for the transverse part at large r
        Ap = -cumulative_trapezoid(Y, t * 1e-9, axis=0, initial=0.0)
    k = max(len(t) // 10, 3)
    A_e, A_l = Ap[:k].mean(0), Ap[-k:].mean(0)
    scale = r / (MU0_4PI * Q) * 1e-9            # A -> v (m/ns), alpha ~ 1
    if data.observable == "A":
        v_in0, v_out0 = A_e * scale, A_l * scale
    else:
        v_in0, v_out0 = np.zeros(3), (A_l - A_e) * scale
    span = t[-1] - t[0]
    tau0 = max(span / 50.0, 3 * np.median(np.diff(t)))
    best, t0o = np.inf, t[len(t) // 2]
    for tc in t[::max(len(t) // 400, 1)]:
        s = expit((t - tc) / tau0)
        X = np.column_stack([np.ones_like(t), s])
        rss = float(((Ap - X @ np.linalg.lstsq(X, Ap, rcond=None)[0]) ** 2).sum())
        if rss < best:
            best, t0o = rss, tc
    return v_in0, v_out0, t0o - r / C, tau0


def _ls(fun, x0, bounds=None, max_nfev=None, **kw):
    if bounds is not None:
        lb, ub = bounds
        eps = 1e-6 * (ub - lb)
        eps = np.where(np.isfinite(eps), eps, 0.0)
        x0 = np.minimum(np.maximum(x0, lb + eps), ub - eps)
    else:
        bounds = (-np.inf, np.inf)
    return least_squares(fun, x0, bounds=bounds, x_scale="jac", method="trf",
                         max_nfev=max_nfev or 100 * (len(x0) + 1), **kw)


def step_bounds(data, K=0):
    """Physical ranges for the step parameters (velocities unbounded: |v| < c by construction).
    Without them, noise-only data let the step drift to absurd widths/tails, which wastes time
    and has no physical meaning."""
    r = np.linalg.norm(data.x_obs)
    u0, u1 = data.t[0] - r / C, data.t[-1] - r / C
    span = u1 - u0
    dt = np.median(np.diff(data.t))
    lb = np.concatenate([np.full(6, -np.inf), [u0 - 0.1 * span, np.log(dt / 10), -3.0],
                         np.full(K, -2.0)])
    ub = np.concatenate([np.full(6, np.inf), [u1 + 0.1 * span, np.log(span), 9.0],
                         np.full(K, 2.0)])
    return lb, ub


class _Fixed:
    """least_squares result with the fixed tail parameter re-inserted (theta always has 9+K)."""
    def __init__(self, res, q):
        self.x = np.insert(res.x, 8, q)
        self.jac = np.insert(res.jac, 8, 0.0, axis=1)
        self.fun, self.cost, self.nfev, self.status = res.fun, res.cost, res.nfev, res.status


def fit_sigmoid(data, n_starts=3, free_tail=True, cap=150, polish=400):
    """Single-step fit. free_tail=True fits the tail weight nu (used for estimation);
    free_tail=False fixes the logistic tail (used for the detection statistic and its
    calibration: fast, and detection power hardly depends on the tails)."""
    m = Model(data, 0)
    lb, ub = step_bounds(data)
    if free_tail:
        fun, bnd, pack = m.residuals, (lb, ub), (lambda r: r)
    else:
        keep = np.arange(9) != 8
        fun = lambda th: m.residuals(np.insert(th, 8, Q_LOGISTIC))
        bnd = (lb[keep], ub[keep])
        pack = lambda r: _Fixed(r, Q_LOGISTIC)
    v_in0, v_out0, t00, tau0 = _initial(data)
    span = data.t[-1] - data.t[0]
    offsets = [0.0, -span / 16, span / 16, -span / 6, span / 6][:max(n_starts, 1)]
    starts = [np.concatenate([to_w(v_in0), to_w(v_out0), [t00 + off, np.log(tau0)]]
                             + ([[Q_LOGISTIC]] if free_tail else [])) for off in offsets]
    best = None
    for x0 in starts:
        res = _ls(fun, x0, bnd, max_nfev=cap)
        if best is None or res.cost < best.cost:
            best = res
    if free_tail:
        for q in (Q_HEAVY, Q_GAUSS):          # tail-weight starts: heavy and Gaussian
            x0 = best.x.copy()
            x0[8] = q
            res = _ls(fun, x0, bnd, max_nfev=cap)
            if res.cost < best.cost:
                best = res
    if best.status == 0:                      # budget ran out: polish the winner
        res = _ls(fun, best.x, bnd, max_nfev=polish)
        if res.cost <= best.cost:
            best = res
    return m, pack(best)


def fit_flexible(data, base_theta, K, window, centre):
    m = Model(data, K, window, centre)
    x0 = np.concatenate([base_theta[:9], np.zeros(K)])
    res = _ls(m.residuals, x0, step_bounds(data, K), max_nfev=80 * (len(x0) + 1),
              ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return m, res


def fit_null(data):
    m = Model(data, 0)
    v_in0, _, _, _ = _initial(data)
    if data.observable == "E":
        v_in0 = np.zeros(3)
    x0w = to_w(v_in0)
    fun = lambda w: m.residuals(np.concatenate([w, w, NULL_TAIL]), null=True)
    res = _ls(fun, x0w)
    return m, res


def chi2_of(res, N):
    return float((res.fun[:N] ** 2).sum())


def summarise(data, m, res, truth_H=None):
    """Physical results with 1-sigma uncertainties from the Fisher matrix."""
    th = res.x
    N = m.N
    J = res.jac[:N]
    cov = np.linalg.pinv(J.T @ J)

    def mem(th_):
        dA, Delta = memory_prediction(data.Q, to_v(th_[0:3]), to_v(th_[3:6]), data.x_obs)
        return np.concatenate([dA, Delta, to_v(th_[0:3]) / C, to_v(th_[3:6]) / C])

    f0 = mem(th)
    G = np.empty((f0.size, th.size))
    for i in range(th.size):
        h = 1e-6 * max(1.0, abs(th[i]))
        e = np.zeros_like(th); e[i] = h
        G[:, i] = (mem(th + e) - mem(th - e)) / (2 * h)
    err = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", G, cov, G), 0, None))
    g = m.g(th)
    t_half, rise = transition_metrics(m.tg, g)
    chi2 = chi2_of(res, N)
    dof = N - th.size
    out = dict(
        observable=data.observable, n_samples=len(data.t), n_modes=m.K,
        dA=f0[0:3], dA_err=err[0:3], Delta=f0[3:6], Delta_err=err[3:6],
        beta_in=f0[6:9], beta_in_err=err[6:9], beta_out=f0[9:12], beta_out_err=err[9:12],
        t0=float(th[6]), tau=float(np.exp(th[7])), nu=float(tail_nu(th[8])), t_half=t_half, rise=rise,
        chi2=chi2, dof=dof, red_chi2=chi2 / dof,
        shape_z=(chi2 - dof) / np.sqrt(2 * dof),
        tg=m.tg, g=g, theta=th,
    )
    return out


def bic(res, N, p):
    return chi2_of(res, N) + p * np.log(N)


def activity_window(data):
    """Centre and half-width (ns, source time) of the region where the signal changes, from
    the smoothed derivative of the A-like proxy (A itself, or -∫E dt). Only a starting point:
    the final window is chosen by BIC among wider candidates."""
    t, Y = data.t, data.Y
    r = np.linalg.norm(data.x_obs)
    Ap = Y if data.observable == "A" else -cumulative_trapezoid(Y, t * 1e-9, axis=0, initial=0.0)
    w = max(len(t) // 50, 3)
    ker = np.ones(w) / w
    act = np.zeros(len(t) - 1)
    for k in range(3):
        if not np.isfinite(data.sigma[k]):
            continue
        d = np.diff(np.convolve(Ap[:, k], ker, mode="same")) / np.diff(t)
        d = d - np.median(d)
        act += d ** 2
    act[: w] = act[-w:] = 0.0
    act = np.clip(act - 5.0 * np.median(act[w:-w]), 0.0, None)   # remove the noise floor
    cdf = np.cumsum(act)
    u = 0.5 * (t[1:] + t[:-1]) - r / C
    if cdf[-1] <= 0:
        return 0.5 * (u[0] + u[-1]), 1.0
    ua, ub = np.interp([0.02, 0.98], cdf / cdf[-1], u)
    return 0.5 * (ua + ub), 0.5 * (ub - ua) * 1.3


def max_window(data, centre):
    """Keep at least ~40% of the data on each side outside the window, so the initial and
    final velocities (and hence the memory) stay pinned by plateau data."""
    r = np.linalg.norm(data.x_obs)
    u0, u1 = data.t[0] - r / C, data.t[-1] - r / C
    return 0.6 * min(centre - u0, u1 - centre)


def fit(data, flexible="check", window=None, verbose=print):
    """Full analysis.
    flexible: 0 (pure sigmoid), K > 0 sine modes, 'auto' (choose K and window by BIC), or
              'check' (single step; switch to 'auto' if chi2 is > 3 sigma high)."""
    m0, r0 = fit_sigmoid(data)
    choice = (m0, r0)
    table = [(0, bic(r0, m0.N, 9), chi2_of(r0, m0.N), None)]
    dof0 = m0.N - 9
    z0 = (chi2_of(r0, m0.N) - dof0) / np.sqrt(2 * dof0)
    switched = False
    if flexible == "check":
        flexible = "auto" if z0 > 3 else 0
        switched = flexible == "auto"
    W = None
    if flexible not in (0, "0", None):
        wc, Wa = activity_window(data)
        Wmax = max_window(data, wc)
        Wa = max(Wa, 4 * float(np.exp(r0.x[7])), 0.5)
        Ks = [4, 8, 12, 16, 24, 32] if flexible == "auto" else [int(flexible)]
        if window:
            Ws = [window]
        elif flexible == "auto":
            Ws = sorted({round(min(Wa * f, Wmax), 6) for f in (1.0, 1.6, 2.4)})
        else:
            Ws = [min(1.6 * Wa, Wmax)]
        v_in0, v_out0, _, tau0 = _initial(data)
        fresh = np.concatenate([to_w(v_in0), to_w(v_out0), [wc, np.log(tau0), r0.x[8]]])
        best_bic = table[0][1] if flexible == "auto" else np.inf
        for Wc in Ws:
            sub = []
            for K in Ks:
                best = None
                for base in (r0.x, fresh):   # two starts: sigmoid optimum and data-driven
                    mk, rk = fit_flexible(data, base, K, Wc, wc)
                    if best is None or rk.cost < best[1].cost:
                        best = (mk, rk)
                mk, rk = best
                b = bic(rk, mk.N, 9 + K)
                table.append((K, b, chi2_of(rk, mk.N), Wc))
                sub.append(b)
                if b < best_bic:
                    choice, best_bic, W = (mk, rk), b, Wc
                if flexible == "auto" and len(sub) > 2 and sub[-1] > sub[-2] > sub[-3]:
                    break                    # BIC rising twice: stop adding modes
    out = summarise(data, *choice)
    out["shape_z_sigmoid"] = z0
    out["auto_switched"] = switched
    out["_model"] = choice[0]
    out["window"] = choice[0].W if out["n_modes"] else None
    out["window_centre"] = choice[0].wc if out["n_modes"] else None
    out["model_table"] = table
    # Detection uses the fixed-tail (logistic) step, so it matches the calibration exactly.
    md, rd = fit_sigmoid(data, free_tail=False)
    mn, rn = fit_null(data)
    out["chi2_null"] = chi2_of(rn, mn.N)
    out["dchi2"] = out["chi2_null"] - chi2_of(rd, md.N)
    out["null_theta"] = rn.x
    out["_sigmoid"] = (m0, r0)
    out["_null"] = (mn, rn)
    return out


# ----------------------------------------------------------------------------- calibration
_PAYLOAD = None


def _init_worker(payload):
    global _PAYLOAD
    _PAYLOAD = payload


def _calib_job(seed):
    t, Y0, observable, x_obs, Q, sigma = _PAYLOAD
    rng = np.random.default_rng(seed)
    Y = Y0 + sigma[None] * rng.standard_normal(Y0.shape) * np.isfinite(sigma)[None]
    d = Data(t, Y, observable, x_obs, Q, sigma)
    m0, r0 = fit_sigmoid(d, free_tail=False)
    mn, rn = fit_null(d)
    return chi2_of(rn, mn.N) - chi2_of(r0, m0.N)


def calibrate(data, result, n=100, workers=None, seed=12345, progress=None):
    """Parametric bootstrap of the detection statistic under 'no memory':
    simulate the best-fit constant-velocity source + noise, refit both models."""
    mn, rn = result["_null"]
    th = np.concatenate([rn.x, rn.x, NULL_TAIL])
    Y0 = mn.predict(th, null=True)[0]
    payload = (data.t, Y0, data.observable, data.x_obs, data.Q, data.sigma)
    seeds = [seed + i for i in range(n)]
    if workers == 1:
        _init_worker(payload)
        vals = []
        for i, s in enumerate(seeds):
            vals.append(_calib_job(s))
            if progress:
                progress(i + 1, n)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        vals = []
        with ctx.Pool(workers, initializer=_init_worker, initargs=(payload,)) as pool:
            for v in pool.imap(_calib_job, seeds, chunksize=2):
                vals.append(v)
                if progress:
                    progress(len(vals), n)
    vals = np.array(vals)
    obs = result["dchi2"]
    p = (1 + int((vals >= obs).sum())) / (n + 1)
    return dict(null_dchi2=vals, p_value=p, threshold_1pct=float(np.percentile(vals, 99)),
                threshold_5pct=float(np.percentile(vals, 95)))
