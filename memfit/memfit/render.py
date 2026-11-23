"""Plots and animations."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from .physics import C, lw_E, lw_at_observer, retarded_grid, memory_prediction
from .hfunc import transition_metrics

INK = "#1f2430"
ACC = "#d1495b"
BLUE = "#2e86ab"
GREY = "#9aa0a6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10})


def _main_component(Y):
    return int(np.argmax(np.abs(Y[-1] - Y[0]) + np.ptp(Y, axis=0) * 1e-6))


# ----------------------------------------------------------------------------- predict
def plot_prediction(src, x_obs, u_window, path):
    x_obs = np.asarray(x_obs, float)
    r = np.linalg.norm(x_obs)
    tr = retarded_grid(src.T, *u_window, r)
    f = lw_at_observer(src, x_obs, tr)
    u = f["t_obs"] - r / C
    sel = (u >= u_window[0]) & (u <= u_window[1])
    u, A, E = u[sel], f["A"][sel], f["E"][sel]
    intE = cumulative_trapezoid(E, u * 1e-9, axis=0, initial=0.0)
    dA, Delta = memory_prediction(src.Q, src.v_in, src.v_out, x_obs)
    k = int(np.argmax(np.abs(dA))) if np.any(dA) else 1
    comp = "xyz"[k]

    fig, ax = plt.subplots(2, 2, figsize=(9, 5.6))
    s = np.linspace(u_window[0], u_window[1], 2000) / src.T
    ax[0, 0].plot(s * src.T, src.H(s), color=INK)
    ax[0, 0].set(title="Input transition H(t/T)", xlabel="source time t (ns)", ylabel="H")
    ax[0, 1].plot(u, A[:, k] - A[0, k], color=BLUE, label=f"A_{comp}(u) - A_{comp}(early)")
    ax[0, 1].axhline(dA[k], color=ACC, ls="--", lw=1, label="Eq. 33 jump")
    ax[0, 1].set(title="Vector potential at observer", xlabel="retarded time u = t - r/c (ns)",
                 ylabel="V s/m")
    ax[0, 1].legend(fontsize=7, frameon=False)
    ax[1, 0].plot(u, E[:, k], color=INK)
    ax[1, 0].set(title=f"Electric field E_{comp} at observer (measurable)",
                 xlabel="u (ns)", ylabel="V/m")
    ax[1, 1].plot(u, intE[:, k], color=BLUE, label=f"∫E_{comp} dt (finite r, exact)")
    ax[1, 1].axhline(Delta[k], color=ACC, ls="--", lw=1, label="Δ, Eq. 38 (r → ∞)")
    ax[1, 1].set(title="Memory: test-charge kick per q/m", xlabel="u (ns)", ylabel="V s/m")
    ax[1, 1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Prediction for an observer at {np.round(x_obs, 2).tolist()} m", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return dict(u=u, A=A, E=E, intE=intE, dA=dA, Delta=Delta)


# ----------------------------------------------------------------------------- fit
def plot_fit(data, result, path, truth=None):
    from .fitting import to_v
    m0, r0 = result["_sigmoid"]
    mn, rn = result["_null"]
    r = np.linalg.norm(data.x_obs)
    u = data.t - r / C
    k = _main_component(data.Y)
    comp = "xyz"[k]
    unit = "V s/m" if data.observable == "A" else "V/m"
    model = result.get("_model")
    th = result["theta"]
    Yfit = (model or m0).predict(th)[0]
    Ynull = mn.predict(np.concatenate([rn.x, rn.x, [0.0, 0.0, 0.0]]), null=True)[0]
    # Show the component where the fitted signal changes most relative to the noise.
    snr = np.ptp(Yfit, axis=0) / np.where(np.isfinite(data.sigma), data.sigma, np.inf)
    k = int(np.argmax(snr))
    comp = "xyz"[k]

    fig = plt.figure(figsize=(9.5, 6.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.1, 0.01], width_ratios=[1.6, 1])
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    a2 = fig.add_subplot(gs[0:2, 1])
    a0.plot(u, data.Y[:, k], ".", ms=2, color=GREY, alpha=0.6, label="data")
    a0.plot(u, Ynull[:, k], color=INK, lw=1, ls=":", label="best fit without memory")
    a0.plot(u, Yfit[:, k], color=ACC, lw=1.6, label="exact-physics fit")
    a0.set(ylabel=f"{data.observable}_{comp} ({unit})",
           title=f"Fit to {data.observable}_{comp} at r = {r:g} m")
    a0.legend(fontsize=7, frameon=False)
    res = (data.Y[:, k] - Yfit[:, k]) / data.sigma[k]
    a1.plot(u, res, ".", ms=1.5, color=GREY)
    a1.axhline(0, color=INK, lw=0.8)
    a1.set(xlabel="retarded time u = t - r/c (ns)", ylabel="residual / σ")

    tg, g = result["tg"], result["g"]
    sel = (tg >= u[0]) & (tg <= u[-1])
    a2.plot(tg[sel], g[sel], color=ACC, lw=1.8, label="recovered H")
    if truth is not None:
        Ht, T = truth
        a2.plot(tg[sel], Ht(tg[sel] / T), color=INK, ls="--", lw=1, label="true H")
    a2.set(title="Recovered transition", xlabel="source time (ns)", ylabel="H")
    a2.legend(fontsize=7, frameon=False, loc="upper left")
    D, De = result["Delta"], result["Delta_err"]
    kd = int(np.argmax(np.abs(D))) if np.any(D) else k
    txt = (f"memory Δ_{'xyz'[kd]} = {D[kd]:.4g} ± {De[kd]:.2g} V s/m      "
           f"β_out,{'xyz'[kd]} = {result['beta_out'][kd]:.4f} ± {result['beta_out_err'][kd]:.2g}      "
           f"Δχ² (memory vs none) = {result['dchi2']:.1f}      "
           f"reduced χ² = {result['red_chi2']:.3f}")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.012, txt, ha="center", va="bottom", fontsize=8, family="monospace",
             color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- animation
def _solve_retarded(P, t, tg, X):
    """Retarded time for every point P at lab time t (vectorised bisection)."""
    lo = np.full(len(P), tg[0])
    hi = np.full(len(P), t)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        Xm = np.stack([np.interp(mid, tg, X[:, k]) for k in range(3)], 1)
        f = C * (t - mid) - np.linalg.norm(P - Xm, axis=1)
        later = f > 0
        lo = np.where(later, mid, lo)
        hi = np.where(later, hi, mid)
    return 0.5 * (lo + hi)


def animate(src, x_obs, path, frames=60, fps=15, extent=None, grid=110, t_range=None,
            dpi=90, progress=None):
    """GIF/MP4: field lines around the charge (left) next to what a distant observer sees."""
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    x_obs = np.asarray(x_obs, float)
    r = np.linalg.norm(x_obs)
    T = src.T
    s = np.linspace(-30, 30, 30001)
    t_half, rise = transition_metrics(s * T, src.H(s))
    if extent is None:
        extent = max(2.5 * C * T * 3, 1.0)
    if t_range is None:
        t_range = (t_half - 2.5 * T, t_half + extent / C * 1.15 + T)
    t0, t1 = t_range

    # Source trajectory on a fine grid that reaches far enough into the past.
    span = t1 - t0
    tg = np.linspace(t0 - (4 * extent + C * span) / C - 3 * T, t1 + 1, 40001)
    src.check_speed(tg)
    X, v, a = src.kinematics(tg)
    Xi = lambda t: np.array([np.interp(t, tg, X[:, k]) for k in range(3)])

    # Plane of the field plot: contains the motion and the observer direction.
    dvec = src.v_out - src.v_in
    e1 = x_obs / r
    e2 = dvec - e1 * (e1 @ dvec)
    if np.linalg.norm(e2) < 1e-12:
        e2 = np.cross(e1, [0, 0, 1.0]) if abs(e1[2]) < 0.9 else np.cross(e1, [0, 1.0, 0])
    e2 = e2 / np.linalg.norm(e2)
    mid = 0.5 * (Xi(t0) + Xi(t1))
    c1, c2 = mid @ e1, mid @ e2
    g1 = np.linspace(c1 - extent, c1 + extent, grid)
    g2 = np.linspace(c2 - extent, c2 + extent, grid)
    G1, G2 = np.meshgrid(g1, g2)
    P = G1.ravel()[:, None] * e1 + G2.ravel()[:, None] * e2

    def field(t):
        tr = _solve_retarded(P, t, tg, X)
        Xr = np.stack([np.interp(tr, tg, X[:, k]) for k in range(3)], 1)
        vr = np.stack([np.interp(tr, tg, v[:, k]) for k in range(3)], 1)
        ar = np.stack([np.interp(tr, tg, a[:, k]) for k in range(3)], 1)
        E = lw_E(src.Q, P, Xr, vr, ar)[0]
        return (E @ e1).reshape(G1.shape), (E @ e2).reshape(G1.shape)

    # Observer time series vs u = t - r/c (same clock as the left panel).
    tro = retarded_grid(T, t0, t1, r)
    fo = lw_at_observer(src, x_obs, tro)
    uo = fo["t_obs"] - r / C
    sel = (uo >= t0 - 1) & (uo <= t1 + 1)
    uo, Ao, Eo = uo[sel], fo["A"][sel], fo["E"][sel]
    dA, Delta = memory_prediction(src.Q, src.v_in, src.v_out, x_obs)
    k = int(np.argmax(np.abs(Delta))) if np.any(Delta) else 1
    comp = "xyz"[k]
    intE = cumulative_trapezoid(Eo[:, k], uo * 1e-9, initial=0.0)

    times = np.linspace(t0, t1, frames)
    fields = []
    for i, t in enumerate(times):
        fields.append(field(t))
        if progress:
            progress(i + 1, frames)
    mags = [np.hypot(*f) for f in fields]
    allm = np.concatenate([m.ravel() for m in mags])
    allm = allm[np.isfinite(allm) & (allm > 0)]
    vmin, vmax = np.log10(np.percentile(allm, [5, 99]))

    fig = plt.figure(figsize=(10.5, 4.6))
    gsp = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
    axf = fig.add_subplot(gsp[:, 0])
    ax1 = fig.add_subplot(gsp[0, 1])
    ax2 = fig.add_subplot(gsp[1, 1], sharex=ax1)
    obs_lab = f"observer at {r:g} m"

    ax1.plot(uo, Eo[:, k], color=GREY, lw=0.8)
    (l1,) = ax1.plot([], [], color=INK, lw=1.4)
    ax1.set(ylabel=f"E_{comp} (V/m)", title=f"What the {obs_lab} sees (shifted by r/c)")
    ax2.plot(uo, intE, color=GREY, lw=0.8)
    (l2,) = ax2.plot([], [], color=BLUE, lw=1.8)
    ax2.axhline(Delta[k], color=ACC, ls="--", lw=1)
    ax2.text(0.99, 0.97, "dashed: memory Δ (Eq. 38, r → ∞)\n"
             "late slope: 1/r² Coulomb drift at finite r", transform=ax2.transAxes,
             color=ACC, va="top", ha="right", fontsize=7)
    ax2.set(xlabel="u = t - r/c (ns)", ylabel=f"∫E_{comp} dt (V s/m)")
    cur = [a_.axvline(t0, color=ACC, lw=0.8) for a_ in (ax1, ax2)]
    ax1.set_xlim(t0, t1)

    def draw(i):
        axf.clear()
        t = times[i]
        Ex, Ey = fields[i]
        lm = np.log10(np.clip(mags[i], 1e-300, None))
        axf.pcolormesh(g1, g2, lm, cmap="magma", vmin=vmin, vmax=vmax, shading="auto")
        q = Xi(t)
        q1, q2 = q @ e1, q @ e2
        ang = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        rad = 0.06 * extent
        sp = np.column_stack([q1 + rad * np.cos(ang), q2 + rad * np.sin(ang)])
        sp = sp[(sp[:, 0] > g1[0]) & (sp[:, 0] < g1[-1]) & (sp[:, 1] > g2[0]) & (sp[:, 1] < g2[-1])]
        try:
            axf.streamplot(g1, g2, Ex, Ey, start_points=sp, color="white", linewidth=0.7,
                           arrowsize=0.01, integration_direction="both",
                           broken_streamlines=False, density=4)
        except Exception:
            pass
        for tt, ls in [(t_half - rise / 2, "--"), (t_half + rise / 2, ":")]:
            if t > tt:
                qq = Xi(tt)
                axf.add_patch(plt.Circle((qq @ e1, qq @ e2), C * (t - tt), fill=False,
                                         color="#6ec6ff", lw=1, ls=ls))
        axf.plot(q1, q2, "o", color=ACC, ms=7, mec="white")
        axf.annotate("", xy=(c1 + extent * 0.97, c2), xytext=(c1 + extent * 0.72, c2),
                     arrowprops=dict(arrowstyle="->", color="white"))
        axf.text(c1 + extent * 0.72, c2 + 0.04 * extent, "to observer", color="white", fontsize=7)
        axf.text(g1[0] + 0.03 * extent, g2[0] + 0.04 * extent,
                 "circles: light fronts from the start/end of the acceleration;\n"
                 "the kink between them is the radiation that carries the memory",
                 color="white", fontsize=6.5, va="bottom")
        axf.set_xlim(g1[0], g1[-1]); axf.set_ylim(g2[0], g2[-1])
        axf.set_aspect("equal")
        axf.set_xlabel("m (towards observer)"); axf.set_ylabel("m")
        axf.set_title(f"Field lines around the charge   t = {t:5.2f} ns", fontsize=10)
        m = uo <= t
        l1.set_data(uo[m], Eo[m, k])
        l2.set_data(uo[m], intE[m])
        for c in cur:
            c.set_xdata([t, t])
        return []

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    if str(path).lower().endswith(".mp4"):
        anim.save(path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    else:
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
