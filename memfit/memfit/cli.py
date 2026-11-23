"""memfit command line: predict, simulate, fit and animate electromagnetic memory."""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import __version__
from .physics import C, Source, observe, memory_prediction
from .hfunc import PRESETS, compile_expr, validate, transition_metrics


# ----------------------------------------------------------------------------- helpers
def vec(text):
    try:
        v = np.array([float(x) for x in text.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected three comma-separated numbers, got '{text}'")
    if v.size != 3:
        raise argparse.ArgumentTypeError(f"expected three comma-separated numbers, got '{text}'")
    return v


def pair(text):
    v = [float(x) for x in text.split(",")]
    if len(v) != 2 or v[0] >= v[1]:
        raise argparse.ArgumentTypeError("expected 'start,end' with start < end")
    return tuple(v)


def add_source_args(p):
    g = p.add_argument_group("source (paper Eq. 30: v(t) = v_in + H(t/T)(v_out - v_in))")
    g.add_argument("--H", default="smooth",
                   help="transition function of s = t/T: a preset name (see 'memfit presets') "
                        "or an expression, e.g. \"0.5*(1+tanh(3*s))\" (default: smooth)")
    g.add_argument("--T", type=float, default=1.0, help="transition time scale T in ns (default 1)")
    g.add_argument("--v-in", type=vec, default=vec("0,0,0"),
                   help="initial velocity in units of c, 'x,y,z' (default 0,0,0)")
    g.add_argument("--v-out", type=vec, default=vec("0,0.5,0"),
                   help="final velocity in units of c, 'x,y,z' (default 0,0.5,0)")
    g.add_argument("--Q", type=float, default=1.0, help="source charge in coulombs (default 1)")
    g.add_argument("--observer", type=vec, default=vec("10,0,0"),
                   help="observer position in metres, 'x,y,z' (default 10,0,0)")


def build_source(a):
    H, expr = compile_expr(a.H)
    validate(H)
    src = Source(H, a.T, a.v_in * C, a.v_out * C, a.Q, label=expr)
    src.check_speed(np.linspace(-60 * a.T, 60 * a.T, 20001))
    return src, expr


def fmt_vec(v, e=None, digits=5):
    v = np.asarray(v, float) + 0.0          # avoid printing -0
    if e is None:
        return "(" + ", ".join(f"{x:.{digits}g}" for x in v) + ")"
    return "(" + ", ".join(f"{x:.{digits}g} ± {y:.2g}" for x, y in zip(v, e)) + ")"


def hr(title):
    print(f"\n{title}\n" + "-" * len(title))


def done(path):
    print(f"  wrote {os.path.abspath(path)}")


# ----------------------------------------------------------------------------- commands
def cmd_presets(a):
    print("Transition presets (use with --H NAME; s = t/T):\n")
    for k, (expr, desc) in PRESETS.items():
        print(f"  {k:<12} {desc}\n  {'':<12} H(s) = {expr}\n")
    print("Or pass your own expression in s. Allowed: + - * / ** %, comparisons, pi, e, and\n"
          "sin cos tan exp log log10 sqrt abs tanh sinh cosh arctan arcsin arccos erf erfc\n"
          "sigmoid heaviside where clip minimum maximum sign. H must go from 0 to 1.")


def cmd_predict(a):
    src, expr = build_source(a)
    dA, Delta = memory_prediction(src.Q, src.v_in, src.v_out, a.observer)
    s = np.linspace(-30, 30, 60001)
    th, rise = transition_metrics(s * src.T, src.H(s))
    hr("Source")
    print(f"  H(s) = {expr}")
    print(f"  T = {src.T:g} ns | midpoint t_half = {th:+.3f} ns | 10-90% rise = {rise:.3f} ns")
    print(f"  v_in = {fmt_vec(a.v_in)} c,  v_out = {fmt_vec(a.v_out)} c,  Q = {src.Q:g} C")
    hr(f"Memory at observer {fmt_vec(a.observer)} m (leading order in 1/r)")
    print(f"  jump in A (Eq. 33):            ΔA = {fmt_vec(dA)} V s/m")
    print(f"  memory vector (Eq. 38):        Δ  = {fmt_vec(Delta)} V s/m")
    print(f"  test-charge kick:              Δv = (q/m) Δ;  |Δ| = {np.linalg.norm(Delta):.5g} V s/m")
    print("  (independent of H: only v_in and v_out matter)")
    if a.plot:
        from .render import plot_prediction
        out = plot_prediction(src, a.observer, a.window, a.plot)
        k = int(np.argmax(np.abs(Delta))) if np.any(Delta) else 1
        print(f"\n  exact ∫E_{'xyz'[k]} dt over the window at finite r: {out['intE'][-1, k]:.5g} "
              f"(differs from Δ by the 1/r² Coulomb drift; shrinks as r grows)")
        done(a.plot)


def cmd_simulate(a):
    src, expr = build_source(a)
    o = observe(src, a.observer, a.window, a.n)
    Y = o[a.observable]
    clean_scale = float(np.abs(Y - Y[0]).max())
    sigma = a.sigma if a.sigma is not None else a.noise * clean_scale
    rng = np.random.default_rng(a.seed)
    Yn = Y + sigma * rng.standard_normal(Y.shape)
    unit = "V_s_per_m" if a.observable == "A" else "V_per_m"
    header = "t_ns," + ",".join(f"{a.observable}_{c}_{unit}" for c in "xyz")
    np.savetxt(a.out, np.column_stack([o["t"], Yn]), delimiter=",", header=header, comments="")
    dA, Delta = memory_prediction(src.Q, src.v_in, src.v_out, a.observer)
    meta = dict(memfit_version=__version__, observable=a.observable,
                observer_m=a.observer.tolist(), Q_C=src.Q, sigma=sigma,
                truth=dict(H=expr, T_ns=src.T, beta_in=a.v_in.tolist(), beta_out=a.v_out.tolist(),
                           dA_Vs_per_m=dA.tolist(), Delta_Vs_per_m=Delta.tolist()),
                noise_fraction=a.noise if a.sigma is None else None, seed=a.seed,
                window_u_ns=list(a.window), n=a.n)
    meta_path = os.path.splitext(a.out)[0] + ".json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Simulated {a.n} samples of {a.observable} at {fmt_vec(a.observer)} m "
          f"with noise σ = {sigma:.4g} ({unit.replace('_', ' ')})")
    print(f"  true memory Δ = {fmt_vec(Delta)} V s/m")
    done(a.out)
    done(meta_path)


def load_data(a):
    from .fitting import Data
    raw = np.genfromtxt(a.data, delimiter=",", names=True)
    names = raw.dtype.names
    if names is None or len(names) < 4:
        sys.exit("Expected a CSV with a header and 4 columns: t_ns, then x, y, z of A or E.")
    t = raw[names[0]]
    Y = np.column_stack([raw[n] for n in names[1:4]])
    meta = {}
    meta_path = os.path.splitext(a.data)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    observable = a.observable or meta.get("observable") or names[1][0].upper()
    if observable not in ("A", "E"):
        sys.exit("Cannot tell whether the data are A or E; pass --observable A or E.")
    x_obs = a.observer if a.observer is not None else np.array(meta.get("observer_m", [10, 0, 0]))
    Q = a.Q if a.Q is not None else meta.get("Q_C", 1.0)
    sigma = a.sigma if a.sigma is not None else meta.get("sigma")
    if a.observer is None and "observer_m" not in meta:
        print("  note: no observer position given; assuming 10,0,0 m (use --observer)")
    return Data(t, Y, observable, x_obs, Q, sigma), meta


def cmd_fit(a):
    from .fitting import fit, calibrate
    t_start = time.time()
    data, meta = load_data(a)
    flexible = a.flexible if a.flexible == "auto" else int(a.flexible)
    res = fit(data, flexible=flexible, window=a.window)
    k = int(np.argmax(np.abs(res["Delta"]))) if np.any(res["Delta"]) else 1
    hr(f"Fit: {len(data.t)} samples of {data.observable} at observer {fmt_vec(data.x_obs)} m")
    print(f"  noise σ used:        {fmt_vec(data.sigma, digits=4)}")
    print(f"  shape model:         " + ("single step (tail weight fitted)" if res["n_modes"] == 0 else
                                         f"step + {res['n_modes']} modes (window {res['window_centre']:+.2f} ± {res['window']:.2f} ns)"))
    if res.get("auto_switched"):
        print(f"  (single-step misfit at {res['shape_z_sigmoid']:.0f} σ, so the shape was made "
              f"flexible; {len(res['model_table'])} models compared by BIC)")
    print(f"  reduced χ²:          {res['red_chi2']:.4f}  ({res['shape_z']:+.1f} σ from 1)")
    if res["shape_z"] > 3:
        if data.observable == "E":
            print("  ! the model still misfits. E follows the acceleration, so a wrong shape biases\n"
                  "    the memory: try --flexible 24 or a wider --window, and treat Δ with caution.")
        else:
            print("  ! the shape is not fully captured. For A data the memory is set by the plateaus\n"
                  "    and stays reliable; use --flexible auto or --window to recover the shape.")
    hr("Results")
    print(f"  memory Δ (Eq. 38):   {fmt_vec(res['Delta'], res['Delta_err'])} V s/m")
    print(f"  jump in A (Eq. 33):  {fmt_vec(res['dA'], res['dA_err'])} V s/m")
    print(f"  β_in:                {fmt_vec(res['beta_in'], res['beta_in_err'], 4)}")
    print(f"  β_out:               {fmt_vec(res['beta_out'], res['beta_out_err'], 4)}")
    print(f"  transition midpoint: {res['t_half']:+.4f} ns (source time)")
    print(f"  10-90% rise time:    {res['rise']:.4f} ns")
    nu = res["nu"]
    print(f"  step tail weight:    ν = {nu:.3g}  " + ("(Gaussian-like tails)" if nu > 50 else
          "(logistic-like tails)" if nu > 5 else "(heavy tails)"))
    hr("Detection")
    print(f"  Δχ² (memory vs constant velocity) = {res['dchi2']:.2f}")
    cal = None
    if a.calibrate:
        print(f"  calibrating with {a.calibrate} noise-only simulations "
              f"({a.workers or os.cpu_count()} workers) ...")
        bar = lambda i, n: print(f"\r  {i}/{n}", end="", flush=True)
        cal = calibrate(data, res, a.calibrate, a.workers, progress=bar)
        print()
        mx = float(cal["null_dchi2"].max())
        detected = res["dchi2"] > cal["threshold_1pct"]
        print(f"  noise-only Δχ²:           median {np.median(cal['null_dchi2']):.2f}, "
              f"max {mx:.2f}")
        print(f"  1% false-alarm threshold: {cal['threshold_1pct']:.2f}")
        if res["dchi2"] > mx:
            print(f"  false-alarm probability:  p < {1 / (a.calibrate + 1):.3g} "
                  f"(no simulation reached the observed Δχ²)")
        else:
            print(f"  false-alarm probability:  p = {cal['p_value']:.3g}")
        print("  verdict:                  " +
              ("MEMORY DETECTED (above the 1% false-alarm threshold)" if detected
               else "not detected at 1% false-alarm"))
        if a.calibrate < 100:
            print("  (fewer than 100 simulations: the threshold is rough; use --calibrate 200+)")
    else:
        print("  for a false-alarm probability, add --calibrate 200 (reference: the 1% threshold\n"
              "  was 20.4 for 2000 samples at r = 10 m; it depends on your data)")
    if meta.get("truth"):
        tr = meta["truth"]
        D = np.array(tr["Delta_Vs_per_m"])
        hr("Compared with the simulation truth")
        if abs(D[k]) > 0:
            pull = (res["Delta"][k] - D[k]) / res["Delta_err"][k]
            print(f"  Δ_{'xyz'[k]}: fitted {res['Delta'][k]:.5g}, true {D[k]:.5g} "
                  f"(bias {res['Delta'][k] / D[k] - 1:+.3%}, {pull:+.1f} σ)")
        Ht, _ = compile_expr(tr["H"])
        s = np.linspace(-30, 30, 60001)
        th, rise = transition_metrics(s * tr["T_ns"], Ht(s))
        print(f"  midpoint: fitted {res['t_half']:+.4f}, true {th:+.4f} ns | "
              f"rise: fitted {res['rise']:.4f}, true {rise:.4f} ns")
    if a.plot:
        from .render import plot_fit
        truth = None
        if meta.get("truth"):
            truth = (compile_expr(meta["truth"]["H"])[0], meta["truth"]["T_ns"])
        plot_fit(data, res, a.plot, truth)
        done(a.plot)
    if a.json:
        keep = {k_: (v.tolist() if isinstance(v, np.ndarray) else v) for k_, v in res.items()
                if not k_.startswith("_") and k_ not in ("tg", "g", "theta", "null_theta")}
        if cal:
            keep.update(p_value=cal["p_value"], threshold_1pct=cal["threshold_1pct"])
        with open(a.json, "w") as fh:
            json.dump(keep, fh, indent=2, default=float)
        done(a.json)
    print(f"\n  ({time.time() - t_start:.1f} s)")


def cmd_animate(a):
    from .render import animate
    src, expr = build_source(a)
    print(f"Rendering {a.frames} frames for H(s) = {expr} ...")
    bar = lambda i, n: print(f"\r  computing fields {i}/{n}", end="", flush=True)
    animate(src, a.observer, a.out, frames=a.frames, fps=a.fps, extent=a.extent, grid=a.grid,
            dpi=a.dpi, progress=bar)
    print()
    done(a.out)


def cmd_demo(a):
    os.makedirs(a.outdir, exist_ok=True)
    p = lambda name: os.path.join(a.outdir, name)
    base = ["--H", a.H, "--T", str(a.T), "--observer", "10,0,0"]
    steps = [
        ["predict", *base, "--plot", p("predict.png")],
        ["simulate", *base, "--observable", "E", "--noise", "0.05", "--out", p("data.csv")],
        ["fit", p("data.csv"), "--plot", p("fit.png"), "--json", p("fit.json")],
        ["animate", *base, "--frames", str(a.frames), "--out", p("memory.gif")],
    ]
    for argv in steps:
        print("\n$ memfit " + " ".join(argv))
        main(argv)


# ----------------------------------------------------------------------------- parser
def build_parser():
    ap = argparse.ArgumentParser(
        prog="memfit",
        description="Predict, simulate, detect and animate the electromagnetic memory effect "
                    "(de Aguiar Alves & Arderucio Costa, Am. J. Phys. 94, 973 (2026)).")
    ap.add_argument("--version", action="version", version=f"memfit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets", help="list built-in transition functions H").set_defaults(fn=cmd_presets)

    p = sub.add_parser("predict", help="memory and fields for an input H (no noise)")
    add_source_args(p)
    p.add_argument("--plot", metavar="PNG", help="save a figure of H, A, E and ∫E dt")
    p.add_argument("--window", type=pair, default=(-8.0, 8.0),
                   help="retarded-time window u for the plot, 'start,end' in ns (default -8,8)")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("simulate", help="write noisy observer data to CSV (+ JSON with the truth)")
    add_source_args(p)
    p.add_argument("--observable", choices=["A", "E"], default="E",
                   help="E (measurable, default) or A (vector potential, Lorenz gauge)")
    p.add_argument("--noise", type=float, default=0.05,
                   help="noise std as a fraction of the clean signal's largest excursion (default 0.05)")
    p.add_argument("--sigma", type=float, help="absolute noise std instead of --noise")
    p.add_argument("--n", type=int, default=2000, help="number of samples (default 2000)")
    p.add_argument("--window", type=pair, default=(-8.0, 8.0),
                   help="retarded-time window u, 'start,end' in ns (default -8,8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data.csv")
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("fit", help="recover the source and memory from CSV data; test for detection")
    p.add_argument("data", help="CSV: header, then columns t_ns, x, y, z of A (V s/m) or E (V/m)")
    p.add_argument("--observable", choices=["A", "E"], help="override detection from header/JSON")
    p.add_argument("--observer", type=vec, help="observer position in m (default: from JSON, else 10,0,0)")
    p.add_argument("--Q", type=float, help="source charge in C (default: from JSON, else 1)")
    p.add_argument("--sigma", type=float, help="noise std (default: from JSON, else estimated)")
    p.add_argument("--flexible", default="0",
                   help="shape model: 0 = pure sigmoid (default, optimal for smooth steps), "
                        "K = sigmoid + K sine modes, auto = choose by BIC")
    p.add_argument("--window", type=float, help="half-width (ns) of the flexible-shape window")
    p.add_argument("--calibrate", type=int, default=0, metavar="N",
                   help="simulate N noise-only datasets to get a false-alarm probability")
    p.add_argument("--workers", type=int, help="parallel processes for --calibrate (default: all cores)")
    p.add_argument("--plot", metavar="PNG", help="save a figure of the fit")
    p.add_argument("--json", metavar="FILE", help="save results as JSON")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("animate", help="GIF (or .mp4) of field lines and the observer's signal")
    add_source_args(p)
    p.add_argument("--out", default="memory.gif", help="output .gif or .mp4 (mp4 needs ffmpeg)")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--extent", type=float, help="half-width of the field view in m (default: auto)")
    p.add_argument("--grid", type=int, default=110, help="field grid points per side (default 110)")
    p.add_argument("--dpi", type=int, default=90)
    p.set_defaults(fn=cmd_animate)

    p = sub.add_parser("demo", help="run predict, simulate, fit and animate into one folder")
    p.add_argument("--outdir", default="memfit_demo")
    p.add_argument("--H", default="smooth")
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--frames", type=int, default=50)
    p.set_defaults(fn=cmd_demo)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except (ValueError, RuntimeError) as err:
        sys.exit(f"memfit: error: {err}")


if __name__ == "__main__":
    main()
