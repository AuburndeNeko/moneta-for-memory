# memfit

Predict, simulate, detect and animate the **electromagnetic memory effect**: the permanent
velocity kick a distant test charge receives when a source charge changes velocity.

Physics follows de Aguiar Alves & Arderucio Costa, *Liénard–Wiechert potentials and the
electromagnetic memory effect*, Am. J. Phys. **94**, 973 (2026). The source velocity is

    v(t) = v_in + H(t/T) (v_out − v_in)          (paper Eq. 30)

and **you choose H**: a preset or any formula in `s = t/T` that goes from 0 to 1.

## Install

Needs Python 3.9+ with NumPy, SciPy, Matplotlib and Pillow. No PyTorch, so it works on
Intel Macs. From this folder:

```bash
pip install .            # or: pip install -e .   to edit the code
memfit --help
```

`python -m memfit ...` works too, without installing.

## Quick start

```bash
memfit demo                 # predict + simulate + fit + animate into ./memfit_demo  (~40 s)
```

## Commands

| Command | What it does |
|---|---|
| `memfit presets` | list built-in transition functions |
| `memfit predict` | memory (Eq. 33 / 38) and the fields an observer sees, for your H |
| `memfit simulate` | noisy observer data → `data.csv` + `data.json` (the truth) |
| `memfit fit data.csv` | recover the source, transition and memory; test whether memory is detected |
| `memfit animate` | GIF (or MP4) of the field lines kinking, next to the observer's signal |

### Your own transition function

```bash
memfit predict  --H "0.5*(1+erf(2*s)) + 0.1*exp(-(s-1)**2)*sin(6*s)" --plot pred.png
memfit animate  --H "sigmoid(5*s)" --v-out 0,0.7,0 --out kick.gif
```

Allowed in `--H`: `s`, numbers, `+ - * / ** %`, comparisons, `pi`, `e`, and
`sin cos tan exp log log10 sqrt abs tanh sinh cosh arctan arcsin arccos erf erfc sigmoid
heaviside where clip minimum maximum sign`. Nothing else is evaluated. H must settle at 0
before and 1 after the transition, and the source must stay below c.

Other source options: `--T` (ns), `--v-in`, `--v-out` (units of c, `x,y,z`), `--Q` (C),
`--observer` (m, `x,y,z`).

### Fitting data

```bash
memfit simulate --H oscillating --observable E --noise 0.05 --out data.csv
memfit fit data.csv --plot fit.png --json fit.json --calibrate 200
```

* **Input**: CSV with a header and columns `t_ns, x, y, z` of either **E** (V/m, the measurable
  field) or **A** (V s/m, Lorenz gauge). If `data.json` sits next to the CSV (as `simulate`
  writes), the observer position, charge, noise and truth are read from it; otherwise pass
  `--observer`, `--Q`, `--sigma` (noise is estimated from the data if not given).
* **Shape model**: a single smooth step (a Student-t CDF) whose midpoint, width and **tail
  weight ν** are fitted. ν ≈ 9 behaves like the logistic sigmoid and ν → ∞ like an erf step.
  The tail weight matters for **E** data: E follows the acceleration, and a model with the
  wrong tails moves part of the velocity change into (or out of) the low-level tails, which
  biased the memory by about 5% for an erf transition before ν was fitted. If the fit is
  still statistically poor, `fit` switches automatically to the step plus sine modes,
  choosing the number of modes and the window by BIC (`--flexible auto` forces this,
  `--flexible 0` turns it off, `--flexible K` fixes K).
* **Detection vs estimation**: the detection statistic Δχ² always uses the step with
  logistic tails (fast, and detection power barely depends on the tails), so `--calibrate`
  simulates exactly the same statistic. The memory value comes from the tail-fitted model.
* **Output**: memory vector Δ with uncertainty, β_in and β_out, transition midpoint and rise
  time, reduced χ², and Δχ² for "memory" against "constant velocity".
* **Detection**: `--calibrate N` simulates N noise-only datasets like yours and gives a
  false-alarm probability and the 1% threshold. Use 200+ for a reliable threshold
  (several minutes; runs on all cores).

### Typical run times (2 cores)

| Task | Time |
|---|---|
| `fit`, smooth / erf / sharp transition | ~2–15 s |
| `fit`, oscillating or two-stage transition | ~1–5 min |
| `fit`, piecewise-linear `ramp` with E data | ~3–5 min (fits less well; memory still OK) |
| `fit --calibrate 200` | ~5–15 min |
| `animate`, 60 frames | ~30–40 s, 3–4 MB GIF |

## Validation

Memory recovered from simulated data, 5% noise, r = 10 m, 2000 samples, 3 noise seeds per
case. "Pull" is (fitted − true) / reported error; an RMS pull near 1 means the error bars
are honest.

| Transition | A: mean bias | A: RMS pull | E: mean bias | E: RMS pull |
|---|---|---|---|---|
| smooth | −0.01% | 1.2 | +0.39% | 1.4 |
| erf | +0.01% | 1.1 | +0.34% | 1.0 |
| sharp | +0.05% | 1.3 | +0.68% | 0.3 |
| ramp | +0.22% | 1.5 | +0.34% | 1.0 |
| oscillating | ≈ +0.1% | ≈ 1.0 | +0.18% | 1.2 |
| two-stage | +0.18% | 0.9 | +0.35% | 0.9 |

Before the tail weight was fitted, the erf transition with E data gave +5.2% (8σ); that
case is now a regression test. Single-step fits sit at the Cramér–Rao bound, and the
detection statistic matches the known-shape matched filter (see the project's benchmark
scripts). `pytest tests` runs the unit tests (~30 s).

## Limitations

* For **E** data the memory estimate depends on modelling the transition's shape, tails
  included. Shapes outside the model family (step + sine modes) can still bias it; the
  reduced χ² and the `fit` warning are the guard. **A**-type data are far less sensitive.
* Single point charge with known charge Q, known observer position, and X(0) = x0.
  (Q and v enter A as the product Qv, so they can't both be fitted.)
* White Gaussian noise; no detector response, bandwidth limit or drift model yet.
* The tool works with simulated data. Real measurements would need an instrument model
  (antenna response, filtering) in front of the fit.
