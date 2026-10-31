# Damped Oscillator: Numerical Integration

Compares numerical integration methods for a damped, optionally driven mass-spring system,

    m x'' + b x' + k x = F(t),

against its **exact** solution, with an interactive web interface for changing the system, the driving force,
the step size and the methods.

The project report is `Numerical_integration.ipynb`. It has been corrected for the problems listed in the
app's **Notebook review** tab (see the revision note at the top of the notebook); the package and app below
were written with the same fixes (summary at the end).

## Quick start

```bash
pip install -r requirements.txt
python server.py --open
```

This opens http://localhost:8000 in your browser (use `--port` to pick another port).
The front-end is plain HTML, CSS and JavaScript in `web/`. `server.py` serves it and runs the calculations
through a small JSON API, using only the Python standard library. Plotly.js is served from the installed
`plotly` package, so the page also works offline.

## The interface

| Tab | What it does |
| --- | --- |
| **Sidebar** | Mass, spring constant, damping (with presets 0.1, 0.5, 2.0, ½b<sub>cr</sub>, b<sub>cr</sub>, 2b<sub>cr</sub>) and initial conditions. Shows ω₀, γ, b<sub>cr</sub>, ζ, Q, ω<sub>d</sub>, decay time and the damping regime. |
| **Method comparison** | Runs the chosen methods (updates as you type) with no force, a rectangular pulse or a sinusoidal force. Ranks the methods by MAE, MSE, RMSE or max error, and plots position, error (log scale), energy and phase portrait against the exact solution. CSV download. |
| **Convergence** | Error against step size on log-log axes with reference slopes, and the observed order of each method. |
| **Resonance** | Exact steady-state amplitude against driving frequency for several dampings, optionally checked with RK4 runs; phase lag; Q from the formula and from the peak width. |
| **Methods** | The equations, the exact solutions and the methods. |
| **Notebook review** | Problems in the original notebook. |

Inputs are saved in the browser.

### Methods

| Method | Global order | Notes |
| --- | --- | --- |
| Euler | 1 | Explicit; gains energy without damping. |
| Improved Euler (notebook) | 1 | The notebook's variant: h²a/2 added to x only. |
| Heun (RK2) | 2 | The standard "improved Euler". |
| Euler-Cromer | 1 | Semi-implicit; symplectic for b = 0. |
| Velocity Verlet | 2 | Damping treated implicitly so it stays 2nd order; self-starting. |
| Runge-Kutta 4 | 4 | Classical fixed-step RK4. |
| SciPy RK45 | adaptive | `solve_ivp` as in the notebook; accuracy set by its tolerance, not by h. |

### API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/defaults` | Methods and the notebook's parameter values |
| `POST /api/system` | Derived quantities for m, k, b |
| `POST /api/simulate` | Run methods and compare with the exact solution |
| `POST /api/convergence` | Error against step size and observed orders |
| `POST /api/resonance` | Resonance curves, phase lag and Q factor |

## Using the package from Python

```python
import numpy as np
from oscillator import Oscillator, PulseForce, SinusoidalForce, compare_methods, convergence_study, integrate

system = Oscillator(mass=5.58, spring_constant=1.81, damping=0.1)
print(system.regime, system.quality_factor)

# One method
trajectory = integrate('rk4', system, x0=0, v0=-1, total_time=150, step=0.05)

# All methods against the exact solution, with a sinusoidal driving force
comparison = compare_methods(system, 0, -1, 150, 0.1, force=SinusoidalForce(2.5, 0.9))
print(comparison.ranking('rmse'))

# Observed order of accuracy
study = convergence_study(system, 0, -1, 50, np.geomspace(0.01, 0.1, 5), force=PulseForce(2.5, 20, 3))
print(study['orders'])
```

## Project layout

```
oscillator/
  system.py       Oscillator class, derived quantities, exact solutions (all regimes, pulse, sinusoid)
  forces.py       No force, rectangular pulse, sinusoidal force
  integrators.py  Euler, Improved Euler, Heun, Euler-Cromer, Velocity Verlet, RK4, SciPy RK45
  analysis.py     Error metrics, method comparison, convergence study, resonance curves
server.py         Web server and JSON API
web/              HTML/CSS/JS front-end
tests/            pytest tests: exact solutions satisfy the equation, observed orders, API
```

Run the tests with `pytest`.

## Changes from the original notebook

- **Exact reference for every case.** One formula covers under-, critically and overdamped motion (no `b == b_cr`
  float comparison), and exact solutions for the pulse and sinusoidal forces mean the driven runs have real errors
  instead of being compared with the unforced solution.
- **Same time grid for everything.** The notebook's RK results were reported on a `linspace` grid with spacing
  nsteps·h/(nsteps−1) but compared with an analytic solution on `arange(0, T, h)`. That doubled the RK error and made
  Euler-Cromer look best for b = 0.1. Here every method and the exact solution use t = 0, h, …, T.
- **Real RK4.** The notebook's "Runge-Kutta" was SciPy's adaptive RK45 at default tolerance, whose error does not depend
  on h. Fixed-step RK4 is added, and RK45 is kept with a tolerance setting.
- **Second-order Verlet.** The notebook's Verlet used a lagged velocity in the damping force, which made it first order.
- **Heun's method** added as the standard 2nd-order improved Euler (the notebook's "Improved Euler" is first order).
- **Convergence orders measured** on log-log axes instead of judged from linear MAE-vs-h plots.
- **Resonance** uses the steady-state amplitude (exact, and from RK4 after transients have decayed) instead of the
  largest |x| over a run that includes the start-up transient.
- Pulse forces: fixed-step methods use the force averaged over each step, and the convergence study puts the pulse
  edges on grid points so the methods keep their order.
