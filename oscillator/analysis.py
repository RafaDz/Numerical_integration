"""Comparisons against the exact solution: errors, convergence orders and resonance curves."""

import math
import time
from fractions import Fraction
from dataclasses import dataclass, field

import numpy as np

from .forces import NoForce, PulseForce, SinusoidalForce
from .integrators import METHODS, integrate, rk4_sinusoidal_batch, time_grid
from .system import Oscillator


def error_metrics(exact, numeric):
    """MAE, MSE, RMSE and maximum absolute error of numeric against exact (same grid)."""
    difference = np.asarray(numeric) - np.asarray(exact)
    with np.errstate(over='ignore', invalid='ignore'):
        mse = float(np.mean(np.square(difference)))
        return {
            'mae': float(np.mean(np.abs(difference))),
            'mse': mse,
            'rmse': math.sqrt(mse) if math.isfinite(mse) else math.inf,
            'max': float(np.max(np.abs(difference))),
        }


@dataclass
class MethodResult:
    key: str
    label: str
    x: np.ndarray
    v: np.ndarray
    energy: np.ndarray
    errors: dict
    final_energy_error: float  # relative to the exact energy at the end
    runtime_ms: float


@dataclass
class Comparison:
    t: np.ndarray
    step: float
    exact_x: np.ndarray
    exact_v: np.ndarray
    exact_energy: np.ndarray
    force: np.ndarray
    results: list = field(default_factory=list)

    def ranking(self, metric='rmse'):
        """Method keys from most to least accurate."""
        return [r.key for r in sorted(self.results, key=lambda r: _sort_value(r.errors[metric]))]


def _sort_value(value):
    return value if math.isfinite(value) else math.inf


def compare_methods(oscillator, x0, v0, total_time, step, methods=None, force=None, rtol=1e-3, atol=1e-6):
    """Run each method on the same grid and compare it with the exact solution."""
    force = force or NoForce()
    methods = list(methods or METHODS)
    t, h = time_grid(total_time, step)
    exact_x, exact_v = oscillator.exact_response(x0, v0, t, force)
    exact_energy = oscillator.energy(exact_x, exact_v)
    comparison = Comparison(t, h, exact_x, exact_v, exact_energy, np.asarray(force(t), dtype=float))
    for key in methods:
        started = time.perf_counter()
        with np.errstate(over='ignore', invalid='ignore'):
            trajectory = integrate(key, oscillator, x0, v0, total_time, step, force, rtol, atol)
            energy = oscillator.energy(trajectory.x, trajectory.v)
        runtime = 1000 * (time.perf_counter() - started)
        scale = max(abs(exact_energy[-1]), 1e-12 * max(np.max(exact_energy), 1e-300))
        comparison.results.append(MethodResult(
            key, METHODS[key].label, trajectory.x, trajectory.v, energy,
            error_metrics(exact_x, trajectory.x), float((energy[-1] - exact_energy[-1]) / scale), runtime))
    return comparison


def aligned_steps(total_time, times, steps):
    """
    Replace each step by base/n, where base is the largest step that fits a whole number of times into the
    total time and every force jump time, so the jumps fall on grid points. Duplicates are removed.
    Returns the steps unchanged if the times have no common divisor of reasonable size.
    """
    fractions = [Fraction(value).limit_denominator(10_000) for value in (total_time, *times) if value > 0]
    if any(abs(float(f) - value) > 1e-9 * max(1.0, value)
           for f, value in zip(fractions, [v for v in (total_time, *times) if v > 0])):
        return np.asarray(steps, dtype=float)
    base = fractions[0]
    for f in fractions[1:]:
        base = Fraction(math.gcd(base.numerator * f.denominator, f.numerator * base.denominator),
                        base.denominator * f.denominator)
    base = float(base)
    return np.unique([base / max(1, round(base / h)) for h in steps])


def convergence_study(oscillator, x0, v0, total_time, steps, methods=None, force=None, metric='max',
                      rtol=1e-3, atol=1e-6, align=True):
    """
    Error against step size for each method, and the observed order p from a fit of log(error) = p log(h) + c.
    The fit skips errors that are not finite or are at the round-off level. With align=True, steps are
    adjusted so the force's jump times (if any) fall on grid points, where fixed-step methods keep their order.
    """
    force = force or NoForce()
    methods = list(methods or METHODS)
    steps = np.sort(np.asarray(steps, dtype=float))
    jumps = [p for p in force.breakpoints() if 0 < p < total_time]
    if align and jumps:
        steps = aligned_steps(total_time, jumps, steps)
    used_steps = np.array([time_grid(total_time, h)[1] for h in steps])
    errors = {key: [] for key in methods}
    for h in steps:
        t, _ = time_grid(total_time, h)
        exact_x, _ = oscillator.exact_response(x0, v0, t, force)
        scale = max(np.max(np.abs(exact_x)), 1e-300)
        for key in methods:
            with np.errstate(over='ignore', invalid='ignore'):
                trajectory = integrate(key, oscillator, x0, v0, total_time, h, force, rtol, atol)
            errors[key].append(error_metrics(exact_x, trajectory.x)[metric] / scale)

    orders = {}
    for key, values in errors.items():
        values = np.asarray(values)
        usable = np.isfinite(values) & (values > 1e-12) & (values < 1)
        if np.count_nonzero(usable) >= 2 and np.ptp(np.log(used_steps[usable])) > 0:
            orders[key] = float(np.polyfit(np.log(used_steps[usable]), np.log(values[usable]), 1)[0])
        else:
            orders[key] = math.nan
    return {'steps': used_steps, 'errors': {key: np.asarray(v) for key, v in errors.items()}, 'orders': orders}


def resonance_curves(mass, spring_constant, dampings, amplitude, omegas, numeric=False, x0=0.0, v0=0.0,
                     total_time=300.0, step=0.05, phase=0.0):
    """
    Steady-state amplitude against driving frequency for several damping coefficients.

    The exact curve is A(ω) = F0 / √((k - mω²)² + (bω)²). With numeric=True the amplitude is also measured
    from RK4 runs as the largest |x| over the last quarter of the run (after transients have decayed, if the
    run is long enough compared with the decay time 2m/b).
    """
    omegas = np.asarray(omegas, dtype=float)
    curves = []
    for b in dampings:
        oscillator = Oscillator(mass, spring_constant, float(b))
        curve = {
            'damping': float(b),
            'exact': oscillator.steady_state_amplitude(amplitude, omegas),
            'quality_factor': oscillator.quality_factor,
            'resonance_frequency': oscillator.resonance_frequency,
            'peak_amplitude': (float(oscillator.steady_state_amplitude(amplitude, oscillator.resonance_frequency))
                               if math.isfinite(oscillator.resonance_frequency) else abs(amplitude) / spring_constant),
            'half_power_q': half_power_quality_factor(oscillator),
            'decay_time': oscillator.decay_time,
            'settled': 5 * oscillator.decay_time < 0.75 * total_time,
        }
        curves.append(curve)

    if numeric:
        b_grid = np.asarray([float(b) for b in dampings])[:, None]
        peaks = rk4_sinusoidal_batch(mass, spring_constant, b_grid, amplitude, omegas[None, :], phase,
                                     x0, v0, total_time, step, window_start=0.75 * total_time)
        for curve, row in zip(curves, peaks):
            curve['numeric'] = row
    return {'omegas': omegas, 'natural_frequency': math.sqrt(spring_constant / mass), 'curves': curves}


def half_power_quality_factor(oscillator, samples=200_001):
    """
    Q measured from the exact response curve as ω_peak / Δω, where Δω is the full width of the energy
    (amplitude²) peak at half its height. Returns nan when the peak is too broad to have two half-power points.
    """
    w0 = oscillator.natural_frequency
    omegas = np.linspace(0, 3 * w0, samples)[1:]
    power = oscillator.steady_state_amplitude(1.0, omegas) ** 2
    peak = int(np.argmax(power))
    if not np.isfinite(power[peak]):
        return math.inf
    above = power >= power[peak] / 2
    if above[0] or above[-1]:
        return math.nan
    left = np.flatnonzero(~above[:peak])[-1]
    right = peak + np.flatnonzero(~above[peak:])[0]
    return float(omegas[peak] / (omegas[right] - omegas[left]))


def make_force(kind, amplitude=0.0, start=0.0, duration=1.0, angular_frequency=1.0, phase=0.0):
    """Build a force from simple parameters ('none', 'pulse' or 'sinusoidal')."""
    if kind == 'none':
        return NoForce()
    if kind == 'pulse':
        return PulseForce(amplitude, start, duration)
    if kind == 'sinusoidal':
        return SinusoidalForce(amplitude, angular_frequency, phase)
    raise ValueError(f'Unknown force type {kind!r}.')
