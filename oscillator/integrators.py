"""
Fixed-step integrators for  m x'' + b x' + k x = F(t),  plus SciPy's adaptive RK45 for reference.

Every method returns a Trajectory on the grid t_n = n h, n = 0..n_steps.
The force is evaluated at the start, middle and end of each step (see forces.step_values).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .forces import NoForce, step_values


@dataclass
class Trajectory:
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray


def time_grid(total_time, step):
    """Grid of n+1 points from 0 to total_time; the step is adjusted slightly so the grid ends exactly at T."""
    if not (total_time > 0 and step > 0):
        raise ValueError('Total time and step size must be positive.')
    n_steps = max(1, round(total_time / step))
    return np.linspace(0.0, total_time, n_steps + 1), total_time / n_steps


def _constants(oscillator):
    return oscillator.mass, oscillator.spring_constant, oscillator.damping


def euler(oscillator, force, x0, v0, t, h):
    """Explicit (forward) Euler: both x and v are advanced with values from the start of the step. Order 1."""
    m, k, b = _constants(oscillator)
    fs, fm, fe = step_values(force, t)
    x, v = [float(x0)], [float(v0)]
    xi, vi = x[0], v[0]
    for i in range(len(t) - 1):
        a = (fs[i] - b * vi - k * xi) / m
        xi, vi = xi + h * vi, vi + h * a
        x.append(xi)
        v.append(vi)
    return Trajectory(t, np.array(x), np.array(v))


def improved_euler(oscillator, force, x0, v0, t, h):
    """The notebook's 'Improved Euler': Euler with the h²a/2 Taylor term added to x only. Still order 1."""
    m, k, b = _constants(oscillator)
    fs, fm, fe = step_values(force, t)
    x, v = [float(x0)], [float(v0)]
    xi, vi = x[0], v[0]
    for i in range(len(t) - 1):
        a = (fs[i] - b * vi - k * xi) / m
        xi, vi = xi + h * vi + 0.5 * h * h * a, vi + h * a
        x.append(xi)
        v.append(vi)
    return Trajectory(t, np.array(x), np.array(v))


def heun(oscillator, force, x0, v0, t, h):
    """Heun's method (the standard 'improved Euler', a 2nd-order Runge-Kutta): Euler predictor, trapezoidal corrector."""
    m, k, b = _constants(oscillator)
    fs, fm, fe = step_values(force, t)
    x, v = [float(x0)], [float(v0)]
    xi, vi = x[0], v[0]
    for i in range(len(t) - 1):
        a0 = (fs[i] - b * vi - k * xi) / m
        xp, vp = xi + h * vi, vi + h * a0
        a1 = (fe[i] - b * vp - k * xp) / m
        xi, vi = xi + 0.5 * h * (vi + vp), vi + 0.5 * h * (a0 + a1)
        x.append(xi)
        v.append(vi)
    return Trajectory(t, np.array(x), np.array(v))


def euler_cromer(oscillator, force, x0, v0, t, h):
    """Euler-Cromer (semi-implicit Euler): update v first, then x with the new v. Order 1, symplectic for b = 0."""
    m, k, b = _constants(oscillator)
    fs, fm, fe = step_values(force, t)
    x, v = [float(x0)], [float(v0)]
    xi, vi = x[0], v[0]
    for i in range(len(t) - 1):
        vi = vi + h * (fs[i] - b * vi - k * xi) / m
        xi = xi + h * vi
        x.append(xi)
        v.append(vi)
    return Trajectory(t, np.array(x), np.array(v))


def verlet(oscillator, force, x0, v0, t, h):
    """
    Velocity Verlet. The damping force is taken at the end of the step implicitly,

        v_half  = v_n + (h/2) a_n
        x_n+1   = x_n + h v_half
        v_n+1   = v_half + (h/2) a(x_n+1, v_n+1, t_n+1)   (solved for v_n+1, which is linear)

    so the method stays 2nd order with damping and is self-starting. Symplectic for b = 0.
    """
    m, k, b = _constants(oscillator)
    fs, fm, fe = step_values(force, t)
    x, v = [float(x0)], [float(v0)]
    xi, vi = x[0], v[0]
    a = (fs[0] - b * vi - k * xi) / m if len(fs) else 0.0
    implicit = 1 + 0.5 * h * b / m
    for i in range(len(t) - 1):
        v_half = vi + 0.5 * h * a
        xi = xi + h * v_half
        vi = (v_half + 0.5 * h * (fe[i] - k * xi) / m) / implicit
        a = ((fs[i + 1] if i + 1 < len(fs) else fe[i]) - b * vi - k * xi) / m
        x.append(xi)
        v.append(vi)
    return Trajectory(t, np.array(x), np.array(v))


def rk4(oscillator, force, x0, v0, t, h):
    """Classical fixed-step 4th-order Runge-Kutta. Order 4."""
    m, k, b = _constants(oscillator)
    fs, fm, fe = step_values(force, t)
    x, v = [float(x0)], [float(v0)]
    xi, vi = x[0], v[0]
    for i in range(len(t) - 1):
        k1x, k1v = vi, (fs[i] - b * vi - k * xi) / m
        x2, v2 = xi + 0.5 * h * k1x, vi + 0.5 * h * k1v
        k2x, k2v = v2, (fm[i] - b * v2 - k * x2) / m
        x3, v3 = xi + 0.5 * h * k2x, vi + 0.5 * h * k2v
        k3x, k3v = v3, (fm[i] - b * v3 - k * x3) / m
        x4, v4 = xi + h * k3x, vi + h * k3v
        k4x, k4v = v4, (fe[i] - b * v4 - k * x4) / m
        xi = xi + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        vi = vi + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        x.append(xi)
        v.append(vi)
    return Trajectory(t, np.array(x), np.array(v))


def scipy_rk45(oscillator, force, x0, v0, t, h, rtol=1e-3, atol=1e-6):
    """
    SciPy's adaptive RK45 (Dormand-Prince), as used in the notebook. It picks its own steps to meet
    rtol/atol, so h only sets the output times. The integration is restarted at force discontinuities.
    """
    m, k, b = _constants(oscillator)

    def rhs(time, y):
        return [y[1], (float(force(time)) - b * y[1] - k * y[0]) / m]

    t_end = t[-1]
    edges = sorted({0.0, t_end, *(p for p in force.breakpoints() if 0 < p < t_end)})
    x, v = np.empty_like(t), np.empty_like(t)
    state = [float(x0), float(v0)]
    for start, end in zip(edges[:-1], edges[1:]):
        inside = (t >= start) & (t <= end)
        t_eval = np.union1d(t[inside], [end])
        solution = solve_ivp(rhs, (start, end), state, method='RK45', t_eval=t_eval, rtol=rtol, atol=atol)
        if not solution.success:
            raise RuntimeError(f'RK45 failed: {solution.message}')
        x[inside] = np.interp(t[inside], solution.t, solution.y[0])  # exact match: t[inside] ⊂ solution.t
        v[inside] = np.interp(t[inside], solution.t, solution.y[1])
        state = solution.y[:, -1].tolist()
    return Trajectory(t, x, v)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    function: object
    order: float  # global order of accuracy (nan for the adaptive method)
    description: str


METHODS = {method.key: method for method in [
    Method('euler', 'Euler', euler, 1, 'x and v both advanced with values at the start of the step.'),
    Method('improved_euler', 'Improved Euler (notebook)', improved_euler, 1,
           'Euler plus the h²a/2 term in x only, as in the notebook. The velocity update is still Euler, so it is first order.'),
    Method('heun', 'Heun (RK2)', heun, 2, 'Standard improved Euler: Euler predictor followed by a trapezoidal corrector.'),
    Method('euler_cromer', 'Euler-Cromer', euler_cromer, 1,
           'Semi-implicit Euler: new v first, then x with the new v. Symplectic when b = 0, so energy does not drift.'),
    Method('verlet', 'Velocity Verlet', verlet, 2,
           'Half-step velocity update with the damping force treated implicitly. Symplectic when b = 0.'),
    Method('rk4', 'Runge-Kutta 4', rk4, 4, 'Classical fixed-step fourth-order Runge-Kutta.'),
    Method('rk45', 'SciPy RK45 (adaptive)', scipy_rk45, math.nan,
           'scipy.integrate.solve_ivp with adaptive steps. Accuracy is set by the tolerances, not by h.'),
]}


def integrate(method, oscillator, x0, v0, total_time, step, force=None, rtol=1e-3, atol=1e-6):
    """Integrate with a method key from METHODS from t = 0 to total_time."""
    if method not in METHODS:
        raise ValueError(f'Unknown method {method!r}. Choose from {", ".join(METHODS)}.')
    force = force or NoForce()
    t, h = time_grid(total_time, step)
    if method == 'rk45':
        return scipy_rk45(oscillator, force, x0, v0, t, h, rtol=rtol, atol=atol)
    return METHODS[method].function(oscillator, force, x0, v0, t, h)


def rk4_sinusoidal_batch(masses, springs, dampings, amplitude, omegas, phase, x0, v0, total_time, step,
                         window_start):
    """
    RK4 for many sinusoidally driven oscillators at once (arrays broadcast together).
    Returns max |x| over t ≥ window_start for each oscillator.
    """
    t, h = time_grid(total_time, step)
    m, k, b, omega = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (masses, springs, dampings, omegas)))
    x = np.full(m.shape, float(x0))
    v = np.full(m.shape, float(v0))
    peak = np.zeros(m.shape)

    def accel(xx, vv, time):
        return (amplitude * np.cos(omega * time + phase) - b * vv - k * xx) / m

    for i in range(len(t) - 1):
        time = t[i]
        k1x, k1v = v, accel(x, v, time)
        k2x, k2v = v + 0.5 * h * k1v, accel(x + 0.5 * h * k1x, v + 0.5 * h * k1v, time + 0.5 * h)
        k3x, k3v = v + 0.5 * h * k2v, accel(x + 0.5 * h * k2x, v + 0.5 * h * k2v, time + 0.5 * h)
        k4x, k4v = v + h * k3v, accel(x + h * k3x, v + h * k3v, time + h)
        x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if t[i + 1] >= window_start:
            np.maximum(peak, np.abs(x), out=peak)
    return peak
