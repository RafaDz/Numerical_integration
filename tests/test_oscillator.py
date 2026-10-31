import math

import numpy as np
import pytest

from oscillator import (METHODS, NOTEBOOK_DEFAULTS, NoForce, Oscillator, PulseForce, SinusoidalForce, compare_methods,
                        convergence_study, integrate, resonance_curves, time_grid)
from oscillator.analysis import aligned_steps, half_power_quality_factor

M, K = NOTEBOOK_DEFAULTS['mass'], NOTEBOOK_DEFAULTS['spring_constant']
B_CRIT = 2 * math.sqrt(K * M)
DAMPINGS = [0.0, 0.1, 2.0, B_CRIT / 2, B_CRIT, B_CRIT * (1 + 1e-6), 2 * B_CRIT]
FORCES = [NoForce(), PulseForce(2.5, 2.0, 3.0), SinusoidalForce(2.5, 0.9, 0.3)]


def residual(oscillator, force, x0, v0, t_end=20.0, n=20001):
    """Largest residual of m x'' + b x' + k x - F (x'' from finite differences of the exact v), relative to the terms."""
    t = np.linspace(0, t_end, n)
    x, v = oscillator.exact_response(x0, v0, t, force)
    a = np.gradient(v, t, edge_order=2)
    r = oscillator.mass * a + oscillator.damping * v + oscillator.spring_constant * x - force(t)
    away = np.ones_like(t, dtype=bool)
    for point in force.breakpoints():  # the derivative is not defined at a jump in the force
        away &= np.abs(t - point) > 3 * (t[1] - t[0])
    scale = np.max(np.abs(oscillator.spring_constant * x)) + np.max(np.abs(oscillator.damping * v)) \
        + np.max(np.abs(force(t))) + 1e-300
    return np.max(np.abs(r[away])) / scale, x, v


@pytest.mark.parametrize('b', DAMPINGS)
@pytest.mark.parametrize('force', FORCES, ids=lambda f: f.kind)
def test_exact_solution_satisfies_equation_and_initial_conditions(b, force):
    oscillator = Oscillator(M, K, b)
    error, x, v = residual(oscillator, force, 0.3, -1.0)
    assert error < 1e-5
    assert x[0] == pytest.approx(0.3, abs=1e-12)
    assert v[0] == pytest.approx(-1.0, abs=1e-12)


def test_regimes_and_derived_quantities():
    assert Oscillator(M, K, 0).regime == 'undamped'
    assert Oscillator(M, K, 0.1).regime == 'underdamped'
    assert Oscillator(M, K, B_CRIT).regime == 'critically damped'
    assert Oscillator(M, K, 2 * B_CRIT).regime == 'overdamped'
    o = Oscillator(M, K, 0.1)
    assert o.critical_damping == pytest.approx(6.36, abs=0.01)          # notebook value
    assert o.quality_factor == pytest.approx(31.8, abs=0.05)            # notebook value
    assert o.natural_frequency == pytest.approx(0.5695, abs=1e-4)
    assert Oscillator(M, K, B_CRIT).damping_ratio == pytest.approx(1)


def test_near_critical_solution_is_continuous():
    t = np.linspace(0, 30, 301)
    x_crit, _ = Oscillator(M, K, B_CRIT).free_response(0, -1, t)
    for factor in (1 - 1e-7, 1 + 1e-7):
        x, _ = Oscillator(M, K, B_CRIT * factor).free_response(0, -1, t)
        assert np.max(np.abs(x - x_crit)) < 1e-6


def test_overdamped_solution_does_not_overflow():
    x, v = Oscillator(1.0, 1.0, 1e4).free_response(1.0, 0.0, np.array([0, 1e3, 1e6]))
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(v))


def test_undamped_resonance_grows_linearly():
    o = Oscillator(M, K, 0)
    force = SinusoidalForce(1.0, o.natural_frequency)
    error, x, _ = residual(o, force, 0, 0, t_end=200)
    assert error < 1e-5
    assert np.max(np.abs(x[-2000:])) > 10 * np.max(np.abs(x[:2000]))


def test_time_grid_ends_exactly_at_total_time():
    t, h = time_grid(1.0, 0.3)
    assert t[-1] == 1.0 and len(t) == 4 and h == pytest.approx(1 / 3)
    t, h = time_grid(300, 0.1)
    assert len(t) == 3001 and h == pytest.approx(0.1)


@pytest.mark.parametrize('force', FORCES, ids=lambda f: f.kind)
@pytest.mark.parametrize('b', [0.0, 0.5, 2 * B_CRIT])
def test_observed_orders(force, b):
    study = convergence_study(Oscillator(M, K, b), 0, -1, 30, [0.1, 0.05, 0.025, 0.0125], force=force)
    for key, method in METHODS.items():
        if key == 'rk45':
            continue
        expected = method.order
        if key == 'euler_cromer' and b == 0 and isinstance(force, NoForce):
            expected = 2  # with x0 = 0 and b = 0 the positions coincide with Verlet's
        assert study['orders'][key] == pytest.approx(expected, abs=0.15), key


def test_pulse_between_grid_points_keeps_second_order():
    steps = np.geomspace(0.0071, 0.093, 6)  # the pulse edges never fall on the grid
    study = convergence_study(Oscillator(M, K, 0.5), 0, -1, 30, steps, methods=['heun', 'verlet', 'rk4'],
                              force=PulseForce(2.5, 2.013, 3.07), align=False)
    assert study['orders']['heun'] == pytest.approx(2, abs=0.2)
    assert study['orders']['verlet'] == pytest.approx(2, abs=0.2)
    assert study['orders']['rk4'] > 1.7


def test_aligned_steps_put_pulse_edges_on_grid():
    steps = aligned_steps(150, [20, 23], np.geomspace(0.005, 0.2, 10))
    for h in steps:
        for point in (20, 23, 150):
            assert point / h == pytest.approx(round(point / h), abs=1e-9)
    study = convergence_study(Oscillator(M, K, 3.178), 0, -1, 50, np.geomspace(0.01, 0.2, 6), methods=['rk4'],
                              force=PulseForce(2.5, 20, 3))
    assert study['orders']['rk4'] == pytest.approx(4, abs=0.2)


def test_pulse_step_average():
    force = PulseForce(2.0, 1.0, 2.0)
    np.testing.assert_allclose(force.step_average(np.array([0.0, 0.5, 1.5, 2.9]), np.array([0.5, 1.5, 2.5, 3.1])),
                               [0.0, 1.0, 2.0, 1.0])


def test_rk45_error_does_not_depend_on_step():
    study = convergence_study(Oscillator(M, K, 0.5), 0, -1, 50, [0.1, 0.05, 0.01], methods=['rk45'])
    errors = study['errors']['rk45']
    assert errors.max() / errors.min() < 1.1
    tight = convergence_study(Oscillator(M, K, 0.5), 0, -1, 50, [0.1], methods=['rk45'], rtol=1e-9, atol=1e-12)
    assert tight['errors']['rk45'][0] < errors.min() / 1000


def test_euler_cromer_conserves_energy_without_damping():
    o = Oscillator(M, K, 0)
    euler = integrate('euler', o, 0, -1, 300, 0.1)
    cromer = integrate('euler_cromer', o, 0, -1, 300, 0.1)
    e0 = o.energy(0, -1)
    assert o.energy(euler.x[-1], euler.v[-1]) > 2 * e0              # explicit Euler gains energy
    assert np.max(np.abs(o.energy(cromer.x, cromer.v) / e0 - 1)) < 0.05


def test_compare_methods_ranks_rk4_first_for_notebook_case():
    comparison = compare_methods(Oscillator(M, K, 0.1), 0, -1, 300, 0.1)
    assert comparison.ranking('rmse')[0] == 'rk4'
    assert comparison.ranking('rmse')[-1] == 'euler'
    assert len(comparison.t) == 3001


def test_resonance_curves_match_rk4_when_settled():
    w0 = math.sqrt(K / M)
    omegas = np.linspace(0.5 * w0, 1.5 * w0, 9)
    result = resonance_curves(M, K, [0.5, 2.0], 2.5, omegas, numeric=True, total_time=300, step=0.05)
    for curve in result['curves']:
        assert curve['settled']
        np.testing.assert_allclose(curve['numeric'], curve['exact'], rtol=1e-3)


def test_half_power_q_matches_formula_for_light_damping():
    o = Oscillator(M, K, 0.1)
    assert half_power_quality_factor(o) == pytest.approx(o.quality_factor, rel=0.01)
    assert math.isnan(half_power_quality_factor(Oscillator(M, K, 2 * B_CRIT)))


@pytest.mark.parametrize('args', [(0, K, 0.1), (M, -1, 0.1), (M, K, -0.1), (M, K, math.nan)])
def test_invalid_oscillator(args):
    with pytest.raises(ValueError):
        Oscillator(*args)
