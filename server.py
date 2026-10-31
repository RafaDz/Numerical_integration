"""
Web server for the numerical integration comparison.

Serves the HTML/CSS/JS front-end in web/ and a small JSON API that calls the oscillator package.
Uses only the Python standard library (plus numpy/scipy for the calculations).

Run with:  python server.py            (then open http://localhost:8000)
Options:   python server.py --port 8080 --open
"""

import argparse
import json
import math
import webbrowser
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from oscillator import (METHODS, NOTEBOOK_DEFAULTS, Oscillator, compare_methods, convergence_study, make_force,
                        resonance_curves, time_grid)
from oscillator.analysis import half_power_quality_factor

WEB_DIR = Path(__file__).parent / 'web'
MAX_STEPS = 300_000              # steps per method in a single run
MAX_CONVERGENCE_STEPS = 2_000_000  # total steps over all step sizes and methods in a convergence study
MAX_RESONANCE_WORK = 20_000_000  # RK4 steps x driving frequencies x dampings for numeric resonance curves
MAX_PLOT_POINTS = 4000
METRICS = ('mae', 'mse', 'rmse', 'max')


class ApiError(ValueError):
    """An invalid request; reported to the client with status 400."""


# --- JSON helpers ---------------------------------------------------------------------
def finite_or_none(value):
    """JSON has no inf/nan, so send them as null."""
    value = float(value)
    return value if math.isfinite(value) else None


def clean_list(values):
    array = np.asarray(values, dtype=float)
    if np.all(np.isfinite(array)):
        return array.tolist()
    return [value if math.isfinite(value) else None for value in array.tolist()]


def plot_indices(n_points):
    """At most MAX_PLOT_POINTS evenly spread indices (always including the first and last point)."""
    if n_points <= MAX_PLOT_POINTS:
        return np.arange(n_points)
    return np.unique(np.linspace(0, n_points - 1, MAX_PLOT_POINTS).round().astype(int))


# --- Parsing --------------------------------------------------------------------------
def parse_float(payload, name, low=-math.inf, high=math.inf, default=None, low_inclusive=True):
    value = payload.get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ApiError(f'{name} must be a number.') from error
    if not math.isfinite(value):
        raise ApiError(f'{name} must be a finite number.')
    if value > high or value < low or (value == low and not low_inclusive):
        bound = '>' if not low_inclusive else '≥'
        raise ApiError(f'{name} must be {bound} {low:g}' + (f' and ≤ {high:g}.' if math.isfinite(high) else '.'))
    return value


def parse_int(payload, name, low, high, default=None):
    try:
        value = int(payload.get(name, default))
    except (TypeError, ValueError) as error:
        raise ApiError(f'{name} must be an integer.') from error
    if not low <= value <= high:
        raise ApiError(f'{name} must be between {low} and {high}.')
    return value


def parse_system(data):
    if not isinstance(data, dict):
        raise ApiError('system must be an object with mass, spring_constant and damping.')
    try:
        return Oscillator(parse_float(data, 'mass'), parse_float(data, 'spring_constant'),
                          parse_float(data, 'damping'))
    except ValueError as error:
        raise ApiError(str(error)) from error


def parse_force(data):
    data = data or {'kind': 'none'}
    if not isinstance(data, dict):
        raise ApiError('force must be an object.')
    kind = data.get('kind', 'none')
    try:
        if kind == 'none':
            return make_force('none')
        if kind == 'pulse':
            return make_force('pulse', amplitude=parse_float(data, 'amplitude'), start=parse_float(data, 'start'),
                              duration=parse_float(data, 'duration'))
        if kind == 'sinusoidal':
            return make_force('sinusoidal', amplitude=parse_float(data, 'amplitude'),
                              angular_frequency=parse_float(data, 'angular_frequency'),
                              phase=parse_float(data, 'phase', default=0.0))
    except ValueError as error:
        raise ApiError(str(error)) from error
    raise ApiError(f'Unknown force type {kind!r}.')


def parse_methods(payload):
    methods = payload.get('methods') or []
    if not isinstance(methods, list) or not methods:
        raise ApiError('Choose at least one method.')
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ApiError(f'Unknown method(s): {", ".join(map(str, unknown))}.')
    return tuple(dict.fromkeys(methods))


def parse_run(payload):
    """Initial conditions, time range and tolerances shared by simulate and convergence."""
    return {
        'x0': parse_float(payload, 'x0', -1e6, 1e6),
        'v0': parse_float(payload, 'v0', -1e6, 1e6),
        'total_time': parse_float(payload, 'total_time', 0, 1e5, low_inclusive=False),
        'rtol': parse_float(payload, 'rtol', 1e-12, 0.1, default=1e-3),
    }


def system_summary(oscillator):
    return {key: (finite_or_none(value) if isinstance(value, float) else value)
            for key, value in oscillator.summary().items()}


# --- Endpoints ------------------------------------------------------------------------
def api_defaults(_payload=None):
    return {
        'methods': [{'key': m.key, 'label': m.label, 'order': finite_or_none(m.order), 'description': m.description}
                    for m in METHODS.values()],
        'notebook': NOTEBOOK_DEFAULTS,
    }


def api_system(payload):
    oscillator = parse_system(payload.get('system'))
    return {'summary': system_summary(oscillator), 'half_power_q': finite_or_none(half_power_quality_factor(oscillator))}


@lru_cache(maxsize=64)
def cached_simulation(oscillator, force, x0, v0, total_time, step, methods, rtol):
    comparison = compare_methods(oscillator, x0, v0, total_time, step, methods, force, rtol=rtol, atol=rtol * 1e-3)
    idx = plot_indices(len(comparison.t))
    results = []
    for result in comparison.results:
        with np.errstate(invalid='ignore', over='ignore'):
            error = result.x - comparison.exact_x
        results.append({
            'key': result.key,
            'label': result.label,
            'x': clean_list(result.x[idx]),
            'v': clean_list(result.v[idx]),
            'energy': clean_list(result.energy[idx]),
            'error': clean_list(error[idx]),
            'errors': {name: finite_or_none(value) for name, value in result.errors.items()},
            'final_energy_error': finite_or_none(result.final_energy_error),
            'runtime_ms': result.runtime_ms,
        })
    return {
        't': comparison.t[idx].tolist(),
        'step': comparison.step,
        'n_steps': len(comparison.t) - 1,
        'exact': {'x': clean_list(comparison.exact_x[idx]), 'v': clean_list(comparison.exact_v[idx]),
                  'energy': clean_list(comparison.exact_energy[idx])},
        'force': clean_list(comparison.force[idx]),
        'results': results,
        'ranking': {metric: comparison.ranking(metric) for metric in METRICS},
        'summary': system_summary(oscillator),
    }


def api_simulate(payload):
    """Run the chosen methods once and compare them with the exact solution."""
    oscillator = parse_system(payload.get('system'))
    force = parse_force(payload.get('force'))
    run = parse_run(payload)
    step = parse_float(payload, 'step', 0, run['total_time'], low_inclusive=False)
    methods = parse_methods(payload)
    if run['total_time'] / step > MAX_STEPS:
        raise ApiError(f'Total time / step must be at most {MAX_STEPS:,} steps.')
    return cached_simulation(oscillator, force, run['x0'], run['v0'], run['total_time'], step, methods, run['rtol'])


@lru_cache(maxsize=64)
def cached_convergence(oscillator, force, x0, v0, total_time, steps, methods, metric, rtol):
    study = convergence_study(oscillator, x0, v0, total_time, steps, methods, force, metric, rtol=rtol,
                              atol=rtol * 1e-3)
    return {
        'steps': study['steps'].tolist(),
        'errors': {key: clean_list(values) for key, values in study['errors'].items()},
        'orders': {key: finite_or_none(value) for key, value in study['orders'].items()},
        'metric': metric,
    }


def api_convergence(payload):
    """Error against step size over a log-spaced range of steps."""
    oscillator = parse_system(payload.get('system'))
    force = parse_force(payload.get('force'))
    run = parse_run(payload)
    h_min = parse_float(payload, 'h_min', 0, run['total_time'], low_inclusive=False)
    h_max = parse_float(payload, 'h_max', h_min, run['total_time'])
    if h_max <= h_min:
        raise ApiError('h_max must be larger than h_min.')
    n_points = parse_int(payload, 'n_points', 3, 30)
    metric = payload.get('metric', 'max')
    if metric not in METRICS:
        raise ApiError(f'metric must be one of {", ".join(METRICS)}.')
    methods = parse_methods(payload)
    steps = tuple(np.geomspace(h_min, h_max, n_points).tolist())
    work = sum(len(time_grid(run['total_time'], h)[0]) for h in steps) * len(methods)
    if work > MAX_CONVERGENCE_STEPS:
        raise ApiError(f'This study needs {work:,} steps; the limit is {MAX_CONVERGENCE_STEPS:,}. '
                       'Increase the smallest step or shorten the total time.')
    return cached_convergence(oscillator, force, run['x0'], run['v0'], run['total_time'], steps, methods, metric,
                              run['rtol'])


@lru_cache(maxsize=32)
def cached_resonance(mass, spring_constant, dampings, amplitude, omega_range, n_points, numeric, x0, v0,
                     total_time, step, phase):
    w0 = math.sqrt(spring_constant / mass)
    omegas = np.linspace(omega_range[0] * w0, omega_range[1] * w0, n_points)
    result = resonance_curves(mass, spring_constant, dampings, amplitude, omegas, numeric, x0, v0,
                              total_time, step, phase)
    curves = []
    for curve in result['curves']:
        oscillator = Oscillator(mass, spring_constant, curve['damping'])
        curves.append({
            'damping': curve['damping'],
            'exact': clean_list(curve['exact']),
            'numeric': clean_list(curve['numeric']) if numeric else None,
            'phase_lag': np.degrees(oscillator.phase_lag(omegas)).tolist(),
            'quality_factor': finite_or_none(curve['quality_factor']),
            'half_power_q': finite_or_none(curve['half_power_q']),
            'resonance_frequency': finite_or_none(curve['resonance_frequency']),
            'peak_amplitude': finite_or_none(curve['peak_amplitude']),
            'decay_time': finite_or_none(curve['decay_time']),
            'settled': bool(curve['settled']),
            'regime': oscillator.regime,
        })

    # Quality factor against damping, from the formula and from the width of the exact resonance peak
    b_crit = 2 * math.sqrt(spring_constant * mass)
    b_values = np.linspace(b_crit / 60, 2 * b_crit, 120)
    q_half = [half_power_quality_factor(Oscillator(mass, spring_constant, b), samples=20_001) for b in b_values]
    return {
        'omegas': omegas.tolist(),
        'natural_frequency': w0,
        'curves': curves,
        'q_curve': {'damping': b_values.tolist(), 'formula': (mass * w0 / b_values).tolist(),
                    'half_power': clean_list(q_half), 'critical_damping': b_crit},
    }


def api_resonance(payload):
    """Steady-state amplitude against driving frequency for several dampings."""
    mass = parse_float(payload, 'mass', 0, 1e6, low_inclusive=False)
    spring_constant = parse_float(payload, 'spring_constant', 0, 1e6, low_inclusive=False)
    raw = payload.get('dampings')
    if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
        raise ApiError('Give between 1 and 8 damping coefficients.')
    dampings = tuple(parse_float({'damping': b}, 'damping', 0, 1e6) for b in raw)
    if any(b == 0 for b in dampings):
        raise ApiError('Damping coefficients must be positive: without damping the resonance peak is infinite.')
    amplitude = parse_float(payload, 'amplitude', -1e6, 1e6)
    omega_min = parse_float(payload, 'omega_min', 0, 100, low_inclusive=False)
    omega_max = parse_float(payload, 'omega_max', omega_min, 100)
    if omega_max <= omega_min:
        raise ApiError('omega_max must be larger than omega_min.')
    n_points = parse_int(payload, 'n_points', 5, 400)
    numeric = bool(payload.get('numeric', False))
    total_time = parse_float(payload, 'total_time', 0, 1e5, default=300.0, low_inclusive=False)
    step = parse_float(payload, 'step', 0, total_time, default=0.05, low_inclusive=False)
    if numeric and total_time / step * n_points * len(dampings) > MAX_RESONANCE_WORK:
        raise ApiError('Numeric resonance curves need too many steps: use fewer frequencies, a larger step '
                       'or a shorter total time.')
    return cached_resonance(mass, spring_constant, dampings, amplitude, (omega_min, omega_max), n_points, numeric,
                            parse_float(payload, 'x0', -1e6, 1e6, default=0.0),
                            parse_float(payload, 'v0', -1e6, 1e6, default=0.0), total_time, step,
                            parse_float(payload, 'phase', -1e3, 1e3, default=0.0))


ROUTES = {
    ('GET', '/api/defaults'): api_defaults,
    ('POST', '/api/system'): api_system,
    ('POST', '/api/simulate'): api_simulate,
    ('POST', '/api/convergence'): api_convergence,
    ('POST', '/api/resonance'): api_resonance,
}


def plotly_bundle():
    """Path of the plotly.js bundle shipped with the plotly Python package, if installed."""
    try:
        import plotly
    except ImportError:
        return None
    path = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
    return path if path.exists() else None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_api(self, method):
        route = ROUTES.get((method, self.path.split('?')[0]))
        if route is None:
            self.send_json(404, {'error': 'Unknown API endpoint.'})
            return
        try:
            payload = {}
            if method == 'POST':
                length = int(self.headers.get('Content-Length', 0))
                payload = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(payload, dict):
                    raise ApiError('Request body must be a JSON object.')
            self.send_json(200, route(payload))
        except json.JSONDecodeError:
            self.send_json(400, {'error': 'Request body is not valid JSON.'})
        except ApiError as error:
            self.send_json(400, {'error': str(error)})
        except Exception as error:  # Report unexpected failures instead of dropping the connection
            self.log_error('Error handling %s: %r', self.path, error)
            self.send_json(500, {'error': f'Internal error: {error}'})

    def do_GET(self):
        if self.path.startswith('/api/'):
            self.handle_api('GET')
        elif self.path == '/vendor/plotly.min.js' and plotly_bundle():
            body = plotly_bundle().read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/javascript')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'max-age=86400')
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def do_POST(self):
        self.handle_api('POST')

    def log_message(self, format, *args):
        if self.path.startswith('/api/'):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description='Numerical integration of a damped mass-spring system.')
    parser.add_argument('--host', default='127.0.0.1', help='Address to listen on (default 127.0.0.1).')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on (default 8000).')
    parser.add_argument('--open', action='store_true', help='Open the page in a web browser.')
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f'http://{"localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host}:{args.port}'
    print(f'Damped oscillator integrators running at {url}  (Ctrl+C to stop)')
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
