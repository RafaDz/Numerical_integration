import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import server

SYSTEM = {'mass': 5.58, 'spring_constant': 1.81, 'damping': 0.5}
RUN = {'system': SYSTEM, 'x0': 0, 'v0': -1, 'total_time': 50, 'rtol': 1e-3}


def test_defaults_list_methods_and_notebook_values():
    data = server.api_defaults()
    assert [m['key'] for m in data['methods']] == ['euler', 'improved_euler', 'heun', 'euler_cromer', 'verlet', 'rk4', 'rk45']
    assert data['methods'][-1]['order'] is None  # adaptive: nan sent as null
    assert data['notebook']['mass'] == 5.58
    json.dumps(data)


def test_system_summary_sends_infinite_q_as_null():
    summary = server.api_system({'system': {**SYSTEM, 'damping': 0}})['summary']
    assert summary['regime'] == 'undamped'
    assert summary['quality_factor'] is None
    assert server.api_system({'system': SYSTEM})['summary']['regime'] == 'underdamped'


def test_simulate_returns_consistent_result():
    result = server.api_simulate({**RUN, 'step': 0.01, 'methods': ['euler', 'rk4'],
                                  'force': {'kind': 'pulse', 'amplitude': 2.5, 'start': 20, 'duration': 3}})
    assert result['n_steps'] == 5000
    assert len(result['t']) == server.MAX_PLOT_POINTS
    assert result['t'][0] == 0 and result['t'][-1] == 50
    assert result['ranking']['rmse'] == ['rk4', 'euler']
    assert all(len(r['x']) == len(result['t']) for r in result['results'])
    assert max(result['force']) == 2.5
    json.dumps(result)


def test_simulate_sends_blown_up_values_as_null():
    result = server.api_simulate({**RUN, 'system': {**SYSTEM, 'damping': 0}, 'total_time': 20000, 'step': 1.5,
                                  'methods': ['euler']})
    json.dumps(result, allow_nan=False)
    assert None in result['results'][0]['x']


@pytest.mark.parametrize('changes, message', [
    ({'methods': []}, 'at least one method'),
    ({'methods': ['magic']}, 'Unknown method'),
    ({'step': 0}, 'step'),
    ({'step': 1e-5}, 'at most'),
    ({'system': {**SYSTEM, 'mass': -1}}, 'Mass'),
    ({'force': {'kind': 'pulse', 'amplitude': 1, 'start': 1, 'duration': -1}}, 'duration'),
    ({'force': {'kind': 'laser'}}, 'Unknown force'),
])
def test_simulate_rejects_bad_requests(changes, message):
    payload = {**RUN, 'step': 0.1, 'methods': ['rk4'], **changes}
    with pytest.raises(server.ApiError, match=message):
        server.api_simulate(payload)


def test_convergence_reports_orders():
    result = server.api_convergence({**RUN, 'h_min': 0.01, 'h_max': 0.1, 'n_points': 4, 'metric': 'max',
                                     'methods': ['euler', 'verlet', 'rk4']})
    assert len(result['steps']) == 4
    assert result['orders']['euler'] == pytest.approx(1, abs=0.15)
    assert result['orders']['verlet'] == pytest.approx(2, abs=0.15)
    assert result['orders']['rk4'] == pytest.approx(4, abs=0.15)


def test_convergence_limits_work():
    with pytest.raises(server.ApiError, match='limit'):
        server.api_convergence({**RUN, 'total_time': 1000, 'h_min': 1e-4, 'h_max': 0.1, 'n_points': 10,
                                'methods': ['rk4']})


def test_resonance():
    result = server.api_resonance({'mass': 5.58, 'spring_constant': 1.81, 'dampings': [0.1, 2.0], 'amplitude': 2.5,
                                   'omega_min': 0.5, 'omega_max': 1.5, 'n_points': 11, 'numeric': True,
                                   'total_time': 200, 'step': 0.1})
    assert len(result['omegas']) == 11
    light, heavy = result['curves']
    assert light['quality_factor'] == pytest.approx(31.78, abs=0.01)
    assert not light['settled'] and heavy['settled']
    assert len(heavy['numeric']) == 11
    json.dumps(result, allow_nan=False)


def test_resonance_rejects_zero_damping():
    with pytest.raises(server.ApiError, match='positive'):
        server.api_resonance({'mass': 1, 'spring_constant': 1, 'dampings': [0], 'amplitude': 1,
                              'omega_min': 0.5, 'omega_max': 1.5, 'n_points': 11})


@pytest.fixture
def running_server():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()


def post(url, payload):
    request = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request) as response:
        return json.load(response)


def test_http_round_trip(running_server):
    with urlopen(running_server + '/') as response:
        assert b'Damped Oscillator' in response.read()
    with urlopen(running_server + '/app.js') as response:
        assert response.status == 200
    with urlopen(running_server + '/api/defaults') as response:
        assert json.load(response)['methods']
    result = post(running_server + '/api/simulate', {**RUN, 'step': 0.1, 'methods': ['rk4']})
    assert result['results'][0]['key'] == 'rk4'


def test_http_errors_are_json(running_server):
    with pytest.raises(HTTPError) as error:
        post(running_server + '/api/simulate', {**RUN, 'step': 0.1, 'methods': []})
    assert error.value.code == 400
    assert 'method' in json.load(error.value)['error']
