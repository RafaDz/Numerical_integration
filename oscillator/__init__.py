"""Numerical integration of the damped, driven mass-spring system, checked against exact solutions."""

from .analysis import (compare_methods, convergence_study, error_metrics, half_power_quality_factor, make_force,
                       resonance_curves)
from .forces import NoForce, PulseForce, SinusoidalForce
from .integrators import METHODS, Trajectory, integrate, time_grid
from .system import Oscillator

# Values used in the original notebook
NOTEBOOK_DEFAULTS = {
    'mass': 5.58,             # kg
    'spring_constant': 1.81,  # N/m
    'x0': 0.0,                # m
    'v0': -1.0,               # m/s
    'dampings': [0.1, 0.5, 2.0],      # kg/s
    'total_times': [50.0, 150.0, 300.0],  # s
    'steps': [0.1, 0.05, 0.01],       # s
}

__all__ = ['METHODS', 'NOTEBOOK_DEFAULTS', 'NoForce', 'Oscillator', 'PulseForce', 'SinusoidalForce', 'Trajectory',
           'compare_methods', 'convergence_study', 'error_metrics', 'half_power_quality_factor', 'integrate',
           'make_force', 'resonance_curves', 'time_grid']
