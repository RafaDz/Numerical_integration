"""External driving forces F(t)."""

import math
from dataclasses import dataclass

import numpy as np


def step_values(force, t):
    """
    Force at the start, middle and end of each step of the grid t, as lists.

    For a piecewise-constant force (one with breakpoints) all three values are the force averaged over the
    step, so every step receives the exact impulse even when a jump falls inside it. Away from the jumps this
    is just the constant value, so fixed-step methods do not sample the force exactly at a discontinuity.
    """
    t = np.asarray(t, dtype=float)
    if force.breakpoints():
        values = np.asarray(force.step_average(t[:-1], t[1:]), dtype=float).tolist()
        return values, values, values
    middle = np.asarray(force(0.5 * (t[:-1] + t[1:])), dtype=float)
    start = np.asarray(force(t[:-1]), dtype=float)
    end = np.asarray(force(t[1:]), dtype=float)
    return start.tolist(), middle.tolist(), end.tolist()


@dataclass(frozen=True)
class NoForce:
    kind = 'none'

    def __call__(self, t):
        return np.zeros_like(np.asarray(t, dtype=float))

    def breakpoints(self):
        """Times at which the force is discontinuous."""
        return ()


@dataclass(frozen=True)
class PulseForce:
    """A constant force F that acts for `duration` seconds from time `start` (a rectangular push)."""

    amplitude: float
    start: float
    duration: float
    kind = 'pulse'

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.amplitude, self.start, self.duration)):
            raise ValueError('Pulse parameters must be finite numbers.')
        if self.start < 0:
            raise ValueError('Pulse start time cannot be negative.')
        if self.duration <= 0:
            raise ValueError('Pulse duration must be positive.')

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        return np.where((t >= self.start) & (t < self.start + self.duration), float(self.amplitude), 0.0)

    def breakpoints(self):
        return (self.start, self.start + self.duration)

    def step_average(self, t0, t1):
        """Average force over each interval [t0, t1]."""
        overlap = np.clip(np.minimum(t1, self.start + self.duration) - np.maximum(t0, self.start), 0, None)
        return self.amplitude * overlap / (np.asarray(t1) - np.asarray(t0))


@dataclass(frozen=True)
class SinusoidalForce:
    """F(t) = F0 cos(ωt + φ)."""

    amplitude: float
    angular_frequency: float
    phase: float = 0.0
    kind = 'sinusoidal'

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.amplitude, self.angular_frequency, self.phase)):
            raise ValueError('Sinusoidal force parameters must be finite numbers.')
        if self.angular_frequency <= 0:
            raise ValueError('Driving angular frequency must be positive.')

    def __call__(self, t):
        return self.amplitude * np.cos(self.angular_frequency * np.asarray(t, dtype=float) + self.phase)

    def breakpoints(self):
        return ()
