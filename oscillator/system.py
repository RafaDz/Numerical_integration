"""
The damped mass-spring system  m x'' + b x' + k x = F(t)  and its exact solutions.

The free response is written in a single form that covers every damping regime,

    x(t) = e^(-γt) [ x0 C(t) + (v0 + γ x0) S(t) ]
    v(t) = e^(-γt) [ v0 C(t) - (γ v0 + ω0² x0) S(t) ]

with γ = b/2m, ω0² = k/m and

    underdamped  (ω0 > γ):  C = cos(ωd t),  S = sin(ωd t)/ωd,  ωd = √(ω0² - γ²)
    critical     (ω0 = γ):  C = 1,          S = t
    overdamped   (ω0 < γ):  C = cosh(r t),  S = sinh(r t)/r,   r  = √(γ² - ω0²)

The overdamped case is evaluated with exponentials of (r - γ)t ≤ 0 so it cannot overflow.
"""

import math
from dataclasses import dataclass

import numpy as np

from .forces import NoForce, PulseForce, SinusoidalForce

# Relative tolerance on |ω0² - γ²| / ω0² below which the system is treated as critically damped
CRITICAL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Oscillator:
    """A damped mass-spring system. Units: kg, N/m (= kg/s²), kg/s."""

    mass: float
    spring_constant: float
    damping: float

    def __post_init__(self):
        for name in ('mass', 'spring_constant', 'damping'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number.')
        if self.mass <= 0:
            raise ValueError('Mass must be positive.')
        if self.spring_constant <= 0:
            raise ValueError('Spring constant must be positive.')
        if self.damping < 0:
            raise ValueError('Damping coefficient cannot be negative.')

    # --- Derived quantities -----------------------------------------------------------
    @property
    def natural_frequency(self):
        """Undamped angular frequency ω0 = √(k/m), in rad/s."""
        return math.sqrt(self.spring_constant / self.mass)

    @property
    def gamma(self):
        """Decay rate of the amplitude, γ = b/2m, in 1/s."""
        return self.damping / (2 * self.mass)

    @property
    def critical_damping(self):
        """Damping coefficient for critical damping, b_cr = 2√(km)."""
        return 2 * math.sqrt(self.spring_constant * self.mass)

    @property
    def damping_ratio(self):
        """ζ = b / b_cr (ζ < 1 underdamped, ζ = 1 critical, ζ > 1 overdamped)."""
        return self.damping / self.critical_damping

    @property
    def quality_factor(self):
        """Q = m ω0 / b = 1/(2ζ); infinite without damping."""
        return math.inf if self.damping == 0 else self.mass * self.natural_frequency / self.damping

    @property
    def regime(self):
        if self.damping == 0:
            return 'undamped'
        s = self._discriminant
        if abs(s) <= CRITICAL_TOLERANCE * self.natural_frequency ** 2:
            return 'critically damped'
        return 'underdamped' if s > 0 else 'overdamped'

    @property
    def damped_frequency(self):
        """ωd = √(ω0² - γ²) for an underdamped system, otherwise nan."""
        return math.sqrt(self._discriminant) if self.regime in ('undamped', 'underdamped') else math.nan

    @property
    def decay_time(self):
        """Time constant of the slowest-decaying free solution (inf without damping)."""
        if self.damping == 0:
            return math.inf
        s = self._discriminant
        if s >= 0 or self.regime == 'critically damped':
            return 1 / self.gamma
        return 1 / (self.gamma - math.sqrt(-s))

    @property
    def resonance_frequency(self):
        """Driving frequency of the largest steady-state amplitude, √(ω0² - 2γ²); nan if there is no peak."""
        value = self.natural_frequency ** 2 - 2 * self.gamma ** 2
        return math.sqrt(value) if value > 0 else math.nan

    @property
    def _discriminant(self):
        return self.natural_frequency ** 2 - self.gamma ** 2

    def summary(self):
        return {
            'natural_frequency': self.natural_frequency,
            'natural_period': 2 * math.pi / self.natural_frequency,
            'gamma': self.gamma,
            'critical_damping': self.critical_damping,
            'damping_ratio': self.damping_ratio,
            'quality_factor': self.quality_factor,
            'regime': self.regime,
            'damped_frequency': self.damped_frequency,
            'decay_time': self.decay_time,
            'resonance_frequency': self.resonance_frequency,
        }

    # --- Dynamics ---------------------------------------------------------------------
    def acceleration(self, x, v, force=0.0):
        return (force - self.damping * v - self.spring_constant * x) / self.mass

    def energy(self, x, v):
        """Mechanical energy ½mv² + ½kx² (J)."""
        return 0.5 * self.mass * np.square(v) + 0.5 * self.spring_constant * np.square(x)

    def steady_state_amplitude(self, amplitude, omega):
        """Amplitude of the steady-state response to F0 cos(ωt + φ)."""
        omega = np.asarray(omega, dtype=float)
        k, m, b = self.spring_constant, self.mass, self.damping
        with np.errstate(divide='ignore'):
            return np.abs(amplitude) / np.hypot(k - m * omega ** 2, b * omega)

    def phase_lag(self, omega):
        """Phase δ by which the steady-state response lags the driving force, in [0, π]."""
        omega = np.asarray(omega, dtype=float)
        return np.arctan2(self.damping * omega, self.spring_constant - self.mass * omega ** 2)

    # --- Exact solutions --------------------------------------------------------------
    def free_response(self, x0, v0, t):
        """Exact (x, v) at times t ≥ 0 without an external force."""
        t = np.asarray(t, dtype=float)
        gamma, w0_sq, s = self.gamma, self.natural_frequency ** 2, self._discriminant
        regime = self.regime
        if regime == 'critically damped':
            decay = np.exp(-gamma * t)
            c, sn = decay, t * decay
        elif s > 0:
            wd = math.sqrt(s)
            decay = np.exp(-gamma * t)
            c, sn = decay * np.cos(wd * t), decay * np.sin(wd * t) / wd
        else:
            r = math.sqrt(-s)
            slow = np.exp((r - gamma) * t)          # r < γ, so this never overflows
            fast = np.exp(-(r + gamma) * t)
            c = 0.5 * (slow + fast)
            sn = -slow * np.expm1(-2 * r * t) / (2 * r)  # = e^(-γt) sinh(rt)/r, accurate for small r
        x = x0 * c + (v0 + gamma * x0) * sn
        v = v0 * c - (gamma * v0 + w0_sq * x0) * sn
        return x, v

    def exact_response(self, x0, v0, t, force=None):
        """Exact (x, v) at times t ≥ 0 for no force, a rectangular pulse or a sinusoidal force."""
        force = force or NoForce()
        t = np.asarray(t, dtype=float)
        if isinstance(force, NoForce):
            return self.free_response(x0, v0, t)
        if isinstance(force, PulseForce):
            return self._pulse_response(x0, v0, t, force)
        if isinstance(force, SinusoidalForce):
            return self._sinusoidal_response(x0, v0, t, force)
        raise TypeError(f'No exact solution for {type(force).__name__}.')

    def _pulse_response(self, x0, v0, t, force):
        # While the force acts, x - F/k obeys the free equation; before and after it, x does.
        start, end = force.start, force.start + force.duration
        shift = force.amplitude / self.spring_constant
        x1, v1 = self.free_response(x0, v0, start)
        x2, v2 = self.free_response(x1 - shift, v1, force.duration)
        x2 = x2 + shift

        x, v = np.empty_like(t), np.empty_like(t)
        before, during, after = t < start, (t >= start) & (t < end), t >= end
        x[before], v[before] = self.free_response(x0, v0, t[before])
        x[during], v[during] = self.free_response(x1 - shift, v1, t[during] - start)
        x[during] += shift
        x[after], v[after] = self.free_response(x2, v2, t[after] - end)
        return x, v

    def _sinusoidal_response(self, x0, v0, t, force):
        f0, omega, phi = force.amplitude, force.angular_frequency, force.phase
        w0 = self.natural_frequency
        if self.damping == 0 and math.isclose(omega, w0, rel_tol=1e-12):
            # Undamped resonance: the particular solution grows linearly in time
            c = f0 / (2 * self.mass * w0)
            xp = lambda tt: c * tt * np.sin(w0 * tt + phi)
            vp = lambda tt: c * (np.sin(w0 * tt + phi) + tt * w0 * np.cos(w0 * tt + phi))
        else:
            amplitude = f0 / math.hypot(self.spring_constant - self.mass * omega ** 2, self.damping * omega)
            delta = float(self.phase_lag(omega))
            xp = lambda tt: amplitude * np.cos(omega * tt + phi - delta)
            vp = lambda tt: -amplitude * omega * np.sin(omega * tt + phi - delta)
        xh, vh = self.free_response(x0 - float(xp(0.0)), v0 - float(vp(0.0)), t)
        return xh + xp(t), vh + vp(t)
