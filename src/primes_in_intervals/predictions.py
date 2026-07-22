"""Theoretical predictions for the distribution of primes in intervals.

Three probability estimates are compared against the empirical data.  With
``lambda = H * p`` for a prime density ``p``:

* :func:`binom_pmf` is the naive prediction from Cramér's model: the number of
  primes in an interval of length ``H`` is ``Binom(H, p)``.
* :func:`frei` is the project's refined prediction ``F(H, m, lambda)``: the
  Poisson weight ``exp(-lambda) lambda^m / m!`` corrected by a second-order
  term of size ``(log H - MS) / H`` times the polynomial
  ``Q_2 = ((m - lambda)^2 - m) / 2``, where :data:`MS` is the
  Montgomery-Soundararajan constant ``1 - gamma - log(2 pi)`` (``gamma`` the
  Euler-Mascheroni constant).  It is intended for overlapping intervals with
  ``lambda = H / (log N - 1)``, ``N`` the center of the sampled range.
* :func:`frei_alt` is the alternative form ``F*(H, m, lambda*)`` appropriate
  to the density ``lambda* = H / log N``: the same Poisson weight with the
  additional first-order correction ``(t / H) Q_1``, ``Q_1 = m - t``.

All three accept a real (non-integer) ``m`` so they can be drawn as smooth
curves over the histograms; :func:`binom_pmf` uses the generalized binomial
coefficient ``Gamma(H + 1) / (Gamma(m + 1) Gamma(H - m + 1))`` for the same
reason.
"""

from __future__ import annotations

import numpy as np
import sympy
from scipy.special import binom, gamma

__all__ = ["MS", "binom_pmf", "frei", "frei_alt"]

#: The "Montgomery-Soundararajan" constant 1 - gamma - log(2*pi).
MS = 1 - sympy.EulerGamma.evalf() - np.log(2 * (np.pi))


def binom_pmf(H, m, p):
    """Binomial probability of ``m`` successes in ``H`` trials with probability ``p``.

    This is the prediction of Cramér's model for the number of primes in an
    interval of length ``H``, with ``p`` the local density of primes.

    Parameters
    ----------
    H : int
        Number of trials (interval length).
    m : int, float, or array
        Number of successes; real values are allowed (the generalized binomial
        coefficient is used), so the curve can be plotted continuously.
    p : float
        Success probability.

    Returns
    -------
    float or array
        ``binom(H, m) * p**m * (1 - p)**(H - m)``.
    """
    return binom(H, m) * (p**m) * (1 - p) ** (H - m)


def frei(H, m, t):
    """Evaluate the prediction ``F(H, m, t)`` for overlapping intervals.

    The Poisson weight ``exp(-t) t^m / Gamma(m + 1)`` corrected at second
    order:

    ``F = exp(-t) * t^m / m! * (1 - ((log H - MS) / H) * Q_2)``

    with ``Q_2 = ((m - t)^2 - m) / 2`` and :data:`MS` the
    Montgomery-Soundararajan constant.  Intended with ``t = H * p`` and
    ``p = 1 / (log N - 1)``.

    Parameters
    ----------
    H : int
        Interval length.
    m : int, float, or array
        Number of primes; real values allowed for plotting.
    t : float
        The Poisson parameter ``lambda``.

    Returns
    -------
    float or array
        The predicted probability.
    """
    Q_2 = ((m - t) ** 2 - m) / 2
    return np.exp(-t) * (t**m / gamma(m + 1)) * (1 - ((np.log(H) - MS) / (H)) * Q_2)


def frei_alt(H, m, t):
    """Evaluate the alternative prediction ``F*(H, m, t)`` for density ``1 / log N``.

    As :func:`frei`, but with the additional first-order correction term
    ``(t / H) * Q_1``:

    ``F* = exp(-t) * t^m / m! * (1 + (t / H) * Q_1 - ((log H - MS) / H) * Q_2)``

    with ``Q_1 = m - t`` and ``Q_2 = ((m - t)^2 - m) / 2``.  Intended with
    ``t = H * p_alt`` and ``p_alt = 1 / log N``.

    Parameters
    ----------
    H : int
        Interval length.
    m : int, float, or array
        Number of primes; real values allowed for plotting.
    t : float
        The Poisson parameter ``lambda*``.

    Returns
    -------
    float or array
        The predicted probability.
    """
    Q_1 = m - t
    Q_2 = ((m - t) ** 2 - m) / 2
    return (
        np.exp(-t)
        * (t**m / gamma(m + 1))
        * (1 + (t / H) * Q_1 - ((np.log(H) - MS) / (H)) * Q_2)
    )
