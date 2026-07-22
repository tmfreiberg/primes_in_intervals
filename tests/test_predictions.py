"""Tests for the prediction functions and the Montgomery-Soundararajan constant."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma

import primes_in_intervals as pii


class TestConstant:
    def test_value(self):
        # 1 - EulerGamma - log(2*pi) = 1 - 0.5772156... - 1.8378770... = -1.4150927...
        assert float(pii.MS) == pytest.approx(-1.4150927, abs=1e-6)


class TestBinomPmf:
    def test_sums_to_one(self):
        H, p = 30, 0.07
        total = sum(pii.binom_pmf(H, m, p) for m in range(H + 1))
        assert total == pytest.approx(1.0)

    def test_matches_direct_formula(self):
        from math import comb

        H, p = 12, 0.3
        for m in range(H + 1):
            assert pii.binom_pmf(H, m, p) == pytest.approx(
                comb(H, m) * p**m * (1 - p) ** (H - m)
            )

    def test_accepts_real_m(self):
        # Continuous in m for the plotted curves.
        val = pii.binom_pmf(30, 2.5, 0.07)
        assert 0 < val < 1


class TestFrei:
    def test_formula(self):
        H, m, t = 76, 5, 4.75
        Q2 = ((m - t) ** 2 - m) / 2
        expected = np.exp(-t) * (t**m / gamma(m + 1)) * (
            1 - ((np.log(H) - float(pii.MS)) / H) * Q2
        )
        assert float(pii.frei(H, m, t)) == pytest.approx(float(expected))

    def test_reduces_to_poisson_at_large_H(self):
        m, t = 4, 4.0
        poisson = np.exp(-t) * t**m / gamma(m + 1)
        assert float(pii.frei(10**9, m, t)) == pytest.approx(float(poisson), rel=1e-6)


class TestFreiAlt:
    def test_formula(self):
        H, m, t = 76, 5, 4.4
        Q1 = m - t
        Q2 = ((m - t) ** 2 - m) / 2
        expected = np.exp(-t) * (t**m / gamma(m + 1)) * (
            1 + (t / H) * Q1 - ((np.log(H) - float(pii.MS)) / H) * Q2
        )
        assert float(pii.frei_alt(H, m, t)) == pytest.approx(float(expected))

    def test_extra_term_is_the_difference(self):
        H, m, t = 50, 6, 5.0
        Q1 = m - t
        base = np.exp(-t) * t**m / gamma(m + 1)
        diff = float(pii.frei_alt(H, m, t)) - float(pii.frei(H, m, t))
        assert diff == pytest.approx(float(base * (t / H) * Q1))
