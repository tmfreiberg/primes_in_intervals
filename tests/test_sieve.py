"""Tests for the prime generator and the counting helpers."""

from __future__ import annotations

from sympy import isprime, prime

import primes_in_intervals as pii


def test_first_primes():
    P = pii.postponed_sieve()
    assert [next(P) for _ in range(10)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_thousandth_prime_and_primality():
    P = pii.postponed_sieve()
    primes = [next(P) for _ in range(1000)]
    assert primes[-1] == prime(1000)  # 7919
    assert all(isprime(p) for p in primes[::37])
    assert primes == sorted(set(primes))


def test_next_prime_readme_values():
    assert (pii.next_prime(100), pii.next_prime(101)) == (101, 103)


def test_prime_pi_readme_values():
    assert (pii.prime_pi(1, 100), pii.prime_pi(1, 101)) == (25, 26)


def test_prime_pi_interval_convention():
    # (x, y]: endpoints excluded on the left, included on the right.
    assert pii.prime_pi(2, 3) == 1
    assert pii.prime_pi(3, 3) == 0
    assert pii.prime_pi(1, 2) == 1
