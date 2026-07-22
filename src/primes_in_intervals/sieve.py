"""Prime generation and elementary prime counting.

The heart of this module is :func:`postponed_sieve`, an infinite generator of
primes due to Will Ness (building on code by David Eppstein and Alex Martelli).
It is an incremental sieve of Eratosthenes that postpones adding a prime's
multiples to the sieve dictionary until the prime's square is reached, so the
dictionary holds one entry per prime below the square root of the current
candidate.  This allows prime enumeration far beyond the memory limits of a
preallocated array sieve.

The helpers :func:`next_prime` and :func:`prime_pi` are deliberately naive
(each call runs a fresh sieve from scratch).  They exist to provide independent
ground truth when testing the interval-counting functions in
:mod:`primes_in_intervals.intervals`, not for production use.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import count

__all__ = ["next_prime", "postponed_sieve", "prime_pi"]


def postponed_sieve() -> Iterator[int]:
    """Yield the primes 2, 3, 5, 7, 11, ... indefinitely.

    This is Will Ness's postponed sieve, transcribed verbatim from
    https://stackoverflow.com/questions/2211990/ (see ideone.com/aVndFM), where
    the original code is credited to David Eppstein and Alex Martelli
    (ActiveState recipe, 2002).  See that page for a discussion of who
    contributed what, and of the algorithm's time and space complexity.

    The generator maintains a dictionary ``sieve`` mapping each known composite
    to an iterator over further multiples of one of its prime factors.  A
    separate, recursively created base prime supply ``ps`` provides the primes
    whose squares delimit when new multiples must start being tracked; a
    prime's multiples are only entered into the dictionary once the candidate
    sequence reaches its square (hence "postponed"), keeping the dictionary
    small.

    Yields
    ------
    int
        The primes in increasing order, without end.

    Examples
    --------
    >>> P = postponed_sieve()
    >>> [next(P) for _ in range(10)]
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    yield 2
    yield 3
    yield 5
    yield 7
    sieve: dict[int, Iterator[int]] = {}
    ps = postponed_sieve()  # a separate base Primes Supply:
    p = next(ps) and next(ps)  # (3) a Prime to add to dict
    q = p * p  # (9) its sQuare
    for r in count(9, 2):  # the Candidate
        if r in sieve:  # r's a multiple of some base prime
            s = sieve.pop(r)  # i.e. a composite; or
        elif r < q:
            yield r  # a prime
            continue
        else:  # (r == q): or the next base prime's square:
            s = count(q + 2 * p, 2 * p)  # (9 + 6, by 6 : 15, 21, 27, 33, ...)
            p = next(ps)  # (5)
            q = p * p  # (25)
        for m in s:  # the next multiple
            if m not in sieve:  # no duplicates
                break
        sieve[m] = s  # original test entry: ideone.com/WFv4f


def next_prime(a: int) -> int:
    """Return the first prime strictly greater than ``a``.

    Runs a fresh sieve from scratch on every call: fine for testing, wasteful
    otherwise.

    Parameters
    ----------
    a : int
        The threshold.

    Returns
    -------
    int
        The smallest prime ``p`` with ``p > a``.

    Examples
    --------
    >>> next_prime(100), next_prime(101)
    (101, 103)
    """
    primes = postponed_sieve()
    p = next(primes)
    while p <= a:
        p = next(primes)
    return p


def prime_pi(x: int, y: int) -> int:
    """Count the primes in the half-open interval ``(x, y]``.

    In the usual notation, this is ``pi(y) - pi(x)``.  Runs a fresh sieve from
    scratch on every call: fine for testing, wasteful otherwise.

    Parameters
    ----------
    x : int
        Lower endpoint (excluded).
    y : int
        Upper endpoint (included).

    Returns
    -------
    int
        The number of primes ``p`` with ``x < p <= y``.

    Examples
    --------
    >>> prime_pi(1, 100), prime_pi(1, 101)
    (25, 26)
    """
    primes = postponed_sieve()
    c = 0
    p = next(primes)
    while p <= x:
        p = next(primes)
    while p <= y:
        c += 1
        p = next(primes)
    return c
