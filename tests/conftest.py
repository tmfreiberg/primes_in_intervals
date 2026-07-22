"""Shared fixtures: small analyzed datasets, computed once per session."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import primes_in_intervals as pii

N0 = int(np.exp(12))
CHECKPOINTS = list(range(N0 - 2000, N0 + 2001, 100))
H0 = 30


@pytest.fixture(scope="session")
def overlap_dataset():
    """Return a flat overlap dataset over 41 checkpoints around e^12."""
    return pii.intervals(list(CHECKPOINTS), H0, "overlap")


@pytest.fixture(scope="session")
def disjoint_dataset():
    return pii.intervals(list(CHECKPOINTS), H0, "disjoint")


@pytest.fixture(scope="session")
def prime_start_dataset():
    return pii.intervals(list(CHECKPOINTS), H0, "prime_start")


@pytest.fixture()
def analyzed_overlap(overlap_dataset):
    """Return a fresh analyzed copy (analyze modifies in place)."""
    ds = copy.deepcopy(overlap_dataset)
    pii.analyze(ds)
    return ds


@pytest.fixture()
def nested_overlap(overlap_dataset):
    """Return a fresh analyzed nested dataset."""
    ds = pii.nest(copy.deepcopy(overlap_dataset))
    pii.analyze(ds)
    return ds
