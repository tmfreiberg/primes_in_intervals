"""Tests for compare and winners."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import primes_in_intervals as pii


class TestCompare:
    def test_structure_and_values_flat(self, analyzed_overlap):
        ds = analyzed_overlap
        assert pii.compare(ds) is ds
        C = list(ds["data"].keys())
        A = C[0]
        H = ds["header"]["interval_length"]
        c = C[-1]
        N = (A + c) // 2
        p = 1 / (np.log(N) - 1)
        p_alt = 1 / np.log(N)
        multiplier = c - A  # overlap
        for m in ds["data"][c]:
            probs, preds = ds["comparison"][c][m]
            assert probs[0] == ds["distribution"][c][m]
            assert probs[1] == pytest.approx(float(pii.binom_pmf(H, m, p)))
            assert probs[2] == pytest.approx(float(pii.frei(H, m, H * p)))
            assert probs[3] == pytest.approx(float(pii.frei_alt(H, m, H * p_alt)))
            assert preds[0] == ds["data"][c][m]
            assert preds[1] == int(probs[1] * multiplier)
            assert preds[2] == int(probs[2] * multiplier)
            assert preds[3] == int(probs[3] * multiplier)
        assert ds["header"]["contents"][-1] == "comparison - actual, binomial, frei, frei_alt"

    def test_multiplier_disjoint(self, disjoint_dataset):
        ds = copy.deepcopy(disjoint_dataset)
        pii.analyze(ds)
        pii.compare(ds)
        C = list(ds["data"].keys())
        c = C[-1]
        H = ds["header"]["interval_length"]
        # Total intervals = (c - A) // H; the binomial counts should sum close
        # to it (up to truncation), and no prediction can exceed it.
        n_intervals = (c - C[0]) // H
        preds = [ds["comparison"][c][m][1][1] for m in ds["data"][c]]
        assert sum(preds) <= n_intervals
        assert sum(preds) > 0.9 * n_intervals

    def test_multiplier_prime_start(self, prime_start_dataset):
        ds = copy.deepcopy(prime_start_dataset)
        pii.analyze(ds)
        pii.compare(ds)
        C = list(ds["data"].keys())
        c = C[-1]
        n_intervals = sum(ds["data"][c].values())
        preds = [ds["comparison"][c][m][1][1] for m in ds["data"][c]]
        assert sum(preds) <= n_intervals

    def test_nested(self, nested_overlap):
        ds = nested_overlap
        pii.compare(ds)
        keys = list(ds["nested_interval_data"].keys())
        c = keys[-1]
        assert set(ds["comparison"][c]) == set(ds["nested_interval_data"][c])
        probs, preds = ds["comparison"][c][max(ds["nested_interval_data"][c])]
        assert len(probs) == 4 and len(preds) == 4

    def test_requires_analyze(self, overlap_dataset, capsys):
        ds = copy.deepcopy(overlap_dataset)
        assert pii.compare(ds) is None
        assert "Analyze data first" in capsys.readouterr().out

    def test_no_data(self, capsys):
        assert pii.compare({"header": {}}) is None
        assert "No data to compare." in capsys.readouterr().out


class TestWinners:
    @pytest.fixture()
    def compared(self, nested_overlap):
        pii.compare(nested_overlap)
        return nested_overlap

    def test_rankings_match_square_errors(self, compared):
        ds = compared
        assert pii.winners(ds) is ds
        for _c, w in ds["winners"].items():
            errors = {"B": w["B sq error"], "F": w["F sq error"], "F*": w["F* sq error"]}
            ranked = sorted(errors.values())
            assert errors[w[1]] == ranked[0]
            assert errors[w[2]] == ranked[1]
            assert errors[w[3]] == ranked[2]

    def test_win_lists_partition_the_range(self, compared):
        ds = pii.winners(compared) if "winners" not in compared else compared
        for c, w in ds["winners"].items():
            M = [m for m in ds["comparison"][c] if ds["comparison"][c][m] != 0]
            full = list(range(min(M), max(M) + 1))
            union = (
                set(w["B wins for m in "])
                | set(w["F wins for m in "])
                | set(w["F* wins for m in "])
            )
            assert union == set(full)
            tallies = sorted(
                [
                    len(w["B wins for m in "]),
                    len(w["F wins for m in "]),
                    len(w["F* wins for m in "]),
                ],
                reverse=True,
            )
            assert w["most wins"] != ""
            assert tallies[0] >= tallies[1] >= tallies[2]

    def test_guards(self, compared, capsys):
        pii.winners(compared)
        capsys.readouterr()
        assert pii.winners(compared) is None
        assert "already been applied" in capsys.readouterr().out
        assert pii.winners({"header": {}}) is None
        assert "Compare the data first" in capsys.readouterr().out
