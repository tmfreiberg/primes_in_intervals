"""Tests for dictionary statistics and analyze."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import primes_in_intervals as pii


class TestDictionaryStatistics:
    def test_hand_computed(self):
        # Data: 1, 2, 2, 3, 3, 3 (six objects).
        stats = pii.dictionary_statistics({1: 1, 2: 2, 3: 3})
        assert stats["dist"] == {1: 1 / 6, 2: 2 / 6, 3: 3 / 6}
        assert stats["mean"] == pytest.approx(14 / 6)
        assert stats["2ndmom"] == pytest.approx(36 / 6)
        assert stats["var"] == pytest.approx(36 / 6 - (14 / 6) ** 2)
        assert stats["sdv"] == pytest.approx(np.sqrt(36 / 6 - (14 / 6) ** 2))
        assert stats["med"] == pytest.approx(2.5)  # even count: mean of 3rd, 4th
        assert stats["mode"] == [3]

    def test_median_odd_count(self):
        # Data: 1, 2, 2, 3, 3 (five objects); median is the 3rd value.
        stats = pii.dictionary_statistics({1: 1, 2: 2, 3: 2})
        assert stats["med"] == 2
        assert stats["mode"] == [2, 3]

    def test_dictionary_sort(self):
        assert list(pii.dictionary_sort({3: "c", 1: "a", 2: "b"})) == [1, 2, 3]


class TestAnalyze:
    def test_flat(self, overlap_dataset):
        ds = copy.deepcopy(overlap_dataset)
        assert pii.analyze(ds) is ds
        C = list(ds["data"].keys())
        assert ds["distribution"][C[0]] == {}
        assert ds["statistics"][C[0]] == {}
        for c in C[1:]:
            total = sum(ds["data"][c].values())
            assert sum(ds["distribution"][c].values()) == pytest.approx(1.0)
            for m, v in ds["data"][c].items():
                assert ds["distribution"][c][m] == pytest.approx(v / total)
            assert set(ds["statistics"][c]) == {"mean", "2ndmom", "var", "sdv", "med", "mode"}
        assert ds["header"]["contents"][-2:] == ["distribution", "statistics"]

    def test_nested(self, nested_overlap):
        for _c, dist in nested_overlap["distribution"].items():
            assert sum(dist.values()) == pytest.approx(1.0)

    def test_guards(self, analyzed_overlap, capsys):
        assert pii.analyze(analyzed_overlap) is None
        assert "already been analyzed" in capsys.readouterr().out
        assert pii.analyze({"header": {}}) is None
        assert "No data to analyze." in capsys.readouterr().out
