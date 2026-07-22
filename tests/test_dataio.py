"""Tests for the SQLite save/retrieve layer, on temporary databases."""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import primes_in_intervals as pii


def _small(itype):
    return pii.intervals(list(range(0, 1001, 250)), 25, itype)


class TestSaveAndRetrieve:
    def test_roundtrip_all_types(self, tmp_path, capsys):
        db = tmp_path / "db"
        for itype in ["disjoint", "overlap", "prime_start"]:
            ds = _small(itype)
            pii.save(ds, db_path=db)
            got = pii.retrieve(25, itype, db_path=db)
            out = capsys.readouterr().out
            assert "Found 1 dataset" in out
            assert "'header'" in out
            assert got["data"] == ds["data"]
            assert got["header"] == ds["header"]

    def test_multiple_lower_bounds_return_list(self, tmp_path, capsys):
        db = tmp_path / "db"
        ds1 = pii.intervals([0, 500, 1000], 25, "overlap")
        ds2 = pii.intervals([5000, 5500, 6000], 25, "overlap")
        pii.save(ds1, db_path=db)
        pii.save(ds2, db_path=db)
        got = pii.retrieve(25, "overlap", db_path=db)
        out = capsys.readouterr().out
        assert "Found 2 datasets" in out
        assert isinstance(got, list) and len(got) == 2
        assert got[0]["data"] == ds1["data"]
        assert got[1]["data"] == ds2["data"]

    def test_insert_or_ignore(self, tmp_path):
        db = tmp_path / "db"
        ds = _small("overlap")
        pii.save(ds, db_path=db)
        pii.save(copy.deepcopy(ds), db_path=db)  # second save is a no-op
        conn = sqlite3.connect(db)
        (count,) = conn.execute("SELECT COUNT(*) FROM overlap_raw").fetchone()
        conn.close()
        assert count == len(ds["data"]) - 1  # one row per non-initial checkpoint

    def test_row_layout(self, tmp_path):
        db = tmp_path / "db"
        ds = _small("disjoint")
        pii.save(ds, db_path=db)
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT * FROM disjoint_raw ORDER BY upper_bound LIMIT 1"
        ).fetchone()
        conn.close()
        C = list(ds["data"].keys())
        assert row[:3] == (C[0], C[1], 25)
        assert len(row) == pii.max_primes + 4
        for m, v in ds["data"][C[1]].items():
            assert row[m + 3] == v

    def test_save_without_data_prints_message(self, tmp_path, capsys):
        assert pii.save({"header": {}}, db_path=tmp_path / "db") is None
        assert "No data to save" in capsys.readouterr().out


class TestMissingTables:
    def test_retrieve_message(self, tmp_path, capsys):
        db = tmp_path / "empty"
        assert pii.retrieve(25, "overlap", db_path=db) is None
        assert (
            capsys.readouterr().out
            == "Database contains no table for overlapping intervals.\n"
        )

    def test_show_table_messages(self, tmp_path, capsys):
        db = tmp_path / "empty"
        expected = {
            "disjoint": "Database contains no table for disjoint intervals.",
            "overlap": "Database contains no table for overlapping intervals.",
            "prime_start": "Database contains no table for prime-starting intervals.",
        }
        for itype, message in expected.items():
            assert pii.show_table(itype, db_path=db) is None
            assert capsys.readouterr().out == message + "\n"


class TestShowTable:
    def test_bare_dataframe(self, tmp_path):
        db = tmp_path / "db"
        ds = _small("overlap")
        pii.save(ds, db_path=db)
        df = pii.show_table("overlap", description="no description", db_path=db)
        assert list(df.columns[:3]) == ["A", "B", "H"]
        assert len(df) == len(ds["data"]) - 1

    def test_caption(self, tmp_path):
        db = tmp_path / "db"
        pii.save(_small("disjoint"), db_path=db)
        styled = pii.show_table("disjoint", db_path=db)
        assert "Disjoint intervals." in styled.caption


class TestSetDb:
    def test_default_path_is_switchable(self, tmp_path):
        old = pii.DB_PATH
        try:
            db = tmp_path / "switched"
            pii.set_db(db)
            ds = _small("overlap")
            pii.save(ds)
            assert db.exists()
            got = pii.retrieve(25, "overlap")
            assert got["data"] == ds["data"]
        finally:
            pii.set_db(old)


class TestPathResolution:
    def test_default_location(self):
        assert pii.DB_PATH == Path("data") / "primes_in_intervals_db"

    def test_env_var_is_used(self, tmp_path, monkeypatch):
        # Neutralize any set_db from earlier tests so the environment
        # variable is actually consulted; monkeypatch restores it after.
        monkeypatch.setattr(pii.dataio, "_DB_OVERRIDE", None)
        db = tmp_path / "made" / "by" / "env"
        monkeypatch.setenv("PII_DB", str(db))
        ds = _small("overlap")
        pii.save(ds)
        assert db.exists()
        got = pii.retrieve(25, "overlap")
        assert got["data"] == ds["data"]

    def test_set_db_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pii.dataio, "_DB_OVERRIDE", None)
        via_env = tmp_path / "env_db"
        via_set = tmp_path / "set_db"
        monkeypatch.setenv("PII_DB", str(via_env))
        pii.set_db(via_set)
        pii.save(_small("overlap"))
        assert via_set.exists()
        assert not via_env.exists()