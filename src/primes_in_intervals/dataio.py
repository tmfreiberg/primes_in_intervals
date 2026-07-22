"""SQLite persistence for interval-count datasets.

The database (by default ``data/primes_in_intervals_db``, resolved relative
to the current working directory, so run from the repository root; override
with :func:`set_db`, the ``PII_DB`` environment variable, or a ``db_path``
argument) holds one table per interval type:

* ``disjoint_raw``
* ``overlap_raw``
* ``prime_start_raw``

Each table has columns ``lower_bound``, ``upper_bound``, ``interval_length``
(together the primary key) followed by ``m0, m1, ..., m100``: the number of
intervals ``(a, a + H]`` with ``a`` in ``(lower_bound, upper_bound]`` (``a``
in an arithmetic progression mod ``H`` in the disjoint case, ``a`` prime in
the prime-start case) containing exactly ``m`` primes.  The cap
:data:`max_primes` = 100 comfortably covers every computation in the project;
raise it (and alter the tables) if that ever changes.

So ``lower_bound``, ``upper_bound``, ``interval_length``, ``m0``, ...,
``m100`` are columns ``0, 1, 2, 3, ..., 103`` respectively: ``mi`` is column
``i + 3``.

A dataset's checkpoint rows share their ``lower_bound``; :func:`save` writes
one row per checkpoint (``INSERT OR IGNORE``, so re-saving is harmless), and
:func:`retrieve` groups rows by ``lower_bound`` to reconstruct the original
meta-dictionaries.

Unlike the original script, importing this module does not touch the
filesystem; tables are created on first use by :func:`save` (or explicitly by
:func:`ensure_tables`), and every function accepts a ``db_path`` argument,
defaulting to :data:`DB_PATH`, which can be changed globally with
:func:`set_db`.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from primes_in_intervals.intervals import Dataset, zeros

__all__ = [
    "DB_PATH",
    "ensure_tables",
    "max_primes",
    "retrieve",
    "save",
    "set_db",
    "show_table",
]

#: Largest per-interval prime count stored (columns ``m0`` .. ``m{max_primes}``).
max_primes = 100

#: Default database location: ``data/primes_in_intervals_db``, resolved
#: relative to the current working directory (so run from the repo root).
DB_PATH: Path = Path("data") / "primes_in_intervals_db"

#: Set by :func:`set_db`; when not ``None`` it overrides both the ``PII_DB``
#: environment variable and the :data:`DB_PATH` default.
_DB_OVERRIDE: Path | None = None

_TABLES = {
    "disjoint": "disjoint_raw",
    "overlap": "overlap_raw",
    "prime_start": "prime_start_raw",
}

_MISSING_TABLE_MESSAGE = {
    "disjoint": "Database contains no table for disjoint intervals.",
    "overlap": "Database contains no table for overlapping intervals.",
    "prime_start": "Database contains no table for prime-starting intervals.",
}

_CAPTION = {
    "disjoint": (
        "Disjoint intervals. "
        r"Column with label $m$ shows $\#\{1 \le k \le (B - A)/H : "
        r"\pi(A + kH) - \pi(A + (k - 1)H) = m \}$"
    ),
    "overlap": (
        "Overlapping intervals. "
        r"Column with label $m$ shows $\#\{A < a \le B : "
        r"\pi(a + H) - \pi(a) = m \}$"
    ),
    "prime_start": (
        "Prime-starting intervals. "
        r"Column with label $m$ shows $\#\{A < p \le B : "
        r"\pi(p + H) - \pi(p) = m \}$, $p$ prime."
    ),
}


def _resolve(db_path: str | Path | None) -> str:
    """Return the database path to use, as a string for ``sqlite3.connect``.

    Resolution order, first match wins:

    1. an explicit ``db_path`` argument;
    2. a path set by :func:`set_db`;
    3. the ``PII_DB`` environment variable;
    4. the default :data:`DB_PATH` (``data/primes_in_intervals_db``).
    """
    if db_path is not None:
        return str(db_path)
    if _DB_OVERRIDE is not None:
        return str(_DB_OVERRIDE)
    env = os.environ.get("PII_DB")
    if env:
        return env
    return str(DB_PATH)


def set_db(db_path: str | Path) -> Path:
    """Set the process-wide database location.

    The location set here takes precedence over the ``PII_DB`` environment
    variable and the :data:`DB_PATH` default, and applies to every subsequent
    call that does not pass an explicit ``db_path``.

    Parameters
    ----------
    db_path : str or Path
        New location; subsequent calls without an explicit ``db_path`` use it.

    Returns
    -------
    Path
        The new location, for convenience.
    """
    global _DB_OVERRIDE
    _DB_OVERRIDE = Path(db_path)
    return _DB_OVERRIDE


def ensure_tables(db_path: str | Path | None = None) -> None:
    """Create the three raw tables if they do not already exist.

    The schema is exactly the original project's: three integer key columns
    forming the primary key, then ``m0`` through ``m{max_primes}``.

    Parameters
    ----------
    db_path : str, Path, or None, optional
        Database file; resolved as described in :func:`set_db`.
    """
    resolved = _resolve(db_path)
    # A fresh clone will not have the data/ directory yet; without this,
    # sqlite3 fails with an unhelpful "unable to open database file".
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    # Generate the string 'm0 int, m1 int, m2 int, ... '
    cols = ""
    for i in range(max_primes + 1):
        cols = cols + "m" + f"{i}" + " int, "
    conn = sqlite3.connect(resolved)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS disjoint_raw "
        "(lower_bound int, upper_bound int, interval_length int," + cols
        + "PRIMARY KEY(lower_bound, upper_bound, interval_length))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS overlap_raw "
        "(lower_bound int, upper_bound int, interval_length int," + cols
        + "PRIMARY KEY(lower_bound, upper_bound, interval_length))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS prime_start_raw "
        "(lower_bound int, upper_bound int, interval_length int," + cols
        + "PRIMARY KEY(lower_bound, upper_bound, interval_length))"
    )
    conn.commit()
    conn.close()


def save(data: Dataset, db_path: str | Path | None = None) -> None:
    """Store a dataset's checkpoint rows in the appropriate raw table.

    One row is inserted per checkpoint ``C[k]`` (``k >= 1``), of the form
    ``(C[0], C[k], H, g(0), g(1), ..., g(max_primes))``, into the table named
    by the dataset's ``'interval_type'``.  ``INSERT OR IGNORE`` is used, so
    rows whose ``(lower_bound, upper_bound, interval_length)`` key already
    exists are left untouched and re-saving a dataset is harmless.

    Parameters
    ----------
    data : dict
        A meta-dictionary with ``'header'`` and ``'data'`` items, as produced
        by the ``_cp`` counting functions.
    db_path : str, Path, or None, optional
        Database file; defaults to :data:`DB_PATH`.  Tables are created if
        absent.
    """
    if "data" not in data.keys():
        return print("No data to save. Check contents.")
    ensure_tables(db_path)
    C = list(data["data"].keys())
    H = data["header"]["interval_length"]
    # We'll insert rows of the form C[0], C[k], H, g(0), g(1), ..., g(max_primes).
    # Thus, there are max_primes + 4 columns total. For the SQL string...
    qstring = ""
    for _ in range(max_primes + 4):
        qstring += "?,"
    qstring = qstring[:-1]
    conn = sqlite3.connect(_resolve(db_path))
    for k in range(1, len(C)):
        row = [0] * (max_primes + 4)
        row[0], row[1], row[2] = C[0], C[k], H
        for m in data["data"][C[k]].keys():
            row[m + 3] = data["data"][C[k]][m]
        if data["header"]["interval_type"] == "disjoint":
            conn.executemany(
                "INSERT OR IGNORE INTO disjoint_raw VALUES(" + qstring + ")",
                [tuple(row)],
            )
        if data["header"]["interval_type"] == "overlap":
            conn.executemany(
                "INSERT OR IGNORE INTO overlap_raw VALUES(" + qstring + ")",
                [tuple(row)],
            )
        if data["header"]["interval_type"] == "prime_start":
            conn.executemany(
                "INSERT OR IGNORE INTO prime_start_raw VALUES(" + qstring + ")",
                [tuple(row)],
            )
    conn.commit()
    conn.close()
    return None


def show_table(
    interval_type: str,
    description: str = "description",
    db_path: str | Path | None = None,
) -> Any:
    """Return an entire raw table as a DataFrame.

    Parameters
    ----------
    interval_type : str
        ``'disjoint'``, ``'overlap'``, or ``'prime_start'``.
    description : str, optional
        ``'no description'`` for the bare DataFrame; anything else (the
        default) returns a styled DataFrame with an explanatory caption.
    db_path : str, Path, or None, optional
        Database file; defaults to :data:`DB_PATH`.

    Returns
    -------
    pandas.DataFrame or pandas Styler or None
        The table, ordered by ``lower_bound``, ``upper_bound``,
        ``interval_length``, with columns ``A``, ``B``, ``H``, ``0``, ...,
        ``max_primes``; or ``None`` (with a message) if the table is absent.
    """
    if interval_type not in _TABLES:
        return None
    table = _TABLES[interval_type]
    resolved = _resolve(db_path)
    # Connecting to a nonexistent file would create an empty database as a
    # side effect (and fail outright if its directory is missing); a file
    # that is not there certainly contains no tables, so short-circuit.
    if not Path(resolved).exists():
        print(_MISSING_TABLE_MESSAGE[interval_type])
        return None
    conn = sqlite3.connect(resolved)
    c = conn.cursor()
    existence_check = c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchall()
    if existence_check == []:
        print(_MISSING_TABLE_MESSAGE[interval_type])
        c.close()
        conn.close()
        return None
    res = conn.execute(
        f"SELECT * FROM {table} "  # noqa: S608 - table name from fixed mapping
        "ORDER BY lower_bound ASC, upper_bound ASC, interval_length ASC"
    )
    rows = res.fetchall()
    c.close()
    conn.close()
    cols: list[Any] = ["A", "B", "H"]
    for m in range(0, max_primes + 1):
        cols.append(m)
    df = pd.DataFrame(rows, columns=cols)
    if description == "no description":
        return df
    else:
        return df.style.set_caption(_CAPTION[interval_type])


def retrieve(
    H: int,
    interval_type: str = "overlap",
    db_path: str | Path | None = None,
) -> Dataset | list[Dataset] | None:
    """Reconstruct the dataset(s) with interval length ``H`` from the database.

    Rows are grouped by ``lower_bound``: each group of rows
    ``(A, C[k], H, g(0), ..., g(max_primes))`` becomes one meta-dictionary with
    the usual ``'header'`` and a ``'data'`` item mapping each checkpoint to its
    frequency dictionary (zero-count keys re-trimmed by
    :func:`~primes_in_intervals.intervals.zeros`, exactly as when the data was
    first computed).

    A summary of what was found, including each dataset's header, is printed.

    Parameters
    ----------
    H : int
        Interval length to look up.
    interval_type : str, optional
        ``'disjoint'``, ``'overlap'`` (the default), or ``'prime_start'``.
    db_path : str, Path, or None, optional
        Database file; defaults to :data:`DB_PATH`.

    Returns
    -------
    dict, list of dict, or None
        A single meta-dictionary if exactly one dataset (one distinct
        ``lower_bound``) matches; a list of meta-dictionaries if several do;
        ``None`` (with a message) if the table is absent.
    """
    if interval_type not in _TABLES:
        return None
    table = _TABLES[interval_type]
    resolved = _resolve(db_path)
    # See show_table: avoid creating an empty database on a read.
    if not Path(resolved).exists():
        print(_MISSING_TABLE_MESSAGE[interval_type])
        return None
    conn = sqlite3.connect(resolved)
    c = conn.cursor()
    existence_check = c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchall()
    if existence_check == []:
        print(_MISSING_TABLE_MESSAGE[interval_type])
        c.close()
        conn.close()
        return None
    res = conn.execute(
        f"SELECT * FROM {table} "  # noqa: S608 - table name from fixed mapping
        "WHERE (interval_length) = (?) ORDER BY lower_bound ASC, upper_bound ASC",
        (H,),
    )
    rows = res.fetchall()
    # rows = [(C[0], C[k], H, g(0), ..., g(100)), k = 0,1,...),
    #         (C'[0], C'[k], H, g(0), ..., g(100)), k = 0,1,...), ...]
    c.close()
    conn.close()
    found: dict[int, dict[int, dict[int, int]]] = {}
    i = 0
    while i < len(rows):
        A = rows[i][0]  # C[0]
        found[A] = {}
        j = i
        while j < len(rows) and rows[j][0] == A:
            B = rows[j][1]
            found[A][B] = {m - 3: rows[j][m] for m in range(3, max_primes + 4)}
            j += 1
        i = j
    output = []
    for A in found.keys():
        C = list(found[A].keys())
        C.insert(0, A)
        outputA: Dataset = {
            "header": {
                "interval_type": interval_type,
                "lower_bound": A,
                "upper_bound": C[-1],
                "interval_length": H,
                "no_of_checkpoints": len(C),
                "contents": ["data"],
            }
        }
        data = {C[0]: {m: 0 for m in range(H + 1)}}
        for cc in C[1:]:
            data[cc] = found[A][cc]
        trimmed_data = zeros(data)
        outputA["data"] = trimmed_data
        output.append(outputA)
    if len(output) == 1:
        print(
            f"Found {len(output)} dataset corresponding to interval of "
            f"length {H} ({interval_type} intervals)."
        )
        print(f"\n 'header' : {output[0]['header']}\n")
        return output[0]
    else:
        print(
            f"Found {len(output)} datasets corresponding to interval of "
            f"length {H} ({interval_type} intervals)."
        )
        for i in range(len(output)):
            print(f"\n [{i}] 'header' : {output[i]['header']}\n")
        return output