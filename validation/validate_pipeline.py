"""Compare original vs package across the full pipeline and the SQLite layer."""

import copy
import importlib.util
import io
import os
import sqlite3
import sys
import types
from contextlib import redirect_stdout

import matplotlib

matplotlib.use("Agg")

if "dataframe_image" not in sys.modules:
    try:
        import dataframe_image  # noqa: F401
    except ImportError:
        stub = types.ModuleType("dataframe_image")
        stub.export = lambda *a, **k: None
        sys.modules["dataframe_image"] = stub
# The original module does `from IPython.display import HTML` at import time.
# If IPython is absent we stub it just long enough to import the original, then
# remove the stub again: matplotlib probes sys.modules for IPython when it
# switches backends and expects a real module, so a fake one left in place
# breaks figure creation.
STUBBED_IPYTHON = False
try:
    import IPython.display  # noqa: F401
except ImportError:
    ipy = types.ModuleType("IPython")
    disp = types.ModuleType("IPython.display")
    disp.HTML = lambda *a, **k: None
    ipy.display = disp
    sys.modules["IPython"] = ipy
    sys.modules["IPython.display"] = disp
    STUBBED_IPYTHON = True


def _remove_ipython_stub():
    """Drop the temporary IPython stub, if we installed one."""
    if STUBBED_IPYTHON:
        sys.modules.pop("IPython", None)
        sys.modules.pop("IPython.display", None)

import tempfile


def _find_original():
    """Locate the original 2023 script.

    Looks at $PII_ORIGINAL first, then at the two sensible places in the
    repository: original/primes_in_intervals.py (recommended, since a copy
    at the repository root shadows the installed package when Python is
    started from there) and the repository root itself.
    """
    from_env = os.environ.get("PII_ORIGINAL")
    if from_env:
        return os.path.abspath(from_env)
    for candidate in (
        os.path.join("original", "primes_in_intervals.py"),
        "primes_in_intervals.py",
    ):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise SystemExit(
        "Cannot find the original primes_in_intervals.py; set PII_ORIGINAL "
        "to its path, or run this from the repository root."
    )


ORIG_SOURCE = _find_original()
SCRATCH = tempfile.mkdtemp(prefix="pii_validate_")
os.chdir(SCRATCH)

spec = importlib.util.spec_from_file_location("orig", ORIG_SOURCE)
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)
_remove_ipython_stub()


import primes_in_intervals as pii  # noqa: E402

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(label)
        print(f"FAIL {label}\n  new : {got!r}\n  orig: {want!r}")
    else:
        print(f"ok   {label}")


def check_df(label, got, want):
    got_df = got.data if hasattr(got, "data") else got
    want_df = want.data if hasattr(want, "data") else want
    same = got_df.equals(want_df)
    # caption text, if styled
    got_cap = getattr(got, "caption", None)
    want_cap = getattr(want, "caption", None)
    if got_cap != want_cap:
        same = False
    if not same:
        FAILURES.append(label)
        print(f"FAIL {label}")
        print("  new :", got_df.to_dict(), got_cap)
        print("  orig:", want_df.to_dict(), want_cap)
    else:
        print(f"ok   {label}")


# ------------------------------------------------ mini EXP17-style pipelines
# A smaller stand-in for the examples: N near e^12, checkpoints every 100 over
# +-2000, H = 30, so it runs fast but exercises every code path incl. nesting.
import numpy as np  # noqa: E402

N = int(np.exp(12))
C0 = list(range(N - 2000, N + 2001, 100))

for itype in ["overlap", "disjoint", "prime_start"]:
    ds_o = orig.intervals(list(C0), 30, itype)
    ds_n = pii.intervals(list(C0), 30, itype)
    check(f"[{itype}] intervals equal", ds_n, ds_o)

    # extract narrow + filter
    lo, hi = C0[5], C0[-5]
    ex_o = orig.extract(ds_o, [lo, hi], option="narrow")
    ex_n = pii.extract(ds_n, [lo, hi], option="narrow")
    check(f"[{itype}] extract narrow", ex_n, ex_o)
    fl_o = orig.extract(ds_o, C0[::2], option="filter")
    fl_n = pii.extract(ds_n, C0[::2], option="filter")
    check(f"[{itype}] extract filter", fl_n, fl_o)

    # partition + unpartition roundtrip (in place)
    pt_o = copy.deepcopy(ex_o)
    pt_n = copy.deepcopy(ex_n)
    orig.partition(pt_o)
    pii.partition(pt_n)
    check(f"[{itype}] partition", pt_n, pt_o)
    del pt_o["data"]
    del pt_n["data"]
    orig.unpartition(pt_o)
    pii.unpartition(pt_n)
    check(f"[{itype}] unpartition", pt_n, pt_o)

    # nest (even and odd checkpoint counts)
    nest_o = orig.nest(copy.deepcopy(ds_o))
    nest_n = pii.nest(copy.deepcopy(ds_n))
    check(f"[{itype}] nest (odd cp count)", nest_n, nest_o)
    ds41_o = orig.extract(ds_o, C0[:-1], option="filter")
    ds41_n = pii.extract(ds_n, C0[:-1], option="filter")
    check(f"[{itype}] nest input (even)", ds41_n, ds41_o)
    nest2_o = orig.nest(copy.deepcopy(ds41_o))
    nest2_n = pii.nest(copy.deepcopy(ds41_n))
    check(f"[{itype}] nest (even cp count)", nest2_n, nest2_o)

    # analyze + compare + winners: flat
    flat_o, flat_n = copy.deepcopy(ds_o), copy.deepcopy(ds_n)
    orig.analyze(flat_o)
    pii.analyze(flat_n)
    check(f"[{itype}] analyze flat", flat_n, flat_o)
    orig.compare(flat_o)
    pii.compare(flat_n)
    check(f"[{itype}] compare flat", flat_n, flat_o)
    orig.winners(flat_o)
    pii.winners(flat_n)
    check(f"[{itype}] winners flat", flat_n, flat_o)

    # analyze + compare + winners: nested
    orig.analyze(nest_o)
    pii.analyze(nest_n)
    check(f"[{itype}] analyze nested", nest_n, nest_o)
    orig.compare(nest_o)
    pii.compare(nest_n)
    check(f"[{itype}] compare nested", nest_n, nest_o)
    orig.winners(nest_o)
    pii.winners(nest_n)
    check(f"[{itype}] winners nested", nest_n, nest_o)

    # display: the full option grid that makes sense per structure
    for kwargs in [
        {},
        {"orient": "columns"},
        {"description": "off"},
        {"description": "off", "zeroth_item": "no show"},
        {"zeroth_item": "no show"},
        {"zeroth_item": "no show", "orient": "columns"},
        {"comparisons": "absolute"},
        {"comparisons": "probabilities"},
        {"winners": "show"},
    ]:
        try:
            d_o = orig.display(flat_o, **kwargs)
            err_o = None
        except Exception as e:  # noqa: BLE001
            d_o, err_o = None, type(e).__name__
        try:
            d_n = pii.display(flat_n, **kwargs)
            err_n = None
        except Exception as e:  # noqa: BLE001
            d_n, err_n = None, type(e).__name__
        if err_o is not None or err_n is not None:
            check(f"[{itype}] display flat {kwargs} raises identically", err_n, err_o)
        else:
            check_df(f"[{itype}] display flat {kwargs}", d_n, d_o)
    # partitioned display
    part_o, part_n = copy.deepcopy(ds_o), copy.deepcopy(ds_n)
    orig.partition(part_o)
    pii.partition(part_n)
    for kwargs in [{"count": "partition"}, {"count": "partition", "orient": "columns"}]:
        d_o = orig.display(part_o, **kwargs)
        d_n = pii.display(part_n, **kwargs)
        check_df(f"[{itype}] display partition {kwargs}", d_n, d_o)
    # nested display
    for kwargs in [
        {},
        {"comparisons": "absolute"},
        {"comparisons": "probabilities"},
        {"comparisons": "absolute", "single_cell": "false"},
        {"winners": "show"},
    ]:
        d_o = orig.display(nest_o, **kwargs)
        d_n = pii.display(nest_n, **kwargs)
        check_df(f"[{itype}] display nested {kwargs}", d_n, d_o)

# --------------------------------------------------------------- SQLite
# Original writes to CWD db (created at its import). Point the package at a
# separate file, save identical datasets through both, compare raw tables,
# then compare retrieve() output and printed text.
orig_db = os.path.join(SCRATCH, "primes_in_intervals_db")
new_db = os.path.join(SCRATCH, "new_db")
for f in [orig_db, new_db]:
    if os.path.exists(f):
        os.remove(f)
# re-create original's tables (import already did, but we removed the file)
conn = sqlite3.connect(orig_db)
cols = "".join(f"m{i} int, " for i in range(orig.max_primes + 1))
for t in ["disjoint_raw", "overlap_raw", "prime_start_raw"]:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {t} (lower_bound int, upper_bound int, "
        f"interval_length int,{cols}PRIMARY KEY(lower_bound, upper_bound, interval_length))"
    )
conn.commit()
conn.close()
pii.set_db(new_db)

datasets = {}
for itype in ["disjoint", "overlap", "prime_start"]:
    for C, H in [(list(range(0, 2001, 500)), 40), (list(range(5000, 8001, 1000)), 40), (list(range(0, 1001, 250)), 25)]:
        ds = orig.intervals(list(C), H, itype)
        datasets[(itype, H, C[0])] = ds
        orig.save(ds)
        pii.save(copy.deepcopy(ds))

for t in ["disjoint_raw", "overlap_raw", "prime_start_raw"]:
    co, cn = sqlite3.connect(orig_db), sqlite3.connect(new_db)
    rows_o = co.execute(f"SELECT * FROM {t} ORDER BY lower_bound, upper_bound, interval_length").fetchall()
    rows_n = cn.execute(f"SELECT * FROM {t} ORDER BY lower_bound, upper_bound, interval_length").fetchall()
    co.close()
    cn.close()
    check(f"raw table {t} identical", rows_n, rows_o)

# duplicate-save is ignored in both
orig.save(datasets[("overlap", 40, 0)])
pii.save(copy.deepcopy(datasets[("overlap", 40, 0)]))
co, cn = sqlite3.connect(orig_db), sqlite3.connect(new_db)
check(
    "INSERT OR IGNORE keeps row counts equal",
    cn.execute("SELECT COUNT(*) FROM overlap_raw").fetchone(),
    co.execute("SELECT COUNT(*) FROM overlap_raw").fetchone(),
)
co.close()
cn.close()

# retrieve: values and printed output
for itype in ["disjoint", "overlap", "prime_start"]:
    for H in [40, 25]:
        buf_o, buf_n = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_o):
            r_o = orig.retrieve(H, itype)
        with redirect_stdout(buf_n):
            r_n = pii.retrieve(H, itype)
        check(f"retrieve({H},'{itype}') values", r_n, r_o)
        check(f"retrieve({H},'{itype}') printed text", buf_n.getvalue(), buf_o.getvalue())

# show_table
for itype in ["disjoint", "overlap", "prime_start"]:
    t_o = orig.show_table(itype)
    t_n = pii.show_table(itype)
    check_df(f"show_table('{itype}') styled", t_n, t_o)
    t_o = orig.show_table(itype, description="no description")
    t_n = pii.show_table(itype, description="no description")
    check_df(f"show_table('{itype}') bare", t_n, t_o)

# missing-table messages
empty_db = os.path.join(SCRATCH, "empty_db")
if os.path.exists(empty_db):
    os.remove(empty_db)
pii.set_db(empty_db)
buf = io.StringIO()
with redirect_stdout(buf):
    r = pii.retrieve(40, "overlap")
check("retrieve on missing table returns None", r, None)
check(
    "retrieve missing-table message",
    buf.getvalue(),
    "Database contains no table for overlapping intervals.\n",
)

print()
print("=" * 60)
print(f"{len(FAILURES)} FAILURES: {FAILURES}" if FAILURES else "ALL PIPELINE AND SQLITE COMPARISONS PASS")