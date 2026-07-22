"""Compare the original primes_in_intervals.py against the new package, exactly."""

import importlib.util
import os
import sys
import types

import matplotlib

matplotlib.use("Agg")

# Stub the two imports the original makes that we may not have installed.
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

# Import the original from a scratch CWD (its import creates a database file).
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

spec = importlib.util.spec_from_file_location(
    "orig", ORIG_SOURCE
)
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)
_remove_ipython_stub()

import primes_in_intervals as pii  # the new package  # noqa: E402

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append((label, got, want))
        print(f"FAIL {label}\n  new : {got}\n  orig: {want}")
    else:
        print(f"ok   {label}")


# ----------------------------------------------------------------- sieve
P_old, P_new = orig.postponed_sieve(), pii.postponed_sieve()
check("first 1000 primes", [next(P_new) for _ in range(1000)], [next(P_old) for _ in range(1000)])

# ------------------------------------------------- plain counters, battery
cases = [
    (0, 10, 10),
    (0, 5, 5),
    (0, 100, 10),
    (999, 1000, 1000),
    (0, 97, 10),      # B - A not a multiple of H
    (1, 2, 3),        # tiny
    (10, 20, 20),
    (89, 199, 7),     # awkward endpoints around primes
    (113, 114, 5),    # single left endpoint, prime boundary
    (100, 1000, 57),  # ragged
    (2, 3, 2),
    (0, 1, 1),
    (7919, 8000, 13),
]
# The original has no plain disjoint/overlap/prime_start (README-only), so
# compare the package's plain counters against the original checkpoint
# versions evaluated at the endpoint, and against orig-independent identities.
for A, B, H in cases:
    got = pii.disjoint(A, B, H)
    K = (B - A) // H
    if K >= 1:
        want = orig.disjoint_cp([A, A + K * H], H)["data"][A + K * H]
        want = {m: v for m, v in want.items() if v != 0}
        check(f"disjoint({A},{B},{H}) vs orig disjoint_cp", got, want)
for A, B, H in cases:
    if B > A:
        got = pii.overlap(A, B, H)
        want = orig.overlap_cp([A, B], H)["data"][B]
        want = {m: v for m, v in want.items() if v != 0}
        check(f"overlap({A},{B},{H}) vs orig overlap_cp", got, want)
for A, B, H in cases:
    if B > A:
        got = pii.prime_start(A, B, H)
        want = orig.prime_start_cp([A, B], H)["data"][B]
        want = {m: v for m, v in want.items() if v != 0}
        check(f"prime_start({A},{B},{H}) vs orig prime_start_cp", got, want)

# ------------------------------------------- checkpoint counters, head-on
cp_cases = [
    ([0, 10, 100, 210, 350, 400], 100),          # the docstring's snapping example
    ([0, 100, 200, 300, 400], 100),
    (list(range(0, 1001, 100)), 57),             # checkpoints not aligned to H
    ([400, 0, 200, 100, 300], 50),               # unsorted input
    ([0, 1, 2, 3], 1),
    ([1000, 2000, 3000, 4000, 5000], 100),
    (list(range(10000, 20001, 1000)), 76),
    ([0, 7], 3),
    ([5, 5, 10, 15], 5),                          # duplicate checkpoint
]
for C, H in cp_cases:
    got = pii.disjoint_cp(list(C), H)
    want = orig.disjoint_cp(list(C), H)
    check(f"disjoint_cp({C},{H})", got, want)
for C, H in cp_cases:
    got = pii.overlap_cp(list(C), H)
    want = orig.overlap_cp(list(C), H)
    check(f"overlap_cp({C},{H})", got, want)
for C, H in cp_cases:
    got = pii.prime_start_cp(list(C), H)
    want = orig.prime_start_cp(list(C), H)
    check(f"prime_start_cp({C},{H})", got, want)

# dispatcher
for t in ["disjoint", "overlap", "prime_start"]:
    got = pii.intervals(list(range(0, 501, 100)), 20, t)
    want = orig.intervals(list(range(0, 501, 100)), 20, t)
    check(f"intervals(..., '{t}')", got, want)

# ------------------------------------------------------------ anyIntervals
from itertools import count as itcount  # noqa: E402


def naturals():
    return itcount(1)

got = pii.anyIntervals(0, 300, 20, orig.postponed_sieve(), orig.postponed_sieve())
want = orig.anyIntervals(0, 300, 20, orig.postponed_sieve(), orig.postponed_sieve())
check("anyIntervals(primes, primes)", got, want)
check(
    "anyIntervals(primes,primes) == prime_start",
    pii.anyIntervals(0, 300, 20, pii.postponed_sieve(), pii.postponed_sieve()),
    pii.prime_start(0, 300, 20),
)
got = pii.anyIntervals(0, 200, 10, naturals(), orig.postponed_sieve())
want = orig.anyIntervals(0, 200, 10, naturals(), orig.postponed_sieve())
check("anyIntervals(naturals, primes)", got, want)
C = list(range(0, 301, 50))
got = pii.anyIntervals_cp(list(C), 20, pii.postponed_sieve(), pii.postponed_sieve())
want = orig.anyIntervals_cp(list(C), 20, orig.postponed_sieve(), orig.postponed_sieve())
check("anyIntervals_cp(primes, primes)", got, want)

# --------------------------------------------------------------- zeros
md = {1: {0: 0, 1: 2, 2: 0}, 2: {1: 0, 3: 4}}
check("zeros pad", pii.zeros({k: dict(v) for k, v in md.items()}), orig.zeros({k: dict(v) for k, v in md.items()}))
check("zeros no pad", pii.zeros({k: dict(v) for k, v in md.items()}, pad="no"), orig.zeros({k: dict(v) for k, v in md.items()}, pad="no"))

# ------------------------------------------------- overlap_extension
got = pii.overlap_extension(1000, 2000, 50, [3, 11])
want = orig.overlap_extension(1000, 2000, 50, [3, 11])
check("overlap_extension(1000,2000,50,[3,11])", got, want)
got = pii.overlap_extension(0, 100, 10, [0, 1, 5])
want = orig.overlap_extension(0, 100, 10, [0, 1, 5])
check("overlap_extension(0,100,10,[0,1,5])", got, want)

print()
print("=" * 60)
print(f"{len(FAILURES)} failures in counter comparison" if FAILURES else "ALL COUNTER COMPARISONS PASS")