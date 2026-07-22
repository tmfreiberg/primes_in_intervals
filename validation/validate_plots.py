"""Pixel-compare the original README plotting scripts with the new plotting API."""

import importlib.util
import os
import sys
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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

# Shared analyzed data: an overlap dataset (flat) and its nest.
N = int(np.exp(12))
C0 = list(range(N - 2000, N + 2001, 100))
X_flat = orig.intervals(list(C0), 30, "overlap")
orig.analyze(X_flat)
X_nest = orig.nest(orig.intervals(list(C0), 30, "overlap"))
orig.analyze(X_nest)

# ---------------------------------------------------------------- FLAT
# Original: the README Example 1 script, verbatim (overlap branch).
X = X_flat
interval_type = X["header"]["interval_type"]
A = X["header"]["lower_bound"]
H = X["header"]["interval_length"]
C = list(X["distribution"].keys())

plt.rcParams.update({"font.size": 22})
fig, ax = plt.subplots(figsize=(22, 11))
fig.suptitle("Primes in intervals")

hor_axis = list(X["distribution"][C[-1]].keys())
y_min, y_max = 0, 0
for c in C:
    for m in X["distribution"][c].keys():
        if y_max < X["distribution"][c][m]:
            y_max = X["distribution"][c][m]


def plot(cp):
    ax.clear()
    mu = X["statistics"][cp]["mean"]
    sigma = X["statistics"][cp]["var"]
    med = X["statistics"][cp]["med"]
    if med == int(med):
        med = int(med)
    modes = X["statistics"][cp]["mode"]
    ax.set(xlim=(hor_axis[0], hor_axis[-1]), ylim=(0, np.ceil(100 * y_max) / 100))
    ver_axis = list(X["distribution"][cp].values())
    ax.bar(hor_axis, ver_axis, color="#e0249a", zorder=2.5, alpha=0.3, label=r"$\mathrm{Prob}(X = m)$")
    ax.plot(hor_axis, ver_axis, "o", color="red", zorder=2.5)
    B = cp
    NN = (A + B) / 2
    p = 1 / (np.log(NN) - 1)
    p_alt = 1 / np.log(NN)
    x = np.linspace(hor_axis[0], hor_axis[-1], 100)
    ax.plot(x, orig.binom_pmf(H, x, p), "--", color="orange", zorder=3.5, label=r"$\mathrm{Binom}(H,\lambda/H)$")
    if interval_type == "overlap":
        ax.plot(x, orig.frei(H, x, H * p), "--", color="green", zorder=3.5, label=r"$\mathrm{F}(H,m,\lambda)$")
    if interval_type == "overlap":
        ax.text(
            0.70,
            0.15,
            r"$X = \pi(a + H) - \pi(a)$, "
            + r"$A < a \leq B$"
            + "\n\n"
            + rf"$H = {H}$"
            + "\n\n"
            + rf"$A = {A}$"
            + "\n\n"
            + rf"$B = {B}$"
            + "\n\n"
            + r"$N = (A + B)/2$"
            + "\n\n"
            + rf"$\lambda = H/(\log N - 1) = {H * p:.5f}$"
            + "\n\n"
            + r"$\mathbb{E}[X] = $"
            + f"{mu:.5f}"
            + "\n\n"
            + r"$\mathrm{Var}(X) = $"
            + f"{sigma:.5f}"
            + "\n\n"
            + rf"median : ${med}$"
            + "\n\n"
            + rf"mode(s): ${modes}$",
            bbox=dict(facecolor="white", edgecolor="white", alpha=0.5),
            transform=ax.transAxes,
        )
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r"$m$ (number of primes in an interval)")
    ax.set_ylabel("prop'n of intervals with" + r" $m$ " + "primes")
    ax.legend(loc=2, ncol=1, framealpha=0.5)
    ax.grid(True, zorder=0, alpha=0.7)


plt.rcParams["savefig.facecolor"] = "white"
plot(C[-1])
fig.savefig("orig_flat.png", dpi=80)
plt.close(fig)

# New: same frame through the package function.
plt.rcParams.update({"font.size": 22})
fig2, ax2 = plt.subplots(figsize=(22, 11))
fig2.suptitle("Primes in intervals")
pii.plot_distribution_frame(ax2, X_flat, C[-1], x_pad=0, ylim_decimals=2)
fig2.savefig("new_flat.png", dpi=80)
plt.close(fig2)

# ---------------------------------------------------------------- NESTED
# Original: the examples.md Example 2 script, verbatim (with its custom text
# replaced by generic A/B since our data is not around e^17; we validate the
# machinery with a matched custom overlay instead).
X = X_nest
A = X["header"]["lower_bound"]
H = X["header"]["interval_length"]
C = list(X["distribution"].keys())
plt.rcParams.update({"font.size": 22})
fig, ax = plt.subplots(figsize=(22, 11))
fig.suptitle("Primes in intervals")
hor_axis = list(X["distribution"][C[-1]].keys())
y_min, y_max = 0, 0
for c in C:
    for m in X["distribution"][c].keys():
        if y_max < X["distribution"][c][m]:
            y_max = X["distribution"][c][m]


def custom_text(mu, sigma, med, modes, H, k, p):
    return (
        r"$X = \pi(a + H) - \pi(a)$"
        + "\n\n"
        + r"$N - M < a \leq N + M$"
        + "\n\n"
        + rf"$H = {H}$"
        + "\n\n"
        + rf"$M = 10^2k$, $k = {k}$"
        + "\n\n"
        + rf"$\lambda = H/(\log N - 1) = {H * p:.5f}$"
        + "\n\n"
        + r"$\mathbb{E}[X] = $"
        + f"{mu:.5f}"
        + "\n\n"
        + r"$\mathrm{Var}(X) = $"
        + f"{sigma:.5f}"
        + "\n\n"
        + rf"median : ${med}$"
        + "\n\n"
        + rf"mode(s): ${modes}$"
    )


def plot_nest(c):
    ax.clear()
    mu = X["statistics"][c]["mean"]
    sigma = X["statistics"][c]["var"]
    med = X["statistics"][c]["med"]
    if med == int(med):
        med = int(med)
    modes = X["statistics"][c]["mode"]
    ax.set(xlim=(hor_axis[0] - 0.5, hor_axis[-1] + 0.5), ylim=(0, np.ceil(1000 * y_max) / 1000))
    ver_axis = list(X["distribution"][c].values())
    ax.bar(hor_axis, ver_axis, color="#e0249a", zorder=2.5, alpha=0.3, label=r"$\mathrm{Prob}(X = m)$")
    ax.plot(hor_axis, ver_axis, "o", color="red", zorder=2.5)
    AA = c[0]
    BB = c[1]
    NN = (AA + BB) // 2
    M = NN - AA
    k = M // 10**2
    p = 1 / (np.log(NN) - 1)
    x = np.linspace(hor_axis[0], hor_axis[-1], 100)
    ax.plot(x, orig.binom_pmf(H, x, p), "--", color="orange", zorder=3.5, label=r"$\mathrm{Binom}(H,\lambda/H)$")
    ax.plot(x, orig.frei(H, x, H * p), "--", color="green", zorder=3.5, label=r"$\mathrm{F}(H,m,\lambda)$")
    ax.text(
        0.75,
        0.15,
        custom_text(mu, sigma, med, modes, H, k, p),
        bbox=dict(facecolor="white", edgecolor="white", alpha=0.5),
        transform=ax.transAxes,
    )
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r"$m$ (number of primes in an interval)")
    ax.set_ylabel("prop'n of intervals with" + r" $m$ " + "primes")
    ax.legend(loc=2, ncol=1, framealpha=0.5)
    ax.grid(True, zorder=0, alpha=0.7)


plt.rcParams["savefig.facecolor"] = "white"
plot_nest(C[-1])
fig.savefig("orig_nest.png", dpi=80)
plt.close(fig)


# New: same frame, custom overlay callable reproducing the same text.
def my_overlay(Xd, c, H, p, p_alt):
    mu = Xd["statistics"][c]["mean"]
    sigma = Xd["statistics"][c]["var"]
    med = Xd["statistics"][c]["med"]
    if med == int(med):
        med = int(med)
    modes = Xd["statistics"][c]["mode"]
    AA = c[0]
    NN = (c[0] + c[1]) // 2
    k = (NN - AA) // 10**2
    return custom_text(mu, sigma, med, modes, H, k, p)


plt.rcParams.update({"font.size": 22})
fig2, ax2 = plt.subplots(figsize=(22, 11))
fig2.suptitle("Primes in intervals")
pii.plot_distribution_frame(
    ax2,
    X_nest,
    C[-1],
    show_frei=True,
    overlay=my_overlay,
    overlay_position=(0.75, 0.15),
    x_pad=0.5,
    ylim_decimals=3,
)
fig2.savefig("new_nest.png", dpi=80)
plt.close(fig2)

# ------------------------------------------------------------- pixel diff
import numpy as npx  # noqa: E402
from PIL import Image  # noqa: E402

for name in ["flat", "nest"]:
    a = npx.asarray(Image.open(f"orig_{name}.png").convert("RGB"), dtype=npx.int16)
    b = npx.asarray(Image.open(f"new_{name}.png").convert("RGB"), dtype=npx.int16)
    same_shape = a.shape == b.shape
    diff = npx.abs(a - b) if same_shape else None
    print(
        name,
        "shape match:",
        same_shape,
        "| max pixel diff:",
        int(diff.max()) if same_shape else "n/a",
        "| differing pixels:",
        int((diff.sum(axis=2) > 0).sum()) if same_shape else "n/a",
    )

# Animation smoke: 3 frames end to end, gif written.
fig3, anim = pii.animate_distribution(X_nest, frames=C[:3], show_frei=True)
pii.save_gif(anim, "smoke.gif", fps=5, dpi=40)
plt.close(fig3)
print("animation smoke gif bytes:", os.path.getsize("smoke.gif"))