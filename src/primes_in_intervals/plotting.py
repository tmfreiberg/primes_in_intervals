"""Plotting and animating the distribution of primes in intervals.

The exposition builds its figures from free-standing scripts; this module
packages the same code as reusable functions with the same visual output: for
each checkpoint, a bar-and-dot histogram of the empirical distribution with
the theoretical prediction curves dashed over it, summary statistics overlaid
in a text box, and the whole sequence of checkpoints strung into an animation
whose frames show the distribution settling as the sample grows.

Layout, colors, z-orders, labels, and axis conventions are kept exactly as in
the original scripts (magenta bars ``#e0249a`` with red dots, orange dashed
binomial, green dashed ``F``, blue dashed ``F*``, yellow dashed alternative
binomial, legend upper-left, grid underneath).  The overlay text adapts to the
dataset: it distinguishes overlapping, disjoint, and prime-start interval
types, and integer versus nested ``(lower, upper)`` checkpoints.  For a fully
custom overlay (the exposition's figures hard-code strings like
``N = [e^17]``), pass a callable.

Typical use::

    X = pii.nest(pii.intervals(C, H, 'overlap'))
    pii.analyze(X)
    fig, anim = pii.animate_distribution(X)
    pii.save_gif(anim, 'distribution.gif')

Matplotlib is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from primes_in_intervals.intervals import Dataset
from primes_in_intervals.predictions import binom_pmf, frei, frei_alt

__all__ = [
    "animate_distribution",
    "distribution_axes_limits",
    "plot_distribution_frame",
    "save_gif",
    "save_mp4",
]


def distribution_axes_limits(X: Dataset) -> tuple[list[int], float]:
    """Return the common horizontal axis and maximum height for a dataset.

    The horizontal axis is the key list of the final checkpoint's
    distribution (every checkpoint shares it, by
    :func:`~primes_in_intervals.intervals.zeros`), and the height is the
    largest relative frequency over all checkpoints, so an animation's axes
    can stay fixed across frames.

    Parameters
    ----------
    X : dict
        An analyzed dataset (with a ``'distribution'`` item).

    Returns
    -------
    tuple
        ``(hor_axis, y_max)``.
    """
    C = list(X["distribution"].keys())
    hor_axis = list(X["distribution"][C[-1]].keys())
    y_max = 0.0
    for c in C:
        for m in X["distribution"][c].keys():
            if y_max < X["distribution"][c][m]:
                y_max = X["distribution"][c][m]
    return hor_axis, y_max


def _default_overlay(X: Dataset, c: Any, H: int, p: float, p_alt: float) -> str:
    """Build the statistics text box for a frame (the generic form).

    Adapts the exposition's overlay to any dataset: the random-variable line
    matches the interval type, the range line shows the actual bounds, and the
    statistics lines show ``lambda`` (and ``lambda*`` when the alternative
    density is in play via a prime-start dataset), mean, variance, median, and
    mode(s).
    """
    mu = X["statistics"][c]["mean"]
    sigma = X["statistics"][c]["var"]
    med = X["statistics"][c]["med"]
    if med == int(med):
        med = int(med)
    modes = X["statistics"][c]["mode"]
    interval_type = X["header"]["interval_type"]
    if isinstance(c, tuple):
        A, B = c[0], c[1]
    else:
        A, B = X["header"]["lower_bound"], c
    if interval_type == "overlap":
        variable = r"$X = \pi(a + H) - \pi(a)$, " + r"$A < a \leq B$"
    elif interval_type == "disjoint":
        variable = (
            r"$X = \pi(a + H) - \pi(a)$, "
            + r"$a = A + kH$, $0 \leq k \leq (B - A)/H$"
        )
    else:  # prime_start
        variable = r"$X = \pi(p + H) - \pi(p)$, " + r"$A < p \leq B$, $p$ prime"
    return (
        variable
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
        + rf"mode(s): ${modes}$"
    )


def plot_distribution_frame(
    ax: Any,
    X: Dataset,
    c: Any,
    hor_axis: list[int] | None = None,
    y_max: float | None = None,
    show_binom: bool = True,
    show_binom_alt: bool = False,
    show_frei: bool | None = None,
    show_frei_alt: bool = False,
    overlay: str | Callable[..., str] | None = "auto",
    overlay_position: tuple[float, float] = (0.70, 0.15),
    note: str | None = None,
    x_pad: float = 0.5,
    ylim_decimals: int = 2,
) -> None:
    """Draw one checkpoint's distribution onto an existing axis.

    Renders the empirical distribution at checkpoint ``c`` as translucent
    bars with red dots, overlays the requested prediction curves as smooth
    dashed lines, and adds the statistics text box, exactly as in the
    exposition's figures.  The axis is cleared first, so this doubles as the
    animation's frame function.

    Parameters
    ----------
    ax : matplotlib axis
        The target axis (cleared before drawing).
    X : dict
        An analyzed dataset.
    c : int or tuple
        The checkpoint (or nested interval) to draw.
    hor_axis, y_max : list and float, optional
        The fixed axes from :func:`distribution_axes_limits`; computed from
        ``X`` when omitted.  Pass them explicitly when animating, so every
        frame shares the same axes.
    show_binom : bool, optional
        Draw the binomial prediction at density ``1/(log N - 1)`` (default
        True).
    show_binom_alt : bool, optional
        Also draw the binomial at the alternative density ``1/log N``
        (default False).
    show_frei : bool or None, optional
        Draw the refined prediction ``F``.  The default ``None`` draws it
        exactly when the dataset's interval type is ``'overlap'``, matching
        the exposition (the prediction was derived for overlapping
        intervals).
    show_frei_alt : bool, optional
        Draw the alternative prediction ``F*`` (default False).
    overlay : str, callable, or None, optional
        ``'auto'`` (default) builds the generic statistics box; a callable is
        called as ``overlay(X, c, H, p, p_alt)`` and must return the text; any
        other string is used verbatim as the text; ``None`` or ``'off'``
        suppresses the box.
    overlay_position : tuple, optional
        Axes-fraction position of the text box (default ``(0.70, 0.15)``).
    note : str, optional
        Extra text drawn at the top of the axes (the exposition uses this for
        the "F and F* might not be applicable" reminder on prime-start data).
    x_pad : float, optional
        Horizontal padding beyond the first and last ``m`` (default 0.5; the
        exposition's first example used 0).
    ylim_decimals : int, optional
        The vertical limit is ``y_max`` rounded up at this many decimals
        (default 2; the nested examples used 3).
    """
    ax.clear()

    H = X["header"]["interval_length"]
    interval_type = X["header"]["interval_type"]
    if hor_axis is None or y_max is None:
        hor_axis, y_max = distribution_axes_limits(X)

    # Bounds for the plot, and horizontal axis tick marks.
    scale = 10**ylim_decimals
    ax.set(
        xlim=(hor_axis[0] - x_pad, hor_axis[-1] + x_pad),
        ylim=(0, np.ceil(scale * y_max) / scale),
    )

    # The data and histogram
    ver_axis = list(X["distribution"][c].values())
    ax.bar(
        hor_axis,
        ver_axis,
        color="#e0249a",
        zorder=2.5,
        alpha=0.3,
        label=r"$\mathrm{Prob}(X = m)$",
    )
    ax.plot(hor_axis, ver_axis, "o", color="red", zorder=2.5)

    # Predictions for comparison
    if isinstance(c, tuple):
        A, B = c[0], c[1]
        N = (A + B) // 2
    else:
        A, B = X["header"]["lower_bound"], c
        N = (A + B) / 2
    p = 1 / (np.log(N) - 1)
    p_alt = 1 / np.log(N)
    x = np.linspace(hor_axis[0], hor_axis[-1], 100)
    if show_binom:
        ax.plot(
            x,
            binom_pmf(H, x, p),
            "--",
            color="orange",
            zorder=3.5,
            label=r"$\mathrm{Binom}(H,\lambda/H)$",
        )
    if show_binom_alt:
        ax.plot(
            x,
            binom_pmf(H, x, p_alt),
            "--",
            color="yellow",
            zorder=3.5,
            label=r"$\mathrm{Binom}(H,\lambda^*/H)$",
        )
    if show_frei is None:
        show_frei = interval_type == "overlap"
    if show_frei:
        ax.plot(
            x,
            frei(H, x, H * p),
            "--",
            color="green",
            zorder=3.5,
            label=r"$\mathrm{F}(H,m,\lambda)$",
        )
    if show_frei_alt:
        ax.plot(
            x,
            frei_alt(H, x, H * p_alt),
            "--",
            color="blue",
            zorder=3.5,
            label=r"$\mathrm{F^*}(H,m,\lambda^*)$",
        )

    # Overlay information
    if note is not None:
        ax.text(
            0.25,
            0.90,
            note,
            bbox=dict(facecolor="white", edgecolor="white", alpha=0.5),
            transform=ax.transAxes,
        )
    if overlay is not None and overlay != "off":
        if overlay == "auto":
            text = _default_overlay(X, c, H, p, p_alt)
        elif callable(overlay):
            text = overlay(X, c, H, p, p_alt)
        else:
            text = overlay  # a literal string, used as-is
        ax.text(
            overlay_position[0],
            overlay_position[1],
            text,
            bbox=dict(facecolor="white", edgecolor="white", alpha=0.5),
            transform=ax.transAxes,
        )

    # Formatting/labeling
    ax.set_xticks(hor_axis)
    ax.set_xlabel(r"$m$ (number of primes in an interval)")
    ax.set_ylabel("prop'n of intervals with" + r" $m$ " + "primes")
    ax.legend(loc=2, ncol=1, framealpha=0.5)

    # A grid is helpful, but we want it underneath everything else.
    ax.grid(True, zorder=0, alpha=0.7)


def animate_distribution(
    X: Dataset,
    frames: list | None = None,
    interval: int = 100,
    figsize: tuple[float, float] = (22, 11),
    font_size: int = 22,
    suptitle: str = "Primes in intervals",
    **frame_kwargs: Any,
) -> tuple[Any, Any]:
    """Animate a dataset's distribution across its checkpoints.

    One frame per checkpoint, each drawn by :func:`plot_distribution_frame`
    with axes held fixed across the animation.  For an ordinary dataset the
    first (all-zero) checkpoint is skipped; for a nested dataset every
    interval is a frame, from the innermost outward.  The figure is left
    showing the final frame, as in the exposition.

    Note that this updates ``matplotlib.rcParams`` (font size, white save
    background), as the original scripts did.

    Parameters
    ----------
    X : dict
        An analyzed dataset.
    frames : list, optional
        The checkpoints to animate; defaults as described above.
    interval : int, optional
        Delay between frames in milliseconds (default 100).
    figsize : tuple, optional
        Figure size in inches (default ``(22, 11)``).
    font_size : int, optional
        Global font size (default 22).
    suptitle : str, optional
        Figure title (default ``'Primes in intervals'``).
    **frame_kwargs
        Passed through to :func:`plot_distribution_frame` (curve toggles,
        overlay, padding, and so on).

    Returns
    -------
    tuple
        ``(fig, anim)``: the matplotlib figure and the ``FuncAnimation``.
        Keep a reference to ``anim`` until it has been saved or displayed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.rcParams.update({"font.size": font_size})

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(suptitle)

    hor_axis, y_max = distribution_axes_limits(X)
    C = list(X["distribution"].keys())
    if frames is None:
        if "nested_interval_data" in X.keys():
            frames = C
        else:
            frames = C[1:]

    def draw(c: Any) -> None:
        plot_distribution_frame(
            ax, X, c, hor_axis=hor_axis, y_max=y_max, **frame_kwargs
        )

    anim = FuncAnimation(
        fig,
        # Some matplotlib versions type the frame function more narrowly than
        # the callback we pass; the extra 'unused-ignore' code keeps this
        # comment harmless on versions where no error is raised at all.
        draw,  # type: ignore[arg-type, unused-ignore]
        frames=frames,
        interval=interval,
        blit=False,
        repeat=False,
    )

    # This is supposed to remedy the blurry axis ticks/labels.
    plt.rcParams["savefig.facecolor"] = "white"

    draw(frames[-1])
    return fig, anim


def save_gif(anim: Any, path: str, fps: int = 10, dpi: int = 100) -> None:
    """Save an animation as a GIF with matplotlib's ``PillowWriter``.

    Parameters
    ----------
    anim : matplotlib animation
        As returned by :func:`animate_distribution`.
    path : str
        Output filename (conventionally ending in ``.gif``).
    fps : int, optional
        Frames per second (default 10).
    dpi : int, optional
        Resolution (default 100).
    """
    from matplotlib.animation import PillowWriter

    anim.save(path, dpi=dpi, writer=PillowWriter(fps=fps))


def save_mp4(anim: Any, path: str, fps: int = 10, dpi: int = 100) -> None:
    """Save an animation as an MP4 with matplotlib's ``FFMpegWriter``.

    Requires ``ffmpeg`` to be installed and on the path.

    Parameters
    ----------
    anim : matplotlib animation
        As returned by :func:`animate_distribution`.
    path : str
        Output filename (conventionally ending in ``.mp4``).
    fps : int, optional
        Frames per second (default 10).
    dpi : int, optional
        Resolution (default 100).
    """
    from matplotlib.animation import FFMpegWriter

    anim.save(path, dpi=dpi, writer=FFMpegWriter(fps=fps))