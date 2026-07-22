"""Smoke tests for the plotting layer (Agg backend, no display)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import primes_in_intervals as pii  # noqa: E402


class TestAxesLimits:
    def test_common_axis_and_height(self, analyzed_overlap):
        hor_axis, y_max = pii.distribution_axes_limits(analyzed_overlap)
        C = list(analyzed_overlap["distribution"].keys())
        assert hor_axis == list(analyzed_overlap["distribution"][C[-1]].keys())
        top = max(
            v
            for c in C
            for v in analyzed_overlap["distribution"][c].values()
        )
        assert y_max == top


class TestFrame:
    def test_flat_frame_renders(self, analyzed_overlap):
        fig, ax = plt.subplots()
        C = list(analyzed_overlap["distribution"].keys())
        pii.plot_distribution_frame(ax, analyzed_overlap, C[-1])
        assert ax.get_legend() is not None
        assert ax.get_xlabel() == r"$m$ (number of primes in an interval)"
        # bars + dots + at least the two default curves for overlap data
        assert len(ax.lines) >= 3
        plt.close(fig)

    def test_nested_frame_with_all_curves_and_note(self, nested_overlap):
        fig, ax = plt.subplots()
        keys = list(nested_overlap["distribution"].keys())
        pii.plot_distribution_frame(
            ax,
            nested_overlap,
            keys[0],
            show_binom_alt=True,
            show_frei=True,
            show_frei_alt=True,
            note="NB: a reminder",
            ylim_decimals=3,
        )
        labels = [line.get_label() for line in ax.lines]
        assert r"$\mathrm{Binom}(H,\lambda^*/H)$" in labels
        assert r"$\mathrm{F^*}(H,m,\lambda^*)$" in labels
        plt.close(fig)

    def test_frei_suppressed_off_overlap(self, prime_start_dataset):
        import copy

        ds = copy.deepcopy(prime_start_dataset)
        pii.analyze(ds)
        fig, ax = plt.subplots()
        C = list(ds["distribution"].keys())
        pii.plot_distribution_frame(ax, ds, C[-1])
        labels = [line.get_label() for line in ax.lines]
        assert r"$\mathrm{F}(H,m,\lambda)$" not in labels
        plt.close(fig)


class TestAnimation:
    def test_animate_and_save_gif(self, nested_overlap, tmp_path):
        keys = list(nested_overlap["distribution"].keys())
        fig, anim = pii.animate_distribution(nested_overlap, frames=keys[:3])
        out = tmp_path / "anim.gif"
        pii.save_gif(anim, str(out), fps=5, dpi=30)
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)

    @pytest.mark.filterwarnings("ignore:Animation was deleted:UserWarning")
    def test_default_frames_skip_trivial_checkpoint(self, analyzed_overlap):
        fig, anim = pii.animate_distribution(analyzed_overlap)
        C = list(analyzed_overlap["distribution"].keys())
        assert list(anim._iter_gen()) == C[1:]  # noqa: SLF001 - matplotlib internal, test only
        plt.close(fig)
