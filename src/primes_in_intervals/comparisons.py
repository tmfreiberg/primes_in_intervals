"""Comparing empirical counts against the theoretical predictions.

:func:`compare` attaches, for every checkpoint (or nested interval) and every
prime count ``m``, a pair of tuples: the probabilities
``(actual, binomial, frei, frei_alt)`` and the corresponding absolute counts
``(actual, binomial, frei, frei_alt)``.  :func:`winners` then scores the three
predictions per interval, both by total squared error over ``m`` and by the
number of ``m`` at which each prediction lands closest.

The refined predictions were derived for overlapping intervals only; the
comparisons are nevertheless permitted for disjoint and prime-start data (the
second-order terms may differ in those cases), and the display layer carries a
reminder to that effect.
"""

from __future__ import annotations

import numpy as np

from primes_in_intervals.intervals import Dataset
from primes_in_intervals.predictions import binom_pmf, frei, frei_alt

__all__ = ["compare", "winners"]


def compare(dataset: Dataset) -> Dataset | None:
    """Attach prediction comparisons to an analyzed dataset, in place.

    For each checkpoint ``c`` (or nested interval ``c = (c[0], c[1])``), the
    density of primes near the range's midpoint ``N`` is estimated two ways,
    ``p = 1 / (log N - 1)`` and ``p_alt = 1 / log N``, and each observed count
    ``m`` receives the item::

        (dist, binom_prob, frei_prob, frei_alt_prob),
        (count, binom_pred, frei_pred, frei_alt_pred)

    where the first tuple holds probabilities (the empirical relative
    frequency against :func:`~primes_in_intervals.predictions.binom_pmf` at
    ``p``, :func:`~primes_in_intervals.predictions.frei` at ``H * p``, and
    :func:`~primes_in_intervals.predictions.frei_alt` at ``H * p_alt``) and
    the second holds absolute numbers of intervals (each probability times the
    number of intervals sampled, truncated to an integer).  The number of
    intervals sampled is ``c - A`` for overlapping data, ``(c - A) // H`` for
    disjoint data, and the total prime count for prime-start data.

    The result is stored as a ``'comparison'`` item and noted in the header's
    ``'contents'``.

    Parameters
    ----------
    dataset : dict
        An analyzed dataset (run
        :func:`~primes_in_intervals.statistics.analyze` first).

    Returns
    -------
    dict or None
        The same dataset, modified; ``None`` with a message if the dataset has
        no data or has not been analyzed.
    """
    if "data" in dataset.keys():
        if "distribution" not in dataset.keys():
            return print(
                "Analyze data first, to obtain distribution data for comparison "
                "with theoretical predictions."
            )
        C = list(dataset["data"].keys())
        C.sort()  # just in case --- this is important
        interval_type = dataset["header"]["interval_type"]
        A = C[0]
        H = dataset["header"]["interval_length"]
        # For consistency with the keys:
        comparison: dict = {C[0]: {m: 0 for m in dataset["data"][C[0]].keys()}}
        for c in C[1:]:
            comparison[c] = {}
            N = (A + c) // 2  # midpoint of the interval (A, c]
            # more accurate estimate for the density of primes around (A, c]:
            p = 1 / (np.log(N) - 1)
            p_alt = 1 / np.log(N)  # estimate for the density
            if interval_type == "overlap":
                # the number of intervals considered, in the overlapping case
                multiplier = c - A
            if interval_type == "disjoint":
                # the number of intervals considered, in the disjoint case
                multiplier = (c - A) // H
            if interval_type == "prime_start":
                # the number of intervals considered, in the prime-start case
                multiplier = sum(dataset["data"][c].values())
            for m in dataset["data"][c].keys():
                binom_prob = binom_pmf(H, m, p)
                frei_prob = frei(H, m, H * p)
                frei_alt_prob = frei_alt(H, m, H * p_alt)
                # what dataset['data'][c][m] should be according to Cramer's model:
                binom_pred = int(binom_prob * multiplier)
                # what it should be up to second-order approximation, at least
                # around the centre of the distribution (overlapping intervals):
                frei_pred = int(frei_prob * multiplier)
                # the alternative estimate (overlapping intervals):
                frei_alt_pred = int(frei_alt_prob * multiplier)
                comparison[c][m] = (
                    (dataset["distribution"][c][m], binom_prob, frei_prob, frei_alt_prob),
                    (dataset["data"][c][m], binom_pred, frei_pred, frei_alt_pred),
                )
        dataset["comparison"] = {}
        for c in C:
            dataset["comparison"][c] = {}
            for m in comparison[c].keys():
                dataset["comparison"][c][m] = comparison[c][m]
        dataset["header"]["contents"].append("comparison - actual, binomial, frei, frei_alt")
        return dataset
    if "nested_interval_data" in dataset.keys():
        if "distribution" not in dataset.keys():
            return print(
                "Analyze data first, to obtain distribution data for comparison "
                "with theoretical predictions."
            )
        C = list(dataset["nested_interval_data"].keys())
        interval_type = dataset["header"]["interval_type"]
        H = dataset["header"]["interval_length"]
        comparison2: dict = {}
        for c in C:
            comparison2[c] = {}
            N = (c[0] + c[1]) // 2  # midpoint of the interval c = (c[0], c[1]].
            # more accurate estimate for the density of primes around the interval:
            p = 1 / (np.log(N) - 1)
            p_alt = 1 / np.log(N)  # estimate for the density
            if interval_type == "overlap":
                multiplier = c[1] - c[0]
            if interval_type == "disjoint":
                multiplier = (c[1] - c[0]) // H
            if interval_type == "prime_start":
                multiplier = sum(dataset["nested_interval_data"][c].values())
            for m in dataset["nested_interval_data"][c].keys():
                binom_prob = binom_pmf(H, m, p)
                frei_prob = frei(H, m, H * p)
                frei_alt_prob = frei_alt(H, m, H * p_alt)
                binom_pred = int(binom_prob * multiplier)
                frei_pred = int(frei_prob * multiplier)
                frei_alt_pred = int(frei_alt_prob * multiplier)
                comparison2[c][m] = (
                    (dataset["distribution"][c][m], binom_prob, frei_prob, frei_alt_prob),
                    (
                        dataset["nested_interval_data"][c][m],
                        binom_pred,
                        frei_pred,
                        frei_alt_pred,
                    ),
                )
        dataset["comparison"] = {}
        for c in C:
            dataset["comparison"][c] = {}
            for m in comparison2[c].keys():
                dataset["comparison"][c][m] = comparison2[c][m]
        dataset["header"]["contents"].append("comparison - actual, binomial, frei, frei_alt")
        return dataset
    if "data" not in dataset.keys() and "new_interval_data" not in dataset.keys():
        return print("No data to compare.")
    return None


def winners(dataset: Dataset) -> Dataset | None:
    """Score the three predictions per interval, in place.

    "Best" is judged in two senses.  First, the sum over ``m`` of the squared
    error between the actual and predicted interval counts (over the full run
    of ``m`` from the smallest to the largest with a nonzero comparison
    entry); the three predictions are ranked 1, 2, 3 by this score.  Second,
    for each ``m`` in that run, whichever prediction's count lands closest to
    the actual count "wins" that ``m`` (ties shared); the lists of ``m`` won
    and the resulting most/2nd-most/least tallies are recorded.

    The result is stored as a ``'winners'`` item, with keys ``'B sq error'``,
    ``'F sq error'``, ``'F* sq error'``, ``1``, ``2``, ``3``,
    ``'B wins for m in '``, ``'F wins for m in '``, ``'F* wins for m in '``,
    ``'most wins'``, ``'2nd most wins'``, and ``'least wins'`` per interval
    (``'B'`` the binomial, ``'F'`` the refined prediction, ``'F*'`` its
    alternative form), and noted in the header's ``'contents'``.

    Parameters
    ----------
    dataset : dict
        A dataset to which :func:`compare` has been applied.

    Returns
    -------
    dict or None
        The same dataset, modified; ``None`` with a message if comparisons are
        missing or winners already computed.
    """
    if "winners" in dataset.keys():
        return print("This function has already been applied to the data.")
    if "comparison" not in dataset.keys():
        return print(
            "Compare the data first, to obtain distribution data for comparison "
            "with theoretical predictions."
        )
    if "nested_interval_data" in dataset.keys():
        datakey = "nested_interval_data"
    elif "data" in dataset.keys():
        datakey = "data"
    else:
        return print("No data.")
    C = list(dataset[datakey].keys())
    win: dict = {}
    for c in C:
        win[c] = {}
        M = [m for m in dataset["comparison"][c].keys() if dataset["comparison"][c][m] != 0]
        if M != []:
            min_m, max_m = min(M), max(M)
            M = list(range(min_m, max_m + 1))
            square_error_binom = sum(
                [
                    (dataset["comparison"][c][m][1][0] - dataset["comparison"][c][m][1][1]) ** 2
                    for m in M
                ]
            )
            square_error_frei = sum(
                [
                    (dataset["comparison"][c][m][1][0] - dataset["comparison"][c][m][1][2]) ** 2
                    for m in M
                ]
            )
            square_error_frei_alt = sum(
                [
                    (dataset["comparison"][c][m][1][0] - dataset["comparison"][c][m][1][3]) ** 2
                    for m in M
                ]
            )
            win[c]["B sq error"] = square_error_binom
            win[c]["F sq error"] = square_error_frei
            win[c]["F* sq error"] = square_error_frei_alt
            square_error = [square_error_binom, square_error_frei, square_error_frei_alt]
            square_error.sort()
            for i in [0, 1, 2]:
                if square_error[i] == square_error_frei:
                    win[c][i + 1] = "F"
                if square_error[i] == square_error_frei_alt:
                    win[c][i + 1] = "F*"
                if square_error[i] == square_error_binom:
                    win[c][i + 1] = "B"
            win[c]["B wins for m in "] = []
            win[c]["F wins for m in "] = []
            win[c]["F* wins for m in "] = []
            mB, mF, mFalt = 0, 0, 0
            for m in M:
                temp_list = [
                    abs(dataset["comparison"][c][m][1][0] - dataset["comparison"][c][m][1][i])
                    for i in range(1, 4)
                ]
                min_diff = min(temp_list)
                diff_1 = abs(
                    dataset["comparison"][c][m][1][0] - dataset["comparison"][c][m][1][1]
                )
                if diff_1 == min_diff:
                    win[c]["B wins for m in "].append(m)
                    mB += 1
                diff_2 = abs(
                    dataset["comparison"][c][m][1][0] - dataset["comparison"][c][m][1][2]
                )
                if diff_2 == min_diff:
                    win[c]["F wins for m in "].append(m)
                    mF += 1
                diff_3 = abs(
                    dataset["comparison"][c][m][1][0] - dataset["comparison"][c][m][1][3]
                )
                if diff_3 == min_diff:
                    win[c]["F* wins for m in "].append(m)
                    mFalt += 1
            max_wins = [mB, mF, mFalt]
            max_wins.sort(reverse=True)
            win[c]["most wins"] = ""
            win[c]["2nd most wins"] = ""
            win[c]["least wins"] = ""
            if mB == max_wins[0]:
                win[c]["most wins"] += "B"
            if mF == max_wins[0]:
                win[c]["most wins"] += "F"
            if mFalt == max_wins[0]:
                win[c]["most wins"] += "F*"
            if mB == max_wins[1]:
                win[c]["2nd most wins"] += "B"
            if mF == max_wins[1]:
                win[c]["2nd most wins"] += "F"
            if mFalt == max_wins[1]:
                win[c]["2nd most wins"] += "F*"
            if mB == max_wins[2]:
                win[c]["least wins"] += "B"
            if mF == max_wins[2]:
                win[c]["least wins"] += "F"
            if mFalt == max_wins[2]:
                win[c]["least wins"] += "F*"

        if M == []:
            win[c] = {
                "B sq error": "-",
                "F sq error": "-",
                "F* sq error": "-",
                1: "-",
                2: "-",
                3: "-",
                "B wins for m in ": "-",
                "F wins for m in ": "-",
                "F* wins for m in ": "-",
                "most wins": "-",
                "2nd most wins": "-",
                "least wins": "-",
            }
    dataset["winners"] = win
    dataset["header"]["contents"].append("winners")
    return dataset
