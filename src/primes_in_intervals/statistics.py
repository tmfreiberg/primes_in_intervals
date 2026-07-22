"""Summary statistics for interval-count frequency dictionaries.

:func:`dictionary_statistics` turns one frequency dictionary ``{m: count}``
into relative frequencies plus its mean, second moment, variance, standard
deviation, median, and mode(s).  :func:`analyze` applies it across every
checkpoint of a dataset, adding ``'distribution'`` and ``'statistics'`` items
in place; these are what the display and plotting layers consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from primes_in_intervals.intervals import Dataset

__all__ = ["analyze", "dictionary_sort", "dictionary_statistics"]


def dictionary_sort(dictionary: dict) -> dict:
    """Return a copy of ``dictionary`` with items in sorted key order.

    Parameters
    ----------
    dictionary : dict
        Any dictionary with sortable keys.

    Returns
    -------
    dict
        A new dictionary; insertion order is the sorted key order.
    """
    L = list(dictionary.keys())
    L.sort()
    sorted_dictionary = {}
    for k in L:
        sorted_dictionary[k] = dictionary[k]
    return sorted_dictionary


def dictionary_statistics(dictionary: dict[int, int]) -> dict[str, Any]:
    """Summarize a frequency dictionary.

    The input maps values to their frequencies (how many times each value
    occurs in some data).  The output collects the relative frequencies and
    the standard summary statistics of the underlying data.

    The median follows the usual convention: the middle value when the number
    of objects counted is odd, and the average of the two middle values when
    it is even.  The mode is returned as a list, since it need not be unique.

    Parameters
    ----------
    dictionary : dict
        ``{value: frequency}`` with numeric keys and nonnegative counts.

    Returns
    -------
    dict
        With items ``'dist'`` (relative frequencies, in sorted key order),
        ``'mean'``, ``'2ndmom'``, ``'var'``, ``'sdv'``, ``'med'``, and
        ``'mode'``.
    """
    frequencies = dictionary_sort(dictionary)
    relative_frequencies: dict[int, float] = {}
    number_of_objects_counted = 0
    mean = 0.0
    median: float = 0
    mode = []
    second_moment = 0.0
    variance = 0.0
    standard_deviation = 0.0
    M = max(frequencies.values())
    for s in frequencies.keys():
        number_of_objects_counted += frequencies[s]
        mean += s * frequencies[s]
        second_moment += (s**2) * frequencies[s]
        if frequencies[s] == M:
            mode.append(s)
    mean = mean / number_of_objects_counted
    second_moment = second_moment / number_of_objects_counted
    variance = second_moment - mean**2
    standard_deviation = np.sqrt(variance)

    # A little subroutine for computing the median...
    temp_counter = 0
    if number_of_objects_counted % 2 == 1:
        for s in frequencies.keys():
            if temp_counter < number_of_objects_counted / 2:
                temp_counter += frequencies[s]
                if temp_counter > number_of_objects_counted / 2:
                    median = s
    if number_of_objects_counted % 2 == 0:
        for s in frequencies.keys():
            if temp_counter < number_of_objects_counted / 2:
                temp_counter += frequencies[s]
                if temp_counter >= number_of_objects_counted / 2:
                    median = s
        temp_counter = 0
        for s in frequencies.keys():
            if temp_counter < 1 + (number_of_objects_counted / 2):
                temp_counter += frequencies[s]
                if temp_counter >= 1 + (number_of_objects_counted / 2):
                    median = (median + s) / 2

    # Finally, let's get the relative frequencies.
    for s in frequencies.keys():
        relative_frequencies[s] = frequencies[s] / number_of_objects_counted

    output_dictionary: dict[str, Any] = {}
    output_dictionary["dist"] = relative_frequencies
    output_dictionary["mean"] = mean
    output_dictionary["2ndmom"] = second_moment
    output_dictionary["var"] = variance
    output_dictionary["sdv"] = standard_deviation
    output_dictionary["med"] = median
    output_dictionary["mode"] = mode

    return output_dictionary


def analyze(dataset: Dataset) -> Dataset | None:
    """Add per-checkpoint distributions and statistics to a dataset, in place.

    For a dataset with a ``'data'`` item, the first (all-zero) checkpoint gets
    empty placeholders and every later checkpoint gets its relative-frequency
    distribution and summary statistics.  For a nested dataset (with a
    ``'nested_interval_data'`` item), every nested interval is summarized.
    The new ``'distribution'`` and ``'statistics'`` items are recorded in the
    header's ``'contents'``.

    Parameters
    ----------
    dataset : dict
        A dataset with a ``'data'`` or ``'nested_interval_data'`` item, not
        yet analyzed.

    Returns
    -------
    dict or None
        The same dataset, modified; ``None`` with a message if there is
        nothing to analyze or it has been analyzed already.
    """
    if "distribution" in dataset.keys() and "statistics" in dataset.keys():
        return print("Data has already been analyzed.")
    if "data" in dataset.keys():
        C = list(dataset["data"].keys())
        # No meaningful statistics for the trivial item.
        dataset["distribution"] = {C[0]: {}}
        dataset["statistics"] = {C[0]: {}}
        for c in C[1:]:
            temp_dict = dictionary_statistics(dataset["data"][c])
            dataset["distribution"][c] = temp_dict["dist"]
            dataset["statistics"][c] = {}
            dataset["statistics"][c]["mean"] = temp_dict["mean"]
            dataset["statistics"][c]["2ndmom"] = temp_dict["2ndmom"]
            dataset["statistics"][c]["var"] = temp_dict["var"]
            dataset["statistics"][c]["sdv"] = temp_dict["sdv"]
            dataset["statistics"][c]["med"] = temp_dict["med"]
            dataset["statistics"][c]["mode"] = temp_dict["mode"]
        dataset["header"]["contents"].append("distribution")
        dataset["header"]["contents"].append("statistics")
        return dataset
    if "nested_interval_data" in dataset.keys():
        C = list(dataset["nested_interval_data"].keys())
        dataset["distribution"] = {}
        dataset["statistics"] = {}
        for c in C:
            temp_dict = dictionary_statistics(dataset["nested_interval_data"][c])
            dataset["distribution"][c] = temp_dict["dist"]
            dataset["statistics"][c] = {}
            dataset["statistics"][c]["mean"] = temp_dict["mean"]
            dataset["statistics"][c]["2ndmom"] = temp_dict["2ndmom"]
            dataset["statistics"][c]["var"] = temp_dict["var"]
            dataset["statistics"][c]["sdv"] = temp_dict["sdv"]
            dataset["statistics"][c]["med"] = temp_dict["med"]
            dataset["statistics"][c]["mode"] = temp_dict["mode"]
        dataset["header"]["contents"].append("distribution")
        dataset["header"]["contents"].append("statistics")
        return dataset
    return print("No data to analyze.")
