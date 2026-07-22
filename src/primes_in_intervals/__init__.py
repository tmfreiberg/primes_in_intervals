"""Primes in intervals: computation, storage, analysis, and visualization.

Count how many intervals of a given length contain each possible number of
primes, under three sampling schemes (disjoint blocks, a sliding window, and
windows started at primes); persist the counts in SQLite; reorganize them
(sub-ranges, partitions, nested centered intervals); summarize the resulting
distributions; compare them against the Cramér-model binomial and two refined
predictions; and present everything as tables and animations.

The public API is flat, exactly as in the original single-file project::

    import primes_in_intervals as pii

    C = list(range(0, 10**6 + 1, 10**5))
    X = pii.intervals(C, 100, 'disjoint')
    pii.save(X)
    pii.analyze(X)
    pii.display(X)

For the full exposition with worked examples, see the project README.
"""

from primes_in_intervals.comparisons import compare, winners
from primes_in_intervals.dataio import (
    DB_PATH,
    ensure_tables,
    max_primes,
    retrieve,
    save,
    set_db,
    show_table,
)
from primes_in_intervals.display import display
from primes_in_intervals.intervals import (
    Dataset,
    MetaDict,
    anyIntervals,
    anyIntervals_cp,
    disjoint,
    disjoint_cp,
    intervals,
    overlap,
    overlap_cp,
    overlap_extension,
    prime_start,
    prime_start_cp,
    zeros,
)
from primes_in_intervals.plotting import (
    animate_distribution,
    distribution_axes_limits,
    plot_distribution_frame,
    save_gif,
    save_mp4,
)
from primes_in_intervals.predictions import MS, binom_pmf, frei, frei_alt
from primes_in_intervals.serialize import (
    dataset_from_json,
    dataset_to_json,
    read_dataset_json,
    write_dataset_json,
)
from primes_in_intervals.sieve import next_prime, postponed_sieve, prime_pi
from primes_in_intervals.statistics import analyze, dictionary_sort, dictionary_statistics
from primes_in_intervals.transforms import extract, nest, partition, unpartition

try:  # optional: lets `pii.dfi.export(df, 'table.png')` work as in the examples
    import dataframe_image as dfi  # noqa: F401
except ImportError:  # pragma: no cover - exercised only without the extra
    dfi = None

__version__ = "1.0.0"

__all__ = [
    "DB_PATH",
    "MS",
    "Dataset",
    "MetaDict",
    "analyze",
    "animate_distribution",
    "anyIntervals",
    "anyIntervals_cp",
    "binom_pmf",
    "compare",
    "dataset_from_json",
    "dataset_to_json",
    "dfi",
    "dictionary_sort",
    "dictionary_statistics",
    "disjoint",
    "disjoint_cp",
    "display",
    "distribution_axes_limits",
    "ensure_tables",
    "extract",
    "frei",
    "frei_alt",
    "intervals",
    "max_primes",
    "nest",
    "next_prime",
    "overlap",
    "overlap_cp",
    "overlap_extension",
    "partition",
    "plot_distribution_frame",
    "postponed_sieve",
    "prime_pi",
    "prime_start",
    "prime_start_cp",
    "read_dataset_json",
    "retrieve",
    "save",
    "save_gif",
    "save_mp4",
    "set_db",
    "show_table",
    "unpartition",
    "winners",
    "write_dataset_json",
    "zeros",
]