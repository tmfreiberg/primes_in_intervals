r"""Command-line interface for primes_in_intervals.

Every public function in the package is reachable from the shell, so any
claim in the exposition can be checked with a one-line command.  Two entry
points are installed, ``primes-in-intervals`` and the short ``pii``, both
running :func:`main`.  Every hyphenated command also answers to its Python
function name (``prime-start-cp`` and ``prime_start_cp`` are the same
command).

**Numbers.** Anywhere a number is expected, a small arithmetic expression is
accepted: ``24154952``, ``1_000_000``, ``2e6``, ``10**7``, ``exp(17)``,
``exp(17)-10**4``.  Only numeric literals, the operators ``+ - * / // **``,
unary minus, parentheses, and single-argument ``exp``, ``log``, and ``sqrt``
are allowed; anything else is rejected, so nothing resembling arbitrary code
is ever evaluated.  Where an integer is required, a fractional result is
truncated toward zero, exactly like Python's ``int()``, so ``exp(17)``
denotes ``int(np.exp(17)) = 24154952`` as in the exposition.

**Checkpoints.** Checkpointed commands take either ``--range START STOP
STEP``, which is inclusive of ``STOP`` (checkpoint lists always want both
endpoints, so this differs from Python's ``range`` on purpose), or an
explicit ``--checkpoints`` list of comma-separated expressions.

**Chaining.** Counting commands write to SQLite with ``--save`` and every
dataset can be dumped as JSON with ``--json [FILE]`` (``-`` or no argument
means standard output).  Dataset-consuming commands read their input either
from the database (``--retrieve H --type T [--index I]``) or from JSON
(``--from-json FILE``, with ``-`` meaning standard input), so pipelines can
be run step by step::

    pii intervals --range 0 10**6 10**5 -H 100 --type disjoint --save
    pii retrieve 100 --type disjoint --json | pii nest --from-json - | ...

or in one shot with the pipeline flags::

    pii intervals --range "exp(17)-10**4" "exp(17)+10**4" 100 -H 76 \\
        --nest --analyze --compare --winners --display --view winners

When JSON is written to standard output, informational messages from the
library (the ``retrieve`` summary, guard messages) are redirected to standard
error so the JSON stream stays clean for piping.

**Database.** Storage commands honor ``--db PATH``, then the ``PII_DB``
environment variable, then the default ``data/primes_in_intervals_db``
relative to the working directory.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import math
import sys
from collections.abc import Iterator
from itertools import count
from typing import Any

import primes_in_intervals as pii
from primes_in_intervals.dataio import _CAPTION
from primes_in_intervals.intervals import Dataset
from primes_in_intervals.serialize import read_dataset_json, write_dataset_json

__all__ = ["main"]


# --------------------------------------------------------------------------
# Safe arithmetic expressions
# --------------------------------------------------------------------------

#: Binary operators the expression parser accepts, mapped to evaluators.
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Pow: lambda a, b: a**b,
}

#: Single-argument functions the expression parser accepts.
_FUNCTIONS = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
}


def _eval_node(node: ast.AST, source: str) -> float:
    """Evaluate one node of a vetted arithmetic expression tree.

    Only the constructs listed in the module docstring are accepted; any
    other node type (names, attributes, subscripts, comprehensions, and so
    on) raises ``ValueError``, so the parser cannot be used to run code.

    Parameters
    ----------
    node : ast.AST
        The node to evaluate.
    source : str
        The original expression, for error messages.

    Returns
    -------
    int or float
        The node's value.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, source)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left, source)
        right = _eval_node(node.right, source)
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_node(node.operand, source)
        return -value if isinstance(node.op, ast.USub) else value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](_eval_node(node.args[0], source))
    raise ValueError(
        f"unsupported construct in expression {source!r}: only numbers, "
        "+ - * / // **, parentheses, and exp/log/sqrt are allowed"
    )


def parse_number(text: str) -> float:
    """Evaluate a numeric command-line argument to a float.

    See the module docstring for the accepted grammar.

    Parameters
    ----------
    text : str
        The argument, e.g. ``"2e6"`` or ``"exp(17)"``.

    Returns
    -------
    float
        The value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the expression is malformed or uses a disallowed construct, so
        argparse reports the problem against the right argument.
    """
    try:
        tree = ast.parse(text, mode="eval")
        return float(_eval_node(tree, text))
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_int(text: str) -> int:
    """Evaluate a numeric command-line argument to an integer.

    Fractional results are truncated toward zero, exactly like Python's
    ``int()``; in particular ``exp(17)`` gives ``24154952``, matching the
    exposition's ``int(np.exp(17))``.

    Parameters
    ----------
    text : str
        The argument.

    Returns
    -------
    int
        The truncated value.
    """
    return int(parse_number(text))


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated list of integer expressions.

    Parameters
    ----------
    text : str
        For example ``"0,10**5,2*10**5"``.  Commas separate items only;
        the accepted functions take a single argument, so no expression
        contains a comma of its own.

    Returns
    -------
    list of int
        The values, in the order given.
    """
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [parse_int(item) for item in items]


def parse_checkpoint(text: str) -> int | tuple[int, int] | str:
    """Parse a ``--checkpoint`` selector for the plot command.

    Parameters
    ----------
    text : str
        ``"last"`` for the final checkpoint, an integer expression for a
        flat dataset's checkpoint, or ``"lower,upper"`` for a nested
        dataset's interval.

    Returns
    -------
    str, int, or tuple
        The selector, ready for :func:`_pick_checkpoint`.
    """
    if text == "last":
        return "last"
    if "," in text:
        values = parse_int_list(text)
        if len(values) != 2:
            raise argparse.ArgumentTypeError(
                "a nested checkpoint is written as 'lower,upper'"
            )
        return (values[0], values[1])
    return parse_int(text)


# --------------------------------------------------------------------------
# Named generators for the anyIntervals commands
# --------------------------------------------------------------------------


def _squares() -> Iterator[int]:
    """Yield the positive perfect squares 1, 4, 9, 16, ..."""
    return (k * k for k in count(1))


#: Sequences selectable by name on ``any-intervals`` and
#: ``any-intervals-cp``.  Each entry is a zero-argument factory returning a
#: fresh strictly increasing generator, as those functions require.
GENERATORS: dict[str, Any] = {
    "primes": pii.postponed_sieve,
    "naturals": lambda: count(1),
    "odds": lambda: count(1, 2),
    "evens": lambda: count(2, 2),
    "squares": _squares,
}


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def _emit_mapping(mapping: dict[int, int], fmt: str, header: tuple[str, str]) -> None:
    """Print a frequency dictionary in the requested format.

    Parameters
    ----------
    mapping : dict
        ``{m: count}`` (or any two-column integer mapping).
    fmt : str
        ``'table'`` (aligned columns), ``'json'``, or ``'csv'``.
    header : tuple of str
        The two column names, e.g. ``("m", "count")``.
    """
    if fmt == "json":
        import json

        print(json.dumps({str(k): v for k, v in mapping.items()}, indent=2))
        return
    if fmt == "csv":
        print(f"{header[0]},{header[1]}")
        for k, v in mapping.items():
            print(f"{k},{v}")
        return
    width = max([len(header[0])] + [len(str(k)) for k in mapping]) if mapping else len(header[0])
    print(f"{header[0]:>{width}}  {header[1]}")
    for k, v in mapping.items():
        print(f"{k:>{width}}  {v}")


def _emit_dataframe(df: Any, fmt: str) -> None:
    """Print a pandas DataFrame as an aligned table or as CSV."""
    if fmt == "csv":
        sys.stdout.write(df.to_csv())
    else:
        print(df.to_string())


def _fail(message: str) -> int:
    """Report an error on standard error and return the failure exit code."""
    print(f"error: {message}", file=sys.stderr)
    return 1


def _stdout_guard(json_dest: str | None) -> contextlib.AbstractContextManager:
    """Redirect library prints to stderr when JSON is bound for stdout.

    The library reports through ``print`` (the ``retrieve`` summary, the
    guard messages).  When a command's JSON output goes to standard output,
    those messages would corrupt the stream, so they are diverted to
    standard error; they remain visible either way.

    Parameters
    ----------
    json_dest : str or None
        The resolved ``--json`` destination (``"-"`` means stdout).

    Returns
    -------
    context manager
        A redirection to stderr, or a null context.
    """
    if json_dest == "-":
        return contextlib.redirect_stdout(sys.stderr)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------
# Dataset flow: input, pipeline, display
# --------------------------------------------------------------------------


def _json_dest(args: argparse.Namespace) -> str | None:
    """Return the resolved ``--json`` destination, or ``None`` if absent.

    ``--json`` without a value means standard output (``"-"``).
    """
    dest = getattr(args, "json", None)
    return dest


def _load_dataset(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Dataset | None:
    """Load the input dataset named by ``--from-json`` or ``--retrieve``.

    Exactly one input source must be given.  For ``--retrieve``, if several
    datasets share the interval length, ``--index`` selects one; without it,
    the summaries are printed and the command fails so the user can choose.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments carrying ``from_json``, ``retrieve``, ``type``,
        ``index``, and ``db``.
    parser : argparse.ArgumentParser
        The command's parser, used to report usage errors.

    Returns
    -------
    dict or None
        The dataset, or ``None`` after printing a reason (the caller should
        exit with failure).
    """
    if (args.from_json is None) == (args.retrieve is None):
        parser.error("give exactly one input: --from-json FILE or --retrieve H")
    if args.from_json is not None:
        return read_dataset_json(args.from_json)
    with _stdout_guard(_json_dest(args)):
        found = pii.retrieve(args.retrieve, args.type, db_path=args.db)
    if found is None:
        return None
    if isinstance(found, list):
        if args.index is None:
            print(
                f"error: {len(found)} datasets found; select one with --index I",
                file=sys.stderr,
            )
            return None
        if not 0 <= args.index < len(found):
            print(
                f"error: --index must be in 0..{len(found) - 1}", file=sys.stderr
            )
            return None
        return found[args.index]
    return found


def _run_pipeline(dataset: Dataset, args: argparse.Namespace) -> Dataset | None:
    """Apply the requested pipeline steps to a dataset, in the fixed order.

    The order is nest, analyze, compare, winners; later steps require
    earlier ones, and the library's own guards report anything out of
    sequence.  ``nest`` returns a new dataset; the other steps modify in
    place.

    Parameters
    ----------
    dataset : dict
        The starting dataset.
    args : argparse.Namespace
        Parsed arguments carrying the boolean pipeline flags.

    Returns
    -------
    dict or None
        The resulting dataset, or ``None`` if a step declined (its message
        has already been printed).
    """
    guard = _stdout_guard(_json_dest(args))
    with guard:
        if getattr(args, "nest", False):
            nested = pii.nest(dataset)
            if nested is None:
                return None
            dataset = nested
        if getattr(args, "analyze", False):
            if pii.analyze(dataset) is None:
                return None
        if getattr(args, "compare", False):
            if pii.compare(dataset) is None:
                return None
        if getattr(args, "winners", False):
            if pii.winners(dataset) is None:
                return None
    return dataset


def _do_display(dataset: Dataset, args: argparse.Namespace) -> int:
    """Render a dataset with :func:`primes_in_intervals.display` and print it.

    The library's caption (a pandas Styler in a notebook) becomes a plain
    text line above the table here, since HTML styling has no terminal
    rendering.

    Parameters
    ----------
    dataset : dict
        The dataset to display.
    args : argparse.Namespace
        Parsed arguments carrying the display options.

    Returns
    -------
    int
        Process exit code.
    """
    view_winners = "show" if getattr(args, "view", "data") == "winners" else "no show"
    description = "off" if getattr(args, "no_description", False) else "on"
    zeroth = "no show" if getattr(args, "hide_zeroth", False) else "show"
    single_cell = "false" if getattr(args, "multi_cell", False) else "true"
    result = pii.display(
        dataset,
        orient=getattr(args, "orient", "index"),
        description="off",  # captions are printed separately below
        zeroth_item=zeroth,
        count=getattr(args, "count", "cumulative"),
        comparisons=getattr(args, "comparisons", "off"),
        single_cell=single_cell,
        winners=view_winners,
    )
    if result is None:
        return 1
    if description == "on":
        header = dataset["header"]
        interval_type = header["interval_type"]
        word = {
            "overlap": "overlapping",
            "disjoint": "disjoint",
            "prime_start": "left endpoint prime",
        }.get(interval_type, interval_type)
        counts = "non-cumulative" if getattr(args, "count", "cumulative") == "partition" else (
            "cumulative"
        )
        print(
            f"Interval type: {word}. Lower bound: {header['lower_bound']}. "
            f"Upper bound: {header['upper_bound']}. "
            f"Interval length: {header['interval_length']}. Partial counts: {counts}."
        )
        if getattr(args, "comparisons", "off") in ("absolute", "probabilities"):
            print(
                "In tuple (a,b,c,d), a is actual data, b is Binomial prediction, "
                "c is frei prediction, and d is frei_alt prediction."
            )
    _emit_dataframe(result, getattr(args, "format", "table"))
    return 0


# --------------------------------------------------------------------------
# Command handlers: sieve
# --------------------------------------------------------------------------


def cmd_primes(args: argparse.Namespace) -> int:
    """Print the first N primes, one per line."""
    P = pii.postponed_sieve()
    for _ in range(args.n):
        print(next(P))
    return 0


def cmd_next_prime(args: argparse.Namespace) -> int:
    """Print the first prime strictly greater than A."""
    print(pii.next_prime(args.a))
    return 0


def cmd_prime_pi(args: argparse.Namespace) -> int:
    """Print the number of primes in (X, Y]."""
    print(pii.prime_pi(args.x, args.y))
    return 0


# --------------------------------------------------------------------------
# Command handlers: plain counters
# --------------------------------------------------------------------------


def cmd_disjoint(args: argparse.Namespace) -> int:
    """Count primes in disjoint intervals and print the frequencies."""
    _emit_mapping(pii.disjoint(args.A, args.B, args.H), args.format, ("m", "g(m)"))
    return 0


def cmd_overlap(args: argparse.Namespace) -> int:
    """Count primes in the sliding window and print the frequencies."""
    _emit_mapping(pii.overlap(args.A, args.B, args.H), args.format, ("m", "h(m)"))
    return 0


def cmd_prime_start(args: argparse.Namespace) -> int:
    """Count primes in prime-starting intervals and print the frequencies."""
    _emit_mapping(pii.prime_start(args.M, args.N, args.H), args.format, ("m", "count"))
    return 0


def cmd_any_intervals(args: argparse.Namespace) -> int:
    """Run anyIntervals with two named generators and print the frequencies."""
    gen1 = GENERATORS[args.gen1]()
    gen2 = GENERATORS[args.gen2]()
    output = pii.anyIntervals(args.M, args.N, args.H, gen1, gen2)
    _emit_mapping(output, args.format, ("m", "count"))
    return 0


def cmd_overlap_extension(args: argparse.Namespace) -> int:
    """Run overlap_extension and print counts plus the realizing endpoints."""
    show_me, output = pii.overlap_extension(args.A, args.B, args.H, args.m)
    if args.format == "json":
        import json

        print(
            json.dumps(
                {
                    "counts": {str(k): v for k, v in output.items()},
                    "endpoints": {str(k): v for k, v in show_me.items()},
                },
                indent=2,
            )
        )
        return 0
    _emit_mapping(output, args.format, ("m", "h(m)"))
    for m in args.m:
        endpoints = show_me[m]
        print(f"a with exactly {m} primes in (a, a+{args.H}]: {endpoints}")
    return 0


# --------------------------------------------------------------------------
# Command handlers: checkpointed counters
# --------------------------------------------------------------------------


def _resolve_checkpoints(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[int]:
    """Build the checkpoint list from ``--range`` or ``--checkpoints``.

    ``--range START STOP STEP`` is inclusive of ``STOP`` (see the module
    docstring); exactly one of the two options must be given.
    """
    if (args.range is None) == (args.checkpoints is None):
        parser.error("give exactly one of --range START STOP STEP or --checkpoints LIST")
    if args.checkpoints is not None:
        return list(args.checkpoints)
    start, stop, step = args.range
    if step <= 0:
        parser.error("--range STEP must be positive")
    if stop < start:
        parser.error("--range STOP must not be less than START")
    return list(range(start, stop + 1, step))


def _finish_dataset(
    dataset: Dataset, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    """Run the pipeline, then emit JSON and/or the display table.

    Shared tail of ``intervals`` and ``retrieve``: applies the pipeline
    flags, honors ``--json``, and prints the display table when requested,
    or by default when no JSON was asked for (a bare sanity check should
    show something).
    """
    result = _run_pipeline(dataset, args)
    if result is None:
        return 1
    dest = _json_dest(args)
    if dest is not None:
        write_dataset_json(result, dest)
    if args.display or dest is None:
        return _do_display(result, args)
    return 0


def cmd_intervals(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Compute a checkpointed dataset; optionally save, chain, and show it."""
    C = _resolve_checkpoints(args, parser)
    dataset = pii.intervals(C, args.H, args.type)
    if dataset is None:  # unreachable through the CLI's choices, kept for safety
        return _fail(f"unknown interval type {args.type!r}")
    if args.save:
        with _stdout_guard(_json_dest(args)):
            pii.save(dataset, db_path=args.db)
    return _finish_dataset(dataset, args, parser)


def cmd_any_intervals_cp(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run anyIntervals_cp with named generators and print its mapping.

    The library function returns a bare checkpoint-to-frequencies mapping
    without a header, so the save and pipeline machinery does not apply
    (matching the library's own current limitation).
    """
    C = _resolve_checkpoints(args, parser)
    gen1 = GENERATORS[args.gen1]()
    gen2 = GENERATORS[args.gen2]()
    data = pii.anyIntervals_cp(C, args.H, gen1, gen2)
    if args.format == "json":
        import json

        encoded = {
            str(k): {str(m): v for m, v in row.items()} for k, row in data.items()
        }
        print(json.dumps(encoded, indent=2))
        return 0
    import pandas as pd

    _emit_dataframe(pd.DataFrame.from_dict(data, orient="index"), args.format)
    return 0


# --------------------------------------------------------------------------
# Command handlers: storage
# --------------------------------------------------------------------------


def cmd_save(args: argparse.Namespace) -> int:
    """Save a dataset read from JSON into the database."""
    dataset = read_dataset_json(args.from_json)
    pii.save(dataset, db_path=args.db)
    if "data" not in dataset:
        return 1  # save() has already explained
    return 0


def cmd_retrieve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Retrieve dataset(s) by interval length; optionally chain and show."""
    with _stdout_guard(_json_dest(args)):
        found = pii.retrieve(args.H, args.type, db_path=args.db)
    if found is None:
        return 1
    if isinstance(found, list):
        if len(found) == 0:
            # The library has printed "Found 0 datasets ..."; for scripting,
            # an empty retrieval is a failure.
            return 1
        wants_more = (
            args.index is not None
            or _json_dest(args) is not None
            or args.display
            or args.nest
            or args.analyze
            or args.compare
            or args.winners
        )
        if not wants_more:
            return 0  # the summaries have been printed; nothing else asked
        if args.index is None:
            return _fail(f"{len(found)} datasets found; select one with --index I")
        if not 0 <= args.index < len(found):
            return _fail(f"--index must be in 0..{len(found) - 1}")
        found = found[args.index]
    return _finish_dataset(found, args, parser)


def cmd_show_table(args: argparse.Namespace) -> int:
    """Print an entire raw database table."""
    df = pii.show_table(args.type, description="no description", db_path=args.db)
    if df is None:
        return 1
    if not args.no_description:
        # The library's caption is HTML styling in a notebook; in the
        # terminal, print its text (LaTeX and all) as a plain line.
        print(_CAPTION[args.type])
    _emit_dataframe(df, args.format)
    return 0


def cmd_ensure_tables(args: argparse.Namespace) -> int:
    """Create the database (and its directory) with empty tables if absent."""
    pii.ensure_tables(db_path=args.db)
    return 0


# --------------------------------------------------------------------------
# Command handlers: transforms and analysis (JSON in, JSON out)
# --------------------------------------------------------------------------


def _transform_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    apply: Any,
) -> int:
    """Shared shape of the transform and analysis commands.

    Load a dataset, apply one library function, and write the result as
    JSON (to standard output unless ``--json FILE`` says otherwise), so the
    commands compose with pipes.
    """
    if args.json is None:
        args.json = "-"  # these commands exist to be chained
    dataset = _load_dataset(args, parser)
    if dataset is None:
        return 1
    with _stdout_guard(_json_dest(args)):
        result = apply(dataset)
    if result is None:
        return 1
    write_dataset_json(result, args.json)
    return 0


def cmd_extract(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Extract a sub-range ('narrow') or sub-list ('filter') of checkpoints."""
    if (args.narrow is None) == (args.filter is None):
        parser.error("give exactly one of --narrow A B or --filter LIST")
    if args.narrow is not None:
        newC, option = list(args.narrow), "narrow"
    else:
        newC, option = list(args.filter), "filter"
    return _transform_command(
        args, parser, lambda ds: pii.extract(ds, newC, option=option)
    )


def cmd_partition(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Add per-gap counts to a dataset."""
    return _transform_command(args, parser, pii.partition)


def cmd_unpartition(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Rebuild cumulative counts from per-gap counts."""
    return _transform_command(args, parser, pii.unpartition)


def cmd_nest(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Convert checkpointed counts into nested centered intervals."""
    return _transform_command(args, parser, pii.nest)


def cmd_analyze(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Add distributions and summary statistics to a dataset."""
    return _transform_command(args, parser, pii.analyze)


def cmd_compare(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Attach the three predictions' comparisons to an analyzed dataset."""
    return _transform_command(args, parser, pii.compare)


def cmd_winners(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Score the predictions per interval on a compared dataset."""
    return _transform_command(args, parser, pii.winners)


def cmd_display(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Render a dataset as a table in the terminal."""
    dataset = _load_dataset(args, parser)
    if dataset is None:
        return 1
    return _do_display(dataset, args)


# --------------------------------------------------------------------------
# Command handlers: predictions
# --------------------------------------------------------------------------


def cmd_binom_pmf(args: argparse.Namespace) -> int:
    """Print the binomial probability of m successes in H trials at p."""
    print(float(pii.binom_pmf(args.H, args.m, args.p)))
    return 0


def cmd_frei(args: argparse.Namespace) -> int:
    """Print the prediction F(H, m, t)."""
    print(float(pii.frei(args.H, args.m, args.t)))
    return 0


def cmd_frei_alt(args: argparse.Namespace) -> int:
    """Print the alternative prediction F*(H, m, t)."""
    print(float(pii.frei_alt(args.H, args.m, args.t)))
    return 0


def cmd_ms(args: argparse.Namespace) -> int:
    """Print the Montgomery-Soundararajan constant 1 - gamma - log(2*pi)."""
    print(float(pii.MS))
    return 0


# --------------------------------------------------------------------------
# Command handlers: plotting
# --------------------------------------------------------------------------


def _pick_checkpoint(dataset: Dataset, selector: Any) -> Any:
    """Resolve a ``--checkpoint`` selector against a dataset's checkpoints.

    Parameters
    ----------
    dataset : dict
        An analyzed dataset.
    selector : str, int, or tuple
        ``"last"``, an integer checkpoint, or a nested ``(lower, upper)``.

    Returns
    -------
    int or tuple
        The chosen checkpoint.

    Raises
    ------
    KeyError
        If the selector names no checkpoint of the dataset.
    """
    C = list(dataset["distribution"].keys())
    if selector == "last":
        return C[-1]
    if selector in dataset["distribution"]:
        return selector
    raise KeyError(f"checkpoint {selector!r} not in dataset; choices end at {C[-1]!r}")


def _ensure_analyzed(dataset: Dataset) -> Dataset | None:
    """Analyze a dataset in place if it has not been analyzed yet.

    Plotting needs distributions and statistics; running :func:`analyze`
    automatically saves a chaining step, and a note goes to standard error
    so the behavior is visible.
    """
    if "distribution" not in dataset:
        print("note: dataset not analyzed; running analyze first", file=sys.stderr)
        return pii.analyze(dataset)
    return dataset


def _frame_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Translate plot/animate flags into plot_distribution_frame options."""
    overlay: Any
    if args.overlay == "auto":
        overlay = "auto"
    elif args.overlay == "off":
        overlay = None
    else:
        overlay = args.overlay  # a literal text box
    show_frei: bool | None
    if args.frei == "auto":
        show_frei = None
    else:
        show_frei = args.frei == "on"
    return {
        "show_binom": not args.no_binom,
        "show_binom_alt": args.binom_alt,
        "show_frei": show_frei,
        "show_frei_alt": args.frei_alt,
        "overlay": overlay,
        "overlay_position": (args.overlay_x, args.overlay_y),
        "note": args.note,
        "x_pad": args.x_pad,
        "ylim_decimals": args.ylim_decimals,
    }


def cmd_plot(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Render one checkpoint's distribution to an image file."""
    dataset = _load_dataset(args, parser)
    if dataset is None:
        return 1
    if _ensure_analyzed(dataset) is None:
        return 1
    import matplotlib

    matplotlib.use("Agg")  # file output only; no window, works headless
    import matplotlib.pyplot as plt

    try:
        checkpoint = _pick_checkpoint(dataset, args.checkpoint)
    except KeyError as exc:
        return _fail(str(exc))
    plt.rcParams.update({"font.size": args.font_size})
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    fig.suptitle(args.plot_title)
    pii.plot_distribution_frame(ax, dataset, checkpoint, **_frame_kwargs(args))
    plt.rcParams["savefig.facecolor"] = "white"
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {args.output}")
    return 0


def cmd_animate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Render the distribution's evolution to a GIF or MP4."""
    dataset = _load_dataset(args, parser)
    if dataset is None:
        return 1
    if _ensure_analyzed(dataset) is None:
        return 1
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = None
    if args.max_frames is not None:
        C = list(dataset["distribution"].keys())
        if "nested_interval_data" not in dataset:
            C = C[1:]
        frames = C[: args.max_frames]
    fig, anim = pii.animate_distribution(
        dataset,
        frames=frames,
        figsize=tuple(args.figsize),
        font_size=args.font_size,
        suptitle=args.plot_title,
        **_frame_kwargs(args),
    )
    out = str(args.output)
    if out.lower().endswith(".mp4"):
        pii.save_mp4(anim, out, fps=args.fps, dpi=args.dpi)
    else:
        pii.save_gif(anim, out, fps=args.fps, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print the package version."""
    print(pii.__version__)
    return 0


# --------------------------------------------------------------------------
# Parser construction
# --------------------------------------------------------------------------


def _add(subparsers: Any, name: str, aliases: list[str], help_text: str, **kwargs: Any) -> Any:
    """Add a subcommand whose Python-style alias is derived automatically.

    Every hyphenated name also answers to its underscore form, so shell
    usage matches the library's function names (``prime-start-cp`` and
    ``prime_start_cp`` are the same command).
    """
    auto = name.replace("-", "_")
    if auto != name and auto not in aliases:
        aliases = [*aliases, auto]
    return subparsers.add_parser(name, aliases=aliases, help=help_text, **kwargs)


def _build_parents() -> dict[str, argparse.ArgumentParser]:
    """Build the shared option groups reused across subcommands.

    Returns
    -------
    dict
        Parent parsers keyed by role: ``db`` (--db), ``format`` (--format),
        ``dataset_in`` (--from-json / --retrieve / --type / --index / --db),
        ``json_out`` (--json), ``pipeline`` (--nest .. --winners,
        --display), and ``display_opts`` (the display view options).
    """
    db = argparse.ArgumentParser(add_help=False)
    db.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="database file (default: $PII_DB or data/primes_in_intervals_db)",
    )

    fmt = argparse.ArgumentParser(add_help=False)
    fmt.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="output format (default: table)",
    )

    dataset_in = argparse.ArgumentParser(add_help=False, parents=[db])
    group = dataset_in.add_argument_group("dataset input (give exactly one)")
    group.add_argument(
        "--from-json",
        metavar="FILE",
        default=None,
        help="read the dataset from a JSON file ('-' for standard input)",
    )
    group.add_argument(
        "--retrieve",
        metavar="H",
        type=parse_int,
        default=None,
        help="load the dataset with interval length H from the database",
    )
    group.add_argument(
        "--type",
        choices=["disjoint", "overlap", "prime_start"],
        default="overlap",
        help="interval type for --retrieve (default: overlap)",
    )
    group.add_argument(
        "--index",
        type=int,
        default=None,
        help="which dataset to use when --retrieve finds several",
    )

    json_out = argparse.ArgumentParser(add_help=False)
    json_out.add_argument(
        "--json",
        metavar="FILE",
        nargs="?",
        const="-",
        default=None,
        help="write the resulting dataset as JSON ('-' or no value: stdout)",
    )

    display_opts = argparse.ArgumentParser(add_help=False)
    display_opts.add_argument(
        "--view",
        choices=["data", "winners"],
        default="data",
        help="table to show: the counts or the winners scoreboard",
    )
    display_opts.add_argument(
        "--orient",
        choices=["index", "columns"],
        default="index",
        help="checkpoints as rows (index) or columns",
    )
    display_opts.add_argument(
        "--count",
        choices=["cumulative", "partition"],
        default="cumulative",
        help="cumulative counts or per-gap counts (requires partitioned data)",
    )
    display_opts.add_argument(
        "--comparisons",
        choices=["off", "absolute", "probabilities"],
        default="off",
        help="show comparison tuples instead of plain counts",
    )
    display_opts.add_argument(
        "--multi-cell",
        action="store_true",
        help="nested comparisons: one column per prediction instead of tuples",
    )
    display_opts.add_argument(
        "--hide-zeroth",
        action="store_true",
        help="omit the first (all-zero) checkpoint row or column",
    )
    display_opts.add_argument(
        "--no-description",
        action="store_true",
        help="omit the description line above the table",
    )
    display_opts.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        help="table rendering (default: aligned text; csv for piping)",
    )

    pipeline = argparse.ArgumentParser(add_help=False, parents=[display_opts])
    pipeline.add_argument("--nest", action="store_true", help="apply nest()")
    pipeline.add_argument("--analyze", action="store_true", help="apply analyze()")
    pipeline.add_argument("--compare", action="store_true", help="apply compare()")
    pipeline.add_argument("--winners", action="store_true", help="apply winners()")
    pipeline.add_argument(
        "--display",
        action="store_true",
        help="print the display table (default when --json is not given)",
    )

    return {
        "db": db,
        "format": fmt,
        "dataset_in": dataset_in,
        "json_out": json_out,
        "pipeline": pipeline,
        "display_opts": display_opts,
    }


def _add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add the checkpoint options shared by the checkpointed commands."""
    parser.add_argument(
        "-H",
        "--length",
        dest="H",
        type=parse_int,
        required=True,
        help="interval length H",
    )
    parser.add_argument(
        "--range",
        nargs=3,
        type=parse_int,
        metavar=("START", "STOP", "STEP"),
        default=None,
        help="checkpoints START, START+STEP, ..., STOP (inclusive of STOP)",
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_int_list,
        metavar="LIST",
        default=None,
        help="explicit comma-separated checkpoint expressions",
    )


def _add_frame_args(parser: argparse.ArgumentParser) -> None:
    """Add the figure and curve options shared by plot and animate."""
    parser.add_argument("-o", "--output", required=True, help="output image file")
    parser.add_argument("--no-binom", action="store_true", help="omit the binomial curve")
    parser.add_argument(
        "--binom-alt",
        action="store_true",
        help="also draw the binomial at the alternative density 1/log N",
    )
    parser.add_argument(
        "--frei",
        choices=["auto", "on", "off"],
        default="auto",
        help="draw F (auto: only for overlapping intervals, as in the exposition)",
    )
    parser.add_argument("--frei-alt", action="store_true", help="also draw F*")
    parser.add_argument(
        "--overlay",
        default="auto",
        help="statistics box: 'auto', 'off', or literal text",
    )
    parser.add_argument(
        "--overlay-x", type=float, default=0.70, help="overlay x position (axes fraction)"
    )
    parser.add_argument(
        "--overlay-y", type=float, default=0.15, help="overlay y position (axes fraction)"
    )
    parser.add_argument("--note", default=None, help="extra text drawn at the top of the axes")
    parser.add_argument(
        "--x-pad", type=float, default=0.5, help="horizontal padding beyond first/last m"
    )
    parser.add_argument(
        "--ylim-decimals",
        type=int,
        default=2,
        help="round the vertical limit up at this many decimals",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "IN"),
        default=[22.0, 11.0],
        help="figure size in inches (default: 22 11, as in the exposition)",
    )
    parser.add_argument("--font-size", type=int, default=22, help="global font size")
    parser.add_argument(
        "--plot-title", default="Primes in intervals", help="figure title"
    )
    parser.add_argument("--dpi", type=int, default=100, help="output resolution")


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser for the ``pii`` command."""
    parser = argparse.ArgumentParser(
        prog="pii",
        description="Count, store, analyze, and plot primes in intervals.",
        epilog=(
            "Numbers accept expressions like 2e6, 10**7, exp(17)-10**4. "
            "See docs/cli.md for a full reference with examples."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {pii.__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    parents = _build_parents()

    # ---------------- sieve
    p = _add(sub, "primes", [], "print the first N primes")
    p.add_argument("n", type=parse_int, help="how many primes")
    p.set_defaults(func=cmd_primes)

    p = _add(sub, "next-prime", [], "first prime strictly greater than A")
    p.add_argument("a", type=parse_int)
    p.set_defaults(func=cmd_next_prime)

    p = _add(sub, "prime-pi", [], "number of primes in (X, Y]")
    p.add_argument("x", type=parse_int)
    p.add_argument("y", type=parse_int)
    p.set_defaults(func=cmd_prime_pi)

    # ---------------- plain counters
    p = _add(
        sub,
        "disjoint",
        [],
        "primes in disjoint intervals over (A, B]",
        parents=[parents["format"]],
    )
    p.add_argument("A", type=parse_int)
    p.add_argument("B", type=parse_int)
    p.add_argument("H", type=parse_int)
    p.set_defaults(func=cmd_disjoint)

    p = _add(
        sub,
        "overlap",
        [],
        "primes in the sliding window over (A, B]",
        parents=[parents["format"]],
    )
    p.add_argument("A", type=parse_int)
    p.add_argument("B", type=parse_int)
    p.add_argument("H", type=parse_int)
    p.set_defaults(func=cmd_overlap)

    p = _add(
        sub,
        "prime-start",
        [],
        "primes in (p, p+H] for primes p in (M, N]",
        parents=[parents["format"]],
    )
    p.add_argument("M", type=parse_int)
    p.add_argument("N", type=parse_int)
    p.add_argument("H", type=parse_int)
    p.set_defaults(func=cmd_prime_start)

    p = _add(
        sub,
        "any-intervals",
        ["anyIntervals"],
        "count one named sequence inside intervals started by another",
        parents=[parents["format"]],
    )
    p.add_argument("M", type=parse_int)
    p.add_argument("N", type=parse_int)
    p.add_argument("H", type=parse_int)
    p.add_argument("--gen1", choices=sorted(GENERATORS), default="primes", help="left endpoints")
    p.add_argument("--gen2", choices=sorted(GENERATORS), default="primes", help="counted elements")
    p.set_defaults(func=cmd_any_intervals)

    p = _add(
        sub,
        "overlap-extension",
        [],
        "overlap counts plus the endpoints realizing chosen m",
        parents=[parents["format"]],
    )
    p.add_argument("A", type=parse_int)
    p.add_argument("B", type=parse_int)
    p.add_argument("H", type=parse_int)
    p.add_argument("--m", type=parse_int_list, required=True, help="comma-separated m to trace")
    p.set_defaults(func=cmd_overlap_extension)

    # ---------------- checkpointed counters
    intervals_help = "checkpointed counts; can save, chain, display, and dump JSON"
    p = _add(
        sub,
        "intervals",
        [],
        intervals_help,
        parents=[parents["pipeline"], parents["json_out"], parents["db"]],
    )
    _add_checkpoint_args(p)
    p.add_argument(
        "--type",
        choices=["disjoint", "overlap", "prime_start"],
        default="overlap",
        help="interval type (default: overlap)",
    )
    p.add_argument("--save", action="store_true", help="store the raw counts in the database")
    p.set_defaults(func=cmd_intervals, needs_parser=True)

    for name, itype in [
        ("disjoint-cp", "disjoint"),
        ("overlap-cp", "overlap"),
        ("prime-start-cp", "prime_start"),
    ]:
        p = _add(
            sub,
            name,
            [],
            f"intervals --type {itype}",
            parents=[parents["pipeline"], parents["json_out"], parents["db"]],
        )
        _add_checkpoint_args(p)
        p.add_argument("--save", action="store_true", help="store the raw counts in the database")
        p.set_defaults(func=cmd_intervals, needs_parser=True, type=itype)

    p = _add(
        sub,
        "any-intervals-cp",
        ["anyIntervals_cp"],
        "checkpointed anyIntervals (bare mapping; no header, save, or pipeline)",
        parents=[parents["format"]],
    )
    _add_checkpoint_args(p)
    p.add_argument("--gen1", choices=sorted(GENERATORS), default="primes")
    p.add_argument("--gen2", choices=sorted(GENERATORS), default="primes")
    p.set_defaults(func=cmd_any_intervals_cp, needs_parser=True)

    # ---------------- storage
    p = _add(sub, "save", [], "save a JSON dataset into the database", parents=[parents["db"]])
    p.add_argument("--from-json", metavar="FILE", required=True, help="dataset JSON ('-': stdin)")
    p.set_defaults(func=cmd_save)

    p = _add(
        sub,
        "retrieve",
        [],
        "load dataset(s) by interval length; can chain, display, dump JSON",
        parents=[parents["pipeline"], parents["json_out"], parents["db"]],
    )
    p.add_argument("H", type=parse_int, help="interval length")
    p.add_argument(
        "--type",
        choices=["disjoint", "overlap", "prime_start"],
        default="overlap",
        help="interval type (default: overlap)",
    )
    p.add_argument("--index", type=int, default=None, help="pick one of several datasets")
    p.set_defaults(func=cmd_retrieve, needs_parser=True)

    p = _add(
        sub,
        "show-table",
        [],
        "print an entire raw database table",
        parents=[parents["db"]],
    )
    p.add_argument(
        "--type",
        choices=["disjoint", "overlap", "prime_start"],
        default="overlap",
    )
    p.add_argument("--no-description", action="store_true", help="omit the caption line")
    p.add_argument("--format", choices=["table", "csv"], default="table")
    p.set_defaults(func=cmd_show_table)

    p = _add(
        sub,
        "ensure-tables",
        [],
        "create the database and tables if absent",
        parents=[parents["db"]],
    )
    p.set_defaults(func=cmd_ensure_tables)

    # ---------------- transforms
    p = _add(
        sub,
        "extract",
        [],
        "restrict a dataset to a sub-range or sub-list of checkpoints",
        parents=[parents["dataset_in"], parents["json_out"]],
    )
    p.add_argument(
        "--narrow",
        nargs=2,
        type=parse_int,
        metavar=("A", "B"),
        default=None,
        help="keep checkpoints in [A, B], re-based at the new lower bound",
    )
    p.add_argument(
        "--filter",
        type=parse_int_list,
        metavar="LIST",
        default=None,
        help="keep exactly these checkpoints (comma-separated)",
    )
    p.set_defaults(func=cmd_extract, needs_parser=True)

    for name, handler, help_text in [
        ("partition", cmd_partition, "add per-gap counts (successive differences)"),
        ("unpartition", cmd_unpartition, "rebuild cumulative counts from per-gap counts"),
        ("nest", cmd_nest, "convert to nested centered intervals"),
        ("analyze", cmd_analyze, "add distributions and summary statistics"),
        ("compare", cmd_compare, "attach the three predictions' comparisons"),
        ("winners", cmd_winners, "score the predictions per interval"),
    ]:
        p = _add(
            sub,
            name,
            [],
            help_text,
            parents=[parents["dataset_in"], parents["json_out"]],
        )
        p.set_defaults(func=handler, needs_parser=True)

    p = _add(
        sub,
        "display",
        [],
        "render a dataset as a table",
        parents=[parents["dataset_in"], parents["display_opts"], parents["json_out"]],
    )
    p.set_defaults(func=cmd_display, needs_parser=True)

    # ---------------- predictions
    p = _add(sub, "binom-pmf", [], "binomial probability of m successes in H trials at p")
    p.add_argument("H", type=parse_int)
    p.add_argument("m", type=parse_number, help="number of successes (may be real)")
    p.add_argument("p", type=parse_number, help="success probability")
    p.set_defaults(func=cmd_binom_pmf)

    p = _add(sub, "frei", [], "the prediction F(H, m, t)")
    p.add_argument("H", type=parse_int)
    p.add_argument("m", type=parse_number, help="number of primes (may be real)")
    p.add_argument("t", type=parse_number, help="the Poisson parameter lambda")
    p.set_defaults(func=cmd_frei)

    p = _add(sub, "frei-alt", [], "the alternative prediction F*(H, m, t)")
    p.add_argument("H", type=parse_int)
    p.add_argument("m", type=parse_number)
    p.add_argument("t", type=parse_number, help="the Poisson parameter lambda*")
    p.set_defaults(func=cmd_frei_alt)

    p = _add(sub, "ms", ["MS"], "the Montgomery-Soundararajan constant")
    p.set_defaults(func=cmd_ms)

    # ---------------- plotting
    p = _add(
        sub,
        "plot",
        [],
        "render one checkpoint's distribution to an image",
        parents=[parents["dataset_in"], parents["json_out"]],
    )
    p.add_argument(
        "--checkpoint",
        type=parse_checkpoint,
        default="last",
        help="'last' (default), an integer checkpoint, or 'lower,upper'",
    )
    _add_frame_args(p)
    p.set_defaults(func=cmd_plot, needs_parser=True)

    p = _add(
        sub,
        "animate",
        [],
        "render the distribution's evolution to a GIF or MP4",
        parents=[parents["dataset_in"], parents["json_out"]],
    )
    p.add_argument("--fps", type=int, default=10, help="frames per second")
    p.add_argument(
        "--max-frames", type=parse_int, default=None, help="animate only the first N frames"
    )
    _add_frame_args(p)
    p.set_defaults(func=cmd_animate, needs_parser=True)

    p = _add(sub, "version", [], "print the package version")
    p.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the process exit code.

    Parameters
    ----------
    argv : list of str, optional
        Arguments, without the program name; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success, 1 on a reported failure (argparse exits with 2 on
        usage errors, as usual).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "needs_parser", False):
            return int(args.func(args, parser))
        return int(args.func(args))
    except BrokenPipeError:
        # The reader went away (e.g. `pii primes 10**6 | head`); die quietly,
        # devnull-ing stdout so the interpreter's shutdown flush cannot raise.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())