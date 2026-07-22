# Command-line interface

Installing the package (`pip install -e .`) provides two identical console
commands: `primes-in-intervals` and the short form `pii`, which this document
uses throughout. Every public function in the package is reachable from the
shell, so any claim in the exposition can be checked with a one-line command.

Every hyphenated command also answers to its Python function name:
`prime-start-cp` and `prime_start_cp` are the same command. Run
`pii --help` for the command list and `pii COMMAND --help` for any
command's options.

## Numbers are small expressions

Anywhere a number is expected, a small arithmetic expression is accepted:

```
24154952    1_000_000    2e6    10**7    exp(17)    exp(17)-10**4    sqrt(49)
```

Only numeric literals, the operators `+ - * / // **`, unary minus,
parentheses, and single-argument `exp`, `log`, and `sqrt` are allowed.
Anything else, names and calls included, is rejected with a usage error, so
nothing resembling arbitrary code is ever evaluated.

Where an integer is required, a fractional result is truncated toward zero,
exactly like Python's `int()`. In particular `exp(17)` denotes
`int(np.exp(17)) = 24154952`, matching the exposition, and
`exp(17)-10**4 = 24144952` is consistent with it, since shifting by an
integer does not move the fractional part.

Quote arguments that contain `*` or parentheses so the shell does not
expand them: `--range "exp(17)-10**4" "exp(17)+10**4" 100`.

## The database

Storage commands resolve the database location in this order:

1. `--db PATH` on the command line;
2. the `PII_DB` environment variable;
3. the default `data/primes_in_intervals_db`, relative to the current
   working directory, so run from the repository root.

`pii ensure-tables` creates the file, its directory, and the three empty
tables; `save` does the same on demand. Reading commands (`retrieve`,
`show-table`) do not create anything: if the file is absent they report the
missing table and fail.

## Checkpoints

Checkpointed commands take the interval length as `-H` (or `--length`) and
the checkpoints in one of two ways:

- `--range START STOP STEP`: the checkpoints `START, START+STEP, ..., STOP`.
  This is **inclusive of STOP**, unlike Python's `range`, because a
  checkpoint list always wants both endpoints; the exposition writes
  `range(A, B+1, step)` for the same reason.
- `--checkpoints LIST`: an explicit comma-separated list of expressions,
  such as `--checkpoints "0,10**5,3*10**5"`.

## Chaining

Datasets flow between commands in two ways, and the two combine freely.

**Through SQLite.** `--save` on a counting command stores the raw counts;
`retrieve` loads them back and can run the whole pipeline in one shot:

```
pii intervals --range 0 10**6 10**5 -H 100 --type disjoint --save
pii retrieve 100 --type disjoint --nest --analyze --compare --winners \
    --display --view winners
```

**Through JSON.** `--json [FILE]` writes the current dataset as JSON
(`-` or no value means standard output); dataset-consuming commands read
with `--from-json FILE` (`-` means standard input). The transform and
analysis commands default to JSON on standard output, so they compose with
pipes:

```
pii retrieve 100 --type disjoint --json |
    pii nest --from-json - |
    pii analyze --from-json - |
    pii display --from-json -
```

When JSON is bound for standard output, informational messages from the
library (the `retrieve` summary, the guard messages) are diverted to
standard error, so the JSON stream stays clean for the next command. On
Windows, PowerShell and `cmd.exe` pipe these commands the same way.

The JSON convention is documented in `primes_in_intervals/serialize.py`:
integer keys become decimal strings, nested-interval keys become
`"lower,upper"`, tuples inside the `'comparison'` item are restored on
reading, and NumPy or SymPy scalars become plain numbers. Reading a file
back therefore gives a dataset numerically equal to, though not always
type-identical with, the one written.

## Command reference

### Sieve

| Command | Meaning |
| --- | --- |
| `pii primes N` | print the first N primes, one per line |
| `pii next-prime A` | first prime strictly greater than A |
| `pii prime-pi X Y` | number of primes in (X, Y] |

### Counters

| Command | Meaning |
| --- | --- |
| `pii disjoint A B H` | primes in the disjoint intervals (A, A+H], (A+H, A+2H], ... |
| `pii overlap A B H` | primes in (a, a+H] for every a in (A, B] |
| `pii prime-start M N H` | primes in (p, p+H] for the primes p in (M, N] |
| `pii any-intervals M N H --gen1 G --gen2 G` | count one named sequence inside intervals started by another |
| `pii overlap-extension A B H --m LIST` | overlap counts, plus the left endpoints realizing each m in LIST |

All five take `--format table|json|csv` (default: an aligned table). The
named generators for `any-intervals` are `primes`, `naturals`, `odds`,
`evens`, and `squares`; with the default `primes` for both, the command
reproduces `prime-start`.

### Checkpointed counters

`pii intervals` computes a checkpointed dataset and is the main entry
point:

```
pii intervals (--range START STOP STEP | --checkpoints LIST) -H LENGTH
    [--type disjoint|overlap|prime_start] [--save] [--db PATH]
    [--nest] [--analyze] [--compare] [--winners]
    [--display] [display options] [--json [FILE]]
```

`--save` stores the raw counts before any pipeline step runs. The pipeline
flags apply in the fixed order nest, analyze, compare, winners; later steps
need earlier ones, and the library's guard messages explain anything out of
sequence. With no `--json`, the command ends by printing the display table,
so a bare invocation is a self-contained sanity check.

`disjoint-cp`, `overlap-cp`, and `prime-start-cp` are the same command with
the type preset. `any-intervals-cp` takes the generator options and prints
the bare checkpoint-to-frequencies mapping; the library function returns no
header, so saving and the pipeline do not apply to it.

### Storage

| Command | Meaning |
| --- | --- |
| `pii save --from-json FILE` | store a dataset's raw counts in the database |
| `pii retrieve H [--type T] [--index I]` | load dataset(s) with interval length H |
| `pii show-table [--type T]` | print an entire raw table (`--format table\|csv`, `--no-description`) |
| `pii ensure-tables` | create the database, its directory, and empty tables |

`retrieve` prints the library's summary of what it found. If several
datasets share the length, the summaries alone are fine, but doing anything
further (pipeline, `--display`, `--json`) requires choosing one with
`--index I` (zero-based, in the order listed). Retrieving nothing exits
with failure, which makes shell scripts honest. `retrieve` accepts the same
pipeline, display, and JSON options as `intervals`.

### Transforms and analysis

`extract`, `partition`, `unpartition`, `nest`, `analyze`, `compare`, and
`winners` each apply one library function. Input comes from
`--from-json FILE` or `--retrieve H [--type T] [--index I]`; output is JSON
on standard output unless `--json FILE` names a file. `extract` takes
either `--narrow A B` (keep the checkpoints in [A, B], re-based) or
`--filter LIST` (keep exactly those checkpoints).

### Display

```
pii display (--from-json FILE | --retrieve H [--type T] [--index I])
    [--view data|winners] [--orient index|columns]
    [--count cumulative|partition] [--comparisons off|absolute|probabilities]
    [--multi-cell] [--hide-zeroth] [--no-description] [--format table|csv]
```

The library's caption is HTML styling for notebooks; in the terminal it
becomes a plain description line above the table (suppress with
`--no-description`). The same options are available wherever `--display`
is, on `intervals` and `retrieve`. The description line is printed for
every view; the library itself attaches captions only to the flat data
view.

### Predictions

| Command | Meaning |
| --- | --- |
| `pii binom-pmf H m p` | binomial probability of m successes in H trials |
| `pii frei H m t` | the prediction F(H, m, t) |
| `pii frei-alt H m t` | the alternative prediction F*(H, m, t) |
| `pii ms` | the constant 1 - gamma - log(2 pi) = -1.415092731310878... |

Here `m` and `t` may be real: `pii frei 76 5 76/16`.

### Plotting

```
pii plot (--from-json ... | --retrieve ...) -o FILE.png
    [--checkpoint last|C|"lower,upper"] [curve and figure options]
pii animate (--from-json ... | --retrieve ...) -o FILE.gif|FILE.mp4
    [--fps N] [--max-frames N] [curve and figure options]
```

If the dataset has not been analyzed, `analyze` runs automatically and a
note goes to standard error. `--checkpoint` selects the frame: `last` (the
default), an integer checkpoint for flat data, or `"lower,upper"` for a
nested interval. The curve options mirror the library:
`--no-binom`, `--binom-alt`, `--frei auto|on|off` (auto draws F only for
overlapping intervals, as in the exposition), `--frei-alt`,
`--overlay auto|off|TEXT` with `--overlay-x/--overlay-y`, `--note TEXT`,
`--x-pad`, and `--ylim-decimals`. Figure options: `--figsize W H` (default
22 11), `--font-size` (default 22), `--plot-title`, `--dpi`. `animate`
writes a GIF, or an MP4 when the output file ends in `.mp4` (requires
ffmpeg); `--max-frames N` animates only the first N frames, which is handy
for quick previews.

## Exposition sanity checks

One-liners matching computations in the article:

```
# Gauss's Nachlass: disjoint intervals of length 100 up to ten million.
pii intervals --range 0 10**7 10**5 -H 100 --type disjoint --save

# The overlap dataset around e^17 with H = 76, saved and displayed.
pii intervals --range "exp(17)-10**4" "exp(17)+10**4" 100 -H 76 --save

# The full pipeline on it, ending at the winners table.
pii retrieve 76 --nest --analyze --compare --winners --display --view winners

# A frame like the article's, written to a file.
pii retrieve 76 --nest --analyze --json | pii plot --from-json - -o exp17.png

# Two readme reference values.
pii overlap 0 5 5
pii prime-start "2*10**6" "3*10**6" 100 --format csv
```